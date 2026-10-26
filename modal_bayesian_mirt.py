"""Bayesian multidimensional IRT on the PAIEC benchmark-held-out protocol.

This implements the model described in the AIMS Bayesian-MIRT design note:

* item content embeddings are reduced with training-only PCA;
* linear item heads map each item embedding to a competency-loading vector
  ``alpha_i`` and an intercept ``beta_i``;
* each training task gets a Gaussian competency vector ``theta_s``;
* ``theta_s`` and the item-head coefficients are sampled jointly with a
  Metropolis-corrected HMC using a diagonal Fisher preconditioner;
* validation-task predictions integrate item-head posterior draws and the
  new task's competency posterior with Gaussian Sobol importance sampling.

The model is evaluated on the same 104 validation tasks, 10 acquisition orders,
and six label budgets as the rest of the repository.  Test labels are never
read for fitting or calibration.

Typical commands
----------------
modal run modal_bayesian_mirt.py --action inspect
modal run modal_bayesian_mirt.py --action run --arm bge_base --dims 1,2
modal run modal_bayesian_mirt.py --action run --arm bge_m3 --dims 1,2
modal run modal_bayesian_mirt.py --action run --arm me5_instruct --dims 1,2
modal run modal_bayesian_mirt.py --action compare --arm me5_instruct --dims 2

The ``smoke`` action runs a small synthetic check of the sampler and predictor
without touching the Modal volume.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import modal
import numpy as np

APP_NAME = "aims-neurips-bayesian-mirt"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDING_ROOT = REMOTE_ROOT / "datasets" / "full"
RESULT_ROOT = REMOTE_ROOT / "results" / "bayesian-mirt-v1"

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
PCA_DIM = 32
N_CHAINS = 4
N_BURN = 150
N_KEEP = 120
THIN = 1
N_LEAPFROG = 8
N_POSTERIOR_DRAWS = 48
N_SOBOL = 32
PRIOR_THETA = 1.0
PRIOR_ALPHA = 0.5
PRIOR_BETA = 0.5
PRIOR_MU = 1.0
MAP_STEPS = 120
MAP_LR = 0.02
SEED = 20261018
MIRT_GPU = os.environ.get("MIRT_GPU", "L4")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2,<3",
        "pandas>=2.2,<3",
        "pyarrow>=17,<24",
        "scipy>=1.13,<2",
        "torch==2.5.1",
    )
    .add_local_dir(PROJECT_ROOT / "src", "/opt/aims/src", copy=True)
    .env(
        {
            "PYTHONPATH": "/opt/aims/src",
            "TOKENIZERS_PARALLELISM": "false",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compact_root(arm: str) -> Path:
    """Resolve a compact store for an arm.

    Legacy v1 stores are already compact.  The v2 encoder stores are raw
    full-corpus tables, so they must first be sliced with
    ``modal_compact_store.py`` into ``allbench-train-<arm>-<cap>-v2``.
    """

    legacy = EMBEDDING_ROOT / f"allbench-embed-{arm}-v1"
    if (legacy / "train_tasks.parquet").exists():
        return legacy
    for suffix in ("cap128", "uncapped"):
        compact = EMBEDDING_ROOT / f"allbench-train-{arm}-{suffix}-v2"
        if (compact / "train_tasks.parquet").exists():
            return compact
    return legacy


def _logit(probability: float | np.ndarray) -> float | np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1.0 - 1e-8)
    return np.log(p / (1.0 - p))


def _sigmoid(value):
    import torch

    return torch.sigmoid(value)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={str(REMOTE_ROOT): volume},
    cpu=2,
    memory=8192,
    timeout=30 * 60,
)
def inspect(arm: str = "bge_base") -> dict[str, Any]:
    import pandas as pd

    root = _compact_root(arm)
    manifest = _read_json(root / "manifest.json")
    train = pd.read_parquet(root / "train_tasks.parquet")
    evaluation = pd.read_parquet(root / "eval_items.parquet")
    orders = pd.read_parquet(root / "eval_orders.parquet")
    items = pd.read_parquet(root / "items.parquet")
    subjects = pd.read_parquet(root / "subjects.parquet")

    validation = evaluation[evaluation["split"] == "validation"]
    test = evaluation[evaluation["split"] == "test"]
    order_lengths = (
        orders.groupby(["task_id", "order_seed"], sort=False).size().rename("n").reset_index()
    )
    task_subject_counts = train.groupby("task_id")["subject_index"].nunique()
    task_item_counts = train.groupby("task_id").size()
    summary = {
        "arm": arm,
        "store": str(root),
        "manifest": manifest,
        "train": {
            "rows": len(train),
            "tasks": int(train["task_id"].nunique()),
            "items": int(train["item_index"].nunique()),
            "subjects": int(train["subject_index"].nunique()),
            "labels": sorted(train["label"].unique().tolist()),
            "task_item_min": int(task_item_counts.min()),
            "task_item_median": float(task_item_counts.median()),
            "task_item_max": int(task_item_counts.max()),
            "task_subject_nunique_max": int(task_subject_counts.max()),
        },
        "evaluation": {
            "rows": len(evaluation),
            "tasks": int(evaluation["task_id"].nunique()),
            "validation_tasks": int(validation["task_id"].nunique()),
            "test_tasks": int(test["task_id"].nunique()),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "pools": {
                f"{split}|{pool}": int(count)
                for (split, pool), count in evaluation.groupby(["split", "pool"]).size().items()
            },
            "validation_benchmarks": int(validation["benchmark_id"].nunique()),
            "test_benchmarks": int(test["benchmark_id"].nunique()),
        },
        "orders": {
            "rows": len(orders),
            "tasks": int(orders["task_id"].nunique()),
            "order_seeds": sorted(orders["order_seed"].unique().tolist()),
            "length_min": int(order_lengths["n"].min()),
            "length_median": float(order_lengths["n"].median()),
            "length_max": int(order_lengths["n"].max()),
            "validation_rows": int(orders[orders["task_id"].isin(validation["task_id"])].shape[0]),
        },
        "schemas": {
            "train_tasks": {c: str(t) for c, t in train.dtypes.items()},
            "eval_items": {c: str(t) for c, t in evaluation.dtypes.items()},
            "eval_orders": {c: str(t) for c, t in orders.dtypes.items()},
            "items": {c: str(t) for c, t in items.dtypes.items()},
            "subjects": {c: str(t) for c, t in subjects.dtypes.items()},
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str), flush=True)
    return summary


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def _fit_pca(item_embeddings, train_item_indices: np.ndarray, dim: int, device):
    """Fit a training-only PCA and return all-item standardized features."""

    import torch

    torch.manual_seed(SEED)
    train_indices = torch.as_tensor(train_item_indices, dtype=torch.long, device=device)
    x_train = torch.as_tensor(
        np.asarray(item_embeddings[train_item_indices]), dtype=torch.float32, device=device
    )
    mean = x_train.mean(dim=0, keepdim=True)
    centered = x_train - mean
    q = min(dim + 8, min(centered.shape) - 1)
    _, singular, components = torch.pca_lowrank(centered, q=q, center=False, niter=4)
    components = components[:, :dim]
    z_train = centered @ components
    scale = z_train.std(dim=0, unbiased=False).clamp_min(1e-6)

    chunks = []
    chunk_size = 50_000
    for start in range(0, item_embeddings.shape[0], chunk_size):
        chunk = torch.as_tensor(
            np.asarray(item_embeddings[start : start + chunk_size]),
            dtype=torch.float32,
            device=device,
        )
        z = ((chunk - mean) @ components) / scale
        chunks.append(torch.clamp(z, -5.0, 5.0))
    z_all = torch.cat(chunks, dim=0)
    metadata = {
        "pca_dim": int(dim),
        "pca_fit_items": int(len(train_item_indices)),
        "pca_singular_values": singular[:dim].detach().cpu().tolist(),
        "pca_train_feature_std": scale.detach().cpu().tolist(),
    }
    return z_all, metadata


# ---------------------------------------------------------------------------
# Bayesian MIRT
# ---------------------------------------------------------------------------


def _unpack(vector, shapes, device):
    import torch

    blocks = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        blocks.append(vector[offset : offset + size].view(shape))
        offset += size
    return blocks


def _pack(blocks):
    import torch

    return torch.cat([block.reshape(-1) for block in blocks])


def _log_prob_blocks(
    theta,
    W_alpha,
    w_beta,
    mu,
    E_rows,
    task_index,
    labels,
    *,
    prior_theta: float,
    prior_alpha: float,
    prior_beta: float,
    prior_mu: float,
):
    import torch
    import torch.nn.functional as F

    alpha = E_rows @ W_alpha
    logits = (alpha * theta[task_index]).sum(dim=1) + (E_rows @ w_beta) + mu.squeeze()
    log_likelihood = -F.binary_cross_entropy_with_logits(
        logits, labels, reduction="sum"
    )
    log_prior = (
        -0.5 * (theta.square().sum() / (prior_theta**2))
        -0.5 * (W_alpha.square().sum() / (prior_alpha**2))
        -0.5 * (w_beta.square().sum() / (prior_beta**2))
        -0.5 * (mu.square().sum() / (prior_mu**2))
    )
    return log_likelihood + log_prior


def _make_log_prob(
    E_rows,
    task_index,
    labels,
    shapes,
    *,
    prior_theta: float,
    prior_alpha: float,
    prior_beta: float,
    prior_mu: float,
):
    def log_prob(vector):
        theta, W_alpha, w_beta, mu = _unpack(vector, shapes, vector.device)
        return _log_prob_blocks(
            theta,
            W_alpha,
            w_beta,
            mu,
            E_rows,
            task_index,
            labels,
            prior_theta=prior_theta,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            prior_mu=prior_mu,
        )

    return log_prob


def _diagonal_mass(
    theta,
    W_alpha,
    w_beta,
    mu,
    E_rows,
    task_index,
    labels,
    shapes,
    *,
    prior_theta: float,
    prior_alpha: float,
    prior_beta: float,
    prior_mu: float,
):
    """Diagonal Fisher information at the current parameters."""

    import torch

    alpha = E_rows @ W_alpha
    logits = (alpha * theta[task_index]).sum(dim=1) + (E_rows @ w_beta) + mu.squeeze()
    probability = torch.sigmoid(logits)
    weight = probability * (1.0 - probability)
    n_tasks, dim = theta.shape

    mass_theta = torch.full_like(theta, 1.0 / (prior_theta**2))
    for d in range(dim):
        contribution = weight * alpha[:, d].square()
        mass_theta[:, d].index_add_(0, task_index, contribution)

    E2 = E_rows.square()
    mass_alpha = torch.full_like(W_alpha, 1.0 / (prior_alpha**2))
    for d in range(dim):
        mass_alpha[:, d] += E2.transpose(0, 1) @ (
            weight * theta[task_index, d].square()
        )
    mass_beta = torch.full_like(w_beta, 1.0 / (prior_beta**2)) + E2.transpose(0, 1) @ weight
    mass_mu = torch.full_like(mu, 1.0 / (prior_mu**2)) + weight.sum().reshape(1)
    return _pack([mass_theta, mass_alpha, mass_beta, mass_mu]).clamp_min(1e-3)


def _map_fit(E_rows, task_index, labels, shapes, device, *, n_tasks: int, dim: int):
    import torch

    torch.manual_seed(SEED)
    # Break the theta / loading symmetry explicitly.  Starting both at zero
    # leaves the interaction gradient exactly zero for every gradient step.
    theta = (0.01 * torch.randn(n_tasks, dim, device=device)).requires_grad_(True)
    W_alpha = (
        0.01 * torch.randn(E_rows.shape[1], dim, device=device)
    ).requires_grad_(True)
    w_beta = torch.zeros(E_rows.shape[1], device=device, requires_grad=True)
    base_rate = float(labels.mean().clamp(1e-4, 1.0 - 1e-4))
    mu = torch.tensor(
        [float(np.log(base_rate / (1.0 - base_rate)))],
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([theta, W_alpha, w_beta, mu], lr=MAP_LR)
    for _ in range(MAP_STEPS):
        optimizer.zero_grad(set_to_none=True)
        log_prob = _log_prob_blocks(
            theta,
            W_alpha,
            w_beta,
            mu,
            E_rows,
            task_index,
            labels,
            prior_theta=PRIOR_THETA,
            prior_alpha=PRIOR_ALPHA,
            prior_beta=PRIOR_BETA,
            prior_mu=PRIOR_MU,
        )
        (-log_prob).backward()
        optimizer.step()
    return [theta.detach(), W_alpha.detach(), w_beta.detach(), mu.detach()]


def _hmc_chain(
    q_init,
    mass,
    log_prob,
    *,
    n_burn: int,
    n_keep: int,
    thin: int,
    n_leapfrog: int,
    seed: int,
    device,
):
    import torch

    torch.manual_seed(seed)
    q = q_init.detach().clone()
    mass = mass.detach().clone()
    step_size = 0.1

    def evaluate(vector):
        vector = vector.detach().requires_grad_(True)
        logp = log_prob(vector)
        gradient = torch.autograd.grad(logp, vector)[0]
        return logp.detach(), gradient.detach()

    current_logp, current_gradient = evaluate(q)
    draws = []
    accepted = []
    for step in range(n_burn + n_keep * thin):
        momentum = torch.sqrt(mass) * torch.randn_like(q)
        current_kinetic = 0.5 * (momentum.square() / mass).sum()
        q_prop = q.clone()
        p_prop = momentum.clone()
        g_prop = current_gradient.clone()
        p_prop = p_prop + 0.5 * step_size * g_prop
        for leap in range(n_leapfrog):
            q_prop = q_prop + step_size * p_prop / mass
            logp_prop, g_prop = evaluate(q_prop)
            if leap < n_leapfrog - 1:
                p_prop = p_prop + step_size * g_prop
            else:
                p_prop = p_prop + 0.5 * step_size * g_prop
        proposal_kinetic = 0.5 * (p_prop.square() / mass).sum()
        log_accept = (logp_prop - current_logp) - (proposal_kinetic - current_kinetic)
        accept = bool(torch.log(torch.rand((), device=device)) < log_accept)
        if accept:
            q = q_prop
            current_logp = logp_prop
            current_gradient = g_prop
        accepted.append(float(accept))

        if step < n_burn and (step + 1) % 25 == 0:
            rate = float(np.mean(accepted[-25:]))
            if rate < 0.6:
                step_size *= 0.8
            elif rate > 0.9:
                step_size *= 1.2
            step_size = float(min(max(step_size, 1e-5), 10.0))
        if step >= n_burn and (step - n_burn) % thin == 0:
            draws.append(q.detach().cpu().clone())
    return torch.stack(draws), accepted, step_size


def _split_rhat(chains):
    """Split R-hat for an array shaped (chains, draws, parameters)."""

    import torch

    n_chains, n_draws, n_params = chains.shape
    half = n_draws // 2
    split = torch.cat([chains[:, :half], chains[:, half : 2 * half]], dim=0)
    means = split.mean(dim=1)
    variances = split.var(dim=1, unbiased=True)
    within = variances.mean(dim=0)
    between = half * means.var(dim=0, unbiased=True)
    variance = ((half - 1) / half) * within + between / half
    # Float32 cancellation can make an exactly constant parameter's variance
    # a tiny negative number; clamp before the square root.
    return torch.sqrt(variance.clamp_min(0.0) / within.clamp_min(1e-12))


def _fit_bayesian_mirt(
    z_all,
    train,
    *,
    dim: int,
    device,
    n_chains: int = N_CHAINS,
):
    import torch

    torch.manual_seed(SEED)
    task_ids = sorted(train["task_id"].astype(str).unique())
    task_map = {task_id: index for index, task_id in enumerate(task_ids)}
    task_index = torch.as_tensor(
        train["task_id"].astype(str).map(task_map).to_numpy(dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    item_index = torch.as_tensor(
        train["item_index"].to_numpy(dtype=np.int64), dtype=torch.long, device=device
    )
    labels = torch.as_tensor(
        train["label"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device
    )
    E_rows = z_all[item_index]
    n_tasks = len(task_ids)
    shapes = [
        (n_tasks, dim),
        (E_rows.shape[1], dim),
        (E_rows.shape[1],),
        (1,),
    ]
    initial = _map_fit(
        E_rows, task_index, labels, shapes, device, n_tasks=n_tasks, dim=dim
    )
    mass = _diagonal_mass(
        *initial,
        E_rows,
        task_index,
        labels,
        shapes,
        prior_theta=PRIOR_THETA,
        prior_alpha=PRIOR_ALPHA,
        prior_beta=PRIOR_BETA,
        prior_mu=PRIOR_MU,
    )
    q_map = _pack(initial)
    log_prob = _make_log_prob(
        E_rows,
        task_index,
        labels,
        shapes,
        prior_theta=PRIOR_THETA,
        prior_alpha=PRIOR_ALPHA,
        prior_beta=PRIOR_BETA,
        prior_mu=PRIOR_MU,
    )
    chain_draws = []
    chain_acceptance = []
    chain_steps = []
    for chain in range(n_chains):
        jitter = 0.1 * torch.randn_like(q_map) / torch.sqrt(mass)
        draws, accepted, step_size = _hmc_chain(
            q_map + jitter,
            mass,
            log_prob,
            n_burn=N_BURN,
            n_keep=N_KEEP,
            thin=THIN,
            n_leapfrog=N_LEAPFROG,
            seed=SEED + chain,
            device=device,
        )
        chain_draws.append(draws)
        chain_acceptance.append(float(np.mean(accepted[N_BURN:])))
        chain_steps.append(step_size)
        print(
            f"dim={dim} chain={chain} accept={chain_acceptance[-1]:.3f} "
            f"step={step_size:.4g}",
            flush=True,
        )
    draws = torch.stack(chain_draws, dim=0)
    rhat = _split_rhat(draws)
    diagnostics = {
        "dim": int(dim),
        "n_tasks": int(n_tasks),
        "n_rows": int(len(train)),
        "n_parameters": int(draws.shape[-1]),
        "n_chains": int(n_chains),
        "n_keep": int(N_KEEP),
        "thin": int(THIN),
        "n_leapfrog": int(N_LEAPFROG),
        "acceptance": chain_acceptance,
        "final_step_size": chain_steps,
        "rhat_raw_max": float(rhat.max()),
        "rhat_raw_median": float(rhat.median()),
        "rhat_blocks": {},
    }
    offset = 0
    for name, shape in zip(
        ("theta", "alpha", "beta", "mu"), shapes, strict=True
    ):
        size = int(np.prod(shape))
        diagnostics["rhat_blocks"][name] = float(rhat[offset : offset + size].max())
        offset += size
    return draws, shapes, mass, diagnostics


# ---------------------------------------------------------------------------
# Posterior predictive integration
# ---------------------------------------------------------------------------


def _sobol_normal(n: int, dim: int, seed: int):
    import torch
    from scipy.stats import qmc
    from scipy.special import ndtri

    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    points = sampler.random(n)
    return torch.as_tensor(ndtri(points), dtype=torch.float32)


def _predict_context(
    alpha_target,
    beta_target,
    alpha_context,
    beta_context,
    labels_context,
    *,
    n_sobol: int,
    seed: int,
    device,
):
    """Posterior predictive for one task/order/budget.

    ``alpha_target`` is ``(draws, target_items, dim)``; ``beta_target`` is
    ``(draws, target_items)``.  Context tensors use the same draw axis.
    """

    import torch
    import torch.nn.functional as F

    n_draws, n_target, dim = alpha_target.shape
    n_context = alpha_context.shape[1]
    if n_context == 0:
        mean = torch.zeros(n_draws, dim, device=device)
        covariance = torch.eye(dim, device=device).expand(n_draws, dim, dim)
    else:
        mean = torch.zeros(n_draws, dim, device=device)
        for _ in range(12):
            context_logits = torch.einsum(
                "snd,sd->sn", alpha_context, mean
            ) + beta_context
            probability = torch.sigmoid(context_logits)
            gradient = torch.einsum(
                "snd,sn->sd", alpha_context, labels_context[None, :] - probability
            ) - mean
            weight = probability * (1.0 - probability)
            precision = torch.eye(dim, device=device).expand(n_draws, dim, dim).clone()
            precision += torch.einsum(
                "snd,sne,sn->sde", alpha_context, alpha_context, weight
            )
            step = torch.linalg.solve(precision, gradient.unsqueeze(-1)).squeeze(-1)
            mean = mean + step
            if float(step.abs().max()) < 1e-6:
                break
        precision = torch.eye(dim, device=device).expand(n_draws, dim, dim).clone()
        precision += torch.einsum(
            "snd,sne,sn->sde", alpha_context, alpha_context, weight
        )
        covariance = torch.linalg.inv(precision)
        covariance = 0.5 * (covariance + covariance.transpose(1, 2))
        covariance = covariance + 1e-6 * torch.eye(dim, device=device)

    sobol = _sobol_normal(n_sobol, dim, seed).to(device)
    chol = torch.linalg.cholesky(covariance)
    theta = mean[:, None, :] + torch.einsum("sde,qe->sqd", chol, sobol)

    if n_context == 0:
        log_weight = torch.zeros(n_draws, n_sobol, device=device)
    else:
        context_logits = torch.einsum(
            "snd,sqd->sqn", alpha_context, theta
        ) + beta_context[:, None, :]
        log_likelihood = (
            labels_context[None, None, :] * F.logsigmoid(context_logits)
            + (1.0 - labels_context[None, None, :]) * F.logsigmoid(-context_logits)
        ).sum(dim=-1)
        log_prior = -0.5 * theta.square().sum(dim=-1)
        log_proposal = (
            -0.5 * sobol.square().sum(dim=-1)[None, :]
            - torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(dim=-1)[:, None]
        )
        log_weight = log_likelihood + log_prior - log_proposal
    weight = torch.softmax(log_weight, dim=1)

    target_logits = torch.einsum(
        "snd,sqd->sqn", alpha_target, theta
    ) + beta_target[:, None, :]
    target_probability = torch.sigmoid(target_logits)
    per_draw = (weight[:, :, None] * target_probability).sum(dim=1)
    return per_draw.mean(dim=0), per_draw


# ---------------------------------------------------------------------------
# Evaluation and metrics
# ---------------------------------------------------------------------------


def _fit_calibrators(records: list[dict[str, Any]]):
    from aims_starter.calibration import AnchorCalibrator

    rng = np.random.default_rng(20261020)
    output: dict[str, Any] = {}
    arms = sorted({record["arm"] for record in records if record["arm"] != "beta"})
    for arm in arms:
        samples = [
            record
            for record in records
            if record["arm"] == arm and record["split"] == "validation"
        ]
        folds = sorted({int(record["fold"]) for record in samples})
        output[arm] = {
            "full": _fit_one_calibrator(samples, rng),
            "folds": {
                fold: _fit_one_calibrator(
                    [record for record in samples if int(record["fold"]) != fold], rng
                )
                for fold in folds
            },
        }
    return output


def _fit_one_calibrator(samples: list[dict[str, Any]], rng):
    from aims_starter.calibration import AnchorCalibrator

    logits = np.concatenate([np.asarray(s["logits"], dtype=float) for s in samples])
    labels = np.concatenate([np.asarray(s["labels"], dtype=float) for s in samples])
    rates = np.concatenate(
        [
            np.full(len(s["labels"]), (1.0 + float(np.sum(s["context_labels"]))) / (2.0 + len(s["context_labels"])))
            for s in samples
        ]
    )
    budgets = np.concatenate(
        [np.full(len(s["labels"]), int(s["budget"]), dtype=int) for s in samples]
    )
    if len(labels) > 1_000_000:
        index = rng.choice(len(labels), size=1_000_000, replace=False)
        logits, labels, rates, budgets = (
            logits[index],
            labels[index],
            rates[index],
            budgets[index],
        )
    return AnchorCalibrator.fit(logits, labels, rates, budgets)


def _metric_rows(records: list[dict[str, Any]], calibrators):
    from aims_starter.faithful_np import binary_auc
    from aims_starter.metrics import calibration_ece

    rows = []
    for record in records:
        labels = np.asarray(record["labels"], dtype=float)
        logits = np.asarray(record["logits"], dtype=float)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        arm = record["arm"]
        if arm == "beta":
            calibrated = probability
        else:
            key = int(record["fold"]) if record["split"] == "validation" else "full"
            calibrator = (
                calibrators[arm]["folds"][key]
                if key != "full"
                else calibrators[arm]["full"]
            )
            calibrated = calibrator.transform(
                logits, record["budget"], record["context_labels"]
            )
        rows.append(
            {
                "arm": arm,
                "split": record["split"],
                "fold": int(record["fold"]),
                "benchmark_id": record["benchmark_id"],
                "task_id": record["task_id"],
                "order_seed": int(record["order_seed"]),
                "budget": int(record["budget"]),
                "brier_raw": float(np.mean((probability - labels) ** 2)),
                "brier_cal": float(np.mean((calibrated - labels) ** 2)),
                "ece_raw": float(calibration_ece(labels.tolist(), probability.tolist())),
                "ece_cal": float(
                    calibration_ece(labels.tolist(), calibrated.tolist())
                ),
                "auc_raw": float(binary_auc(labels, probability)),
                "auc_cal": float(binary_auc(labels, calibrated)),
            }
        )
    return rows


def _summarize(metric_rows: list[dict[str, Any]]):
    import pandas as pd

    frame = pd.DataFrame(metric_rows)
    weight = dict(zip(BUDGETS, WEIGHTS, strict=True))
    frame["weighted_raw"] = frame["budget"].map(weight) * frame["brier_raw"]
    frame["weighted_cal"] = frame["budget"].map(weight) * frame["brier_cal"]

    by_budget = []
    for (arm, split, budget), group in frame.groupby(
        ["arm", "split", "budget"], sort=True
    ):
        task_means = group.groupby("task_id", sort=False)[
            ["brier_raw", "brier_cal", "ece_raw", "ece_cal", "auc_raw", "auc_cal"]
        ].mean()
        by_budget.append(
            {
                "arm": arm,
                "split": split,
                "budget": int(budget),
                "brier_raw": float(task_means["brier_raw"].mean()),
                "brier_cal": float(task_means["brier_cal"].mean()),
                "ece_raw": float(task_means["ece_raw"].mean()),
                "ece_cal": float(task_means["ece_cal"].mean()),
                "auc_raw": float(task_means["auc_raw"].mean()),
                "auc_cal": float(task_means["auc_cal"].mean()),
                "tasks": len(task_means),
            }
        )

    task_alc = (
        frame.groupby(["arm", "split", "task_id", "order_seed"], sort=False)[
            ["weighted_raw", "weighted_cal"]
        ]
        .sum()
        .reset_index()
        .groupby(["arm", "split", "task_id"], sort=False)[
            ["weighted_raw", "weighted_cal"]
        ]
        .mean()
        .reset_index()
    )
    alc = []
    for (arm, split), group in task_alc.groupby(["arm", "split"], sort=True):
        alc.append(
            {
                "arm": arm,
                "split": split,
                "brier_alc_raw": float(group["weighted_raw"].mean()),
                "brier_alc_cal": float(group["weighted_cal"].mean()),
                "brier_alc_raw_se": float(
                    group["weighted_raw"].std(ddof=1) / np.sqrt(len(group))
                ),
                "brier_alc_cal_se": float(
                    group["weighted_cal"].std(ddof=1) / np.sqrt(len(group))
                ),
                "tasks": len(group),
            }
        )
    return by_budget, alc, task_alc.to_dict(orient="records")


def _paired(task_alc: list[dict[str, Any]], split: str, first: str, second: str):
    from scipy import stats

    rows = [row for row in task_alc if row["split"] == split]
    first_rows = {row["task_id"]: row for row in rows if row["arm"] == first}
    second_rows = {row["task_id"]: row for row in rows if row["arm"] == second}
    tasks = sorted(set(first_rows) & set(second_rows))
    output = {"split": split, "first": first, "second": second, "tasks": len(tasks)}
    for metric in ("weighted_raw", "weighted_cal"):
        delta = np.asarray(
            [first_rows[task][metric] - second_rows[task][metric] for task in tasks],
            dtype=float,
        )
        mean = float(delta.mean())
        se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
        critical = float(stats.t.ppf(0.975, len(delta) - 1))
        output[metric] = {
            "mean": mean,
            "se": se,
            "ci_low": mean - critical * se,
            "ci_high": mean + critical * se,
            "wins": int((delta < 0.0).sum()),
            "tasks": len(delta),
        }
    return output


def _bootstrap_ci(
    delta: np.ndarray, n_bootstrap: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(delta)
    if n == 0:
        return float("nan"), float("nan")
    index = rng.integers(0, n, size=(n_bootstrap, n))
    means = delta[index].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _bootstrap_paired(
    task_alc: list[dict[str, Any]],
    split: str,
    first: str,
    second: str,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20261025,
):
    """Task-level paired bootstrap for ALC deltas.

    The t interval in ``_paired`` is the primary interval.  This bootstrap is
    a distribution-free sensitivity check over the same 104 tasks.
    """

    rows = [row for row in task_alc if row["split"] == split]
    first_rows = {row["task_id"]: row for row in rows if row["arm"] == first}
    second_rows = {row["task_id"]: row for row in rows if row["arm"] == second}
    tasks = sorted(set(first_rows) & set(second_rows))
    output = {
        "split": split,
        "first": first,
        "second": second,
        "tasks": len(tasks),
        "n_bootstrap": n_bootstrap,
    }
    for metric in ("weighted_raw", "weighted_cal"):
        delta = np.asarray(
            [first_rows[task][metric] - second_rows[task][metric] for task in tasks],
            dtype=float,
        )
        ci_low, ci_high = _bootstrap_ci(delta, n_bootstrap, seed)
        output[metric] = {
            "mean": float(delta.mean()),
            "se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "wins": int((delta < 0.0).sum()),
            "tasks": len(delta),
        }
    return output


def _per_budget_paired(
    metric_rows: list[dict[str, Any]],
    split: str,
    first: str,
    second: str,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20261025,
):
    """Paired per-budget Brier deltas, averaged over acquisition orders."""

    import pandas as pd

    frame = pd.DataFrame(metric_rows)
    frame = frame[frame["split"] == split]
    task_budget = (
        frame.groupby(["arm", "task_id", "budget"], sort=False)[
            ["brier_raw", "brier_cal"]
        ]
        .mean()
        .reset_index()
    )
    first_rows = task_budget[task_budget["arm"] == first].set_index(
        ["task_id", "budget"]
    )
    second_rows = task_budget[task_budget["arm"] == second].set_index(
        ["task_id", "budget"]
    )
    output = []
    for budget in BUDGETS:
        first_budget = first_rows.xs(budget, level="budget")
        second_budget = second_rows.xs(budget, level="budget")
        tasks = sorted(first_budget.index.intersection(second_budget.index))
        row = {
            "first": first,
            "second": second,
            "split": split,
            "budget": int(budget),
            "tasks": len(tasks),
        }
        for metric in ("brier_raw", "brier_cal"):
            delta = (
                first_budget.loc[tasks, metric] - second_budget.loc[tasks, metric]
            ).to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap_ci(delta, n_bootstrap, seed + budget)
            row[metric] = {
                "first": float(first_budget.loc[tasks, metric].mean()),
                "second": float(second_budget.loc[tasks, metric].mean()),
                "delta": float(delta.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "wins": int((delta < 0.0).sum()),
            }
        output.append(row)
    return output


def _evaluate_arm(
    z_all,
    draws,
    shapes,
    *,
    arm: str,
    model_name: str,
    dim: int,
    validation,
    orders,
    fold_map: dict[str, int],
    device,
):
    import torch

    draws = draws.reshape(-1, draws.shape[-1])
    index = torch.linspace(0, draws.shape[0] - 1, N_POSTERIOR_DRAWS).long()
    selected = draws[index].to(device)
    blocks = [_unpack(row, shapes, device) for row in selected]
    W_alpha = torch.stack([block[1] for block in blocks], dim=0)
    w_beta = torch.stack([block[2] for block in blocks], dim=0)
    mu = torch.stack([block[3].squeeze() for block in blocks], dim=0)

    records: list[dict[str, Any]] = []
    task_ids = sorted(validation["task_id"].astype(str).unique())
    for task_id in task_ids:
        task_rows = validation[validation["task_id"].astype(str) == task_id]
        benchmark_id = str(task_rows["benchmark_id"].iloc[0])
        target_rows = task_rows[task_rows["pool"] == "evaluation"].reset_index(drop=True)
        target_item_index = torch.as_tensor(
            target_rows["item_index"].to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        target_embedding = z_all[target_item_index]
        target_labels = target_rows["label"].to_numpy(dtype=np.float32)
        alpha_target = torch.einsum("nm,smd->snd", target_embedding, W_alpha)
        beta_target = torch.einsum("nm,sm->sn", target_embedding, w_beta) + mu[:, None]

        lookup = task_rows.drop_duplicates("item_index").set_index("item_index")["label"]
        task_orders = orders[orders["task_id"].astype(str) == task_id]
        for order_seed in range(10):
            order_rows = (
                task_orders[task_orders["order_seed"] == order_seed]
                .sort_values("acquisition_rank")
                .reset_index(drop=True)
            )
            if len(order_rows) == 0:
                continue
            context_item_index = torch.as_tensor(
                order_rows["item_index"].to_numpy(dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            context_embedding = z_all[context_item_index]
            context_labels = np.asarray(
                lookup.loc[order_rows["item_index"].to_numpy()], dtype=np.float32
            )
            alpha_context = torch.einsum("nm,smd->snd", context_embedding, W_alpha)
            beta_context = (
                torch.einsum("nm,sm->sn", context_embedding, w_beta) + mu[:, None]
            )
            for budget in BUDGETS:
                y_context = torch.as_tensor(
                    context_labels[:budget], dtype=torch.float32, device=device
                )
                probability, _ = _predict_context(
                    alpha_target,
                    beta_target,
                    alpha_context[:, :budget],
                    beta_context[:, :budget],
                    y_context,
                    n_sobol=N_SOBOL,
                    seed=SEED + int(task_id[-6:]) if task_id[-6:].isdigit() else SEED,
                    device=device,
                )
                records.append(
                    {
                        "arm": model_name,
                        "split": "validation",
                        "benchmark_id": benchmark_id,
                        "task_id": str(task_id),
                        "order_seed": int(order_seed),
                        "fold": int(fold_map.get(str(task_id), -1)),
                        "budget": int(budget),
                        "logits": np.asarray(_logit(probability.detach().cpu().numpy())),
                        "labels": np.asarray(target_labels, dtype=float),
                        "context_labels": np.asarray(context_labels[:budget], dtype=float),
                    }
                )
        print(f"{arm} {model_name} scored {task_id}", flush=True)
    return records


def _evaluate_beta(
    *,
    arm: str,
    validation,
    orders,
    fold_map: dict[str, int],
):
    records: list[dict[str, Any]] = []
    task_ids = sorted(validation["task_id"].astype(str).unique())
    for task_id in task_ids:
        task_rows = validation[validation["task_id"].astype(str) == task_id]
        benchmark_id = str(task_rows["benchmark_id"].iloc[0])
        target_rows = task_rows[task_rows["pool"] == "evaluation"].reset_index(drop=True)
        target_labels = target_rows["label"].to_numpy(dtype=np.float32)
        lookup = task_rows.drop_duplicates("item_index").set_index("item_index")["label"]
        task_orders = orders[orders["task_id"].astype(str) == task_id]
        for order_seed in range(10):
            order_rows = (
                task_orders[task_orders["order_seed"] == order_seed]
                .sort_values("acquisition_rank")
                .reset_index(drop=True)
            )
            if len(order_rows) == 0:
                continue
            context_labels = np.asarray(
                lookup.loc[order_rows["item_index"].to_numpy()], dtype=np.float32
            )
            for budget in BUDGETS:
                probability = (2.0 + float(context_labels[:budget].sum())) / (4.0 + budget)
                records.append(
                    {
                        "arm": "beta",
                        "split": "validation",
                        "benchmark_id": benchmark_id,
                        "task_id": str(task_id),
                        "order_seed": int(order_seed),
                        "fold": int(fold_map.get(str(task_id), -1)),
                        "budget": int(budget),
                        "logits": np.full(len(target_labels), float(_logit(probability))),
                        "labels": np.asarray(target_labels, dtype=float),
                        "context_labels": np.asarray(context_labels[:budget], dtype=float),
                    }
                )
    return records


# ---------------------------------------------------------------------------
# Modal entry points
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=MIRT_GPU,
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=12 * 60 * 60,
)
def run(arm: str = "bge_base", dims: str = "1,2") -> dict[str, Any]:
    import pandas as pd
    import torch

    from aims_starter.uniform_eval import validation_fold_map

    root = _compact_root(arm)
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(f"missing compact store for {arm}: {root}")
    dim_values = tuple(sorted({int(value) for value in dims.split(",") if value.strip()}))
    if not dim_values or any(value < 1 for value in dim_values):
        raise ValueError("dims must be a comma-separated list of positive integers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {arm} on {device}", flush=True)
    train = pd.read_parquet(root / "train_tasks.parquet")
    evaluation = pd.read_parquet(root / "eval_items.parquet")
    orders = pd.read_parquet(root / "eval_orders.parquet")
    item_embeddings = np.load(root / "item_embeddings.npy", mmap_mode="r")
    train_item_indices = np.unique(train["item_index"].to_numpy(dtype=np.int64))
    z_all, pca_metadata = _fit_pca(
        item_embeddings, train_item_indices, PCA_DIM, device
    )
    print(f"PCA ready: {pca_metadata['pca_fit_items']} training items", flush=True)

    validation = evaluation[evaluation["split"] == "validation"].copy()
    validation["task_id"] = validation["task_id"].astype(str)
    orders = orders.copy()
    orders["task_id"] = orders["task_id"].astype(str)
    fold_map = validation_fold_map(validation, n_folds=5)

    all_records: list[dict[str, Any]] = []
    all_diagnostics: dict[str, Any] = {}
    all_records.extend(
        _evaluate_beta(
            arm=arm,
            validation=validation,
            orders=orders,
            fold_map=fold_map,
        )
    )
    for dim in dim_values:
        model_name = f"mirt{dim}"
        print(f"fitting {arm} {model_name}", flush=True)
        draws, shapes, mass, diagnostics = _fit_bayesian_mirt(
            z_all, train, dim=dim, device=device
        )
        all_diagnostics[model_name] = diagnostics
        draw_path = RESULT_ROOT / f"draws-{arm}-d{dim}.npz"
        draw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            draw_path,
            draws=draws.cpu().numpy(),
            mass=mass.cpu().numpy(),
            shapes_json=np.asarray(json.dumps([list(shape) for shape in shapes])),
        )
        print(f"sampling done for {arm} {model_name}", flush=True)
        all_records.extend(
            _evaluate_arm(
                z_all,
                draws,
                shapes,
                arm=arm,
                model_name=model_name,
                dim=dim,
                validation=validation,
                orders=orders,
                fold_map=fold_map,
                device=device,
            )
        )

    calibrators = _fit_calibrators(all_records)
    metric_rows = _metric_rows(all_records, calibrators)
    by_budget, alc, task_alc = _summarize(metric_rows)
    paired = []
    if len(dim_values) > 1:
        for first, second in (
            (f"mirt{max(dim_values)}", f"mirt{min(dim_values)}"),
            (f"mirt{max(dim_values)}", "beta"),
            (f"mirt{min(dim_values)}", "beta"),
        ):
            paired.append(_paired(task_alc, "validation", first, second))
    else:
        paired.append(_paired(task_alc, "validation", f"mirt{dim_values[0]}", "beta"))

    result = {
        "status": "bayesian_mirt_validation",
        "arm": arm,
        "dims": list(dim_values),
        "orders": 10,
        "split": "validation",
        "pca": pca_metadata,
        "diagnostics": all_diagnostics,
        "by_budget": by_budget,
        "alc": alc,
        "paired": paired,
        "fits": {
            key: {
                "full": value["full"].summary(),
                "folds": {
                    str(fold): calibrator.summary()
                    for fold, calibrator in value["folds"].items()
                },
            }
            for key, value in calibrators.items()
        },
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_ROOT / f"validation-{arm}.json"
    _write_json(output_path, result)
    pd.DataFrame(metric_rows).to_parquet(
        output_path.with_suffix(".rows.parquet"), index=False
    )
    volume.commit()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


@app.function(
    image=image,
    gpu=MIRT_GPU,
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=3 * 60 * 60,
)
def compare(
    arm: str = "me5_instruct",
    dims: str = "2",
    n_bootstrap: int = 10_000,
) -> dict[str, Any]:
    """Score Beta-Binomial and saved MIRT draws on identical tasks.

    Beta-Binomial is embedding-free: it only consumes the revealed context
    labels.  This action recomputes both predictors from the same compact
    store, validation tasks, acquisition orders, budgets, and folds, then
    reports paired task-level Brier ALC deltas with t and bootstrap intervals.
    """

    import pandas as pd
    import torch

    from aims_starter.uniform_eval import validation_fold_map

    root = _compact_root(arm)
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(f"missing compact store for {arm}: {root}")
    dim_values = tuple(sorted({int(value) for value in dims.split(",") if value.strip()}))
    if not dim_values or any(value < 1 for value in dim_values):
        raise ValueError("dims must be a comma-separated list of positive integers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {arm} on {device}", flush=True)
    train = pd.read_parquet(root / "train_tasks.parquet")
    evaluation = pd.read_parquet(root / "eval_items.parquet")
    orders = pd.read_parquet(root / "eval_orders.parquet")
    item_embeddings = np.load(root / "item_embeddings.npy", mmap_mode="r")
    train_item_indices = np.unique(train["item_index"].to_numpy(dtype=np.int64))
    z_all, pca_metadata = _fit_pca(
        item_embeddings, train_item_indices, PCA_DIM, device
    )
    print(f"PCA ready: {pca_metadata['pca_fit_items']} training items", flush=True)

    validation = evaluation[evaluation["split"] == "validation"].copy()
    validation["task_id"] = validation["task_id"].astype(str)
    orders = orders.copy()
    orders["task_id"] = orders["task_id"].astype(str)
    fold_map = validation_fold_map(validation, n_folds=5)

    all_records: list[dict[str, Any]] = []
    all_records.extend(
        _evaluate_beta(
            arm=arm,
            validation=validation,
            orders=orders,
            fold_map=fold_map,
        )
    )
    draw_metadata: dict[str, Any] = {}
    for dim in dim_values:
        draw_path = RESULT_ROOT / f"draws-{arm}-d{dim}.npz"
        if not draw_path.exists():
            raise FileNotFoundError(f"missing posterior draws: {draw_path}")
        payload = np.load(draw_path, allow_pickle=False)
        draws = torch.as_tensor(
            payload["draws"], dtype=torch.float32, device=device
        )
        shapes = [
            tuple(int(x) for x in shape)
            for shape in json.loads(str(payload["shapes_json"].item()))
        ]
        draw_metadata[f"mirt{dim}"] = {
            "path": str(draw_path),
            "chains": int(draws.shape[0]),
            "draws": int(draws.shape[1]),
            "parameters": int(draws.shape[2]),
        }
        all_records.extend(
            _evaluate_arm(
                z_all,
                draws,
                shapes,
                arm=arm,
                model_name=f"mirt{dim}",
                dim=dim,
                validation=validation,
                orders=orders,
                fold_map=fold_map,
                device=device,
            )
        )

    calibrators = _fit_calibrators(all_records)
    metric_rows = _metric_rows(all_records, calibrators)
    by_budget, alc, task_alc = _summarize(metric_rows)

    pairs = [(f"mirt{dim}", "beta") for dim in dim_values]
    if len(dim_values) > 1:
        pairs.append((f"mirt{max(dim_values)}", f"mirt{min(dim_values)}"))
    paired_t = [_paired(task_alc, "validation", first, second) for first, second in pairs]
    paired_bootstrap = [
        _bootstrap_paired(
            task_alc,
            "validation",
            first,
            second,
            n_bootstrap=n_bootstrap,
            seed=SEED,
        )
        for first, second in pairs
    ]
    per_budget_paired = []
    for first, second in pairs:
        per_budget_paired.extend(
            _per_budget_paired(
                metric_rows,
                "validation",
                first,
                second,
                n_bootstrap=n_bootstrap,
                seed=SEED,
            )
        )

    order0_rows = [row for row in metric_rows if row["order_seed"] == 0]
    order0_by_budget, order0_alc, order0_task_alc = _summarize(order0_rows)
    order0_paired_t = [
        _paired(order0_task_alc, "validation", first, second)
        for first, second in pairs
    ]
    order0_paired_bootstrap = [
        _bootstrap_paired(
            order0_task_alc,
            "validation",
            first,
            second,
            n_bootstrap=n_bootstrap,
            seed=SEED,
        )
        for first, second in pairs
    ]
    order0_per_budget_paired = []
    for first, second in pairs:
        order0_per_budget_paired.extend(
            _per_budget_paired(
                order0_rows,
                "validation",
                first,
                second,
                n_bootstrap=n_bootstrap,
                seed=SEED,
            )
        )

    result = {
        "status": "mirt_beta_comparison",
        "arm": arm,
        "store": str(root),
        "manifest": _read_json(root / "manifest.json"),
        "dims": list(dim_values),
        "orders": 10,
        "split": "validation",
        "tasks": int(validation["task_id"].nunique()),
        "n_bootstrap": int(n_bootstrap),
        "pca": pca_metadata,
        "draws": draw_metadata,
        "by_budget": by_budget,
        "alc": alc,
        "paired_t": paired_t,
        "paired_bootstrap": paired_bootstrap,
        "per_budget_paired": per_budget_paired,
        "order0": {
            "by_budget": order0_by_budget,
            "alc": order0_alc,
            "paired_t": order0_paired_t,
            "paired_bootstrap": order0_paired_bootstrap,
            "per_budget_paired": order0_per_budget_paired,
        },
        "fits": {
            key: {
                "full": value["full"].summary(),
                "folds": {
                    str(fold): calibrator.summary()
                    for fold, calibrator in value["folds"].items()
                },
            }
            for key, value in calibrators.items()
        },
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_ROOT / f"comparison-{arm}.json"
    _write_json(output_path, result)
    pd.DataFrame(metric_rows).to_parquet(
        output_path.with_suffix(".rows.parquet"), index=False
    )
    volume.commit()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


# ---------------------------------------------------------------------------
# Prediction-level convergence diagnostics
# ---------------------------------------------------------------------------

RHAT_DRAWS = 40
RHAT_SOBOL = 16


@app.function(
    image=image,
    gpu=MIRT_GPU,
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=3 * 60 * 60,
)
def prediction_rhat(
    arm: str = "bge_base",
    dims: str = "1,2",
    max_tasks: int = 0,
    order_seeds: str = "0",
) -> dict[str, Any]:
    """Split R-hat on per-draw target probabilities, by chain.

    Raw-parameter R-hat is sensitive to MIRT sign/rotation symmetry.  This
    diagnostic asks the question that matters for deployment: do the chains
    agree on the probability assigned to the same target item, for the same
    task/order/budget?  The same Sobol points are used for every chain so the
    check isolates posterior variation rather than integration noise.
    """

    import pandas as pd
    import torch

    root = _compact_root(arm)
    dim_values = tuple(sorted({int(value) for value in dims.split(",") if value.strip()}))
    order_values = tuple(
        sorted({int(value) for value in order_seeds.split(",") if value.strip()})
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {arm} on {device}", flush=True)

    train = pd.read_parquet(root / "train_tasks.parquet")
    evaluation = pd.read_parquet(root / "eval_items.parquet")
    orders = pd.read_parquet(root / "eval_orders.parquet")
    item_embeddings = np.load(root / "item_embeddings.npy", mmap_mode="r")
    train_item_indices = np.unique(train["item_index"].to_numpy(dtype=np.int64))
    z_all, pca_metadata = _fit_pca(
        item_embeddings, train_item_indices, PCA_DIM, device
    )
    print(f"PCA ready: {pca_metadata['pca_fit_items']} training items", flush=True)

    validation = evaluation[evaluation["split"] == "validation"].copy()
    validation["task_id"] = validation["task_id"].astype(str)
    orders = orders.copy()
    orders["task_id"] = orders["task_id"].astype(str)
    task_ids = sorted(validation["task_id"].unique())
    if max_tasks > 0:
        task_ids = task_ids[:max_tasks]

    def summarize(values: list[float]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=float)
        array = np.where(array == 0.0, 1.0, array)
        finite = array[np.isfinite(array)]
        positive_inf = np.isposinf(array)
        nan = np.isnan(array)
        # Infinite R-hat means zero within-chain variance but nonzero
        # between-chain variance: the worst possible disagreement.
        comparable = np.concatenate([finite, np.full(int(positive_inf.sum()), 1e6)])
        return {
            "n": int(array.size),
            "n_nan": int(nan.sum()),
            "n_posinf": int(positive_inf.sum()),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "p95": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
            "frac_gt_1_01": float((comparable > 1.01).mean()) if comparable.size else float("nan"),
            "frac_gt_1_05": float((comparable > 1.05).mean()) if comparable.size else float("nan"),
        }

    result: dict[str, Any] = {
        "status": "prediction_rhat",
        "arm": arm,
        "dims": list(dim_values),
        "tasks": len(task_ids),
        "order_seeds": list(order_values),
        "n_chains": N_CHAINS,
        "n_draws_per_chain": RHAT_DRAWS,
        "n_sobol": RHAT_SOBOL,
        "pca": pca_metadata,
        "by_dim": {},
    }

    for dim in dim_values:
        draw_path = RESULT_ROOT / f"draws-{arm}-d{dim}.npz"
        payload = np.load(draw_path, allow_pickle=False)
        draws = torch.as_tensor(
            payload["draws"], dtype=torch.float32, device=device
        )
        shapes = [
            tuple(int(x) for x in shape)
            for shape in json.loads(str(payload["shapes_json"].item()))
        ]
        index = torch.linspace(
            0, draws.shape[1] - 1, RHAT_DRAWS, device=device
        ).long()
        selected = draws[:, index, :]
        chain_parameters = []
        for chain in range(selected.shape[0]):
            blocks = [_unpack(row, shapes, device) for row in selected[chain]]
            chain_parameters.append(
                (
                    torch.stack([block[1] for block in blocks], dim=0),
                    torch.stack([block[2] for block in blocks], dim=0),
                    torch.stack([block[3].squeeze() for block in blocks], dim=0),
                )
            )

        by_budget: dict[int, list[float]] = {budget: [] for budget in BUDGETS}
        nonfinite_by_budget: dict[int, int] = {budget: 0 for budget in BUDGETS}
        overall: list[float] = []
        for task_number, task_id in enumerate(task_ids, start=1):
            task_rows = validation[validation["task_id"] == task_id]
            target_rows = task_rows[task_rows["pool"] == "evaluation"].reset_index(
                drop=True
            )
            target_item_index = torch.as_tensor(
                target_rows["item_index"].to_numpy(dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            target_embedding = z_all[target_item_index]
            alpha_targets = []
            beta_targets = []
            for W_alpha, w_beta, mu in chain_parameters:
                alpha_targets.append(
                    torch.einsum("nm,smd->snd", target_embedding, W_alpha)
                )
                beta_targets.append(
                    torch.einsum("nm,sm->sn", target_embedding, w_beta)
                    + mu[:, None]
                )

            lookup = task_rows.drop_duplicates("item_index").set_index("item_index")[
                "label"
            ]
            task_orders = orders[orders["task_id"] == task_id]
            for order_seed in order_values:
                order_rows = (
                    task_orders[task_orders["order_seed"] == order_seed]
                    .sort_values("acquisition_rank")
                    .reset_index(drop=True)
                )
                if len(order_rows) == 0:
                    continue
                context_item_index = torch.as_tensor(
                    order_rows["item_index"].to_numpy(dtype=np.int64),
                    dtype=torch.long,
                    device=device,
                )
                context_embedding = z_all[context_item_index]
                context_labels = np.asarray(
                    lookup.loc[order_rows["item_index"].to_numpy()], dtype=np.float32
                )
                alpha_contexts = []
                beta_contexts = []
                for W_alpha, w_beta, mu in chain_parameters:
                    alpha_contexts.append(
                        torch.einsum("nm,smd->snd", context_embedding, W_alpha)
                    )
                    beta_contexts.append(
                        torch.einsum("nm,sm->sn", context_embedding, w_beta)
                        + mu[:, None]
                    )

                for budget in BUDGETS:
                    labels_context = torch.as_tensor(
                        context_labels[:budget],
                        dtype=torch.float32,
                        device=device,
                    )
                    per_chain = []
                    with torch.no_grad():
                        for chain in range(len(chain_parameters)):
                            _, per_draw = _predict_context(
                                alpha_targets[chain],
                                beta_targets[chain],
                                alpha_contexts[chain][:, :budget],
                                beta_contexts[chain][:, :budget],
                                labels_context,
                                n_sobol=RHAT_SOBOL,
                                seed=SEED + order_seed * 1000 + budget,
                                device=device,
                            )
                            if not torch.isfinite(per_draw).all():
                                nonfinite_by_budget[budget] += 1
                            per_chain.append(per_draw)
                    # 1e-8 rounds to 1.0 in float32, which makes logit(1)=inf.
                    # 1e-6 is safely representable below one.
                    logits = torch.logit(
                        torch.stack(per_chain, dim=0).clamp(1e-6, 1.0 - 1e-6)
                    )
                    rhat = _split_rhat(logits)
                    if torch.isnan(rhat).any():
                        print(
                            f"NaN rhat task={task_id} order={order_seed} "
                            f"budget={budget} n={int(torch.isnan(rhat).sum())} "
                            f"logits_finite={bool(torch.isfinite(logits).all())} "
                            f"logits_min={float(logits.min()):.3f} "
                            f"logits_max={float(logits.max()):.3f}",
                            flush=True,
                        )
                    rhat = rhat.detach().cpu().numpy()
                    by_budget[budget].extend(rhat.tolist())
                    overall.extend(rhat.tolist())
            if task_number % 10 == 0:
                print(
                    f"{arm} mirt{dim}: rhat {task_number}/{len(task_ids)} tasks",
                    flush=True,
                )

        result["by_dim"][f"mirt{dim}"] = {
            "overall": summarize(overall),
            "by_budget": {
                str(budget): summarize(values)
                for budget, values in sorted(by_budget.items())
            },
            "nonfinite_contexts": {
                str(budget): int(count)
                for budget, count in sorted(nonfinite_by_budget.items())
            },
        }
        print(
            f"{arm} mirt{dim} prediction R-hat: "
            f"median={result['by_dim'][f'mirt{dim}']['overall']['median']:.4f} "
            f"max={result['by_dim'][f'mirt{dim}']['overall']['max']:.4f}",
            flush=True,
        )

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_ROOT / f"prediction-rhat-{arm}.json"
    _write_json(output_path, result)
    volume.commit()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


# ---------------------------------------------------------------------------
# Synthetic smoke test
# ---------------------------------------------------------------------------


def smoke() -> dict[str, Any]:
    """Small synthetic check of the sampler and posterior predictor."""

    import torch

    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_tasks, n_items, dim, n_rows = 20, 40, 2, 400
    z_all = torch.randn(n_items, 8, device=device)
    task_index = torch.randint(0, n_tasks, (n_rows,), device=device)
    item_index = torch.randint(0, n_items, (n_rows,), device=device)
    theta = torch.randn(n_tasks, dim, device=device)
    W_alpha = 0.5 * torch.randn(8, dim, device=device)
    w_beta = 0.5 * torch.randn(8, device=device)
    mu = torch.tensor([0.0], device=device)
    alpha = z_all[item_index] @ W_alpha
    logits = (alpha * theta[task_index]).sum(1) + z_all[item_index] @ w_beta + mu
    labels = torch.bernoulli(torch.sigmoid(logits))
    train = __import__("pandas").DataFrame(
        {
            "task_id": [f"t{int(value)}" for value in task_index.cpu()],
            "item_index": item_index.cpu().numpy(),
            "subject_index": 0,
            "label": labels.cpu().numpy().astype("int8"),
        }
    )
    shapes = [(n_tasks, dim), (8, dim), (8,), (1,)]
    initial = _map_fit(
        z_all[item_index],
        task_index,
        labels,
        shapes,
        device,
        n_tasks=n_tasks,
        dim=dim,
    )
    mass = _diagonal_mass(
        *initial,
        z_all[item_index],
        task_index,
        labels,
        shapes,
        prior_theta=PRIOR_THETA,
        prior_alpha=PRIOR_ALPHA,
        prior_beta=PRIOR_BETA,
        prior_mu=PRIOR_MU,
    )
    log_prob = _make_log_prob(
        z_all[item_index],
        task_index,
        labels,
        shapes,
        prior_theta=PRIOR_THETA,
        prior_alpha=PRIOR_ALPHA,
        prior_beta=PRIOR_BETA,
        prior_mu=PRIOR_MU,
    )
    draws, accepted, step = _hmc_chain(
        _pack(initial),
        mass,
        log_prob,
        n_burn=20,
        n_keep=20,
        thin=1,
        n_leapfrog=5,
        seed=1,
        device=device,
    )
    blocks = [_unpack(row, shapes, device) for row in draws]
    W = torch.stack([block[1] for block in blocks])
    b = torch.stack([block[2] for block in blocks])
    m = torch.stack([block[3].squeeze() for block in blocks])
    alpha_t = torch.einsum("nm,smd->snd", z_all[:5], W)
    beta_t = torch.einsum("nm,sm->sn", z_all[:5], b) + m[:, None]
    alpha_c = torch.einsum("nm,smd->snd", z_all[5:10], W)
    beta_c = torch.einsum("nm,sm->sn", z_all[5:10], b) + m[:, None]
    probability, _ = _predict_context(
        alpha_t,
        beta_t,
        alpha_c,
        beta_c,
        torch.bernoulli(torch.full((5,), 0.5, device=device)),
        n_sobol=16,
        seed=3,
        device=device,
    )
    assert probability.shape == (5,)
    assert torch.isfinite(probability).all()
    return {
        "status": "smoke_ok",
        "acceptance": float(np.mean(accepted[20:])),
        "step_size": float(step),
        "probability_mean": float(probability.mean()),
    }


@app.local_entrypoint()
def main(
    action: str = "inspect",
    arm: str = "bge_base",
    dims: str = "1,2",
    max_tasks: int = 0,
    order_seeds: str = "0",
    n_bootstrap: int = 10_000,
) -> None:
    if action == "inspect":
        print(json.dumps(inspect.remote(arm), indent=2, sort_keys=True, default=str))
        return
    if action == "run":
        print(json.dumps(run.remote(arm, dims), indent=2, sort_keys=True, default=str))
        return
    if action == "rhat":
        print(
            json.dumps(
                prediction_rhat.remote(arm, dims, max_tasks, order_seeds),
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return
    if action == "compare":
        print(
            json.dumps(
                compare.remote(arm, dims, n_bootstrap),
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return
    if action == "smoke":
        print(json.dumps(smoke(), indent=2, sort_keys=True))
        return
    raise ValueError("action must be inspect, run, compare, rhat, or smoke")
# Weight-saving placeholder for modal_bayesian_mirt.py
