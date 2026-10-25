"""Isolated reverse-order ablation for target-specific context representations.

Rung 1 (this file): deterministic ANP attention without latent ``z``.
The model lives in ``aims_starter.r_c_attention`` and is not wired into the
production CNP/ANP classes.  Data, protocol, folds, and calibration match the
uniform validation grid.

Run:
    modal run modal_r_c_ablation.py --rung anp_attn --seeds 20261018,20261019,20261020
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import modal
import numpy as np

APP_NAME = "aims-neurips-r-c-ablation"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDING_ROOT = REMOTE_ROOT / "datasets" / "full"
RESULT_ROOT = REMOTE_ROOT / "results" / "r-c-ablation"
UNIFORM_FOLDS = 5
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
SEEDS = (20261018, 20261019, 20261020)
MODES = ("masked", "visible")
ARM = "minilm512"
FIT_SAMPLE_LIMIT = 1_000_000
RUNGS = {
    "anp_attn": ("anp", None),
    "anp_gelu": ("anp", None, "gelu"),
    "anp_gelu_all": ("anp", None, "gelu_all"),
    "cnp": ("cnp", None),
    # Input-normalization / activation controls on the best structure-free arm.
    "cnp_ln": ("cnp", None, "ln"),
    "cnp_gelu": ("cnp", None, "gelu"),
    "cnp_gelu_all": ("cnp", None, "gelu_all"),
    "cnp_ln_gelu": ("cnp", None, "ln_gelu"),
    # Native CNP/ANP with a Gaussian posterior over a D-dimensional theta.
    # Tuple is (kind, ability_dim).
    "mdcnp_d4": ("mdcnp", 4),
    "mdanp_d4": ("mdanp", 4),
    # Joint Bayesian CNP-IRT: diagonal Gaussian posteriors over both theta and
    # each item's (log a, d). Tuple is (kind, ability_dim).
    "bcirt_d4": ("bcirt", 4),
    "bcirt_d4_ln_gelu": ("bcirt", 4, "ln_gelu"),
    "bcirt_d4_gelu_all": ("bcirt", 4, "gelu_all"),
    "direct_ab": ("irt", "direct_ab"),
    "additive_ab": ("irt", "additive_ab"),
    "residual": ("irt", "residual"),
    "scalar": ("irt", "scalar"),
    "2pl": ("irt", "2pl"),
    "additive_ar": ("ar", "additive_ab", True),
    "additive_noar": ("ar", "additive_ab", False),
    "direct_ar": ("ar", "direct_ab", True),
    "direct_noar": ("ar", "direct_ab", False),
    "additive_loop0": ("loop", "additive_ab", 0),
    "additive_loop1": ("loop", "additive_ab", 1),
    "additive_loop3": ("loop", "additive_ab", 3),
    "direct_loop0": ("loop", "direct_ab", 0),
    "direct_loop1": ("loop", "direct_ab", 1),
    "direct_loop3": ("loop", "direct_ab", 3),
    "gp_2pl": ("gp", "2pl"),
    "gp_additive": ("gp", "additive_ab"),
    "meta_gp_2pl": ("gp", "2pl", True),
    "nr_2pl": ("neural", "nr", False),
    "nr_2pl_prior": ("neural", "nr", True),
    "u_nr_2pl": ("neural", "u_nr", False),
    "u_nr_2pl_prior": ("neural", "u_nr", True),
    # Multidimensional-ability counterparts of the three best scalar rungs.
    # Tuple is (kind, mode, ability_dim, loops).
    "mirt_additive_loop3_d1": ("mirt", "additive_ab", 1, 3),
    "mirt_additive_loop3_d4": ("mirt", "additive_ab", 4, 3),
    "mirt_additive_loop3_d8": ("mirt", "additive_ab", 8, 3),
    "mirt_direct_loop0_d1": ("mirt", "direct_ab", 1, 0),
    "mirt_direct_loop0_d4": ("mirt", "direct_ab", 4, 0),
    "mirt_direct_loop0_d8": ("mirt", "direct_ab", 8, 0),
    "mirt_scalar_d1": ("mirt", "scalar", 1, 0),
    "mirt_scalar_d2": ("mirt", "scalar", 2, 0),
    "mirt_scalar_d4": ("mirt", "scalar", 4, 0),
    "mirt_scalar_d4_tau1": ("mirt", "scalar", 4, 0, 1.0),
    # Joint online MAP over D=4 ability and a centered K=4 difficulty adapter.
    # Tuple is (kind, mode, ability_dim, loops, theta_tau2, adapter_dim, adapter_tau2).
    "mirt_scalar_d4_jointb4_tau1": ("mirt_joint", "scalar", 4, 0, 1.0, 4, 1.0),
    "mirt_scalar_d8": ("mirt", "scalar", 8, 0),
    # Rasch (discrimination pinned to 1) and the ability-prior variance.
    # tau^2 = 4 is the value hard-coded throughout the repo and in the
    # information-ceiling note; it has never been derived or ablated, and the
    # IRT convention is N(0, 1). Tuple is (kind, mode, prior_variance).
    #
    # An unregularized MLE fit of 2PL/Rasch on the training tasks put
    # var(theta_hat) at 9.6 (2PL) to 11.9 (Rasch) -- 2.4-3x the hard-coded
    # tau^2 = 4 -- with the +-4 box binding on ~17% of tasks, so that MLE is
    # itself an underestimate. But tau=1 beat tau=4 on validation Brier ALC
    # for both Rasch and 2PL, the opposite of what the population variance
    # would suggest: a tighter prior can still be the loss-optimal shrinkage
    # when theta is estimated from as few as 0-31 noisy labels, independent
    # of what the true population spread is. tau9/tau16 bracket the measured
    # MLE range and test whether loss keeps improving with tau, or whether
    # tau=1 is a local optimum and shrinkage should go tighter still.
    "rasch_tau4": ("irt", "rasch", 4.0),
    "rasch_tau1": ("irt", "rasch", 1.0),
    "rasch_tau9": ("irt", "rasch", 9.0),
    "rasch_tau16": ("irt", "rasch", 16.0),
    "2pl_tau4": ("irt", "2pl", 4.0),
    "2pl_tau1": ("irt", "2pl", 1.0),
    "2pl_tau9": ("irt", "2pl", 9.0),
    "2pl_tau16": ("irt", "2pl", 16.0),
    "scalar_tau1": ("irt", "scalar", 1.0),
}
HELDOUT_ONLY_RUNGS = {
    "additive_ar",
    "additive_noar",
    "direct_ar",
    "direct_noar",
    "additive_loop0",
    "additive_loop1",
    "additive_loop3",
    "direct_loop0",
    "direct_loop1",
    "direct_loop3",
}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2,<3",
        "pandas>=2.2,<3",
        "pyarrow>=17,<24",
        "torch==2.5.1",
    )
    .add_local_dir(PROJECT_ROOT / "src", "/opt/aims/src", copy=True)
    .env({"PYTHONPATH": "/opt/aims/src", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
)


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-8), 1.0 - 1e-8)
    return math.log(clipped / (1.0 - clipped))


def _compact_root(corpus: str = "") -> Path:
    """Resolve the feature store a run reads.

    The default keeps every published r_C number reproducible; passing a store
    name selects another one, which is how the v2 item_content-only arms are
    evaluated without editing a module constant.
    """

    if not corpus:
        return EMBEDDING_ROOT / f"allbench-embed-{ARM}-v1"
    root = EMBEDDING_ROOT / corpus
    return root


def _features(rows, item_embeddings, subject_embeddings):
    return np.concatenate(
        [
            item_embeddings[rows["item_index"].to_numpy()],
            subject_embeddings[rows["subject_index"].to_numpy()],
        ],
        axis=1,
    )


def _meta_tasks(train, item_embeddings, subject_embeddings):
    return [
        {
            "task_id": str(task_id),
            "x": _features(rows, item_embeddings, subject_embeddings),
            "y": rows["label"].to_numpy(dtype="float32"),
        }
        for task_id, rows in train.groupby("task_id", sort=True)
    ]


def _episode(
    task: dict[str, np.ndarray],
    rng: np.random.Generator,
    device,
    *,
    heldout_only: bool = False,
):
    """Match the production CNP episode sampler (context included in targets)."""

    import torch

    n = len(task["y"])
    order = rng.permutation(n)
    n_c = min(int(rng.choice(BUDGETS)), n - 1)
    n_t = min(n, max(64, n_c + 32))
    indices_c = order[:n_c]
    if heldout_only:
        indices_t = order[n_c:n_t]
    else:
        indices_t = np.concatenate([indices_c, order[n_c:n_t]])
    x = torch.as_tensor(task["x"], dtype=torch.float32, device=device)
    y = torch.as_tensor(task["y"], dtype=torch.float32, device=device)
    return x[indices_c], y[indices_c], x[indices_t], y[indices_t]


def _make_model(
    rung: str,
    x_dim: int,
    subject_feature_slice,
    subject_prior_slice,
):
    from aims_starter.r_c_attention import (
        AutoregressiveContextIRT,
        BayesianCNPIRT,
        ContextConditionedIRT,
        DeterministicANPAttention,
        DeterministicCNP,
        GPContextIRT,
        LoopedContextIRT,
        MultidimANP,
        MultidimCNP,
        MultidimContextIRT,
        NeuralResidualIRT,
    )

    kind, mode, *flags = RUNGS[rung]
    if kind == "mdcnp":
        return MultidimCNP(
            x_dim,
            ability_dim=int(mode),
            d_model=128,
            subject_feature_slice=subject_feature_slice,
            prior_variance=1.0,
        )
    if kind == "mdanp":
        return MultidimANP(
            x_dim,
            ability_dim=int(mode),
            d_model=128,
            n_heads=8,
            n_context_layers=2,
            subject_feature_slice=subject_feature_slice,
            prior_variance=1.0,
        )
    if kind == "bcirt":
        variant = flags[0] if flags else ""
        return BayesianCNPIRT(
            x_dim,
            ability_dim=int(mode),
            d_model=128,
            subject_feature_slice=subject_feature_slice,
            prior_theta_variance=1.0,
            prior_item_variance=0.25,
            max_kl_weight=0.01,
            input_norm="ln" in variant,
            activation="gelu" if "gelu" in variant else "relu",
            activation_all="all" in variant,
        )
    if kind == "anp":
        variant = flags[0] if flags else ""
        return DeterministicANPAttention(
            x_dim,
            d_model=128,
            n_heads=8,
            n_context_layers=2,
            subject_feature_slice=subject_feature_slice,
            activation="gelu" if "gelu" in variant else "relu",
            activation_all="all" in variant,
        )
    if kind == "cnp":
        variant = flags[0] if flags else ""
        return DeterministicCNP(
            x_dim,
            d_model=128,
            subject_feature_slice=subject_feature_slice,
            input_norm="ln" in variant,
            activation="gelu" if "gelu" in variant else "relu",
            activation_all="all" in variant,
        )
    if kind == "ar":
        return AutoregressiveContextIRT(
            x_dim,
            mode=mode,
            ar=bool(flags[0]),
            d_model=128,
            subject_feature_slice=subject_feature_slice,
        )
    if kind == "loop":
        return LoopedContextIRT(
            x_dim,
            mode=mode,
            loops=int(flags[0]),
            d_model=128,
            n_heads=8,
            subject_feature_slice=subject_feature_slice,
        )
    if kind in ("mirt", "mirt_joint"):
        ability_dim, loops = int(flags[0]), int(flags[1])
        prior_variance = float(flags[2]) if len(flags) > 2 else 4.0
        adapter_dim = int(flags[3]) if kind == "mirt_joint" else 0
        adapter_prior_variance = float(flags[4]) if kind == "mirt_joint" else 1.0
        return MultidimContextIRT(
            x_dim,
            mode=mode,
            ability_dim=ability_dim,
            loops=loops,
            d_model=128,
            n_heads=8,
            subject_feature_slice=subject_feature_slice,
            prior_variance=prior_variance,
            difficulty_adapter_dim=adapter_dim,
            adapter_prior_variance=adapter_prior_variance,
        )
    if kind == "gp":
        return GPContextIRT(
            x_dim,
            mode=mode,
            d_model=128,
            basis_size=8,
            lengthscale=1.0,
            gp_lambda=1e-3,
            subject_feature_slice=subject_feature_slice,
            subject_prior_slice=subject_prior_slice,
            use_subject_prior=bool(flags[0]) if flags else False,
        )
    if kind == "neural":
        return NeuralResidualIRT(
            x_dim,
            mode=mode,
            d_model=128,
            basis_size=8,
            lengthscale=1.0,
            residual_l2=1e-3,
            subject_feature_slice=subject_feature_slice,
            subject_prior_slice=subject_prior_slice,
            use_subject_prior=bool(flags[0]) if flags else False,
        )
    return ContextConditionedIRT(
        x_dim,
        mode=mode,
        d_model=128,
        subject_feature_slice=subject_feature_slice,
        prior_variance=float(flags[0]) if flags else 4.0,
    )


def _train_model(
    model,
    tasks,
    *,
    seed: int,
    steps: int = 800,
    episodes_per_step: int = 4,
    heldout_only: bool = False,
):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        losses = []
        if hasattr(model, "set_training_progress"):
            model.set_training_progress(step, steps)
        for _ in range(episodes_per_step):
            task = tasks[int(rng.integers(len(tasks)))]
            x_c, y_c, x_t, y_t = _episode(task, rng, device, heldout_only=heldout_only)
            logits = model(x_c, y_c, x_t)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t)
            if hasattr(model, "gp_penalty"):
                loss = loss + model.gp_penalty()
            if hasattr(model, "aux_loss"):
                loss = loss + model.aux_loss()
            losses.append(loss)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})
            print(f"seed={seed} step={step} loss={float(loss.detach()):.6f}", flush=True)
    return model, history


def _train_model_v41(
    model,
    tasks,
    *,
    seed: int,
    pair_epochs: int = 1,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup: int = 100,
    eta_min_ratio: float = 0.05,
):
    """Train with the v4.1 coverage sampler and disjoint context/target splits.

    Two differences from ``_train_model``:

    * **Coverage.** ``build_pair_epoch_batches`` visits every ``(task, budget)``
      pair exactly once per epoch. The random sampler draws 800 x 4 = 3,200
      episodes against 6,702 tasks x 6 budgets = 40,212 pairs, so roughly 92%
      of pairs are never trained on.
    * **Disjointness.** ``sample_pair_episode`` uses a rotating disjoint C/Q
      split keyed on ``(task, budget, pass)`` and independent of the model
      seed, so every arm sees the same splits. The old sampler put the context
      items back into the targets, which the competition never does -- acquired
      items are not scored.
    """

    import torch

    from aims_starter.faithful_ladder import (
        build_pair_epoch_batches,
        sample_pair_episode,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    indexed = [{**task, "task_index": index} for index, task in enumerate(tasks)]
    scheduled = []
    for epoch in range(int(pair_epochs)):
        for batch in build_pair_epoch_batches(
            indexed, seed=seed + 1009 * epoch, batch_size=int(batch_size)
        ):
            scheduled.append((epoch, batch))
    if not scheduled:
        raise ValueError("no training batches were scheduled")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(scheduled)
    warmup_steps = max(1, min(int(warmup), max(1, total_steps // 10)))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    rng = np.random.default_rng(seed)
    history = []
    print(
        f"seed={seed} v4.1 schedule: {total_steps} batches, "
        f"{sum(len(batch) for _, batch in scheduled)} pair visits",
        flush=True,
    )

    for step, (epoch, batch) in enumerate(scheduled):
        losses = []
        if hasattr(model, "set_training_progress"):
            model.set_training_progress(step, total_steps)
        for visit in batch:
            task = indexed[visit.task_index]
            episode = sample_pair_episode(
                task,
                budget=visit.budget,
                rng=rng,
                visit_key=f"{visit.budget}:{visit.pass_index}",
            )
            if episode["x_t"].shape[0] == 0:
                continue
            x_c = torch.as_tensor(episode["x_c"], dtype=torch.float32, device=device)
            y_c = torch.as_tensor(episode["y_c"], dtype=torch.float32, device=device)
            x_t = torch.as_tensor(episode["x_t"], dtype=torch.float32, device=device)
            y_t = torch.as_tensor(episode["y_t"], dtype=torch.float32, device=device)
            logits = model(x_c, y_c, x_t)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t)
            if hasattr(model, "gp_penalty"):
                loss = loss + model.gp_penalty()
            if hasattr(model, "aux_loss"):
                loss = loss + model.aux_loss()
            losses.append(loss)
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == total_steps - 1:
            history.append({"step": step, "epoch": epoch, "loss": float(loss.detach())})
            print(
                f"seed={seed} epoch={epoch} step={step}/{total_steps} "
                f"loss={float(loss.detach()):.6f}",
                flush=True,
            )
    return model, history


def _fit_calibrator(samples, rng):
    from aims_starter.calibration import AnchorCalibrator

    logits = np.concatenate([sample["logits"] for sample in samples])
    labels = np.concatenate([sample["labels"] for sample in samples])
    rates = np.concatenate(
        [
            np.full(len(sample["labels"]), _context_rate(sample["context_labels"]))
            for sample in samples
        ]
    )
    budgets = np.concatenate(
        [np.full(len(sample["labels"]), sample["budget"], dtype=int) for sample in samples]
    )
    if len(labels) > FIT_SAMPLE_LIMIT:
        index = rng.choice(len(labels), size=FIT_SAMPLE_LIMIT, replace=False)
        logits, labels, rates, budgets = logits[index], labels[index], rates[index], budgets[index]
    return AnchorCalibrator.fit(logits, labels, rates, budgets)


def _context_rate(labels: np.ndarray) -> float:
    values = np.asarray(labels, dtype=float)
    return (1.0 + float(values.sum())) / (2.0 + len(values))


def _calibrators(records):
    rng = np.random.default_rng(20261020)
    folds = sorted({int(record["fold"]) for record in records})
    return {
        "full": _fit_calibrator(records, rng),
        "folds": {
            fold: _fit_calibrator(
                [record for record in records if int(record["fold"]) != fold], rng
            )
            for fold in folds
        },
    }


def _metric_rows(records, calibrators, arm: str):
    rows = []
    for record in records:
        labels = np.asarray(record["labels"], dtype=float)
        logits = np.asarray(record["logits"], dtype=float)
        if arm == "beta":
            probability = _sigmoid(logits)
        else:
            calibrator = calibrators["folds"][int(record["fold"])]
            probability = calibrator.transform(
                logits, int(record["budget"]), np.asarray(record["context_labels"], dtype=float)
            )
        rows.append(
            {
                "arm": arm,
                "task_id": record["task_id"],
                "order_seed": int(record["order_seed"]),
                "budget": int(record["budget"]),
                "brier": float(np.mean((probability - labels) ** 2)),
            }
        )
    return rows


def _summarize(rows):
    import pandas as pd

    frame = pd.DataFrame(rows)
    weight = dict(zip(BUDGETS, WEIGHTS, strict=True))
    frame["weighted"] = frame["budget"].map(weight) * frame["brier"]
    task_alc = (
        frame.groupby(["arm", "task_id", "order_seed"], sort=False)["weighted"]
        .sum()
        .reset_index()
        .groupby(["arm", "task_id"], sort=False)["weighted"]
        .mean()
        .reset_index()
    )
    alc = []
    for arm, group in task_alc.groupby("arm", sort=True):
        alc.append(
            {
                "arm": arm,
                "split": "validation",
                "brier_alc": float(group["weighted"].mean()),
                "se": float(group["weighted"].std(ddof=1) / np.sqrt(len(group))),
                "tasks": len(group),
            }
        )
    task_records = task_alc.to_dict(orient="records")
    for row in task_records:
        row["weighted"] = float(row["weighted"])
    return alc, task_records


def _score_validation(model, seed: int, mode: str, orders: int = 10, corpus: str = ""):
    import pandas as pd
    import torch

    from aims_starter.uniform_eval import validation_fold_map

    root = _compact_root(corpus)
    item_embeddings = np.load(root / "item_embeddings.npy")
    subject_embeddings = np.load(root / "subject_embeddings.npy")
    evaluation = pd.read_parquet(root / "eval_items.parquet")
    order_frame = pd.read_parquet(root / "eval_orders.parquet")
    fold_map = validation_fold_map(evaluation, n_folds=UNIFORM_FOLDS)
    device = next(model.parameters()).device
    model.eval()

    records = []
    beta_records = []
    validation = evaluation[evaluation["split"] == "validation"]
    for task_id in sorted(validation["task_id"].astype(str).unique()):
        task_rows = validation[validation["task_id"].astype(str) == task_id]
        benchmark = str(task_rows["benchmark_id"].iloc[0])
        target_rows = task_rows[task_rows["pool"] == "evaluation"].reset_index(drop=True)
        x_t_np = _features(target_rows, item_embeddings, subject_embeddings)
        y_t = target_rows["label"].to_numpy(dtype="float32")
        lookup = task_rows.set_index("item_index")
        with torch.no_grad():
            x_t = torch.as_tensor(x_t_np, dtype=torch.float32, device=device)
        for order_seed in range(orders):
            order_rows = (
                order_frame[
                    (order_frame["task_id"].astype(str) == task_id)
                    & (order_frame["order_seed"] == order_seed)
                ]
                .sort_values("acquisition_rank")
                .reset_index(drop=True)
            )
            context_rows = lookup.loc[order_rows["item_index"].to_numpy()].reset_index()
            x_c_np = _features(context_rows, item_embeddings, subject_embeddings)
            y_c = context_rows["label"].to_numpy(dtype="float32")
            with torch.no_grad():
                x_c = torch.as_tensor(x_c_np, dtype=torch.float32, device=device)
                y_c_t = torch.as_tensor(y_c, dtype=torch.float32, device=device)
            for budget in BUDGETS:
                base = {
                    "seed": seed,
                    "mode": mode,
                    "split": "validation",
                    "benchmark_id": benchmark,
                    "task_id": task_id,
                    "order_seed": int(order_seed),
                    "fold": int(fold_map.get(task_id, -1)),
                    "budget": int(budget),
                    "labels": np.asarray(y_t, dtype=float),
                    "context_labels": np.asarray(y_c[:budget], dtype=float),
                }
                with torch.no_grad():
                    logits = model(x_c[:budget], y_c_t[:budget], x_t).cpu().numpy()
                records.append({**base, "logits": np.asarray(logits, dtype=float)})
                beta_probability = (2.0 + float(y_c[:budget].sum())) / (4.0 + budget)
                beta_records.append(
                    {
                        **base,
                        "logits": np.full(len(y_t), _logit(beta_probability), dtype=float),
                    }
                )
        print(f"seed={seed} scored {task_id}", flush=True)
    return records, beta_records


@app.function(
    image=image,
    gpu=os.environ.get("R_C_GPU", "L4"),
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=8 * 60 * 60,
)
def run(
    rung: str = "anp_attn",
    seed: int = 20261018,
    mode: str = "masked",
    resume: bool = False,
    corpus: str = "",
    sampler: str = "random",
    pair_epochs: int = 1,
) -> dict[str, Any]:
    if rung not in RUNGS:
        raise ValueError(f"rung must be one of {tuple(RUNGS)}")
    if sampler not in ("random", "v41"):
        raise ValueError("sampler must be 'random' or 'v41'")
    if pair_epochs < 1:
        raise ValueError("pair_epochs must be positive")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    tag = f"-{corpus}" if corpus else ""
    if sampler == "v41":
        tag += f"-v41e{pair_epochs}"
    output = RESULT_ROOT / f"r-c-{rung}-{seed}-{mode}{tag}.json"
    existing = _read_json(output)
    if existing is not None:
        return existing

    import pandas as pd
    import torch

    root = _compact_root(corpus)
    manifest = _read_json(root / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"compact corpus missing: {root}")
    item_embeddings = np.load(root / "item_embeddings.npy")
    subject_embeddings = np.load(root / "subject_embeddings.npy")
    item_dim = int(item_embeddings.shape[1])
    subject_prior_slice = (
        item_dim,
        item_dim + int(subject_embeddings.shape[1]),
    )
    subject_slice = subject_prior_slice if mode == "masked" else None
    train = pd.read_parquet(root / "train_tasks.parquet")
    tasks = _meta_tasks(train, item_embeddings, subject_embeddings)
    model = _make_model(
        rung,
        tasks[0]["x"].shape[1],
        subject_slice,
        subject_prior_slice,
    )
    if sampler == "v41":
        model, history = _train_model_v41(model, tasks, seed=seed, pair_epochs=int(pair_epochs))
    else:
        model, history = _train_model(
            model,
            tasks,
            seed=seed,
            steps=800,
            episodes_per_step=4,
            heldout_only=rung in HELDOUT_ONLY_RUNGS,
        )
    checkpoint_root = RESULT_ROOT / f"seed-{seed}-{mode}{tag}" / f"{rung}-checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_root / "model.pt")

    records, beta_records = _score_validation(model, seed, mode, orders=10, corpus=corpus)
    calibrators = _calibrators(records)
    rows = _metric_rows(records, calibrators, rung) + _metric_rows(
        beta_records, calibrators, "beta"
    )
    alc, task_alc = _summarize(rows)
    result = {
        "status": "complete",
        "rung": rung,
        "arm": ARM,
        "seed": seed,
        "mode": mode,
        "split": "validation",
        "orders": 10,
        "prior_variance": float(getattr(model, "prior_variance", 4.0)),
        "manifest": manifest,
        "training": history,
        "alc": alc,
        "task_alc": task_alc,
    }
    _write_json(output, result)
    volume.commit()
    return result


@app.function(
    image=image,
    gpu=os.environ.get("R_C_GPU", "L4"),
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=8 * 60 * 60,
)
def score_checkpoint(
    rung: str = "bcirt_d4",
    seed: int = 20261018,
    mode: str = "masked",
    corpus: str = "",
    sampler: str = "v41",
    pair_epochs: int = 1,
) -> dict[str, Any]:
    """Rescore a saved checkpoint and report calibrated Brier by budget."""
    import pandas as pd
    import torch

    if rung not in RUNGS:
        raise ValueError(f"rung must be one of {tuple(RUNGS)}")
    if sampler not in ("random", "v41"):
        raise ValueError("sampler must be 'random' or 'v41'")
    tag = f"-{corpus}" if corpus else ""
    if sampler == "v41":
        tag += f"-v41e{pair_epochs}"
    root = _compact_root(corpus)
    manifest = _read_json(root / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"compact corpus missing: {root}")
    item_embeddings = np.load(root / "item_embeddings.npy")
    subject_embeddings = np.load(root / "subject_embeddings.npy")
    item_dim = int(item_embeddings.shape[1])
    subject_prior_slice = (
        item_dim,
        item_dim + int(subject_embeddings.shape[1]),
    )
    subject_slice = subject_prior_slice if mode == "masked" else None
    train = pd.read_parquet(root / "train_tasks.parquet")
    tasks = _meta_tasks(train, item_embeddings, subject_embeddings)
    model = _make_model(
        rung,
        tasks[0]["x"].shape[1],
        subject_slice,
        subject_prior_slice,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = RESULT_ROOT / f"seed-{seed}-{mode}{tag}" / f"{rung}-checkpoints" / "model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)

    records, beta_records = _score_validation(model, seed, mode, orders=10, corpus=corpus)
    calibrators = _calibrators(records)
    rows = _metric_rows(records, calibrators, rung) + _metric_rows(
        beta_records, calibrators, "beta"
    )
    alc, task_alc = _summarize(rows)
    frame = pd.DataFrame(rows)
    by_budget = (
        frame.groupby(["arm", "budget"], sort=True)["brier"]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    result = {
        "status": "complete",
        "rung": rung,
        "arm": ARM,
        "seed": seed,
        "mode": mode,
        "split": "validation",
        "orders": 10,
        "manifest": manifest,
        "alc": alc,
        "by_budget": by_budget,
        "task_alc": task_alc,
    }
    output = RESULT_ROOT / f"by-budget-{rung}-{seed}-{mode}{tag}.json"
    _write_json(output, result)
    volume.commit()
    return result


@app.function(
    image=image,
    gpu=os.environ.get("R_C_GPU", "L4"),
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=2 * 60 * 60,
)
def logit_stats(
    rung: str = "cnp_gelu",
    seed: int = 20261018,
    mode: str = "masked",
    corpus: str = "",
    sampler: str = "v41",
    pair_epochs: int = 1,
) -> dict[str, Any]:
    """Report target-logit margin quantiles for a saved checkpoint.

    The fixed-softmax paper's fp32 precision cliff is a positive-margin
    phenomenon.  Binary CE has the same ``sigmoid(z) - 1`` collapse for the
    positive class, so this check tells us whether any validation margins get
    close to the fp32 cliff (about 16.6).
    """
    import pandas as pd
    import torch

    if rung not in RUNGS:
        raise ValueError(f"rung must be one of {tuple(RUNGS)}")
    if sampler not in ("random", "v41"):
        raise ValueError("sampler must be 'random' or 'v41'")
    tag = f"-{corpus}" if corpus else ""
    if sampler == "v41":
        tag += f"-v41e{pair_epochs}"
    root = _compact_root(corpus)
    manifest = _read_json(root / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"compact corpus missing: {root}")
    item_embeddings = np.load(root / "item_embeddings.npy")
    subject_embeddings = np.load(root / "subject_embeddings.npy")
    item_dim = int(item_embeddings.shape[1])
    subject_prior_slice = (
        item_dim,
        item_dim + int(subject_embeddings.shape[1]),
    )
    subject_slice = subject_prior_slice if mode == "masked" else None
    train = pd.read_parquet(root / "train_tasks.parquet")
    tasks = _meta_tasks(train, item_embeddings, subject_embeddings)
    model = _make_model(
        rung,
        tasks[0]["x"].shape[1],
        subject_slice,
        subject_prior_slice,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = RESULT_ROOT / f"seed-{seed}-{mode}{tag}" / f"{rung}-checkpoints" / "model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)

    records, _ = _score_validation(model, seed, mode, orders=10, corpus=corpus)
    logits = np.concatenate([np.asarray(r["logits"], dtype=float) for r in records])
    labels = np.concatenate([np.asarray(r["labels"], dtype=float) for r in records])
    margins = (2.0 * labels - 1.0) * logits
    quantiles = (0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0)
    result = {
        "status": "complete",
        "rung": rung,
        "seed": seed,
        "mode": mode,
        "n": int(len(logits)),
        "margin_quantiles": {
            str(q): float(np.quantile(margins, q)) for q in quantiles
        },
        "abs_logit_quantiles": {
            str(q): float(np.quantile(np.abs(logits), q)) for q in quantiles
        },
        "fraction_margin_gt_5_5": float((margins > 5.5).mean()),
        "fraction_margin_gt_16_6": float((margins > 16.6).mean()),
        "fraction_margin_gt_36_7": float((margins > 36.7).mean()),
        "max_margin": float(margins.max()),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def score_main(
    rung: str = "bcirt_d4",
    seeds: str = "20261018,20261019,20261020",
    mode: str = "masked",
    corpus: str = "",
    sampler: str = "v41",
    pair_epochs: int = 1,
):
    for seed in [int(value) for value in seeds.split(",") if value]:
        print(
            json.dumps(
                score_checkpoint.remote(rung, seed, mode, corpus, sampler, int(pair_epochs)),
                indent=2,
                sort_keys=True,
            )
        )


@app.local_entrypoint()
def main(
    rungs: str = "anp_attn",
    seeds: str = "20261018,20261019,20261020",
    mode: str = "masked",
    parallel: bool = False,
    corpus: str = "",
    sampler: str = "random",
    pair_epochs: int = 1,
):
    rung_values = [value for value in rungs.split(",") if value]
    seed_values = [int(value) for value in seeds.split(",") if value]
    for rung in rung_values:
        if rung not in RUNGS:
            raise ValueError(f"rung must be one of {tuple(RUNGS)}")
    if parallel:
        calls = [
            (rung, seed, run.spawn(rung, seed, mode, False, corpus, sampler, int(pair_epochs)))
            for rung in rung_values
            for seed in seed_values
        ]
        for rung, seed, call in calls:
            print(f"=== {rung} seed={seed} ===", flush=True)
            print(json.dumps(call.get(), indent=2, sort_keys=True))
        return
    for rung in rung_values:
        for seed in seed_values:
            print(f"=== {rung} seed={seed} ===", flush=True)
            print(
                json.dumps(
                    run.remote(rung, seed, mode, False, corpus, sampler, int(pair_epochs)),
                    indent=2,
                    sort_keys=True,
                )
            )
