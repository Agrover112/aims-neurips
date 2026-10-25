"""Acquisition-policy ablation for the multidimensional-ability IRT model.

The published acquisition ablation (README, "Active-acquisition ablation")
compared policies against a scalar CNP-AIRT predictor on the minilm feature
store and concluded that "the acquisition rules do nothing", with the stated
cause being near-chance item discrimination: no policy can extract signal a
predictor does not have. Two things have changed since: the me5 content-only
embeddings are the first change to move validation ALC (-0.003 against Beta),
and ``MultidimContextIRT`` gives the ability a vector rather than a scalar,
which is what makes a genuinely multidimensional acquisition criterion
meaningful.

Arms
----
``A-R``   random stream order (the platform default).
``A-U``   uncertainty: rank by ``-|p - 0.5|``.
``A-F``   D-optimal Fisher information (see ``_fisher_d_scores``).
``A-MM``  greedy k-center on the item embedding: maximum spread, no labels.

``A-MM`` is included because it is the one coverage policy in the
Clustering-Embed family that is deployable under the competition's streaming
hook: it needs only the items already selected, never the unseen candidate
pool. Per-task k-means is not deployable and is deliberately not an arm here.

Run:
    modal run modal_mirt_acquisition.py --rung mirt_scalar_d4 --streams 10
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import modal
import numpy as np

APP_NAME = "aims-neurips-mirt-acquisition"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDING_ROOT = REMOTE_ROOT / "datasets" / "full"
ABLATION_ROOT = REMOTE_ROOT / "results" / "r-c-ablation"
RESULT_ROOT = REMOTE_ROOT / "results" / "mirt-acquisition"
DEFAULT_CORPUS = "allbench-train-me5_instruct-cap128-v2"
UNIFORM_FOLDS = 5
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
FIT_SAMPLE_LIMIT = 1_000_000

ARMS: dict[str, str] = {
    "A-R": "random",
    "A-U": "uncertainty",
    "A-F": "fisher_d",
    "A-FA": "fisher_a",
    "A-FE": "fisher_e",
    "A-MM": "maxmin",
}


def _rung_config(rung: str) -> tuple[int, float, int, float]:
    """Read ability and online difficulty priors from a MIRT rung name."""

    dimension = re.search(r"_d(\d+)(?:_|$)", rung)
    if dimension is None:
        raise ValueError(f"MIRT rung does not encode an ability dimension: {rung}")
    tau = re.search(r"_tau([0-9]+(?:\.[0-9]+)?)(?:_|$)", rung)
    adapter = re.search(r"_jointb(\d+)(?:_|$)", rung)
    adapter_tau = re.search(r"_btau([0-9]+(?:\.[0-9]+)?)(?:_|$)", rung)
    return (
        int(dimension.group(1)),
        float(tau.group(1)) if tau else 4.0,
        int(adapter.group(1)) if adapter else 0,
        float(adapter_tau.group(1)) if adapter_tau else 1.0,
    )


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scipy>=1.13,<2")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(PROJECT_ROOT / "src", "/opt/aims/src", copy=True)
    .env({"PYTHONPATH": "/opt/aims/src"})
)


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-8), 1.0 - 1e-8)
    return math.log(clipped / (1.0 - clipped))


def _context_rate(labels: np.ndarray) -> float:
    values = np.asarray(labels, dtype=float)
    return (1.0 + float(values.sum())) / (2.0 + len(values))


def _features(rows, item_embeddings, subject_embeddings):
    return np.concatenate(
        [
            item_embeddings[rows["item_index"].to_numpy()],
            subject_embeddings[rows["subject_index"].to_numpy()],
        ],
        axis=1,
    ).astype("float32")


def _posterior_covariance(design_selected, d_selected, state, prior_variance):
    """Laplace covariance of an ability or joint ability/difficulty state.

    This is the inverse of the same Hessian the model's Newton solve builds,
    rebuilt here because ``map_theta`` returns only the mode. For the
    multidimensional model the curvature is a full matrix. For a joint model,
    each row is ``[a_i, -u_i]`` and the state is ``[theta, z]``:

        H = X^T diag(p (1 - p)) X + diag(1 / tau_k^2),

    so the posterior is an ellipsoid rather than an interval, which is what
    makes a D-optimal criterion different from a scalar one.
    """

    dim = state.shape[0]
    eye = np.eye(dim)
    prior_variance = np.broadcast_to(np.asarray(prior_variance, dtype=float), (dim,))
    if design_selected.shape[0] == 0:
        return np.diag(prior_variance)
    probability = _sigmoid(design_selected @ state - d_selected)
    weight = probability * (1.0 - probability)
    hessian = (design_selected.T * weight) @ design_selected
    hessian = hessian + np.diag(1.0 / prior_variance)
    return np.linalg.inv(hessian + 1e-6 * eye)


def _fisher_d_scores(a_candidates, probability, covariance):
    """D-optimal information gain for each candidate item.

    A response to item i adds the rank-one block ``w_i a_i a_i^T`` to the
    ability information matrix, with ``w_i = p_i (1 - p_i)``. By the matrix
    determinant lemma,

        det(H + w_i a_i a_i^T) = det(H) (1 + w_i a_i^T H^-1 a_i),

    so maximizing the determinant of the posterior precision -- shrinking the
    volume of the credible ellipsoid, the D-optimality criterion -- is exactly
    maximizing ``w_i a_i^T Sigma a_i``. No determinant is ever formed.

    At one ability dimension this collapses to ``a_i^2 p_i (1 - p_i)`` scaled
    by the posterior variance, i.e. the classical CAT Fisher rule, so the
    scalar and multidimensional arms are the same criterion where they
    overlap.
    """

    weight = probability * (1.0 - probability)
    quadratic = np.einsum("ij,jk,ik->i", a_candidates, covariance, a_candidates)
    return weight * quadratic


def _fisher_a_scores(a_candidates, probability, covariance):
    """A-optimal information gain: the drop in total posterior variance.

    A-optimality minimizes ``tr(Sigma)``, the sum of the per-dimension
    variances, where D-optimality minimizes the determinant. Sherman-Morrison
    gives the updated covariance after a rank-one information update in closed
    form,

        Sigma' = Sigma - w (Sigma a)(Sigma a)^T / (1 + w a^T Sigma a),

    so the trace reduction this item buys is

        tr(Sigma) - tr(Sigma') = w ||Sigma a||^2 / (1 + w a^T Sigma a).

    The two criteria disagree whenever the posterior is anisotropic: D favours
    the direction that shrinks the ellipsoid's volume, A favours whichever
    single axis is currently widest.
    """

    weight = probability * (1.0 - probability)
    projected = a_candidates @ covariance
    numerator = weight * np.einsum("ij,ij->i", projected, projected)
    quadratic = np.einsum("ij,ij->i", projected, a_candidates)
    return numerator / (1.0 + weight * quadratic)


def _fisher_e_scores(a_candidates, probability, covariance, prior_variance):
    """E-optimal information gain: improve the worst-determined direction.

    E-optimality maximizes the smallest eigenvalue of the information matrix,
    i.e. it attacks whichever ability dimension is currently least pinned
    down, rather than any aggregate. Unlike D and A there is no rank-one
    shortcut, so the eigenvalue is recomputed per candidate; at four ability
    dimensions these are 4x4 symmetric matrices and the cost is negligible.
    """

    weight = probability * (1.0 - probability)
    information = np.linalg.inv(covariance)
    scores = np.empty(len(a_candidates), dtype=float)
    for index, (vector, w) in enumerate(zip(a_candidates, weight, strict=True)):
        updated = information + w * np.outer(vector, vector)
        scores[index] = float(np.linalg.eigvalsh(updated)[0])
    return scores


def _maxmin_position(vectors, remaining, selected):
    """Greedy k-center: the candidate furthest from everything already taken."""

    if not selected:
        return 0
    chosen = vectors[np.asarray(selected, dtype=int)]
    pool = vectors[remaining]
    distance = np.linalg.norm(pool[:, None, :] - chosen[None, :, :], axis=2).min(axis=1)
    return int(np.argmax(distance))


def _fit_calibrator(samples, rng):
    from aims_starter.calibration import AnchorCalibrator

    logits = np.concatenate([sample["logits"] for sample in samples])
    labels = np.concatenate([sample["labels"] for sample in samples])
    rates = np.concatenate(
        [np.full(len(s["labels"]), _context_rate(s["context_labels"])) for s in samples]
    )
    budgets = np.concatenate([np.full(len(s["labels"]), s["budget"], dtype=int) for s in samples])
    if len(labels) > FIT_SAMPLE_LIMIT:
        index = rng.choice(len(labels), size=FIT_SAMPLE_LIMIT, replace=False)
        logits, labels, rates, budgets = logits[index], labels[index], rates[index], budgets[index]
    return AnchorCalibrator.fit(logits, labels, rates, budgets)


def _calibrators(records):
    rng = np.random.default_rng(20261020)
    folds = sorted({int(record["fold"]) for record in records})
    return {
        fold: _fit_calibrator([r for r in records if int(r["fold"]) != fold], rng) for fold in folds
    }


@app.function(
    image=image,
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=8 * 60 * 60,
)
def run_seed(
    rung: str = "mirt_scalar_d4",
    seed: int = 20261018,
    streams: int = 10,
    corpus: str = DEFAULT_CORPUS,
    sampler_tag: str = "-v41e1",
) -> dict[str, Any]:
    import pandas as pd
    import torch

    from aims_starter.metrics import calibration_ece
    from aims_starter.r_c_attention import MultidimContextIRT
    from aims_starter.uniform_eval import validation_fold_map

    # The arm set is part of the cache key: adding a policy must invalidate a
    # previous result rather than silently returning the older, smaller table.
    arm_tag = "-".join(sorted(ARMS))
    output = RESULT_ROOT / f"{rung}-{seed}-{corpus}-{len(ARMS)}arms.json"
    existing = _read_json(output)
    if (
        existing is not None
        and int(existing.get("streams", 0)) == streams
        and "-".join(sorted(existing.get("arms", {}))) == arm_tag
    ):
        return existing

    root = EMBEDDING_ROOT / corpus
    item_embeddings = np.load(root / "item_embeddings.npy")
    subject_embeddings = np.load(root / "subject_embeddings.npy")
    evaluation = pd.read_parquet(root / "eval_items.parquet")
    order_frame = pd.read_parquet(root / "eval_orders.parquet")
    fold_map = validation_fold_map(evaluation, n_folds=UNIFORM_FOLDS)

    checkpoint = (
        ABLATION_ROOT
        / f"seed-{seed}-masked-{corpus}{sampler_tag}"
        / f"{rung}-checkpoints"
        / "model.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}")

    ability_dim, prior_variance, adapter_dim, adapter_prior_variance = _rung_config(rung)
    item_dim = int(item_embeddings.shape[1])
    x_dim = item_dim + int(subject_embeddings.shape[1])
    model = MultidimContextIRT(
        x_dim,
        mode="scalar" if "scalar" in rung else ("direct_ab" if "direct" in rung else "additive_ab"),
        ability_dim=ability_dim,
        loops=3 if "loop3" in rung else 0,
        d_model=128,
        n_heads=8,
        subject_feature_slice=(item_dim, x_dim),
        prior_variance=prior_variance,
        difficulty_adapter_dim=adapter_dim,
        adapter_prior_variance=adapter_prior_variance,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    prior_variance = float(model.prior_variance)
    print(
        f"loaded {rung} D={ability_dim} tau2={prior_variance} "
        f"difficulty_D={adapter_dim} difficulty_tau2={adapter_prior_variance}",
        flush=True,
    )

    validation = evaluation[evaluation["split"] == "validation"]
    task_ids = sorted(validation["task_id"].astype(str).unique())
    records: list[dict[str, Any]] = []

    for task_number, task_id in enumerate(task_ids):
        task_rows = validation[validation["task_id"].astype(str) == task_id]
        benchmark = str(task_rows["benchmark_id"].iloc[0])
        fold = int(fold_map.get(task_id, -1))
        target_rows = task_rows[task_rows["pool"] == "evaluation"].reset_index(drop=True)
        y_target = target_rows["label"].to_numpy(dtype="float32")
        x_target = torch.as_tensor(
            _features(target_rows, item_embeddings, subject_embeddings), dtype=torch.float32
        )
        lookup = task_rows.set_index("item_index")

        # Targets are scored through model.__call__, which recomputes their
        # (a, d) internally; only the context-side parameters are needed here,
        # for the acquisition criteria.
        with torch.no_grad():
            zero_r = model.encode_item(x_target).new_zeros((128,))

        for stream_seed in range(streams):
            order_rows = (
                order_frame[
                    (order_frame["task_id"].astype(str) == task_id)
                    & (order_frame["order_seed"] == stream_seed)
                ]
                .sort_values("acquisition_rank")
                .reset_index(drop=True)
            )
            if order_rows.empty:
                continue
            context_rows = lookup.loc[order_rows["item_index"].to_numpy()].reset_index()
            y_candidate = context_rows["label"].to_numpy(dtype="float32")
            x_candidate_np = _features(context_rows, item_embeddings, subject_embeddings)
            x_candidate = torch.as_tensor(x_candidate_np, dtype=torch.float32)
            with torch.no_grad():
                h_candidate = model.encode_item(x_candidate)
                a_cand_t, d_cand_t = model._head_parameters(h_candidate, zero_r)
                u_cand_t = model._difficulty_loadings(h_candidate)
            a_candidate = a_cand_t.numpy().astype(float)
            d_candidate = d_cand_t.numpy().astype(float)
            u_candidate = u_cand_t.numpy().astype(float)
            embed_candidate = h_candidate.numpy().astype(float)

            for arm, policy in ARMS.items():
                selected: list[int] = []
                remaining = np.arange(len(y_candidate), dtype=int)
                max_labels = min(BUDGETS[-1], len(remaining))

                while len(selected) < max_labels:
                    if policy == "random":
                        position = 0
                    elif policy == "maxmin":
                        position = _maxmin_position(embed_candidate, remaining, selected)
                    else:
                        index = np.asarray(selected, dtype=int)
                        if len(index):
                            loading_mean = u_candidate[index].mean(axis=0, keepdims=True)
                            u_selected = u_candidate[index] - loading_mean
                            u_remaining = u_candidate[remaining] - loading_mean
                        else:
                            u_selected = u_candidate[index]
                            u_remaining = u_candidate[remaining]
                        with torch.no_grad():
                            if adapter_dim:
                                theta_t, adapter_t = model.map_theta_and_difficulty(
                                    torch.as_tensor(a_candidate[index], dtype=torch.float32),
                                    torch.as_tensor(d_candidate[index], dtype=torch.float32),
                                    torch.as_tensor(u_selected, dtype=torch.float32),
                                    torch.as_tensor(y_candidate[index], dtype=torch.float32),
                                    theta_prior_variance=prior_variance,
                                    adapter_prior_variance=adapter_prior_variance,
                                )
                            else:
                                theta_t = model.map_theta(
                                    torch.as_tensor(a_candidate[index], dtype=torch.float32),
                                    torch.as_tensor(d_candidate[index], dtype=torch.float32),
                                    torch.as_tensor(y_candidate[index], dtype=torch.float32),
                                    prior_variance=prior_variance,
                                )
                                adapter_t = theta_t.new_zeros((0,))
                        theta = theta_t.numpy().astype(float)
                        adapter = adapter_t.numpy().astype(float)
                        selected_design = np.concatenate([a_candidate[index], -u_selected], axis=1)
                        candidate_design = np.concatenate(
                            [a_candidate[remaining], -u_remaining], axis=1
                        )
                        state = np.concatenate([theta, adapter])
                        state_prior_variance = np.concatenate(
                            [
                                np.full(ability_dim, prior_variance),
                                np.full(adapter_dim, adapter_prior_variance),
                            ]
                        )
                        probability = _sigmoid(candidate_design @ state - d_candidate[remaining])
                        if policy == "uncertainty":
                            score = -np.abs(probability - 0.5)
                        else:
                            covariance = _posterior_covariance(
                                selected_design,
                                d_candidate[index],
                                state,
                                state_prior_variance,
                            )
                            if policy == "fisher_a":
                                score = _fisher_a_scores(candidate_design, probability, covariance)
                            elif policy == "fisher_e":
                                score = _fisher_e_scores(
                                    candidate_design,
                                    probability,
                                    covariance,
                                    float(state_prior_variance.max()),
                                )
                            else:
                                score = _fisher_d_scores(candidate_design, probability, covariance)
                        position = int(np.argmax(score))
                    selected.append(int(remaining[position]))
                    remaining = np.delete(remaining, position)

                for budget in BUDGETS:
                    prefix = np.asarray(selected[:budget], dtype=int)
                    with torch.no_grad():
                        logits = (
                            model(
                                x_candidate[prefix],
                                torch.as_tensor(y_candidate[prefix], dtype=torch.float32),
                                x_target,
                            )
                            .numpy()
                            .astype(float)
                        )
                    records.append(
                        {
                            "arm": arm,
                            "acquisition": policy,
                            "task_id": task_id,
                            "benchmark_id": benchmark,
                            "fold": fold,
                            "stream_seed": int(stream_seed),
                            "budget": int(budget),
                            "labels": np.asarray(y_target, dtype=float),
                            "context_labels": np.asarray(y_candidate[prefix], dtype=float),
                            "logits": logits,
                        }
                    )
        if task_number % 10 == 0:
            print(f"task {task_number}/{len(task_ids)} {task_id}", flush=True)

    # Out-of-fold calibration, fitted per arm so a policy cannot borrow
    # calibration strength from a different policy's label distribution.
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_records = [r for r in records if r["arm"] == arm]
        fold_calibrators = _calibrators(arm_records)
        for record in arm_records:
            calibrator = fold_calibrators.get(int(record["fold"]))
            raw = _sigmoid(record["logits"])
            calibrated = (
                raw
                if calibrator is None
                else calibrator.transform(
                    record["logits"], int(record["budget"]), record["context_labels"]
                )
            )
            for name, probability in (("raw", raw), ("oof", calibrated)):
                rows.append(
                    {
                        "arm": arm,
                        "calibration": name,
                        "task_id": record["task_id"],
                        "benchmark_id": record["benchmark_id"],
                        "stream_seed": record["stream_seed"],
                        "budget": record["budget"],
                        "brier": float(np.mean((probability - record["labels"]) ** 2)),
                        "ece": float(
                            calibration_ece(
                                record["labels"].tolist(), np.asarray(probability).tolist()
                            )
                        ),
                    }
                )

    import pandas as pd

    frame = pd.DataFrame(rows)
    weight = dict(zip(BUDGETS, WEIGHTS, strict=True))
    frame["weighted"] = frame["budget"].map(weight) * frame["brier"]
    per_task = (
        frame.groupby(["arm", "calibration", "task_id", "stream_seed"], sort=False)["weighted"]
        .sum()
        .reset_index()
        .groupby(["arm", "calibration", "task_id"], sort=False)["weighted"]
        .mean()
        .reset_index()
    )
    summary = []
    for (arm, calibration), group in per_task.groupby(["arm", "calibration"], sort=True):
        summary.append(
            {
                "arm": arm,
                "calibration": calibration,
                "brier_alc": float(group["weighted"].mean()),
                "se": float(group["weighted"].std(ddof=1) / np.sqrt(len(group))),
                "tasks": len(group),
            }
        )

    result = {
        "status": "mirt_acquisition",
        "rung": rung,
        "seed": seed,
        "streams": streams,
        "corpus": corpus,
        "ability_dim": ability_dim,
        "prior_variance": prior_variance,
        "arms": ARMS,
        "summary": summary,
        "task_alc": per_task.to_dict(orient="records"),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
    frame.to_parquet(output.with_suffix(".rows.parquet"), index=False)
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    rung: str = "mirt_scalar_d4",
    seed: int = 20261018,
    seeds: str = "",
    streams: int = 10,
    corpus: str = DEFAULT_CORPUS,
    parallel: bool = False,
) -> None:
    seed_values = [int(value) for value in seeds.split(",") if value] or [int(seed)]
    if parallel:
        calls = [(value, run_seed.spawn(rung, value, streams, corpus)) for value in seed_values]
        results = [(value, call.get()) for value, call in calls]
    else:
        results = [(value, run_seed.remote(rung, value, streams, corpus)) for value in seed_values]
    for value, result in results:
        print(f"\nseed={value} tau2={result['prior_variance']}")
        print(f"{'arm':<8}{'calib':<8}{'Brier ALC':>12}{'se':>10}")
        for row in sorted(result["summary"], key=lambda r: (r["calibration"], r["brier_alc"])):
            print(
                f"{row['arm']:<8}{row['calibration']:<8}{row['brier_alc']:>12.6f}{row['se']:>10.6f}"
            )
