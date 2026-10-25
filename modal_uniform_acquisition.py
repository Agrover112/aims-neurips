"""Uniform acquisition ablation on the fixed validation grid.

Every policy sees the same 104 validation tasks and the same 10 random candidate
streams.  At each step a ranking policy scores every remaining candidate in the
stream and selects the highest-scoring one.  The old threshold-plus-reserve rule
is deliberately removed.

Arms
----
C-R   CNP-AIRT predictor, random stream order.
C-U   CNP-AIRT predictor, rank by closeness to 0.5.
C-F   CNP-AIRT predictor, rank by IRT Fisher information
      a_i^2 p_i (1-p_i) at the current MAP ability.
C-D   CNP-AIRT predictor, rank by |p_C - p_beta|.
beta-R Beta-Binomial predictor, random stream order.

The CNP predictor uses an AnchorCalibrator fitted out of fold on the validation
tasks.  The calibration records come from the same frozen orders and budgets as
the ladder, so the acquisition and architecture ablations share one grid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "aims-neurips-uniform-acquisition"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "allbench-features-v1"
RESULT_ROOT = REMOTE_ROOT / "results" / "allbench-uniform-v1" / "acquisition"
CHECKPOINT_ROOT = REMOTE_ROOT / "results" / "allbench-deterministic-2seed"
SPLIT = "validation"
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
ORDER_SEEDS = tuple(range(10))
UNIFORM_FOLDS = 5
FIT_SAMPLE_LIMIT = 1_000_000
SUBJECT_SLICE = (384, 768)
SEEDS = (20261018, 20261019)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scikit-learn>=1.5,<2")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(PROJECT_ROOT / "src", "/opt/aims/src", copy=True)
    .env({"PYTHONPATH": "/opt/aims/src", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
)

ARMS: dict[str, dict[str, Any]] = {
    "C-R": {
        "predictor": "cnp",
        "acquisition": "random",
        "label": "P_C + random",
        "description": "CNP-AIRT predictor, first remaining candidate in the stream.",
    },
    "C-U": {
        "predictor": "cnp",
        "acquisition": "uncertainty",
        "label": "P_C + uncertainty",
        "description": "CNP-AIRT predictor, rank by |p_C - 0.5|.",
    },
    "C-F": {
        "predictor": "cnp",
        "acquisition": "fisher",
        "label": "P_C + Fisher",
        "description": "CNP-AIRT predictor, rank by a_i^2 p_i (1-p_i).",
    },
    "C-KM": {
        "predictor": "cnp",
        "acquisition": "cluster_embed",
        "label": "P_C + Clustering-Embed",
        "description": (
            "CNP-AIRT predictor, static order from k-means on the item "
            "embeddings: one item per centroid, nearest first. Ignores labels."
        ),
    },
    "C-KI": {
        "predictor": "cnp",
        "acquisition": "cluster_irt",
        "label": "P_C + Clustering-IRT",
        "description": (
            "CNP-AIRT predictor, static order from k-means on the fitted "
            "(a_i, b_i) IRT parameters rather than the raw embedding."
        ),
    },
    "C-MM": {
        "predictor": "cnp",
        "acquisition": "maxmin",
        "label": "P_C + max-min diversity",
        "description": (
            "CNP-AIRT predictor, greedy k-center: each step takes the "
            "candidate furthest from everything already selected."
        ),
    },
    "beta-R": {
        "predictor": "beta",
        "acquisition": "random",
        "label": "P_beta + random",
        "description": "Beta-Binomial predictor, first remaining candidate in the stream.",
    },
}



def _cluster_priority(vectors, budget: int, seed: int):
    """Static acquisition order from k-means: one item per centroid.

    This is the Clustering-Embed / Clustering-IRT family. ``budget`` clusters
    are fitted on the candidate representations, and each cluster contributes
    the single item closest to its centroid. Those representatives are ranked
    by their distance to their own centroid, so the most prototypical items go
    first; everything else is appended afterwards in distance order so the
    policy still has a defined ordering past ``budget`` labels.

    The order does not depend on any acquired label, which is the defining
    property of a static policy: it is computed once and never revised.
    """

    import numpy as np
    from sklearn.cluster import KMeans

    count = len(vectors)
    k = int(min(budget, count))
    if k < 1:
        return np.zeros(count, dtype=float)
    model = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(vectors)
    distance = np.linalg.norm(vectors - model.cluster_centers_[model.labels_], axis=1)

    priority = np.full(count, np.inf, dtype=float)
    representatives = []
    for cluster in range(k):
        members = np.flatnonzero(model.labels_ == cluster)
        if members.size:
            representatives.append(int(members[np.argmin(distance[members])]))

    # Order the representatives by farthest-point traversal rather than by
    # distance to their own centroid. The budget grid is nested (1, 3, 7, ...)
    # and budgets 1 and 3 carry weight 0.2 each, so the first few labels have
    # to span the space: ranking by prototypicality instead puts the tightest
    # clusters first, which at k = 31 over a handful of real groups can spend
    # every early label inside one region.
    ordered = []
    pool = list(representatives)
    if pool:
        start = min(pool, key=lambda index: distance[index])
        ordered.append(start)
        pool.remove(start)
    while pool:
        spread = np.linalg.norm(
            vectors[pool][:, None, :] - vectors[ordered][None, :, :], axis=2
        ).min(axis=1)
        pick = pool[int(np.argmax(spread))]
        ordered.append(pick)
        pool.remove(pick)
    representatives = ordered
    for rank, index in enumerate(representatives):
        priority[index] = float(rank)

    rest = [index for index in range(count) if not np.isfinite(priority[index])]
    rest.sort(key=lambda index: distance[index])
    for offset, index in enumerate(rest):
        priority[index] = float(len(representatives) + offset)
    return priority


def _maxmin_position(vectors, remaining, selected):
    """Greedy k-center: take the candidate furthest from everything chosen.

    Unlike the clustering arms this one updates as items are selected, but it
    still never reads a label -- it is a pure coverage policy over the
    representation space, so it isolates "spread" from "informativeness".
    """

    import numpy as np

    if not selected:
        return 0
    chosen = vectors[np.asarray(selected, dtype=int)]
    pool = vectors[remaining]
    distance = np.linalg.norm(pool[:, None, :] - chosen[None, :, :], axis=2).min(axis=1)
    return int(np.argmax(distance))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _beta_probability(successes: float, count: int) -> float:
    return (2.0 + successes) / (4.0 + count)


def _sigmoid(value):
    import numpy as np

    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def _context_theta(model, encoded_c, labels_c):
    import torch

    if encoded_c.shape[0] == 0:
        return 0.0
    with torch.no_grad():
        theta = model.map_theta(encoded_c, labels_c)
        label_value = labels_c[:, None]
        residual = model.r_encoder(torch.cat([encoded_c, label_value], dim=-1))
        theta = theta + model.theta_delta(residual.mean(dim=0)).squeeze(-1)
    return float(theta.detach().cpu())


def _fit_calibrator(samples: list[dict[str, Any]]):
    import numpy as np

    from aims_starter.calibration import AnchorCalibrator, context_rate

    logits = np.concatenate([sample["logits"] for sample in samples])
    labels = np.concatenate([sample["labels"] for sample in samples])
    rates = np.asarray([context_rate(sample["context_labels"]) for sample in samples])
    rates = np.repeat(rates, [len(sample["labels"]) for sample in samples])
    budgets = np.repeat(
        [sample["budget"] for sample in samples], [len(sample["labels"]) for sample in samples]
    )
    rng = np.random.default_rng(20261020)
    if len(labels) > FIT_SAMPLE_LIMIT:
        index = rng.choice(len(labels), size=FIT_SAMPLE_LIMIT, replace=False)
        logits, labels, rates, budgets = (
            logits[index],
            labels[index],
            rates[index],
            budgets[index],
        )
    return AnchorCalibrator.fit(logits, labels, rates, budgets)


@app.function(
    image=image,
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=6 * 60 * 60,
)
def run_seed(seed: int = 20261018, streams: int = 10) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch

    from aims_starter.calibration import AnchorCalibrator
    from aims_starter.faithful_np import CNPAIRT
    from aims_starter.metrics import calibration_ece
    from aims_starter.uniform_eval import validation_fold_map

    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    output = RESULT_ROOT / f"seed-{seed}.json"
    rows_path = RESULT_ROOT / f"seed-{seed}-rows.parquet"
    existing = _read_json(output)
    if existing is not None and int(existing.get("streams", 0)) == streams:
        return existing

    device = torch.device("cpu")
    item_embeddings = np.load(CORPUS_ROOT / "item_embeddings.npy")
    subject_embeddings = np.load(CORPUS_ROOT / "subject_embeddings.npy")
    evaluation_items = pd.read_parquet(CORPUS_ROOT / "eval_items.parquet")
    evaluation_items = evaluation_items[evaluation_items["split"] == SPLIT].copy()
    orders = pd.read_parquet(CORPUS_ROOT / "eval_orders.parquet")
    fold_map = validation_fold_map(evaluation_items, n_folds=UNIFORM_FOLDS)
    task_ids = sorted(evaluation_items["task_id"].astype(str).unique())
    print(f"seed={seed} tasks={len(task_ids)} streams={streams}", flush=True)

    checkpoint = CHECKPOINT_ROOT / f"seed-{seed}-masked-checkpoints" / "cnp_airt.pt"
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = CNPAIRT(
        item_embeddings.shape[1] + subject_embeddings.shape[1],
        subject_feature_slice=SUBJECT_SLICE,
    )
    model.load_state_dict(state)
    model.eval()

    def features(item_index, subject_index):
        return np.concatenate(
            [item_embeddings[item_index], subject_embeddings[subject_index]], axis=1
        ).astype("float32")

    def encode(x):
        with torch.no_grad():
            encoded = model.encode_item_x(torch.as_tensor(x, dtype=torch.float32, device=device))
            raw = model.irt_parameter_net(encoded)
            a = torch.nn.functional.softplus(raw[:, 0]) + 0.05
            b = raw[:, 1]
        return encoded, a.detach().cpu().numpy(), b.detach().cpu().numpy()

    # Precompute target and candidate features plus item parameters once per task.
    tasks: dict[str, dict[str, Any]] = {}
    calibration_records: list[dict[str, Any]] = []
    for task_id in task_ids:
        task_rows = evaluation_items[evaluation_items["task_id"].astype(str) == task_id]
        target_rows = task_rows[task_rows["pool"] == "evaluation"].reset_index(drop=True)
        candidate_rows = task_rows[task_rows["pool"] == "acquisition"].reset_index(drop=True)
        if target_rows.empty or candidate_rows.empty:
            continue
        bench = str(task_rows["benchmark_id"].iloc[0])
        x_target = features(
            target_rows["item_index"].to_numpy(), target_rows["subject_index"].to_numpy()
        )
        x_candidate = features(
            candidate_rows["item_index"].to_numpy(), candidate_rows["subject_index"].to_numpy()
        )
        encoded_target, a_target, b_target = encode(x_target)
        encoded_candidate, a_candidate, b_candidate = encode(x_candidate)
        candidate_lookup = {
            int(item_index): position
            for position, item_index in enumerate(candidate_rows["item_index"].to_numpy())
        }
        y_target = target_rows["label"].to_numpy(dtype=float)
        y_candidate = candidate_rows["label"].to_numpy(dtype=float)
        task = {
            "task_id": task_id,
            "benchmark_id": bench,
            "fold": int(fold_map[task_id]),
            "y_target": y_target,
            "y_candidate": y_candidate,
            "encoded_target": encoded_target,
            "a_target": a_target,
            "b_target": b_target,
            "encoded_candidate": encoded_candidate,
            "a_candidate": a_candidate,
            "b_candidate": b_candidate,
            "candidate_lookup": candidate_lookup,
        }
        tasks[task_id] = task

        # Calibration records use the same ten frozen orders as the ladder.
        for order_seed in ORDER_SEEDS:
            order_rows = (
                orders[
                    (orders["task_id"].astype(str) == task_id)
                    & (orders["order_seed"] == order_seed)
                ]
                .sort_values("acquisition_rank")
                .reset_index(drop=True)
            )
            positions = np.asarray(
                [candidate_lookup[int(item)] for item in order_rows["item_index"].to_numpy()],
                dtype=int,
            )
            encoded_c = encoded_candidate[positions]
            labels_c = torch.as_tensor(y_candidate[positions], dtype=torch.float32)
            for budget in BUDGETS:
                theta = _context_theta(model, encoded_c[:budget], labels_c[:budget])
                logits = a_target * (theta - b_target)
                calibration_records.append(
                    {
                        "task_id": task_id,
                        "fold": int(fold_map[task_id]),
                        "budget": int(budget),
                        "logits": np.asarray(logits, dtype=float),
                        "labels": y_target,
                        "context_labels": y_candidate[positions[:budget]],
                    }
                )
        print(f"prepared {task_id}", flush=True)

    print(f"fitting fold calibrators on {len(calibration_records)} records", flush=True)
    full_calibrator = _fit_calibrator(calibration_records)
    fold_calibrators = {
        fold: _fit_calibrator(
            [record for record in calibration_records if int(record["fold"]) != fold]
        )
        for fold in range(UNIFORM_FOLDS)
    }
    print("calibrators fitted", flush=True)

    rows: list[dict[str, Any]] = []
    for task_number, task_id in enumerate(task_ids):
        task = tasks.get(task_id)
        if task is None:
            continue
        y_target = task["y_target"]
        y_candidate = task["y_candidate"]
        encoded_target = task["encoded_target"]
        a_target = task["a_target"]
        b_target = task["b_target"]
        encoded_candidate = task["encoded_candidate"]
        a_candidate = task["a_candidate"]
        b_candidate = task["b_candidate"]
        fold = int(task["fold"])
        calibrator = fold_calibrators[fold]

        for stream_seed in range(streams):
            rng = np.random.default_rng([stream_seed, task_number])
            stream = rng.permutation(len(y_candidate))
            for arm, arm_spec in ARMS.items():
                acquisition = arm_spec["acquisition"]
                static_priority = None
                if acquisition == "cluster_embed":
                    static_priority = _cluster_priority(
                        encoded_candidate, BUDGETS[-1], stream_seed
                    )
                elif acquisition == "cluster_irt":
                    static_priority = _cluster_priority(
                        np.stack([a_candidate, b_candidate], axis=1),
                        BUDGETS[-1],
                        stream_seed,
                    )
                selected: list[int] = []
                selected_rank: list[float] = []
                remaining = np.asarray(stream, dtype=int)
                max_labels = min(BUDGETS[-1], len(remaining))
                while len(selected) < max_labels:
                    if static_priority is not None:
                        chosen_position = int(np.argmin(static_priority[remaining]))
                    elif acquisition == "maxmin":
                        chosen_position = _maxmin_position(
                            encoded_candidate, remaining, selected
                        )
                    elif arm_spec["predictor"] == "beta":
                        chosen_position = 0
                        p_beta = _beta_probability(
                            float(y_candidate[selected].sum()) if selected else 0.0,
                            len(selected),
                        )
                    else:
                        if selected:
                            encoded_c = encoded_candidate[np.asarray(selected, dtype=int)]
                            labels_c = torch.as_tensor(
                                y_candidate[np.asarray(selected, dtype=int)], dtype=torch.float32
                            )
                        else:
                            encoded_c = torch.zeros(
                                (0, encoded_target.shape[1]), dtype=torch.float32
                            )
                            labels_c = torch.zeros((0,), dtype=torch.float32)
                        theta = _context_theta(model, encoded_c, labels_c)
                        logits = a_candidate[remaining] * (
                            theta - b_candidate[remaining]
                        )
                        p_raw = _sigmoid(logits)
                        if calibrator is None:
                            p_cal = p_raw
                        else:
                            calibration_budget = max(
                                value for value in BUDGETS if value <= len(selected)
                            )
                            p_cal = calibrator.transform(
                                logits,
                                calibration_budget,
                                labels_c.detach().cpu().numpy(),
                            )
                        successes = float(y_candidate[selected].sum()) if selected else 0.0
                        p_beta = _beta_probability(successes, len(selected))
                        if arm_spec["acquisition"] == "fisher":
                            score = a_candidate[remaining] ** 2 * p_raw * (1.0 - p_raw)
                            chosen_position = int(np.argmax(score))
                        elif arm_spec["acquisition"] == "uncertainty":
                            score = -np.abs(p_cal - 0.5)
                            chosen_position = int(np.argmax(score))
                        else:
                            chosen_position = 0
                    chosen = int(remaining[chosen_position])
                    selected.append(chosen)
                    selected_rank.append(float(chosen_position))
                    remaining = np.delete(remaining, chosen_position)

                for budget in BUDGETS:
                    prefix = selected[:budget]
                    actual = len(prefix)
                    calibration_budget = max(value for value in BUDGETS if value <= actual)
                    successes = float(y_candidate[np.asarray(prefix, dtype=int)].sum())
                    p_beta = _beta_probability(successes, actual)
                    if arm_spec["predictor"] == "beta":
                        p_target = np.full(len(y_target), p_beta)
                    else:
                        encoded_c = encoded_candidate[np.asarray(prefix, dtype=int)]
                        labels_c = torch.as_tensor(
                            y_candidate[np.asarray(prefix, dtype=int)], dtype=torch.float32
                        )
                        theta = _context_theta(model, encoded_c, labels_c)
                        logits = a_target * (theta - b_target)
                        if calibrator is None:
                            p_target = _sigmoid(logits)
                        else:
                            p_target = calibrator.transform(
                                logits,
                                calibration_budget,
                                labels_c.detach().cpu().numpy(),
                            )
                    brier = float(np.mean((p_target - y_target) ** 2))
                    ece = float(calibration_ece(y_target.tolist(), p_target.tolist()))
                    ranks = selected_rank[:budget]
                    rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "predictor": arm_spec["predictor"],
                            "acquisition": arm_spec["acquisition"],
                            "task_id": task_id,
                            "fold": fold,
                            "stream_seed": stream_seed,
                            "budget": budget,
                            "brier": brier,
                            "ece": ece,
                            "labels_acquired": actual,
                            "mean_selected_rank": float(np.mean(ranks)) if ranks else 0.0,
                            "stream_length": len(stream),
                        }
                    )
        print(f"task {task_number + 1}/{len(task_ids)} {task_id}", flush=True)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no acquisition rows produced")
    weights = dict(zip(BUDGETS, WEIGHTS, strict=True))
    frame["weighted_brier"] = frame["brier"] * frame["budget"].map(weights)
    frame["weighted_ece"] = frame["ece"] * frame["budget"].map(weights)
    per_run = (
        frame.groupby(["arm", "task_id", "stream_seed"], sort=False)
        .agg(brier_alc=("weighted_brier", "sum"), ece_alc=("weighted_ece", "sum"))
        .reset_index()
    )
    frame = frame.merge(per_run, on=["arm", "task_id", "stream_seed"], how="left")

    summary_rows = []
    for arm, group in per_run.groupby("arm", sort=False):
        values = group["brier_alc"].to_numpy(dtype=float)
        eces = group["ece_alc"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "arm": arm,
                "label": ARMS[arm]["label"],
                "description": ARMS[arm]["description"],
                "runs": len(group),
                "brier_alc_mean": float(values.mean()),
                "brier_alc_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "brier_alc_se": float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0,
                "ece_alc_mean": float(eces.mean()),
                "ece_alc_sd": float(eces.std(ddof=1)) if len(eces) > 1 else 0.0,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("brier_alc_mean").reset_index(drop=True)

    by_budget_rows = []
    for (arm, budget), group in frame.groupby(["arm", "budget"], sort=False):
        by_budget_rows.append(
            {
                "arm": arm,
                "budget": int(budget),
                "brier_mean": float(group["brier"].mean()),
                "ece_mean": float(group["ece"].mean()),
                "mean_selected_rank": float(group["mean_selected_rank"].mean()),
                "mean_stream_length": float(group["stream_length"].mean()),
            }
        )

    deltas = []
    baseline = per_run[per_run["arm"] == "C-R"][
        ["task_id", "stream_seed", "brier_alc"]
    ].rename(columns={"brier_alc": "baseline_alc"})
    for arm, group in per_run.groupby("arm", sort=False):
        if arm == "C-R":
            continue
        joined = group.merge(baseline, on=["task_id", "stream_seed"], how="inner")
        delta = (joined["brier_alc"] - joined["baseline_alc"]).to_numpy(dtype=float)
        deltas.append(
            {
                "arm": arm,
                "baseline": "C-R",
                "delta_mean": float(delta.mean()),
                "delta_sd": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                "delta_se": float(delta.std(ddof=1) / np.sqrt(len(delta)))
                if len(delta) > 1
                else 0.0,
                "wins": int((delta < 0).sum()),
                "paired_runs": len(delta),
            }
        )

    artifact = {
        "status": "uniform_acquisition_ablation",
        "split": SPLIT,
        "seed": seed,
        "protocol": "top-score ranking over the same random candidate stream; selection removes only the chosen item",
        "predictors": {
            "beta": "Laplace-smoothed Beta-Binomial, p=(2+successes)/(4+n)",
            "cnp": "CNP-AIRT masked checkpoint, out-of-fold AnchorCalibrator",
        },
        "folds": int(UNIFORM_FOLDS),
        "fold_fits": {
            str(fold): calibrator.summary()
            for fold, calibrator in fold_calibrators.items()
        },
        "full_fit": full_calibrator.summary(),
        "tasks": len(task_ids),
        "streams": streams,
        "budgets": list(BUDGETS),
        "weights": list(WEIGHTS),
        "summary": summary.to_dict(orient="records"),
        "deltas_vs_random": deltas,
        "by_budget": by_budget_rows,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(output, artifact)
    frame.to_parquet(rows_path, index=False)
    volume.commit()
    print(json.dumps(artifact["summary"], indent=2), flush=True)
    return artifact


@app.local_entrypoint()
def main(seed: int = 0, streams: int = 10) -> None:
    if seed:
        print(json.dumps(run_seed.remote(seed, streams), indent=2, sort_keys=True))
        return
    handles = [run_seed.spawn(seed_value, streams) for seed_value in SEEDS]
    for handle in handles:
        handle.get()
