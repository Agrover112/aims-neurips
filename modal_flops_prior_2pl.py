"""Very simple 2PL online-MAP probe for a log-FLOPs ability prior.

This is deliberately a *scoring-only* experiment.  It loads the existing
me5 ``2pl_tau4`` checkpoint, estimates one ability per training task, fits a
within-benchmark log-FLOPs prior on those training abilities, and then re-scores
the same 104 validation tasks with different MAP prior means.

No validation target label is used to fit the prior or the calibrator folds.

Run:
    modal run modal_flops_prior_2pl.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import modal
import numpy as np
import pandas as pd

APP_NAME = "aims-neurips-flops-prior-2pl"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS = "allbench-train-me5_instruct-cap128-v2"
STORE_ROOT = REMOTE_ROOT / "datasets" / "full" / CORPUS
CHECKPOINT = (
    REMOTE_ROOT
    / "results"
    / "r-c-ablation"
    / "seed-20261018-masked-allbench-train-me5_instruct-cap128-v2-v41e1"
    / "2pl_tau4-checkpoints"
    / "model.pt"
)
RESULT_ROOT = REMOTE_ROOT / "results" / "flops-prior-2pl"
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
FIT_SAMPLE_LIMIT = 1_000_000
N_FOLDS = 5
ORDERS = 10
PRIOR_TAU = 4.0
PRIOR_TAU_STRONG = 1.0

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


def _context_rate(labels: np.ndarray) -> float:
    values = np.asarray(labels, dtype=float)
    return (1.0 + float(values.sum())) / (2.0 + len(values))


def _normalise_text(value: Any) -> str:
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    text = text.replace(".", " ")
    return re.sub(r"\s+", " ", text).strip()


def _flops_for_row(row: pd.Series) -> float | None:
    """Map a subject row to an approximate total training FLOPs.

    Values are in 1e21 FLOPs, matching the Amortized IRT paper table.  When a
    model is a fine-tune, the base model's pretraining compute is used.  Missing
    means "not safely mappable"; the prior then falls back to the global mean.
    """

    text = " ".join(
        [
            str(row.get("normalized_name", "")),
            str(row.get("subject_features_extra", "")),
            str(row.get("source_directory", "")),
        ]
    )
    norm = _normalise_text(text)

    # Most specific patterns first.  All values are 1e21 FLOPs.
    patterns: tuple[tuple[str, float], ...] = (
        (r"llama 3 1 405b|llama3 405b|llama 3 405b", 36450.0),
        (r"llama 3 1 70b|llama3 1 70b", 6300.0),
        (r"llama 3 1 8b|llama3 1 8b", 720.0),
        (r"llama 3 70b|llama3 70b", 6300.0),
        (r"llama 3 8b|llama3 8b", 720.0),
        (r"llama 2 70b|llama2 70b", 840.0),
        (r"llama 2 13b|llama2 13b", 156.0),
        (r"llama 2 7b|llama2 7b", 84.0),
        (r"code ?llama 34b|codellama 34b", 102.0),
        (r"code ?llama 13b|codellama 13b", 39.0),
        (r"mixtral 8x7b", 280.2),
        (r"mistral 7b", 42.0),
        (r"qwen2 5 72b|qwen 2 5 72b", 7776.0),
        (r"qwen2 5 32b|qwen 2 5 32b", 3456.0),
        (r"qwen2 5 7b|qwen 2 5 7b", 756.0),
        (r"qwen3 32b|qwen 3 32b", 6912.0),
        (r"qwen3 14b|qwen 3 14b", 3024.0),
        (r"qwen3 8b|qwen 3 8b", 1728.0),
        (
            r"deepseek v3|deepseek chat v3|deepseek v3 1|deepseek r1|deepseek reasoner",
            60828.0,
        ),
        (r"deepseek coder 33b", 396.0),
        (r"gemma 2 27b", 2106.0),
        (r"gemma 2 9b", 432.0),
        (r"gemma 7b", 252.0),
        (r"phi 3 medium", 403.2),
        (r"phi 3 small", 201.6),
        (r"phi 3 mini", 75.24),
        (r"phi 2", 22.68),
        (r"yi 34b", 612.0),
        (r"yi 6b", 108.0),
        (r"olmo 7b", 105.0),
        (r"falcon 40b", 240.0),
        (r"falcon 7b", 63.0),
        (r"dbrx", 9504.0),
        (r"openbiollm 70b|openbio llm 70b", 6300.0),
        (r"openbiollm 8b|openbio llm 8b", 720.0),
        (r"biomistral 7b", 42.0),
        (r"meditron 7b", 84.0),
        (r"pmc llama 7b", 84.0),
        (r"jsl med llama 8b", 720.0),
    )
    for pattern, value in patterns:
        if re.search(pattern, norm):
            return float(value)
    return None


def _load_subjects() -> pd.DataFrame:
    subjects = pd.read_parquet(STORE_ROOT / "subjects.parquet").reset_index(drop=True)
    subjects["subject_index"] = np.arange(len(subjects), dtype=int)
    meta = pd.read_parquet(REMOTE_ROOT / "datasets" / "full" / "processed" / "subjects.parquet")
    meta = meta.drop_duplicates("subject_uid").copy()
    meta["flops"] = meta.apply(_flops_for_row, axis=1)
    subjects = subjects.merge(
        meta[
            [
                "subject_uid",
                "normalized_name",
                "provider",
                "subject_features_extra",
                "source_directory",
                "flops",
            ]
        ],
        on="subject_uid",
        how="left",
    )
    subjects["normalized_name"] = (
        subjects["normalized_name"].fillna("").astype(str).str.strip()
    )
    return subjects


def _item_x(item_indices: np.ndarray, item_embeddings: np.ndarray, device) -> Any:
    import torch

    n = len(item_indices)
    x = np.zeros((n, 2 * item_embeddings.shape[1]), dtype=np.float32)
    x[:, : item_embeddings.shape[1]] = item_embeddings[item_indices]
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _estimate_training_theta(
    model,
    train: pd.DataFrame,
    subjects: pd.DataFrame,
    item_embeddings: np.ndarray,
    device,
) -> pd.DataFrame:
    import torch

    names = subjects["normalized_name"].to_numpy()
    flops = subjects["flops"].to_numpy(dtype=float)
    uids = subjects["subject_uid"].to_numpy()
    rows: list[dict[str, Any]] = []

    for count, (task_id, group) in enumerate(train.groupby("task_id", sort=True), start=1):
        subject_index = int(group["subject_index"].iloc[0])
        item_index = group["item_index"].to_numpy()
        y = torch.as_tensor(
            group["label"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device
        )
        x = _item_x(item_index, item_embeddings, device)
        with torch.no_grad():
            h = model.encode_item(x)
            a, b = model._anchor_parameters(h)
            theta = float(
                model.map_theta(
                    a,
                    b,
                    y,
                    prior_mean=0.0,
                    prior_variance=PRIOR_TAU,
                )
                .detach()
                .cpu()
            )
        rows.append(
            {
                "task_id": str(task_id),
                "benchmark": str(task_id).split("::")[-1],
                "subject_uid": str(uids[subject_index]),
                "normalized_name": str(names[subject_index]),
                "flops": flops[subject_index],
                "theta": theta,
                "n_items": int(len(group)),
            }
        )
        if count % 500 == 0:
            print(f"training theta {count}/{train['task_id'].nunique()}", flush=True)

    return pd.DataFrame(rows)


def _fit_priors(train_theta: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    frame = train_theta.copy()
    frame["benchmark_mean"] = frame.groupby("benchmark")["theta"].transform("mean")
    frame["benchmark_residual"] = frame["theta"] - frame["benchmark_mean"]
    known = frame[frame["flops"].notna() & np.isfinite(frame["flops"])].copy()
    known["log_flops"] = np.log(known["flops"].to_numpy(dtype=float))
    log_mean = float(known["log_flops"].mean())
    log_std = float(known["log_flops"].std(ddof=0))
    if log_std <= 0.0:
        raise ValueError("log FLOPs has zero variance")
    known["z"] = (known["log_flops"] - log_mean) / log_std
    frame["z"] = np.nan
    valid_flops = frame["flops"].notna()
    frame.loc[valid_flops, "z"] = (
        np.log(frame.loc[valid_flops, "flops"].to_numpy(dtype=float)) - log_mean
    ) / log_std

    # Within-benchmark slope: benchmark fixed effects plus z.  The validation
    # benchmarks are held out, so only the slope and global level transfer.
    dummies = pd.get_dummies(known["benchmark"], drop_first=True, dtype=float)
    design_bench = np.column_stack([np.ones(len(known)), dummies.to_numpy()])
    design = np.column_stack([design_bench, known["z"].to_numpy()])
    beta, *_ = np.linalg.lstsq(design, known["theta"].to_numpy(), rcond=None)
    slope = float(beta[-1])

    theta_res = known["theta"].to_numpy() - design_bench @ np.linalg.lstsq(
        design_bench, known["theta"].to_numpy(), rcond=None
    )[0]
    z_res = known["z"].to_numpy() - design_bench @ np.linalg.lstsq(
        design_bench, known["z"].to_numpy(), rcond=None
    )[0]
    residual = theta_res - slope * z_res
    r2_within = float(1.0 - np.sum(residual**2) / np.sum(theta_res**2))
    corr_within = float(np.corrcoef(theta_res, z_res)[0, 1])

    global_mean = float(frame["theta"].mean())
    model_stats = (
        frame[frame["normalized_name"] != ""]
        .groupby("normalized_name", sort=False)
        .agg(
            model_resid=("benchmark_residual", "mean"),
            n_tasks=("theta", "size"),
            z_mean=("z", "mean"),
            flops_mean=("flops", "mean"),
        )
        .reset_index()
    )
    # A model prior needs at least two training tasks to be a real cross-task
    # estimate rather than a relabelled single-task ability.
    model_stats = model_stats[model_stats["n_tasks"] >= 2].copy()

    fit = {
        "global_mean": global_mean,
        "log_flops_mean": log_mean,
        "log_flops_std": log_std,
        "slope": slope,
        "r2_within": r2_within,
        "corr_within": corr_within,
        "n_train_tasks": int(len(frame)),
        "n_train_flops": int(len(known)),
        "n_train_benchmarks": int(known["benchmark"].nunique()),
        "n_models_with_prior": int(len(model_stats)),
    }
    return fit, frame, model_stats


def _prior_specs(
    subject_index: int,
    subjects: pd.DataFrame,
    model_stats: pd.DataFrame,
    fit: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    row = subjects.iloc[int(subject_index)]
    flops = row["flops"]
    name = str(row["normalized_name"])
    global_mean = float(fit["global_mean"])
    slope = float(fit["slope"])
    log_mean = float(fit["log_flops_mean"])
    log_std = float(fit["log_flops_std"])

    z = None
    if pd.notna(flops) and float(flops) > 0.0:
        z = (math.log(float(flops)) - log_mean) / log_std

    model_row = model_stats[model_stats["normalized_name"] == name]
    model_resid = None
    model_z_mean = None
    if len(model_row) > 0:
        model_resid = float(model_row["model_resid"].iloc[0])
        if pd.notna(model_row["z_mean"].iloc[0]):
            model_z_mean = float(model_row["z_mean"].iloc[0])

    flops_prior = global_mean + slope * z if z is not None else global_mean
    model_prior = (
        global_mean + model_resid if model_resid is not None else global_mean
    )
    if model_resid is not None and z is not None:
        flops_model_prior = global_mean + model_resid + slope * (z - (model_z_mean or 0.0))
    elif model_resid is not None:
        flops_model_prior = model_prior
    else:
        flops_model_prior = flops_prior

    return {
        "zero": (0.0, PRIOR_TAU),
        "zero_tau1": (0.0, PRIOR_TAU_STRONG),
        "global": (global_mean, PRIOR_TAU),
        "global_tau1": (global_mean, PRIOR_TAU_STRONG),
        "flops": (flops_prior, PRIOR_TAU),
        "flops_tau1": (flops_prior, PRIOR_TAU_STRONG),
        "model": (model_prior, PRIOR_TAU),
        "model_tau1": (model_prior, PRIOR_TAU_STRONG),
        "flops_model": (flops_model_prior, PRIOR_TAU),
        "flops_model_tau1": (flops_model_prior, PRIOR_TAU_STRONG),
    }


def _score_validation(
    model,
    subjects: pd.DataFrame,
    model_stats: pd.DataFrame,
    fit: dict[str, Any],
    evaluation: pd.DataFrame,
    orders: pd.DataFrame,
    item_embeddings: np.ndarray,
    device,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, float]]:
    import torch

    from aims_starter.uniform_eval import validation_fold_map

    fold_map = validation_fold_map(evaluation, n_folds=N_FOLDS)
    validation = evaluation[evaluation["split"] == "validation"].copy()
    task_meta = (
        validation[["task_id", "subject_index", "benchmark_id"]]
        .drop_duplicates("task_id")
        .reset_index(drop=True)
    )
    task_flops = {
        str(task_id): subjects.iloc[int(subject_index)]["flops"]
        for task_id, subject_index in zip(
            task_meta["task_id"].astype(str), task_meta["subject_index"]
        )
    }

    records: dict[str, list[dict[str, Any]]] = {}
    beta_records: list[dict[str, Any]] = []
    for count, task_id in enumerate(sorted(validation["task_id"].astype(str).unique()), start=1):
        task_rows = validation[validation["task_id"].astype(str) == task_id]
        benchmark = str(task_rows["benchmark_id"].iloc[0])
        subject_index = int(task_rows["subject_index"].iloc[0])
        target_rows = task_rows[task_rows["pool"] == "evaluation"].reset_index(drop=True)
        y_t = target_rows["label"].to_numpy(dtype=np.float32)
        x_t = _item_x(target_rows["item_index"].to_numpy(), item_embeddings, device)
        with torch.no_grad():
            h_t = model.encode_item(x_t)
            a_t, b_t = model._anchor_parameters(h_t)

        priors = _prior_specs(subject_index, subjects, model_stats, fit)
        for arm in priors:
            records.setdefault(arm, [])

        lookup = task_rows.set_index("item_index")
        for order_seed in range(ORDERS):
            order_rows = (
                orders[
                    (orders["task_id"].astype(str) == task_id)
                    & (orders["order_seed"] == order_seed)
                ]
                .sort_values("acquisition_rank")
                .reset_index(drop=True)
            )
            context_rows = lookup.loc[order_rows["item_index"].to_numpy()].reset_index()
            y_c = context_rows["label"].to_numpy(dtype=np.float32)
            x_c = _item_x(context_rows["item_index"].to_numpy(), item_embeddings, device)
            with torch.no_grad():
                h_c = model.encode_item(x_c)
                a_c, b_c = model._anchor_parameters(h_c)

            for budget in BUDGETS:
                y_b = torch.as_tensor(
                    y_c[:budget], dtype=torch.float32, device=device
                )
                base = {
                    "split": "validation",
                    "benchmark_id": benchmark,
                    "task_id": task_id,
                    "order_seed": int(order_seed),
                    "fold": int(fold_map.get(task_id, -1)),
                    "budget": int(budget),
                    "labels": np.asarray(y_t, dtype=float),
                    "context_labels": np.asarray(y_c[:budget], dtype=float),
                    "n_target": int(len(y_t)),
                }
                for arm, (prior, tau) in priors.items():
                    with torch.no_grad():
                        theta = model.map_theta(
                            a_c[:budget],
                            b_c[:budget],
                            y_b,
                            prior_mean=prior,
                            prior_variance=tau,
                        )
                        logits = (a_t * (theta - b_t)).detach().cpu().numpy()
                    records[arm].append(
                        {**base, "logits": np.asarray(logits, dtype=float)}
                    )

                beta_probability = (2.0 + float(y_c[:budget].sum())) / (4.0 + budget)
                beta_records.append(
                    {
                        **base,
                        "logits": np.full(len(y_t), _logit(beta_probability), dtype=float),
                    }
                )
        if count % 10 == 0:
            print(f"scored {count}/104 validation tasks", flush=True)

    return records, beta_records, task_flops


def _fit_calibrator(samples: list[dict[str, Any]], rng: np.random.Generator):
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


def _calibrators(records: list[dict[str, Any]]) -> dict[str, Any]:
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


def _metric_rows(
    records: list[dict[str, Any]],
    calibrators: dict[str, Any] | None,
    arm: str,
    *,
    calibrated: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        labels = np.asarray(record["labels"], dtype=float)
        logits = np.asarray(record["logits"], dtype=float)
        if calibrated and calibrators is not None and arm != "beta":
            calibrator = calibrators["folds"][int(record["fold"])]
            probability = calibrator.transform(
                logits,
                int(record["budget"]),
                np.asarray(record["context_labels"], dtype=float),
            )
        else:
            probability = _sigmoid(logits)
        rows.append(
            {
                "arm": arm,
                "benchmark_id": record["benchmark_id"],
                "task_id": record["task_id"],
                "order_seed": int(record["order_seed"]),
                "fold": int(record["fold"]),
                "budget": int(record["budget"]),
                "brier": float(np.mean((probability - labels) ** 2)),
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
                "tasks": int(len(group)),
            }
        )
    task_records = task_alc.to_dict(orient="records")
    for row in task_records:
        row["weighted"] = float(row["weighted"])
    return alc, task_records


def _per_budget(rows: list[dict[str, Any]]) -> dict[str, dict[int, float]]:
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["arm", "budget"], sort=True)["brier"].mean().reset_index()
    out: dict[str, dict[int, float]] = {}
    for arm, group in grouped.groupby("arm", sort=True):
        out[str(arm)] = {
            int(row["budget"]): float(row["brier"])
            for _, row in group.iterrows()
        }
    return out


def _paired(task_alc: list[dict[str, Any]], arm: str, baseline: str) -> dict[str, Any]:
    from aims_starter.uniform_eval import task_bootstrap

    frame = pd.DataFrame(task_alc)
    pivot = frame.pivot(index="task_id", columns="arm", values="weighted")
    diff = (pivot[arm] - pivot[baseline]).dropna()
    stats = task_bootstrap(diff.to_numpy(), draws=10_000, seed=20261025)
    stats.update(
        {
            "arm": arm,
            "baseline": baseline,
            "wins": int((diff < 0.0).sum()),
            "tasks": int(len(diff)),
        }
    )
    return stats


def _evaluate(
    records: dict[str, list[dict[str, Any]]],
    beta_records: list[dict[str, Any]],
    task_subset: set[str] | None,
) -> dict[str, Any]:
    arms = list(records)
    selected = {
        arm: [
            record
            for record in records[arm]
            if task_subset is None or record["task_id"] in task_subset
        ]
        for arm in arms
    }
    beta_selected = [
        record
        for record in beta_records
        if task_subset is None or record["task_id"] in task_subset
    ]

    calibrated_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for arm in arms:
        calibrators = _calibrators(selected[arm])
        calibrated_rows.extend(
            _metric_rows(selected[arm], calibrators, arm, calibrated=True)
        )
        raw_rows.extend(_metric_rows(selected[arm], None, arm, calibrated=False))
    calibrated_rows.extend(
        _metric_rows(beta_selected, None, "beta", calibrated=True)
    )
    raw_rows.extend(_metric_rows(beta_selected, None, "beta", calibrated=False))

    alc_cal, task_cal = _summarize(calibrated_rows)
    alc_raw, task_raw = _summarize(raw_rows)
    paired = []
    for arm in arms:
        paired.append(_paired(task_cal, arm, "zero"))
        paired.append(_paired(task_cal, arm, "beta"))
    paired.append(_paired(task_cal, "flops", "model"))
    return {
        "alc_calibrated": alc_cal,
        "alc_raw": alc_raw,
        "per_budget_calibrated": _per_budget(calibrated_rows),
        "per_budget_raw": _per_budget(raw_rows),
        "paired_calibrated": paired,
        "rows_calibrated": calibrated_rows,
        "rows_raw": raw_rows,
    }


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=2 * 60 * 60,
)
def run() -> dict[str, Any]:
    import torch

    from aims_starter.r_c_attention import ContextConditionedIRT

    print("loading compact store", flush=True)
    manifest = _read_json(STORE_ROOT / "manifest.json")
    subjects = _load_subjects()
    item_embeddings = np.load(STORE_ROOT / "item_embeddings.npy")
    train = pd.read_parquet(STORE_ROOT / "train_tasks.parquet")
    evaluation = pd.read_parquet(STORE_ROOT / "eval_items.parquet")
    orders = pd.read_parquet(STORE_ROOT / "eval_orders.parquet")

    print("loading 2PL checkpoint", flush=True)
    device = torch.device("cuda")
    state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = ContextConditionedIRT(
        item_embeddings.shape[1] * 2,
        mode="2pl",
        d_model=128,
        subject_feature_slice=(item_embeddings.shape[1], item_embeddings.shape[1] * 2),
        prior_variance=PRIOR_TAU,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    cache_theta = RESULT_ROOT / "train_theta.parquet"
    cache_stats = RESULT_ROOT / "model_stats.parquet"
    cache_fit = RESULT_ROOT / "prior_fit.json"
    if cache_theta.exists() and cache_stats.exists() and cache_fit.exists():
        print("loading cached training abilities", flush=True)
        train_theta = pd.read_parquet(cache_theta)
        model_stats = pd.read_parquet(cache_stats)
        fit = _read_json(cache_fit)
    else:
        print("estimating training-task abilities", flush=True)
        train_theta = _estimate_training_theta(
            model, train, subjects, item_embeddings, device
        )
        fit, train_theta, model_stats = _fit_priors(train_theta)
        train_theta.to_parquet(cache_theta, index=False)
        model_stats.to_parquet(cache_stats, index=False)
        _write_json(cache_fit, fit)
    print(json.dumps(fit, indent=2), flush=True)

    print("scoring validation with prior means", flush=True)
    records, beta_records, task_flops = _score_validation(
        model,
        subjects,
        model_stats,
        fit,
        evaluation,
        orders,
        item_embeddings,
        device,
    )

    known_tasks = {
        str(task_id) for task_id, flops in task_flops.items() if pd.notna(flops)
    }
    full = _evaluate(records, beta_records, None)
    known = _evaluate(records, beta_records, known_tasks)

    # Persist the training abilities and the scoring rows.
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    train_theta.to_parquet(RESULT_ROOT / "train_theta.parquet", index=False)
    model_stats.to_parquet(RESULT_ROOT / "model_stats.parquet", index=False)
    pd.DataFrame(full["rows_calibrated"]).to_parquet(
        RESULT_ROOT / "rows-calibrated.parquet", index=False
    )
    pd.DataFrame(full["rows_raw"]).to_parquet(
        RESULT_ROOT / "rows-raw.parquet", index=False
    )

    result = {
        "status": "complete",
        "experiment": "flops-prior-2pl",
        "corpus": CORPUS,
        "checkpoint": str(CHECKPOINT),
        "manifest": manifest,
        "prior_fit": fit,
        "coverage": {
            "train_tasks": int(len(train_theta)),
            "train_tasks_with_flops": int(train_theta["flops"].notna().sum()),
            "validation_tasks": int(len(task_flops)),
            "validation_tasks_with_flops": int(len(known_tasks)),
            "models_with_prior": int(len(model_stats)),
            "validation_tasks_with_model_prior": int(
                sum(
                    str(name) in set(model_stats["normalized_name"])
                    for name in subjects.iloc[
                        evaluation[evaluation["split"] == "validation"]
                        .drop_duplicates("task_id")["subject_index"]
                        .to_numpy()
                    ]["normalized_name"]
                )
            ),
        },
        "full": {
            key: value
            for key, value in full.items()
            if not key.startswith("rows_")
        },
        "known_flops": {
            key: value
            for key, value in known.items()
            if not key.startswith("rows_")
        },
        "training_theta_summary": {
            "mean": float(train_theta["theta"].mean()),
            "std": float(train_theta["theta"].std(ddof=0)),
            "min": float(train_theta["theta"].min()),
            "max": float(train_theta["theta"].max()),
        },
    }
    _write_json(RESULT_ROOT / "flops-prior-2pl.json", result)
    volume.commit()
    print(json.dumps({k: v for k, v in result.items() if k not in ("full", "known_flops", "manifest")}, indent=2)[:8000], flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    result = run.remote()
    print(
        json.dumps(
            {
                "status": result["status"],
                "prior_fit": result["prior_fit"],
                "coverage": result["coverage"],
                "full_alc": result["full"]["alc_calibrated"],
                "known_alc": result["known_flops"]["alc_calibrated"],
            },
            indent=2,
            sort_keys=True,
        )
    )
