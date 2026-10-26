"""Isolate the CNP encoder from the item head with a five-rung ladder.

Rungs, all scored on the same benchmark-held-out tasks, acquisition orders, and
budgets:

1. Beta-Binomial context mean.
2. 1PL item difficulty + regularized MAP ability.
3. 2PL item difficulty/discrimination + regularized MAP ability (Online-AIRT).
4. CNP-AIRT with the learned correction zeroed at inference.
5. CNP-AIRT with the learned correction.

The fourth and fifth rungs share one checkpoint, so their difference is exactly
the contribution of ``Delta theta(C)``.  The script also trains a CNP-AIRT
checkpoint with the correction disabled during training, which checks whether
the item head recovers when the encoder is removed from the start.  It records
the correction's magnitude and dispersion by budget, and fits one
validation-only AnchorCalibrator per arm before scoring test.

All evaluation uses the frozen protocol in ``allbench-features-v1``.  Test
labels are used only for scoring.  The selected submission archives are not
modified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import modal
import numpy as np

APP_NAME = "aims-neurips-cnp-ladder"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "allbench-features-v1"
RESULT_ROOT = REMOTE_ROOT / "results" / "allbench-deterministic-2seed"
OUTPUT_ROOT = REMOTE_ROOT / "results" / "allbench-cnp-ladder"
UNIFORM_ROOT = REMOTE_ROOT / "results" / "allbench-uniform-v1" / "ladder"
UNIFORM_FOLDS = 5
MODES = ("masked", "visible")
SEEDS = (20261018, 20261019)
SUBJECT_SLICE = (384, 768)
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
ORDER_SEEDS = tuple(range(10))
FIT_SAMPLE_LIMIT = 1_000_000
ARMS = ("beta", "irt_1pl", "irt_2pl", "cnp_no_delta", "cnp_no_delta_retrained", "cnp_full")

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
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_path(name: str) -> Path:
    mounted = Path("/opt/aims/src") / name
    if mounted.exists():
        return mounted
    return PROJECT_ROOT / "src" / name


def _training_signature() -> str:
    from aims_starter.caching import content_signature

    return content_signature(
        [_source_path("aims_starter/faithful_np.py")],
        {
            "corpus": CORPUS_ROOT.name,
            "seeds": list(SEEDS),
            "modes": list(MODES),
            "steps": 800,
            "episodes_per_step": 4,
            "budgets": list(BUDGETS),
            "disjoint_context_target": False,
        },
    )


def _result_signature() -> str:
    from aims_starter.caching import content_signature

    return content_signature(
        [_source_path("aims_starter/calibration.py")],
        {
            "training": _training_signature(),
            "orders": len(ORDER_SEEDS),
            "fit_sample_limit": FIT_SAMPLE_LIMIT,
            "uniform_folds": UNIFORM_FOLDS,
        },
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


def _meta_tasks(train, item_embeddings, subject_embeddings):
    tasks = []
    for task_id, rows in train.groupby("task_id", sort=True):
        tasks.append(
            {
                "task_id": str(task_id),
                "x": _features(rows, item_embeddings, subject_embeddings),
                "y": rows["label"].to_numpy(dtype="float32"),
            }
        )
    return tasks


def _load_checkpoint(model, path: Path) -> None:
    import torch

    state = torch.load(path, map_location=next(model.parameters()).device)
    model.load_state_dict(state)
    model.eval()


def _load_or_train_models(seed: int, mode: str, x_dim: int):
    import torch

    from aims_starter.faithful_np import (
        CNPAIRT,
        Amortized1PL,
        Amortized2PL,
        train_amortized_1pl,
        train_cnp,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_slice = SUBJECT_SLICE if mode == "masked" else None
    source_root = RESULT_ROOT / f"seed-{seed}-{mode}-checkpoints"
    ladder_root = OUTPUT_ROOT / f"seed-{seed}-{mode}-checkpoints"
    ladder_root.mkdir(parents=True, exist_ok=True)
    training_signature = _training_signature()

    cnp = CNPAIRT(x_dim, subject_feature_slice=subject_slice).to(device)
    _load_checkpoint(cnp, source_root / "cnp_airt.pt")
    online = Amortized2PL(
        x_dim,
        int(torch.load(source_root / "online_airt_2pl.pt", map_location=device)["theta"].shape[0]),
        subject_feature_slice=subject_slice,
    ).to(device)
    _load_checkpoint(online, source_root / "online_airt_2pl.pt")

    import pandas as pd

    train = pd.read_parquet(CORPUS_ROOT / "train_tasks.parquet")
    item_embeddings = np.load(CORPUS_ROOT / "item_embeddings.npy")
    subject_embeddings = np.load(CORPUS_ROOT / "subject_embeddings.npy")
    tasks = _meta_tasks(train, item_embeddings, subject_embeddings)

    one_pl_path = ladder_root / "irt_1pl.pt"
    one_pl_history_path = ladder_root / "irt_1pl-history.json"
    if one_pl_path.exists():
        one_pl_history = _read_json(one_pl_history_path)
        if (
            one_pl_history is None
            or one_pl_history.get("training_signature") != training_signature
        ):
            raise RuntimeError(
                "cached 1PL checkpoint was produced by different training code or "
                f"protocol; remove {one_pl_path} and {one_pl_history_path} before rerunning"
            )
        one_pl_state = torch.load(one_pl_path, map_location=device)
        one_pl = Amortized1PL(
            x_dim, int(one_pl_state["theta"].shape[0]), subject_feature_slice=subject_slice
        ).to(device)
        one_pl.load_state_dict(one_pl_state)
        one_pl.eval()
    else:
        one_pl, history = train_amortized_1pl(
            tasks, seed=seed, steps=800, subject_feature_slice=subject_slice
        )
        torch.save(one_pl.state_dict(), one_pl_path)
        one_pl_history = {
            "history": history,
            "training_signature": training_signature,
        }
        _write_json(one_pl_history_path, one_pl_history)
        volume.commit()

    no_delta_path = ladder_root / "cnp_no_delta.pt"
    no_delta_history_path = ladder_root / "cnp_no_delta-history.json"
    if no_delta_path.exists():
        no_delta_history = _read_json(no_delta_history_path)
        if (
            no_delta_history is None
            or no_delta_history.get("training_signature") != training_signature
        ):
            raise RuntimeError(
                "cached no-delta checkpoint was produced by different training code "
                f"or protocol; remove {no_delta_path} and {no_delta_history_path} "
                "before rerunning"
            )
        no_delta_state = torch.load(no_delta_path, map_location=device)
        no_delta = CNPAIRT(
            x_dim,
            subject_feature_slice=subject_slice,
            use_theta_delta=False,
        ).to(device)
        no_delta.load_state_dict(no_delta_state)
        no_delta.eval()
    else:
        no_delta, history = train_cnp(
            tasks,
            seed=seed,
            steps=800,
            episodes_per_step=4,
            subject_feature_slice=subject_slice,
            use_theta_delta=False,
        )
        torch.save(no_delta.state_dict(), no_delta_path)
        no_delta_history = {
            "history": history,
            "training_signature": training_signature,
        }
        _write_json(no_delta_history_path, no_delta_history)
        volume.commit()

    return {
        "cnp": cnp,
        "online": online,
        "one_pl": one_pl,
        "cnp_no_delta": no_delta,
        "cnp_full": cnp,
        "histories": {
            "irt_1pl": one_pl_history,
            "cnp_no_delta": no_delta_history,
        },
    }


def _cnp_components(model, x_T, context, budget: int):
    """Return full logits and the two ability components for one context."""

    import torch

    device = next(model.parameters()).device
    x_C = torch.as_tensor(context["x_C_pool"][:budget], dtype=torch.float32, device=device)
    y_C = torch.as_tensor(context["y_C_pool"][:budget], dtype=torch.float32, device=device)
    x_T_tensor = torch.as_tensor(x_T, dtype=torch.float32, device=device)
    x_C_encoded = model.encode_item_x(x_C)
    with torch.no_grad():
        theta_map = float(model.map_theta(x_C_encoded, y_C))
        delta = float(model.delta_theta(x_C_encoded, y_C)) if budget else 0.0
        full = model.logits(x_C, y_C, x_T_tensor).cpu().numpy()
        no_delta = model.logits(x_C, y_C, x_T_tensor, delta_scale=0.0).cpu().numpy()
    return full, no_delta, theta_map, delta


def _arm_logits(arm: str, models, x_T, context, budget: int):
    import torch

    from aims_starter.faithful_np import map_theta_1pl, map_theta_2pl

    if arm == "irt_1pl":
        model = models["irt_1pl"]
        device = next(model.parameters()).device
        x_C = torch.as_tensor(context["x_C_pool"][:budget], dtype=torch.float32, device=device)
        y_C = torch.as_tensor(context["y_C_pool"][:budget], dtype=torch.float32, device=device)
        x_T_tensor = torch.as_tensor(x_T, dtype=torch.float32, device=device)
        theta = map_theta_1pl(model, x_C, y_C)
        with torch.no_grad():
            return (theta - model.item_difficulty(x_T_tensor)).cpu().numpy()
    if arm == "irt_2pl":
        model = models["irt_2pl"]
        device = next(model.parameters()).device
        x_C = torch.as_tensor(context["x_C_pool"][:budget], dtype=torch.float32, device=device)
        y_C = torch.as_tensor(context["y_C_pool"][:budget], dtype=torch.float32, device=device)
        x_T_tensor = torch.as_tensor(x_T, dtype=torch.float32, device=device)
        theta = map_theta_2pl(model, x_C, y_C)
        a_T, b_T = model.item_parameters(x_T_tensor)
        with torch.no_grad():
            return (a_T * (theta - b_T)).cpu().numpy()
    if arm == "cnp_no_delta":
        model = models["cnp_full"]
        device = next(model.parameters()).device
        x_C = torch.as_tensor(context["x_C_pool"][:budget], dtype=torch.float32, device=device)
        y_C = torch.as_tensor(context["y_C_pool"][:budget], dtype=torch.float32, device=device)
        x_T_tensor = torch.as_tensor(x_T, dtype=torch.float32, device=device)
        with torch.no_grad():
            return model.logits(x_C, y_C, x_T_tensor, delta_scale=0.0).cpu().numpy()
    if arm == "cnp_no_delta_retrained":
        model = models["cnp_no_delta"]
        device = next(model.parameters()).device
        x_C = torch.as_tensor(context["x_C_pool"][:budget], dtype=torch.float32, device=device)
        y_C = torch.as_tensor(context["y_C_pool"][:budget], dtype=torch.float32, device=device)
        x_T_tensor = torch.as_tensor(x_T, dtype=torch.float32, device=device)
        with torch.no_grad():
            return model.logits(x_C, y_C, x_T_tensor).cpu().numpy()
    raise ValueError(f"unknown arm: {arm}")


def _flatten(samples: list[dict[str, Any]]):
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
    return logits, labels, rates, budgets


def _fit_calibrators(records: list[dict[str, Any]]):
    from aims_starter.calibration import AnchorCalibrator

    rng = np.random.default_rng(20261019)
    calibrators = {}
    summaries = {}
    for arm in ARMS:
        if arm == "beta":
            continue
        samples = [
            record
            for record in records
            if record["arm"] == arm and record["split"] == "validation"
        ]
        logits, labels, rates, budgets = _flatten(samples)
        if len(labels) > FIT_SAMPLE_LIMIT:
            index = rng.choice(len(labels), size=FIT_SAMPLE_LIMIT, replace=False)
            logits, labels, rates, budgets = (
                logits[index],
                labels[index],
                rates[index],
                budgets[index],
            )
        calibrator = AnchorCalibrator.fit(logits, labels, rates, budgets)
        calibrators[arm] = calibrator
        summaries[arm] = calibrator.summary()
    return calibrators, summaries


def _fit_one_calibrator(samples: list[dict[str, Any]], rng):
    from aims_starter.calibration import AnchorCalibrator

    logits, labels, rates, budgets = _flatten(samples)
    if len(labels) > FIT_SAMPLE_LIMIT:
        index = rng.choice(len(labels), size=FIT_SAMPLE_LIMIT, replace=False)
        logits, labels, rates, budgets = (
            logits[index],
            labels[index],
            rates[index],
            budgets[index],
        )
    return AnchorCalibrator.fit(logits, labels, rates, budgets)


def _fit_calibrators_uniform(records: list[dict[str, Any]]):
    rng = np.random.default_rng(20261020)
    output: dict[str, Any] = {}
    for arm in ARMS:
        if arm == "beta":
            continue
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


def _metric_rows_uniform(records: list[dict[str, Any]], calibrators) -> list[dict[str, Any]]:
    from aims_starter.faithful_np import binary_auc
    from aims_starter.metrics import calibration_ece

    rows = []
    for record in records:
        labels = np.asarray(record["labels"], dtype=float)
        logits = np.asarray(record["logits"], dtype=float)
        probability = _sigmoid(logits)
        arm = record["arm"]
        if arm == "beta":
            probability_calibrated = probability
        else:
            key = (
                int(record["fold"])
                if record["split"] == "validation"
                else "full"
            )
            calibrator = (
                calibrators[arm]["folds"][key]
                if key != "full"
                else calibrators[arm]["full"]
            )
            probability_calibrated = calibrator.transform(
                logits, record["budget"], record["context_labels"]
            )
        rows.append(
            {
                "arm": arm,
                "split": record["split"],
                "fold": int(record["fold"]),
                "benchmark_id": record["benchmark_id"],
                "task_id": record["task_id"],
                "order_seed": record["order_seed"],
                "budget": record["budget"],
                "brier_raw": float(np.mean((probability - labels) ** 2)),
                "brier_cal": float(np.mean((probability_calibrated - labels) ** 2)),
                "ece_raw": float(calibration_ece(labels.tolist(), probability.tolist())),
                "ece_cal": float(
                    calibration_ece(labels.tolist(), probability_calibrated.tolist())
                ),
                "auc_raw": float(binary_auc(labels, probability)),
                "auc_cal": float(binary_auc(labels, probability_calibrated)),
            }
        )
    return rows


def _metric_rows(records: list[dict[str, Any]], calibrators) -> list[dict[str, Any]]:
    from aims_starter.faithful_np import binary_auc
    from aims_starter.metrics import calibration_ece

    rows = []
    for record in records:
        labels = np.asarray(record["labels"], dtype=float)
        logits = np.asarray(record["logits"], dtype=float)
        probability = _sigmoid(logits)
        if record["arm"] == "beta":
            probability_calibrated = probability
        else:
            probability_calibrated = calibrators[record["arm"]].transform(
                logits, record["budget"], record["context_labels"]
            )
        rows.append(
            {
                "arm": record["arm"],
                "split": record["split"],
                "benchmark_id": record["benchmark_id"],
                "task_id": record["task_id"],
                "order_seed": record["order_seed"],
                "budget": record["budget"],
                "brier_raw": float(np.mean((probability - labels) ** 2)),
                "brier_cal": float(np.mean((probability_calibrated - labels) ** 2)),
                "ece_raw": float(calibration_ece(labels.tolist(), probability.tolist())),
                "ece_cal": float(
                    calibration_ece(labels.tolist(), probability_calibrated.tolist())
                ),
                "auc_raw": float(binary_auc(labels, probability)),
                "auc_cal": float(binary_auc(labels, probability_calibrated)),
            }
        )
    return rows


def _summarize(metric_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import pandas as pd

    frame = pd.DataFrame(metric_rows)
    weight = dict(zip(BUDGETS, WEIGHTS, strict=True))
    frame["weighted_raw"] = frame["budget"].map(weight) * frame["brier_raw"]
    frame["weighted_cal"] = frame["budget"].map(weight) * frame["brier_cal"]

    by_budget = []
    for (arm, split, budget), group in frame.groupby(["arm", "split", "budget"], sort=True):
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
        .groupby(["arm", "split", "task_id"], sort=False)[["weighted_raw", "weighted_cal"]]
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
            [first_rows[task][metric] - second_rows[task][metric] for task in tasks], dtype=float
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


def _delta_diagnostics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import pandas as pd

    full = {
        (record["task_id"], record["order_seed"], record["budget"]): record
        for record in records
        if record["arm"] == "cnp_full" and "delta" in record
    }
    rows = []
    for record in records:
        if record["arm"] != "cnp_no_delta":
            continue
        key = (record["task_id"], record["order_seed"], record["budget"])
        if key not in full:
            continue
        source = full[key]
        labels = np.asarray(record["labels"], dtype=float)
        full_probability = _sigmoid(np.asarray(source["logits"], dtype=float))
        zero_probability = _sigmoid(np.asarray(record["logits"], dtype=float))
        rows.append(
            {
                "split": record["split"],
                "benchmark_id": record["benchmark_id"],
                "task_id": record["task_id"],
                "order_seed": record["order_seed"],
                "budget": record["budget"],
                "theta_map": source["theta_map"],
                "delta": source["delta"],
                "mean_abs_logit_shift": float(
                    np.mean(np.abs(source["logits"] - record["logits"]))
                ),
                "mean_abs_probability_shift": float(
                    np.mean(np.abs(full_probability - zero_probability))
                ),
                "brier_full": float(np.mean((full_probability - labels) ** 2)),
                "brier_zero": float(np.mean((zero_probability - labels) ** 2)),
            }
        )
    frame = pd.DataFrame(rows)
    output = []
    for (split, budget), group in frame.groupby(["split", "budget"], sort=True):
        per_task = group.groupby("task_id", sort=False)["delta"].agg(["mean", "std"])
        output.append(
            {
                "split": split,
                "budget": int(budget),
                "contexts": len(group),
                "tasks": len(per_task),
                "delta_mean": float(group["delta"].mean()),
                "delta_sd": float(group["delta"].std(ddof=1)),
                "delta_mean_abs": float(group["delta"].abs().mean()),
                "delta_q05": float(group["delta"].quantile(0.05)),
                "delta_q95": float(group["delta"].quantile(0.95)),
                "delta_min": float(group["delta"].min()),
                "delta_max": float(group["delta"].max()),
                "theta_map_sd": float(group["theta_map"].std(ddof=1)),
                "delta_task_mean_sd": float(per_task["mean"].std(ddof=1)),
                "delta_within_task_sd": float(per_task["std"].mean()),
                "mean_abs_logit_shift": float(group["mean_abs_logit_shift"].mean()),
                "mean_abs_probability_shift": float(
                    group["mean_abs_probability_shift"].mean()
                ),
                "brier_full": float(group["brier_full"].mean()),
                "brier_zero": float(group["brier_zero"].mean()),
                "delta_brier": float((group["brier_full"] - group["brier_zero"]).mean()),
            }
        )
    return output


def _collect(seed: int, mode: str, orders: int, limit_tasks: int):
    import pandas as pd

    from aims_starter.metrics import DEFAULT_BUDGETS

    item_embeddings = np.load(CORPUS_ROOT / "item_embeddings.npy")
    subject_embeddings = np.load(CORPUS_ROOT / "subject_embeddings.npy")
    x_dim = int(item_embeddings.shape[1] + subject_embeddings.shape[1])
    evaluation_items = pd.read_parquet(CORPUS_ROOT / "eval_items.parquet")
    order_frame = pd.read_parquet(CORPUS_ROOT / "eval_orders.parquet")
    from aims_starter.uniform_eval import validation_fold_map

    fold_map = validation_fold_map(evaluation_items, n_folds=UNIFORM_FOLDS)
    models = _load_or_train_models(seed, mode, x_dim)
    models["irt_1pl"] = models["one_pl"]
    models["irt_2pl"] = models["online"]

    records: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        split_rows = evaluation_items[evaluation_items["split"] == split]
        task_ids = sorted(split_rows["task_id"].astype(str).unique())
        if limit_tasks:
            task_ids = task_ids[:limit_tasks]
        for task_id in task_ids:
            task_rows = split_rows[split_rows["task_id"].astype(str) == task_id]
            benchmark = str(task_rows["benchmark_id"].iloc[0])
            target_rows = task_rows[task_rows["pool"] == "evaluation"]
            x_T = _features(target_rows, item_embeddings, subject_embeddings)
            y_T = target_rows["label"].to_numpy(dtype="float32")
            lookup = task_rows.set_index("item_index")
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
                context = {
                    "x_C_pool": _features(context_rows, item_embeddings, subject_embeddings),
                    "y_C_pool": context_rows["label"].to_numpy(dtype="float32"),
                }
                beta_probability = {
                    budget: (2.0 + float(context["y_C_pool"][:budget].sum()))
                    / (4.0 + budget)
                    for budget in DEFAULT_BUDGETS
                }
                for budget in DEFAULT_BUDGETS:
                    records.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "split": split,
                            "benchmark_id": benchmark,
                            "task_id": task_id,
                            "order_seed": int(order_seed),
                            "fold": int(fold_map.get(task_id, -1)),
                            "arm": "beta",
                            "budget": int(budget),
                            "logits": np.full(
                                len(y_T), _logit(beta_probability[budget]), dtype=float
                            ),
                            "labels": np.asarray(y_T, dtype=float),
                            "context_labels": np.asarray(
                                context["y_C_pool"][:budget], dtype=float
                            ),
                        }
                    )
                    full, no_delta, theta_map, delta = _cnp_components(
                        models["cnp_full"], x_T, context, budget
                    )
                    for arm in ARMS[1:]:
                        if arm == "cnp_full":
                            logits = full
                            extra = {"theta_map": theta_map, "delta": delta}
                        elif arm == "cnp_no_delta":
                            logits = no_delta
                            extra = {}
                        else:
                            logits = _arm_logits(arm, models, x_T, context, int(budget))
                            extra = {}
                        record = {
                            "seed": seed,
                            "mode": mode,
                            "split": split,
                            "benchmark_id": benchmark,
                            "task_id": task_id,
                            "order_seed": int(order_seed),
                            "fold": int(fold_map.get(task_id, -1)),
                            "arm": arm,
                            "budget": int(budget),
                            "logits": np.asarray(logits, dtype=float),
                            "labels": np.asarray(y_T, dtype=float),
                            "context_labels": np.asarray(
                                context["y_C_pool"][:budget], dtype=float
                            ),
                        }
                        record.update(extra)
                        records.append(record)
            print(f"{split} {task_id} collected", flush=True)
    return records, models


def _check_raw(records: list[dict[str, Any]], seed: int, mode: str):
    stored = _read_json(RESULT_ROOT / f"seed-{seed}-{mode}.json")
    if stored is None:
        return {"matched": 0, "max_abs_delta": None}
    reference = {
        (row["task_id"], int(row["order_seed"]), int(row["budget"])): float(row["brier"])
        for row in stored["rows"]
        if row["model"] == "cnp_airt" and row["mode"] == mode
    }
    deltas = []
    for record in records:
        if record["arm"] != "cnp_full":
            continue
        key = (record["task_id"], record["order_seed"], record["budget"])
        if key not in reference:
            continue
        probability = _sigmoid(record["logits"])
        brier = float(np.mean((probability - record["labels"]) ** 2))
        deltas.append(abs(brier - reference[key]))
    return {
        "matched": len(deltas),
        "max_abs_delta": float(max(deltas)) if deltas else None,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=6 * 60 * 60,
)
def run(
    seed: int = 20261018,
    mode: str = "masked",
    orders: int = 10,
    limit_tasks: int = 0,
) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not 1 <= orders <= len(ORDER_SEEDS):
        raise ValueError("orders must be between 1 and 10")
    output = OUTPUT_ROOT / f"cnp-ladder-{seed}-{mode}.json"
    result_signature = _result_signature()
    existing = _read_json(output)
    if (
        existing is not None
        and not limit_tasks
        and orders == len(ORDER_SEEDS)
        and int(existing.get("limit_tasks", 0)) == 0
        and int(existing.get("orders", 0)) == len(ORDER_SEEDS)
    ):
        if existing.get("result_signature") != result_signature:
            raise RuntimeError(
                "cached ladder result was produced by different code or protocol; "
                f"remove {output} before rerunning"
            )
        return existing

    records, models = _collect(seed, mode, orders, limit_tasks)
    calibrators, fits = _fit_calibrators(records)
    metric_rows = _metric_rows(records, calibrators)
    by_budget, alc, task_alc = _summarize(metric_rows)
    paired = []
    for split in ("validation", "test"):
        for first, second in (
            ("cnp_full", "cnp_no_delta"),
            ("cnp_no_delta", "irt_2pl"),
            ("irt_2pl", "irt_1pl"),
            ("cnp_full", "cnp_no_delta_retrained"),
            ("cnp_no_delta_retrained", "cnp_no_delta"),
        ):
            paired.append(_paired(task_alc, split, first, second))
    result = {
        "status": "cnp_ladder",
        "seed": seed,
        "mode": mode,
        "orders": orders,
        "limit_tasks": limit_tasks,
        "result_signature": result_signature,
        "arms": list(ARMS),
        "fits": fits,
        "by_budget": by_budget,
        "alc": alc,
        "paired": paired,
        "delta": _delta_diagnostics(records),
        "raw_check": _check_raw(records, seed, mode),
        "training": {
            "irt_1pl": models["histories"]["irt_1pl"],
            "cnp_no_delta": models["histories"]["cnp_no_delta"],
        },
    }
    _write_json(output, result)
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=6 * 60 * 60,
)
def run_uniform(
    seed: int = 20261018,
    mode: str = "masked",
    orders: int = 10,
    limit_tasks: int = 0,
) -> dict[str, Any]:
    import pandas as pd

    from aims_starter.uniform_eval import validation_fold_frame

    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not 1 <= orders <= len(ORDER_SEEDS):
        raise ValueError("orders must be between 1 and 10")
    output = UNIFORM_ROOT / f"cnp-ladder-{seed}-{mode}.json"
    records_path = UNIFORM_ROOT / f"cnp-ladder-{seed}-{mode}-rows.parquet"
    result_signature = _result_signature()
    existing = _read_json(output)
    if (
        existing is not None
        and not limit_tasks
        and orders == len(ORDER_SEEDS)
        and int(existing.get("limit_tasks", 0)) == 0
        and int(existing.get("orders", 0)) == len(ORDER_SEEDS)
    ):
        if existing.get("result_signature") != result_signature:
            raise RuntimeError(
                "cached uniform ladder result was produced by different code or "
                f"protocol; remove {output} before rerunning"
            )
        return existing

    records, models = _collect(seed, mode, orders, limit_tasks)
    calibrators = _fit_calibrators_uniform(records)
    metric_rows = _metric_rows_uniform(records, calibrators)
    by_budget, alc, task_alc = _summarize(metric_rows)
    paired = []
    for split in ("validation", "test"):
        for first, second in (
            ("cnp_full", "cnp_no_delta"),
            ("cnp_no_delta", "irt_2pl"),
            ("irt_2pl", "irt_1pl"),
            ("cnp_full", "cnp_no_delta_retrained"),
            ("cnp_no_delta_retrained", "cnp_no_delta"),
        ):
            paired.append(_paired(task_alc, split, first, second))
    evaluation_items = pd.read_parquet(CORPUS_ROOT / "eval_items.parquet")
    fold_frame = validation_fold_frame(evaluation_items, n_folds=UNIFORM_FOLDS)
    result = {
        "status": "cnp_ladder_uniform",
        "seed": seed,
        "mode": mode,
        "orders": orders,
        "limit_tasks": limit_tasks,
        "result_signature": result_signature,
        "arms": list(ARMS),
        "split_policy": "validation folds are out-of-fold; test uses the full validation fit",
        "folds": int(UNIFORM_FOLDS),
        "fold_tasks": fold_frame.to_dict(orient="records"),
        "fits": {
            arm: {
                "full": fit["full"].summary(),
                "folds": {
                    str(fold): calibrator.summary()
                    for fold, calibrator in fit["folds"].items()
                },
            }
            for arm, fit in calibrators.items()
        },
        "by_budget": by_budget,
        "alc": alc,
        "paired": paired,
        "delta": _delta_diagnostics(records),
        "raw_check": _check_raw(records, seed, mode),
        "training": {
            "irt_1pl": models["histories"]["irt_1pl"],
            "cnp_no_delta": models["histories"]["cnp_no_delta"],
        },
    }
    UNIFORM_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
    frame = pd.DataFrame(metric_rows)
    frame.to_parquet(records_path, index=False)
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    seed: int = 0,
    mode: str = "",
    orders: int = 10,
    limit_tasks: int = 0,
    uniform: bool = False,
) -> None:
    function = run_uniform if uniform else run
    if seed and mode:
        print(
            json.dumps(
                function.remote(seed, mode, orders, limit_tasks),
                indent=2,
                sort_keys=True,
            )
        )
        return
    handles = [
        function.spawn(seed_value, mode_value, orders, limit_tasks)
        for seed_value in SEEDS
        for mode_value in MODES
    ]
    for handle in handles:
        handle.get()
