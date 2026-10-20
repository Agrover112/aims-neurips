"""Fit the per-budget ability calibrator on validation and apply it to test.

The all-benchmark runs store probabilities. Calibration needs the raw logits, so
this app rescoring each trained checkpoint: it collects logits for every task,
order and budget on the validation and test splits, fits one AnchorCalibrator
per model on the validation tasks, and reports raw versus calibrated Brier, ECE
and AUC on both splits. The calibrator is fitted on validation benchmarks only
and applied unchanged to the test benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "aims-neurips-calibration"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "allbench-features-v1"
RESULT_ROOT = REMOTE_ROOT / "results" / "allbench-deterministic-2seed"
MODES = ("masked", "visible")
SEEDS = (20261018, 20261019)
SUBJECT_SLICE = (384, 768)
ORDER_SEEDS = tuple(range(10))
BUDGETS = (0, 1, 3, 7, 15, 31)
FIT_SAMPLE_LIMIT = 1_000_000

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "torch==2.5.1")
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
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cnp_logits(model, x_T, context: dict[str, Any], budget: int):
    """Logits for one budget, reusing the scoring path's MAP and residual head."""

    import torch

    device = next(model.parameters()).device
    x_C = torch.as_tensor(context["x_C_pool"][:budget], dtype=torch.float32, device=device)
    y_C = torch.as_tensor(context["y_C_pool"][:budget], dtype=torch.float32, device=device)
    with torch.no_grad():
        return model.logits(x_C, y_C, x_T).cpu().numpy()


def _online_logits(model, x_T, context: dict[str, Any], budget: int):
    """Logits for one budget, mirroring ``score_amortized_2pl`` exactly."""

    import torch

    from aims_starter.faithful_np import map_theta_2pl

    device = next(model.parameters()).device
    a_T, b_T = model.item_parameters(x_T)
    x_C = torch.as_tensor(context["x_C_pool"][:budget], dtype=torch.float32, device=device)
    y_C = torch.as_tensor(context["y_C_pool"][:budget], dtype=torch.float32, device=device)
    theta = map_theta_2pl(model, x_C, y_C)
    with torch.no_grad():
        return (a_T * (theta - b_T)).cpu().numpy()


def _load_models(seed: int, mode: str, x_dim: int):
    import torch

    from aims_starter.faithful_np import CNPAIRT, Amortized2PL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_slice = SUBJECT_SLICE if mode == "masked" else None
    root = RESULT_ROOT / f"seed-{seed}-{mode}-checkpoints"
    cnp_state = torch.load(root / "cnp_airt.pt", map_location=device)
    cnp = CNPAIRT(x_dim, subject_feature_slice=subject_slice).to(device)
    cnp.load_state_dict(cnp_state)
    cnp.eval()
    online_state = torch.load(root / "online_airt_2pl.pt", map_location=device)
    online = Amortized2PL(
        x_dim,
        int(online_state["theta"].shape[0]),
        subject_feature_slice=subject_slice,
    ).to(device)
    online.load_state_dict(online_state)
    online.eval()
    return cnp, online


def _fit_mode(seed: int, mode: str) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch

    from aims_starter.calibration import AnchorCalibrator, context_rate
    from aims_starter.metrics import DEFAULT_BUDGETS, DEFAULT_WEIGHTS, calibration_ece

    output = RESULT_ROOT / f"calibration-{seed}-{mode}.json"
    existing = _read_json(output)
    if existing is not None:
        return existing

    item_embeddings = np.load(CORPUS_ROOT / "item_embeddings.npy")
    subject_embeddings = np.load(CORPUS_ROOT / "subject_embeddings.npy")
    x_dim = int(item_embeddings.shape[1] + subject_embeddings.shape[1])
    evaluation_items = pd.read_parquet(CORPUS_ROOT / "eval_items.parquet")
    orders = pd.read_parquet(CORPUS_ROOT / "eval_orders.parquet")
    cnp, online = _load_models(seed, mode, x_dim)
    print(f"seed={seed} mode={mode}: checkpoints loaded", flush=True)

    def features(rows):
        return np.concatenate(
            [
                item_embeddings[rows["item_index"].to_numpy()],
                subject_embeddings[rows["subject_index"].to_numpy()],
            ],
            axis=1,
        ).astype("float32")

    records: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        split_rows = evaluation_items[evaluation_items["split"] == split]
        for task_id, task_rows in split_rows.groupby("task_id", sort=True):
            benchmark = str(task_rows["benchmark_id"].iloc[0])
            target_rows = task_rows[task_rows["pool"] == "evaluation"]
            x_T = torch.as_tensor(
                features(target_rows),
                dtype=torch.float32,
                device=next(cnp.parameters()).device,
            )
            y_T = target_rows["label"].to_numpy(dtype="float32")
            lookup = task_rows.set_index("item_index")
            for order_seed in ORDER_SEEDS:
                order_rows = (
                    orders[(orders["task_id"] == task_id) & (orders["order_seed"] == order_seed)]
                    .sort_values("acquisition_rank")
                    .reset_index(drop=True)
                )
                context_rows = lookup.loc[order_rows["item_index"].to_numpy()].reset_index()
                context = {
                    "x_C_pool": features(context_rows),
                    "y_C_pool": context_rows["label"].to_numpy(dtype="float32"),
                }
                for model_name, function in (
                    ("cnp_airt", _cnp_logits),
                    ("online_airt_2pl", _online_logits),
                ):
                    for budget in DEFAULT_BUDGETS:
                        logits = function(
                            cnp if model_name == "cnp_airt" else online,
                            x_T,
                            context,
                            int(budget),
                        )
                        records.append(
                            {
                                "seed": seed,
                                "mode": mode,
                                "split": split,
                                "benchmark_id": benchmark,
                                "task_id": str(task_id),
                                "order_seed": int(order_seed),
                                "model": model_name,
                                "budget": int(budget),
                                "logits": logits.astype("float64"),
                                "labels": y_T.astype("float64"),
                                "context_labels": context["y_C_pool"][:budget].astype("float64"),
                            }
                        )
            print(f"seed={seed} mode={mode} split={split} collected", flush=True)

    def _metrics(probability, labels) -> tuple[float, float, float]:
        brier = float(np.mean((probability - labels) ** 2))
        ece = calibration_ece(labels.tolist(), probability.tolist())
        try:
            from aims_starter.faithful_np import binary_auc

            auc = float(binary_auc(labels, probability))
        except ValueError:
            auc = float("nan")
        return brier, ece, auc

    fitted: dict[str, list[dict[str, Any]]] = {}
    calibrators: dict[str, Any] = {}
    rng = np.random.default_rng(20261019)
    for model_name in ("cnp_airt", "online_airt_2pl"):
        samples = [
            record
            for record in records
            if record["model"] == model_name and record["split"] == "validation"
        ]
        logits = np.concatenate([record["logits"] for record in samples])
        labels = np.concatenate([record["labels"] for record in samples])
        rates = np.array([context_rate(record["context_labels"]) for record in samples])
        rates = np.repeat(rates, [len(record["labels"]) for record in samples])
        budgets = np.repeat(
            [record["budget"] for record in samples],
            [len(record["labels"]) for record in samples],
        )
        if len(labels) > FIT_SAMPLE_LIMIT:
            index = rng.choice(len(labels), size=FIT_SAMPLE_LIMIT, replace=False)
            logits, labels, rates, budgets = (
                logits[index],
                labels[index],
                rates[index],
                budgets[index],
            )
        calibrator = AnchorCalibrator.fit(logits, labels, rates, budgets)
        calibrators[model_name] = calibrator
        fitted[model_name] = calibrator.summary()
        print(f"seed={seed} mode={mode} {model_name}: calibrator fitted", flush=True)

    rows = []
    for record in records:
        calibrated = calibrators[record["model"]].transform(
            record["logits"], record["budget"], record["context_labels"]
        )
        probability = 1.0 / (1.0 + np.exp(-record["logits"]))
        raw_brier, raw_ece, raw_auc = _metrics(probability, record["labels"])
        cal_brier, cal_ece, cal_auc = _metrics(calibrated, record["labels"])
        rows.append(
            {
                "split": record["split"],
                "mode": record["mode"],
                "benchmark_id": record["benchmark_id"],
                "task_id": record["task_id"],
                "order_seed": record["order_seed"],
                "model": record["model"],
                "budget": record["budget"],
                "raw_brier": raw_brier,
                "calibrated_brier": cal_brier,
                "raw_ece": raw_ece,
                "calibrated_ece": cal_ece,
                "raw_auc": raw_auc,
                "calibrated_auc": cal_auc,
            }
        )

    frame = pd.DataFrame(rows)
    weights = dict(zip(DEFAULT_BUDGETS, DEFAULT_WEIGHTS, strict=True))
    frame["raw_alc"] = frame["budget"].map(weights) * frame["raw_brier"]
    frame["calibrated_alc"] = frame["budget"].map(weights) * frame["calibrated_brier"]

    by_budget = []
    for (model, split, budget), group in frame.groupby(["model", "split", "budget"], sort=True):
        task_means = group.groupby("task_id", as_index=False).agg(
            raw_brier=("raw_brier", "mean"),
            calibrated_brier=("calibrated_brier", "mean"),
            raw_ece=("raw_ece", "mean"),
            calibrated_ece=("calibrated_ece", "mean"),
            raw_auc=("raw_auc", "mean"),
            calibrated_auc=("calibrated_auc", "mean"),
        )
        by_budget.append(
            {
                "model": model,
                "mode": mode,
                "split": split,
                "budget": int(budget),
                "raw_brier": float(task_means["raw_brier"].mean()),
                "calibrated_brier": float(task_means["calibrated_brier"].mean()),
                "raw_ece": float(task_means["raw_ece"].mean()),
                "calibrated_ece": float(task_means["calibrated_ece"].mean()),
                "raw_auc": float(task_means["raw_auc"].mean()),
                "calibrated_auc": float(task_means["calibrated_auc"].mean()),
                "tasks": len(task_means),
            }
        )

    alc_rows = []
    for (model, split), group in frame.groupby(["model", "split"], sort=True):
        per_task = (
            group.groupby(["task_id", "order_seed"], as_index=False)[["raw_alc", "calibrated_alc"]]
            .sum()
            .groupby("task_id", as_index=False)
            .mean(numeric_only=True)
        )
        alc_rows.append(
            {
                "model": model,
                "split": split,
                "raw_alc": float(per_task["raw_alc"].mean()),
                "calibrated_alc": float(per_task["calibrated_alc"].mean()),
                "tasks": len(per_task),
            }
        )

    stored = _read_json(RESULT_ROOT / f"seed-{seed}-{mode}.json")
    check = {"max_abs_delta": None, "matched": 0}
    if stored is not None:
        reference = {
            (row["task_id"], row["order_seed"], row["model"], row["budget"]): row["brier"]
            for row in stored["rows"]
            if row["mode"] == mode
        }
        deltas = []
        for row in rows:
            key = (row["task_id"], row["order_seed"], row["model"], row["budget"])
            if key in reference:
                deltas.append(abs(row["raw_brier"] - reference[key]))
        check = {
            "max_abs_delta": float(max(deltas)) if deltas else None,
            "matched": len(deltas),
        }

    result = {
        "status": "allbench_calibration",
        "seed": seed,
        "mode": mode,
        "fitted": fitted,
        "by_budget": by_budget,
        "alc": alc_rows,
        "raw_check": check,
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
    timeout=3 * 60 * 60,
)
def fit_mode(seed: int, mode: str) -> dict[str, Any]:
    return _fit_mode(seed, mode)


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    volumes={str(REMOTE_ROOT): volume},
    timeout=30 * 60,
)
def aggregate_calibration() -> dict[str, Any]:
    import pandas as pd

    results = [
        _read_json(RESULT_ROOT / f"calibration-{seed}-{mode}.json")
        for seed in SEEDS
        for mode in MODES
    ]
    if any(result is None for result in results):
        raise FileNotFoundError("calibration runs incomplete")
    frame = pd.DataFrame(
        [{**row, "mode": result["mode"]} for result in results for row in result["by_budget"]]
    )
    runs = {f"seed-{result['seed']}-{result['mode']}": result for result in results}
    alc_rows = []
    for (mode, model, split), group in pd.DataFrame(
        [
            {
                "mode": result["mode"],
                "model": row["model"],
                "split": row["split"],
                "raw_alc": row["raw_alc"],
                "calibrated_alc": row["calibrated_alc"],
            }
            for result in results
            for row in result["alc"]
        ]
    ).groupby(["mode", "model", "split"], sort=True):
        alc_rows.append(
            {
                "mode": mode,
                "model": model,
                "split": split,
                "raw_alc": float(group["raw_alc"].mean()),
                "raw_alc_seed_std": float(group["raw_alc"].std(ddof=1)),
                "calibrated_alc": float(group["calibrated_alc"].mean()),
                "calibrated_alc_seed_std": float(group["calibrated_alc"].std(ddof=1)),
                "seeds": len(group),
            }
        )
    by_budget = (
        frame.groupby(["mode", "model", "split", "budget"], as_index=False)
        .agg(
            raw_brier=("raw_brier", "mean"),
            calibrated_brier=("calibrated_brier", "mean"),
            raw_ece=("raw_ece", "mean"),
            calibrated_ece=("calibrated_ece", "mean"),
        )
        .round(6)
    )
    artifact = {
        "status": "allbench_calibration_aggregate",
        "modes": list(MODES),
        "seeds": list(SEEDS),
        "alc": alc_rows,
        "by_budget": by_budget.to_dict(orient="records"),
        "fitted": {key: value["fitted"] for key, value in runs.items()},
        "raw_check": [value["raw_check"] for value in runs.values()],
    }
    _write_json(RESULT_ROOT / "metrics-calibration.json", artifact)
    volume.commit()
    return artifact


@app.local_entrypoint()
def main(
    aggregate_only: bool = False,
    fits_only: bool = False,
    seed: int = 0,
    mode: str = "",
) -> None:
    if aggregate_only:
        print(json.dumps(aggregate_calibration.remote(), indent=2, sort_keys=True))
        return
    if seed and mode:
        print(json.dumps(fit_mode.remote(seed, mode), indent=2, sort_keys=True))
        return
    handles = [
        fit_mode.spawn(seed_value, mode_value) for seed_value in SEEDS for mode_value in MODES
    ]
    for handle in handles:
        handle.get()
    if not fits_only:
        print(json.dumps(aggregate_calibration.remote(), indent=2, sort_keys=True))
