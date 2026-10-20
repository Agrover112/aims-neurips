"""CNP-AIRT trained with disjoint context/target episodes (post-hoc ablation).

The main all-benchmark runs train CNP-AIRT with context items included in the
target set, which does not match the evaluation condition where the acquisition
pool and the evaluation pool are disjoint. This app retrains the same model with
``disjoint_context_target=True``, scores it under the identical protocol, and
fits the same per-budget calibrator on validation. Results land in a separate
result root; the frozen submission decision is not affected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "aims-neurips-cnp-disjoint"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "allbench-features-v1"
RESULT_ROOT = REMOTE_ROOT / "results" / "allbench-disjoint-cnp"
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


def _task_batch(train, item_embeddings, subject_embeddings):
    import numpy as np

    tasks = []
    for task_id, rows in train.groupby("task_id", sort=True):
        tasks.append(
            {
                "task_id": str(task_id),
                "x": np.concatenate(
                    [
                        item_embeddings[rows["item_index"].to_numpy()],
                        subject_embeddings[rows["subject_index"].to_numpy()],
                    ],
                    axis=1,
                ).astype("float32"),
                "y": rows["label"].to_numpy(dtype="float32"),
            }
        )
    return tasks


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=3 * 60 * 60,
)
def train_and_score(seed: int, mode: str) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch

    from aims_starter.faithful_np import score_cnp, train_cnp, write_metrics
    from aims_starter.metrics import DEFAULT_BUDGETS, DEFAULT_WEIGHTS, calibration_ece

    output = RESULT_ROOT / f"seed-{seed}-{mode}.json"
    existing = _read_json(output)
    if existing is not None:
        return existing

    manifest = _read_json(CORPUS_ROOT / "manifest.json")
    if manifest is None or "feature_slices" not in manifest:
        raise FileNotFoundError("all-benchmark feature store is missing")
    subject_slice = tuple(manifest["feature_slices"]["subject"]) if mode == "masked" else None
    item_embeddings = np.load(CORPUS_ROOT / "item_embeddings.npy")
    subject_embeddings = np.load(CORPUS_ROOT / "subject_embeddings.npy")
    train = pd.read_parquet(CORPUS_ROOT / "train_tasks.parquet")
    evaluation_items = pd.read_parquet(CORPUS_ROOT / "eval_items.parquet")
    orders = pd.read_parquet(CORPUS_ROOT / "eval_orders.parquet")
    print(
        f"seed={seed} mode={mode}: {train['task_id'].nunique()} train tasks, "
        f"{evaluation_items['task_id'].nunique()} evaluation tasks",
        flush=True,
    )

    meta_tasks = _task_batch(train, item_embeddings, subject_embeddings)
    cnp, cnp_history = train_cnp(
        meta_tasks,
        seed=seed,
        steps=800,
        episodes_per_step=4,
        subject_feature_slice=subject_slice,
        disjoint_context_target=True,
    )
    fold_root = RESULT_ROOT / f"seed-{seed}-{mode}-checkpoints"
    fold_root.mkdir(parents=True, exist_ok=True)
    torch.save(cnp.state_dict(), fold_root / "cnp_airt.pt")
    write_metrics(fold_root / "cnp-history.json", {"history": cnp_history})

    def features(rows):
        return np.concatenate(
            [
                item_embeddings[rows["item_index"].to_numpy()],
                subject_embeddings[rows["subject_index"].to_numpy()],
            ],
            axis=1,
        ).astype("float32")

    rows = []
    for split in ("validation", "test"):
        split_rows = evaluation_items[evaluation_items["split"] == split]
        order_seeds = ORDER_SEEDS if split == "test" else (0,)
        for task_id, task_rows in split_rows.groupby("task_id", sort=True):
            benchmark = str(task_rows["benchmark_id"].iloc[0])
            target_rows = task_rows[task_rows["pool"] == "evaluation"]
            x_T = features(target_rows)
            y_T = target_rows["label"].to_numpy(dtype="float32")
            lookup = task_rows.set_index("item_index")
            for order_seed in order_seeds:
                order_rows = (
                    orders[(orders["task_id"] == task_id) & (orders["order_seed"] == order_seed)]
                    .sort_values("acquisition_rank")
                    .reset_index(drop=True)
                )
                context_rows = lookup.loc[order_rows["item_index"].to_numpy()].reset_index()
                task = {
                    "task_id": str(task_id),
                    "x_C_pool": features(context_rows),
                    "y_C_pool": context_rows["label"].to_numpy(dtype="float32"),
                    "x_T": x_T,
                    "y_T": y_T,
                }
                scores = score_cnp(cnp, task)
                for budget in DEFAULT_BUDGETS:
                    rows.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "split": split,
                            "benchmark_id": benchmark,
                            "task_id": str(task_id),
                            "order_seed": int(order_seed),
                            "model": "cnp_airt",
                            "budget": int(budget),
                            "brier": float(scores[str(budget)]["brier"]),
                            "ece": float(scores[str(budget)]["ece"]),
                            "brier_alc": float(scores["brier_alc"]),
                            "auc": float(scores[str(budget)]["auc"]),
                        }
                    )
        print(f"seed={seed} mode={mode} split={split} scored", flush=True)

    result = {
        "seed": seed,
        "mode": mode,
        "status": "allbench_disjoint_cnp",
        "disjoint_context_target": True,
        "rows": rows,
    }
    write_metrics(output, result)
    volume.commit()
    return result


def _cnp_logits(model, x_T, context: dict[str, Any], budget: int):
    import torch

    device = next(model.parameters()).device
    x_C = torch.as_tensor(context["x_C_pool"][:budget], dtype=torch.float32, device=device)
    y_C = torch.as_tensor(context["y_C_pool"][:budget], dtype=torch.float32, device=device)
    with torch.no_grad():
        return model.logits(x_C, y_C, x_T).cpu().numpy()


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=3 * 60 * 60,
)
def fit_mode(seed: int, mode: str) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch

    from aims_starter.calibration import AnchorCalibrator, context_rate
    from aims_starter.faithful_np import CNPAIRT, binary_auc
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_slice = SUBJECT_SLICE if mode == "masked" else None
    state = torch.load(
        RESULT_ROOT / f"seed-{seed}-{mode}-checkpoints" / "cnp_airt.pt", map_location=device
    )
    cnp = CNPAIRT(x_dim, subject_feature_slice=subject_slice).to(device)
    cnp.load_state_dict(state)
    cnp.eval()
    print(f"seed={seed} mode={mode}: checkpoint loaded", flush=True)

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
            x_T = torch.as_tensor(features(target_rows), dtype=torch.float32, device=device)
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
                for budget in DEFAULT_BUDGETS:
                    logits = _cnp_logits(cnp, x_T, context, int(budget))
                    records.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "split": split,
                            "benchmark_id": benchmark,
                            "task_id": str(task_id),
                            "order_seed": int(order_seed),
                            "budget": int(budget),
                            "logits": logits.astype("float64"),
                            "labels": y_T.astype("float64"),
                            "context_labels": context["y_C_pool"][:budget].astype("float64"),
                        }
                    )
        print(f"seed={seed} mode={mode} split={split} collected", flush=True)

    samples = [record for record in records if record["split"] == "validation"]
    logits = np.concatenate([record["logits"] for record in samples])
    labels = np.concatenate([record["labels"] for record in samples])
    rates = np.array([context_rate(record["context_labels"]) for record in samples])
    rates = np.repeat(rates, [len(record["labels"]) for record in samples])
    budgets = np.repeat(
        [record["budget"] for record in samples], [len(record["labels"]) for record in samples]
    )
    rng = np.random.default_rng(20261019)
    if len(labels) > FIT_SAMPLE_LIMIT:
        index = rng.choice(len(labels), size=FIT_SAMPLE_LIMIT, replace=False)
        logits, labels, rates, budgets = (logits[index], labels[index], rates[index], budgets[index])
    calibrator = AnchorCalibrator.fit(logits, labels, rates, budgets)
    print(f"seed={seed} mode={mode}: calibrator fitted", flush=True)

    def _metrics(probability, y_true) -> tuple[float, float, float]:
        brier = float(np.mean((probability - y_true) ** 2))
        ece = calibration_ece(y_true.tolist(), probability.tolist())
        try:
            auc = float(binary_auc(y_true, probability))
        except ValueError:
            auc = float("nan")
        return brier, ece, auc

    rows = []
    for record in records:
        calibrated = calibrator.transform(
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
                "model": "cnp_airt",
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
    for (split, budget), group in frame.groupby(["split", "budget"], sort=True):
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
                "model": "cnp_airt",
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
    for split, group in frame.groupby("split", sort=True):
        per_task = (
            group.groupby(["task_id", "order_seed"], as_index=False)[["raw_alc", "calibrated_alc"]]
            .sum()
            .groupby("task_id", as_index=False)
            .mean(numeric_only=True)
        )
        alc_rows.append(
            {
                "model": "cnp_airt",
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
            (row["task_id"], row["order_seed"], row["budget"]): row["brier"]
            for row in stored["rows"]
            if row["mode"] == mode
        }
        deltas = []
        for row in rows:
            key = (row["task_id"], row["order_seed"], row["budget"])
            if key in reference:
                deltas.append(abs(row["raw_brier"] - reference[key]))
        check = {
            "max_abs_delta": float(max(deltas)) if deltas else None,
            "matched": len(deltas),
        }

    result = {
        "status": "allbench_disjoint_cnp_calibration",
        "seed": seed,
        "mode": mode,
        "disjoint_context_target": True,
        "fitted": calibrator.summary(),
        "by_budget": by_budget,
        "alc": alc_rows,
        "raw_check": check,
    }
    _write_json(output, result)
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    volumes={str(REMOTE_ROOT): volume},
    timeout=30 * 60,
)
def aggregate() -> dict[str, Any]:
    import pandas as pd

    results = [
        _read_json(RESULT_ROOT / f"calibration-{seed}-{mode}.json")
        for seed in SEEDS
        for mode in MODES
    ]
    if any(result is None for result in results):
        raise FileNotFoundError("disjoint calibration runs incomplete")
    frame = pd.DataFrame(
        [{**row, "mode": result["mode"]} for result in results for row in result["by_budget"]]
    )
    alc_rows = []
    for (mode, split), group in pd.DataFrame(
        [
            {
                "mode": result["mode"],
                "split": row["split"],
                "raw_alc": row["raw_alc"],
                "calibrated_alc": row["calibrated_alc"],
            }
            for result in results
            for row in result["alc"]
        ]
    ).groupby(["mode", "split"], sort=True):
        alc_rows.append(
            {
                "mode": mode,
                "model": "cnp_airt",
                "split": split,
                "raw_alc": float(group["raw_alc"].mean()),
                "raw_alc_seed_std": float(group["raw_alc"].std(ddof=1)),
                "calibrated_alc": float(group["calibrated_alc"].mean()),
                "calibrated_alc_seed_std": float(group["calibrated_alc"].std(ddof=1)),
                "seeds": len(group),
            }
        )
    by_budget = (
        frame.groupby(["mode", "split", "budget"], as_index=False)
        .agg(
            raw_brier=("raw_brier", "mean"),
            calibrated_brier=("calibrated_brier", "mean"),
            raw_ece=("raw_ece", "mean"),
            calibrated_ece=("calibrated_ece", "mean"),
            raw_auc=("raw_auc", "mean"),
            calibrated_auc=("calibrated_auc", "mean"),
        )
        .round(6)
    )
    artifact = {
        "status": "allbench_disjoint_cnp_aggregate",
        "modes": list(MODES),
        "seeds": list(SEEDS),
        "disjoint_context_target": True,
        "alc": alc_rows,
        "by_budget": by_budget.to_dict(orient="records"),
        "fitted": {
            f"seed-{result['seed']}-{result['mode']}": result["fitted"] for result in results
        },
        "raw_check": [result["raw_check"] for result in results],
    }
    _write_json(RESULT_ROOT / "metrics-disjoint.json", artifact)
    volume.commit()
    return artifact


@app.local_entrypoint()
def main(training_only: bool = False) -> None:
    handles = [
        train_and_score.spawn(seed_value, mode_value) for seed_value in SEEDS for mode_value in MODES
    ]
    for handle in handles:
        handle.get()
    if training_only:
        print("training complete")
        return
    handles = [fit_mode.spawn(seed_value, mode_value) for seed_value in SEEDS for mode_value in MODES]
    for handle in handles:
        handle.get()
    print(json.dumps(aggregate.remote(), indent=2, sort_keys=True))
