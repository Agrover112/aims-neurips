"""Calibrated per-order ALC for the full-protocol CNP-AIRT checkpoints.

Appendix E reports raw test ALC for each frozen acquisition order. The saved
calibration artifacts contain only per-budget and overall aggregates, so this
app rescores the test split, applies the stored validation-fitted calibrator
unchanged, and writes per-order raw and calibrated ALC. It does not refit
anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "aims-neurips-order-calibration"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "allbench-features-v1"
SOURCE_ROOT = REMOTE_ROOT / "results" / "allbench-deterministic-2seed"
RESULT_ROOT = REMOTE_ROOT / "results" / "allbench-order-calibration"
MODES = ("masked", "visible")
SEEDS = (20261018, 20261019)
SUBJECT_SLICE = (384, 768)
ORDER_SEEDS = tuple(range(10))

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


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=3 * 60 * 60,
)
def score_mode(seed: int, mode: str) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch

    from aims_starter.calibration import AnchorCalibrator, BudgetCalibrator
    from aims_starter.faithful_np import CNPAIRT
    from aims_starter.metrics import DEFAULT_BUDGETS, DEFAULT_WEIGHTS

    output = RESULT_ROOT / f"order-calibration-{seed}-{mode}.json"
    existing = _read_json(output)
    if existing is not None:
        return existing
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    stored = _read_json(SOURCE_ROOT / f"calibration-{seed}-{mode}.json")
    if stored is None:
        raise FileNotFoundError("calibration artifact is missing")
    entries = stored["fitted"]["cnp_airt"]
    calibrator = AnchorCalibrator(
        budgets=tuple(int(entry["budget"]) for entry in entries),
        budgets_map={
            int(entry["budget"]): BudgetCalibrator(
                float(entry["scale"]),
                float(entry["bias"]),
                float(entry["anchor"]),
                int(entry["samples"]),
                bool(entry["fitted"]),
            )
            for entry in entries
        },
    )

    item_embeddings = np.load(CORPUS_ROOT / "item_embeddings.npy")
    subject_embeddings = np.load(CORPUS_ROOT / "subject_embeddings.npy")
    x_dim = int(item_embeddings.shape[1] + subject_embeddings.shape[1])
    evaluation_items = pd.read_parquet(CORPUS_ROOT / "eval_items.parquet")
    orders = pd.read_parquet(CORPUS_ROOT / "eval_orders.parquet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_slice = SUBJECT_SLICE if mode == "masked" else None
    state = torch.load(
        SOURCE_ROOT / f"seed-{seed}-{mode}-checkpoints" / "cnp_airt.pt", map_location=device
    )
    cnp = CNPAIRT(x_dim, subject_feature_slice=subject_slice).to(device)
    cnp.load_state_dict(state)
    cnp.eval()

    def features(rows):
        return np.concatenate(
            [
                item_embeddings[rows["item_index"].to_numpy()],
                subject_embeddings[rows["subject_index"].to_numpy()],
            ],
            axis=1,
        ).astype("float32")

    split_rows = evaluation_items[evaluation_items["split"] == "test"]
    records = []
    for task_id, task_rows in split_rows.groupby("task_id", sort=True):
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
            x_C_pool = features(context_rows)
            y_C_pool = context_rows["label"].to_numpy(dtype="float32")
            for budget in DEFAULT_BUDGETS:
                x_C = torch.as_tensor(x_C_pool[:budget], dtype=torch.float32, device=device)
                y_C = torch.as_tensor(y_C_pool[:budget], dtype=torch.float32, device=device)
                with torch.no_grad():
                    logits = cnp.logits(x_C, y_C, x_T).cpu().numpy().astype("float64")
                calibrated = calibrator.transform(
                    logits, int(budget), y_C_pool[:budget].astype("float64")
                )
                probability = 1.0 / (1.0 + np.exp(-logits))
                records.append(
                    {
                        "task_id": str(task_id),
                        "order_seed": int(order_seed),
                        "budget": int(budget),
                        "raw_brier": float(np.mean((probability - y_T) ** 2)),
                        "calibrated_brier": float(np.mean((calibrated - y_T) ** 2)),
                    }
                )
        print(f"seed={seed} mode={mode} {task_id} scored", flush=True)

    frame = pd.DataFrame(records)
    weights = dict(zip(DEFAULT_BUDGETS, DEFAULT_WEIGHTS, strict=True))
    frame["raw_alc"] = frame["budget"].map(weights) * frame["raw_brier"]
    frame["calibrated_alc"] = frame["budget"].map(weights) * frame["calibrated_brier"]

    orders_out = []
    for order_seed, group in frame.groupby("order_seed", sort=True):
        per_task = group.groupby("task_id", as_index=False)[["raw_alc", "calibrated_alc"]].sum()
        orders_out.append(
            {
                "order": int(order_seed),
                "raw_alc": float(per_task["raw_alc"].mean()),
                "calibrated_alc": float(per_task["calibrated_alc"].mean()),
                "tasks": len(per_task),
            }
        )

    stored_rows = _read_json(SOURCE_ROOT / f"seed-{seed}-{mode}.json")
    check = {"max_abs_delta": None, "matched": 0}
    if stored_rows is not None:
        reference = {
            (row["task_id"], row["order_seed"], row["budget"]): row["brier"]
            for row in stored_rows["rows"]
            if row["split"] == "test" and row["model"] == "cnp_airt"
        }
        deltas = [
            abs(record["raw_brier"] - reference[(record["task_id"], record["order_seed"], record["budget"])])
            for record in records
            if (record["task_id"], record["order_seed"], record["budget"]) in reference
        ]
        check = {
            "max_abs_delta": float(max(deltas)) if deltas else None,
            "matched": len(deltas),
        }

    result = {
        "status": "allbench_order_calibration",
        "seed": seed,
        "mode": mode,
        "orders": orders_out,
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
        _read_json(RESULT_ROOT / f"order-calibration-{seed}-{mode}.json")
        for seed in SEEDS
        for mode in MODES
    ]
    if any(result is None for result in results):
        raise FileNotFoundError("order calibration runs incomplete")

    frame = pd.DataFrame(
        [
            {
                "mode": result["mode"],
                "seed": result["seed"],
                "order": row["order"],
                "raw_alc": row["raw_alc"],
                "calibrated_alc": row["calibrated_alc"],
            }
            for result in results
            for row in result["orders"]
        ]
    )
    orders_out = []
    for (mode, order_seed), group in frame.groupby(["mode", "order"], sort=True):
        orders_out.append(
            {
                "mode": mode,
                "order": int(order_seed),
                "raw_alc": float(group["raw_alc"].mean()),
                "raw_alc_seed_std": float(group["raw_alc"].std(ddof=1)),
                "calibrated_alc": float(group["calibrated_alc"].mean()),
                "calibrated_alc_seed_std": float(group["calibrated_alc"].std(ddof=1)),
                "seeds": len(group),
            }
        )

    artifact = {
        "status": "allbench_order_calibration_aggregate",
        "seeds": list(SEEDS),
        "modes": list(MODES),
        "orders": orders_out,
        "raw_check": [result["raw_check"] for result in results],
    }
    _write_json(RESULT_ROOT / "metrics-order-calibration.json", artifact)
    volume.commit()
    return artifact


@app.local_entrypoint()
def main() -> None:
    handles = [score_mode.spawn(seed, mode) for seed in SEEDS for mode in MODES]
    for handle in handles:
        handle.get()
    print(json.dumps(aggregate.remote(), indent=2, sort_keys=True))
