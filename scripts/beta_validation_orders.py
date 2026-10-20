"""Beta-Binomial on the ten-order validation collection used for calibration."""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("aims-beta-val-orders")
volume = modal.Volume.from_name("aims-neurips-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pandas>=2.2,<3", "pyarrow>=17,<24", "numpy>=2,<3"
)

CORPUS = "/data/datasets/full/allbench-features-v1"
OUT = "/data/results/allbench-deterministic-2seed/beta-validation-orders.json"
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)


@app.function(image=image, volumes={"/data": volume}, timeout=1800, cpu=4, memory=16_384)
def run() -> dict:
    import numpy as np
    import pandas as pd

    items = pd.read_parquet(f"{CORPUS}/eval_items.parquet")
    orders = pd.read_parquet(f"{CORPUS}/eval_orders.parquet")
    val = items[items["split"] == "validation"]
    rows = []
    for task_id, task_rows in val.groupby("task_id", sort=True):
        target = task_rows[task_rows["pool"] == "evaluation"]
        y_T = target["label"].to_numpy(dtype="float64")
        lookup = task_rows.set_index("item_index")
        for order_seed in range(10):
            order_rows = (
                orders[(orders["task_id"] == task_id) & (orders["order_seed"] == order_seed)]
                .sort_values("acquisition_rank")
                .reset_index(drop=True)
            )
            context = lookup.loc[order_rows["item_index"].to_numpy()].reset_index()
            y_C = context["label"].to_numpy(dtype="float64")
            briers = {}
            for budget in BUDGETS:
                probability = (2.0 + float(y_C[:budget].sum())) / (4.0 + budget)
                briers[budget] = float(np.mean((probability - y_T) ** 2))
            rows.append(
                {
                    "task_id": str(task_id),
                    "order_seed": int(order_seed),
                    "brier_alc": float(
                        sum(w * briers[b] for b, w in zip(BUDGETS, WEIGHTS, strict=True))
                    ),
                    **{f"brier_{b}": briers[b] for b in BUDGETS},
                }
            )
    frame = pd.DataFrame(rows)
    per_order = frame.groupby("order_seed")["brier_alc"].mean()
    summary = {
        "alc_all_orders": float(frame["brier_alc"].mean()),
        "alc_order_0": float(per_order.loc[0]),
        "per_order": {int(k): float(v) for k, v in per_order.items()},
        "tasks": int(frame["task_id"].nunique()),
    }
    Path(OUT).write_text(json.dumps(summary, indent=2))
    volume.commit()
    print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main() -> None:
    run.remote()
