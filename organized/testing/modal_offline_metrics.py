"""Compute standard offline evaluation metrics aggregated from all models.

Metrics computed per model/benchmark/seed from generalization test results:
  - AUC: area under ROC curve (pooled across all instances)
  - Brier: mean squared error (pooled across all instances)
  - ECE: expected calibration error via bin-based method (10 bins, pooled)
  - LogLoss: binary cross-entropy loss (pooled across all instances)
  - Reports aggregated statistics per benchmark/axis/model

Aggregation: pools predictions across all held-out instances for a given
benchmark/model/axis combination and computes metrics on the pooled set.

Run:  modal run --detach modal_offline_metrics.py
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "aims-neurips-offline-metrics"
VOLUME_NAME = "aims-neurips-data"
RESULTS_DIR = Path("/data/results/data-analysis")
OUT = RESULTS_DIR / "offline-metrics-summary.parquet"

def _ece(y_true, y_pred, n_bins=10):
    """Expected calibration error via bin-based method."""
    import numpy as np

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if len(y_pred) < 2:
        return float("nan")

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece_val = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if not mask.any():
            continue
        acc = (y_true[mask] == (y_pred[mask] >= 0.5)).mean()
        conf = y_pred[mask].mean()
        weight = mask.sum() / len(y_pred)
        ece_val += weight * abs(acc - conf)

    return float(ece_val)


def _logloss(y_true, y_pred):
    """Binary cross-entropy loss."""
    import numpy as np

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if len(y_pred) < 1:
        return float("nan")

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return float(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean())


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scikit-learn>=1.5,<2")


@app.function(image=image, cpu=4, memory=32_768, volumes={"/data": volume},
              timeout=30 * 60)
def run() -> dict:
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from sklearn.metrics import roc_auc_score

    results_dir = Path("/data/results/data-analysis")

    # Find all generalization result files
    gen_files = sorted(results_dir.glob("*generalization*seed*.parquet"))

    if not gen_files:
        print("No generalization results found", flush=True)
        return {"error": "No generalization files"}

    print(f"Found {len(gen_files)} generalization result files", flush=True)

    metrics_rows = []

    for gen_file in gen_files:
        fname = gen_file.stem
        print(f"\nProcessing: {fname}", flush=True)

        # Read results
        try:
            df = pd.read_parquet(gen_file)
        except Exception as e:
            print(f"  Error reading {gen_file}: {e}", flush=True)
            continue

        if df.empty:
            print(f"  File empty, skipping", flush=True)
            continue

        # Extract seed from filename
        seed = None
        for part in fname.split("-"):
            if part.startswith("seed"):
                seed = int(part[4:])
                break

        if seed is None:
            print(f"  Could not extract seed, skipping", flush=True)
            continue

        # Group by benchmark/model/axis and pool predictions
        print(f"  Seed: {seed}", flush=True)

        for (bench, model, axis), group in df.groupby(["benchmark_id", "model", "axis"]):
            # Try to pool raw predictions if available
            y_true_all = []
            y_pred_all = []

            if "label" in df.columns and "pred_prob" in df.columns:
                for _, row_data in group.iterrows():
                    label = row_data.get("label")
                    pred = row_data.get("pred_prob")

                    if label is not None and pred is not None:
                        try:
                            if isinstance(label, (list, np.ndarray)):
                                y_true_all.extend(np.asarray(label))
                                y_pred_all.extend(np.asarray(pred))
                            else:
                                y_true_all.append(label)
                                y_pred_all.append(pred)
                        except:
                            pass

            if len(y_true_all) >= 2:
                # Compute metrics on pooled predictions
                y_true_all = np.asarray(y_true_all, dtype=np.float32)
                y_pred_all = np.asarray(y_pred_all, dtype=np.float32)

                try:
                    auc = float(roc_auc_score(y_true_all, y_pred_all))
                except:
                    auc = float("nan")

                brier = float(np.mean((y_pred_all - y_true_all) ** 2))
                ece = _ece(y_true_all, y_pred_all)
                logloss = _logloss(y_true_all, y_pred_all)

                row = {
                    "model": model,
                    "seed": seed,
                    "benchmark_id": bench,
                    "axis": axis,
                    "n_predictions": len(y_true_all),
                    "auc": auc,
                    "brier": brier,
                    "ece": ece,
                    "logloss": logloss,
                }

                print(f"    {bench:20} {model:6} {axis:6} "
                      f"n={len(y_true_all):5} "
                      f"auc={auc:.3f} ece={ece:.3f} "
                      f"brier={brier:.3f} logloss={logloss:.3f}", flush=True)
            else:
                # Fall back to aggregating instance-level AUC/Brier
                auc_vals = group["auc"].dropna()
                brier_vals = group["brier"].dropna()

                if len(auc_vals) < 1:
                    continue

                row = {
                    "model": model,
                    "seed": seed,
                    "benchmark_id": bench,
                    "axis": axis,
                    "n_instances": len(auc_vals),
                    "auc_mean": float(auc_vals.mean()),
                    "auc_std": float(auc_vals.std()),
                    "brier_mean": float(brier_vals.mean()),
                    "brier_std": float(brier_vals.std()),
                }

                print(f"    {bench:20} {model:6} {axis:6} "
                      f"n={len(auc_vals):3} "
                      f"auc={row['auc_mean']:.3f}±{row['auc_std']:.3f} "
                      f"brier={row['brier_mean']:.3f}±{row['brier_std']:.3f}", flush=True)

            metrics_rows.append(row)

    if not metrics_rows:
        print("No metrics computed", flush=True)
        return {"error": "No valid metrics"}

    # Write results
    frame = pd.DataFrame(metrics_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    volume.commit()

    print(f"\n✓ Saved {len(frame)} rows to {OUT.name}", flush=True)
    print(f"  Models: {sorted(frame['model'].unique())}", flush=True)
    print(f"  Benchmarks: {frame['benchmark_id'].nunique()}", flush=True)
    print(f"  Seeds: {sorted(frame['seed'].unique())}", flush=True)

    return {
        "rows": len(frame),
        "models": frame["model"].nunique(),
        "benchmarks": frame["benchmark_id"].nunique(),
        "output": str(OUT)
    }


@app.local_entrypoint()
def main() -> None:
    result = run()
    print("\n=== Result ===")
    print(result)
