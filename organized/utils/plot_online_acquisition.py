"""Visualize online/streaming acquisition budget evolution.

Shows how predictions change as budget increases from 0→1→3→7→15→31.
Recreates the expected online adaptation protocol for the competition.

This would require prediction data at each budget stage. For now, this
is a template showing the structure for online evaluation visualization.

Usage: python plot_online_acquisition.py <budget_predictions.parquet> <output.pdf>
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

if len(sys.argv) < 2:
    print("Usage: python plot_online_acquisition.py <predictions_dir> <output.pdf>")
    print("\nExample structure for online predictions:")
    print("  Each budget stage saved as: predictions_budget{0,1,3,7,15,31}.parquet")
    print("  Columns: benchmark_id, subject_uid, item_uid, label, pred_prob, budget")
    sys.exit(1)

PRED_DIR = Path(sys.argv[1])
OUT_PDF = Path(sys.argv[2]) if len(sys.argv) > 2 else PRED_DIR.parent / "online-acquisition.pdf"

BUDGETS = (0, 1, 3, 7, 15, 31)

# Try to find budget prediction files
budget_files = {}
for budget in BUDGETS:
    pattern = f"*budget{budget}*.parquet"
    matches = list(PRED_DIR.glob(pattern))
    if matches:
        budget_files[budget] = matches[0]

if not budget_files:
    print(f"No budget prediction files found in {PRED_DIR}")
    print(f"Expected patterns: *budget0*.parquet, *budget1*.parquet, etc.")
    print("\nTo use this script, save predictions at each budget stage.")
    sys.exit(1)

print(f"Found predictions for budgets: {sorted(budget_files.keys())}")

with PdfPages(OUT_PDF) as pdf:
    # Page 1: Budget evolution over time
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    all_dfs = {}
    for budget in sorted(budget_files.keys()):
        try:
            df = pd.read_parquet(budget_files[budget])
            all_dfs[budget] = df
        except Exception as e:
            print(f"Error loading budget {budget}: {e}")

    if not all_dfs:
        print("Could not load any budget predictions")
        sys.exit(1)

    # Plot AUC/Brier by budget
    budgets_done = sorted(all_dfs.keys())

    for model_idx, model in enumerate(["1pl", "2pl", "3pl"][:len(axes)]):
        ax = axes[model_idx]

        auc_by_budget = []
        brier_by_budget = []

        for budget in budgets_done:
            df = all_dfs[budget]

            # Filter to just this model
            if "model" in df.columns:
                model_df = df[df["model"] == model]
            else:
                model_df = df

            if "pred_prob" in model_df.columns and "label" in model_df.columns:
                from sklearn.metrics import roc_auc_score
                y_true = []
                y_pred = []

                for _, row in model_df.iterrows():
                    label = row.get("label")
                    pred = row.get("pred_prob")

                    if label is not None and pred is not None:
                        try:
                            if hasattr(label, "__len__"):
                                y_true.extend(np.asarray(label))
                                y_pred.extend(np.asarray(pred))
                            else:
                                y_true.append(label)
                                y_pred.append(pred)
                        except:
                            pass

                if len(y_true) > 1:
                    try:
                        auc = roc_auc_score(y_true, y_pred)
                        auc_by_budget.append(auc)
                    except:
                        auc_by_budget.append(np.nan)
                    brier = np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)
                    brier_by_budget.append(brier)

        if auc_by_budget:
            ax.plot(budgets_done, auc_by_budget, "o-", linewidth=2, markersize=8, label="AUC", color="#2b6ca3")
            ax.set_xlabel("Budget")
            ax.set_ylabel("AUC")
            ax.set_title(f"{model.upper()} - Performance by Budget")
            ax.set_xticks(budgets_done)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])

    # Items acquired summary
    ax = axes[3]
    if all_dfs:
        first_budget = all_dfs[budgets_done[0]]
        if "n_acquired" in first_budget.columns:
            n_acquired = first_budget.groupby("benchmark_id")["n_acquired"].first()
            ax.barh(range(len(n_acquired)), n_acquired.values, color="#4b8b6f")
            ax.set_yticks(range(len(n_acquired)))
            ax.set_yticklabels(n_acquired.index, fontsize=9)
            ax.set_xlabel("Items Acquired (Budget=0)")
            ax.set_title("Acquisition Counts per Benchmark")
        else:
            ax.text(0.5, 0.5, "No acquisition counts available", ha="center", va="center")

    # Info panel
    ax = axes[4]
    ax.axis("off")
    info_text = f"Online Acquisition Analysis\n\n"
    info_text += f"Budgets tested: {', '.join(map(str, budgets_done))}\n"
    info_text += f"Benchmarks: {len(all_dfs[budgets_done[0]]['benchmark_id'].unique()) if budgets_done else 0}\n"
    info_text += f"\nStreaming protocol:\n"
    info_text += f"1. Start with budget=0 items\n"
    info_text += f"2. Acquire 1 item→budget=1\n"
    info_text += f"3. Continue: 1→3→7→15→31\n"
    info_text += f"4. Measure performance at each stage"
    ax.text(0.1, 0.5, info_text, fontsize=11, verticalalignment="center", family="monospace")

    # Hide extra subplot
    axes[5].axis("off")

    fig.suptitle("Online Streaming Acquisition: Budget Evolution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"✓ Saved to {OUT_PDF}")
print("\nNote: To populate online acquisition data, run streaming predictions at each budget.")
