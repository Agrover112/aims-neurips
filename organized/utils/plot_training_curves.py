"""Plot training loss curves from IRT/MIRT model training.

Reads loss_history from saved weights files and plots convergence curves
per model/benchmark/seed.

Usage: python plot_training_curves.py <weights_dir> <output.pdf>
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

if len(sys.argv) < 3:
    print("Usage: python plot_training_curves.py <weights_dir> <output.pdf>")
    sys.exit(1)

WEIGHTS_DIR = Path(sys.argv[1])
OUT_PDF = Path(sys.argv[2])

# Find all weight files with loss_history
weight_files = list(WEIGHTS_DIR.glob("*-weights-seed*.pt"))

if not weight_files:
    print(f"No weight files found in {WEIGHTS_DIR}")
    sys.exit(1)

print(f"Found {len(weight_files)} weight files")

with PdfPages(OUT_PDF) as pdf:
    for weight_file in sorted(weight_files):
        print(f"Processing {weight_file.name}")

        try:
            weights = torch.load(weight_file, map_location="cpu")
        except Exception as e:
            print(f"  Error loading {weight_file}: {e}")
            continue

        # Collect all models with loss_history
        models_with_loss = {}
        for model_key, model_data in weights.items():
            if isinstance(model_data, dict) and "loss_history" in model_data:
                loss_hist = model_data["loss_history"]
                if len(loss_hist) > 0:
                    models_with_loss[model_key] = loss_hist

        if not models_with_loss:
            print(f"  No models with loss_history found")
            continue

        # Create figure with subplots
        n_models = len(models_with_loss)
        fig, axes = plt.subplots(
            (n_models + 2) // 3, 3, figsize=(15, 5 * ((n_models + 2) // 3))
        )
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (model_key, loss_hist) in enumerate(sorted(models_with_loss.items())):
            ax = axes[idx]
            loss_hist = np.asarray(loss_hist)

            ax.plot(loss_hist, linewidth=1.5, color="#2b6ca3")
            ax.fill_between(range(len(loss_hist)), loss_hist, alpha=0.3, color="#2b6ca3")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title(f"{model_key}\nFinal: {loss_hist[-1]:.4f}")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(models_with_loss), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f"{weight_file.stem}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print(f"✓ Saved to {OUT_PDF}")
