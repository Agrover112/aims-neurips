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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import CYCLE, SUBTLE, TEXTWIDTH, header, style

if len(sys.argv) < 3:
    print("Usage: python plot_training_curves.py <weights_dir> <output.pdf>")
    sys.exit(1)

WEIGHTS_DIR = Path(sys.argv[1])
OUT_PDF = Path(sys.argv[2])

weight_files = sorted(WEIGHTS_DIR.glob("*-weights-seed*.pt"))

if not weight_files:
    print(f"No weight files found in {WEIGHTS_DIR}")
    sys.exit(1)

print(f"Found {len(weight_files)} weight files")

NCOL = 3

with PdfPages(OUT_PDF) as pdf:
    for weight_file in weight_files:
        print(f"Processing {weight_file.name}")

        try:
            weights = torch.load(weight_file, map_location="cpu")
        except Exception as exc:
            print(f"  Error loading {weight_file}: {exc}")
            continue

        curves = {
            key: np.asarray(val["loss_history"])
            for key, val in weights.items()
            if isinstance(val, dict) and len(val.get("loss_history", [])) > 0
        }

        if not curves:
            print("  No models with loss_history found")
            continue

        nrow = (len(curves) + NCOL - 1) // NCOL
        fig, axes = plt.subplots(nrow, NCOL, figsize=(TEXTWIDTH, 1.5 * nrow + 0.5),
                                 squeeze=False)
        axes = axes.flatten()

        for idx, (model_key, loss) in enumerate(sorted(curves.items())):
            ax = axes[idx]
            ax.plot(loss, color=CYCLE[idx % len(CYCLE)], lw=1.0)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title(model_key, loc="left", fontweight="bold")
            ax.text(0.97, 0.92, f"final {loss[-1]:.4f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6, color=SUBTLE)
            ax.set_xlim(0, len(loss) - 1)
            style(ax, grid_axis="both")

        for idx in range(len(curves), len(axes)):
            axes[idx].set_visible(False)

        header(fig, weight_file.stem, f"{len(curves)} models")
        fig.tight_layout(pad=0.3, w_pad=1.0, h_pad=0.8, rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved to {OUT_PDF}")
