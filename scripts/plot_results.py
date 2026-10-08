#!/usr/bin/env python3
"""Export a compact learning-curve figure and summary table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    values = json.loads(args.metrics.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [(int(budget), result["brier"], result["ece"]) for budget, result in values["by_budget"].items()]
    rows.sort()
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label_budget", "brier", "ece"])
        writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install the 'experiment' extra to create figures") from exc
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    axis.plot([row[0] for row in rows], [row[1] for row in rows], marker="o")
    axis.set(xlabel="Acquired labels", ylabel="Brier score", xscale="symlog", xticks=[row[0] for row in rows])
    figure.savefig(args.output_dir / "learning_curve.pdf")
    figure.savefig(args.output_dir / "learning_curve.png", dpi=200)


if __name__ == "__main__":
    main()

