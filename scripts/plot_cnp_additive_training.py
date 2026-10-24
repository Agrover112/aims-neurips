#!/usr/bin/env python3
"""Plot three-seed training loss for the context-conditioned additive a/b arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rung", default="additive_ab")
    parser.add_argument("--title", default="CNP–Additive training loss")
    args = parser.parse_args()

    histories = []
    for path in sorted(args.root.glob(f"r-c-{args.rung}-*-masked.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        history = result["training"]
        histories.append(
            (
                int(result["seed"]),
                np.asarray([row["step"] for row in history]),
                np.asarray([row["loss"] for row in history]),
            )
        )
    if not histories:
        raise SystemExit(f"no {args.rung} histories found")

    steps = histories[0][1]
    losses = np.stack([history[2] for history in histories])
    mean = losses.mean(axis=0)
    sd = losses.std(axis=0, ddof=1) if len(histories) > 1 else np.zeros_like(mean)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ("#2F6690", "#3A7D44", "#C65D21")
    for color, (seed, seed_steps, seed_losses) in zip(colors, histories, strict=False):
        ax.plot(seed_steps, seed_losses, color=color, alpha=0.55, linewidth=1.4, label=str(seed))
    ax.plot(steps, mean, color="#171717", linewidth=2.6, marker="o", markersize=3.5, label="mean")
    ax.fill_between(steps, mean - sd, mean + sd, color="#171717", alpha=0.12, label="±1 seed SD")
    ax.set(
        title=args.title,
        xlabel="Optimization step",
        ylabel="Episode BCE",
        xlim=(steps.min(), steps.max()),
    )
    ax.legend(frameon=True, ncol=2)
    ax.text(
        0.01,
        0.02,
        "Each checkpoint is one stochastic 4-task minibatch; fluctuations are expected.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
