#!/usr/bin/env python3
"""Parse Codabench formative scoring output and make diagnostic plots.

Usage:
  .venv/bin/python scripts/plot_codabench_scoring.py \
    results/codabench-1005155/scoring_stdout.txt
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
ROW = re.compile(
    r"^(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
)


def parse(path: Path):
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\nLabel budget: (\d+)\s*\n", text)
    by_budget = {}
    # blocks[0] is the ALC summary; subsequent pairs are budget, table text.
    for i in range(1, len(blocks), 2):
        budget = int(blocks[i])
        rows = {}
        for line in blocks[i + 1].splitlines():
            m = ROW.match(line.strip())
            if m:
                subject, benchmark, n, brier, ece = m.groups()
                key = f"{subject} × {benchmark}"
                rows[key] = (int(n), float(brier), float(ece))
        by_budget[budget] = rows
    alc = {}
    # Summary rows have the same row shape and follow the first header.
    summary = text.split("Label budget: 0", 1)[0]
    for line in summary.splitlines():
        m = ROW.match(line.strip())
        if m:
            subject, benchmark, n, brier, ece = m.groups()
            alc[f"{subject} × {benchmark}"] = (int(n), float(brier), float(ece))
    return alc, by_budget


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.log.parent
    out.mkdir(parents=True, exist_ok=True)
    alc, by_budget = parse(args.log)
    pairs = sorted(set().union(*[set(v) for v in by_budget.values()]))
    budgets = [b for b in BUDGETS if b in by_budget]
    brier = np.array([[by_budget[b][p][1] for b in budgets] for p in pairs])
    ece = np.array([[by_budget[b][p][2] for b in budgets] for p in pairs])

    with (out / "parsed_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "budget", "n_unique_pairs", "brier", "ece"])
        for p in pairs:
            for j, b in enumerate(budgets):
                n, _, _ = by_budget[b][p]
                w.writerow([p, b, n, brier[pairs.index(p), j], ece[pairs.index(p), j]])

    mean_brier, mean_ece = brier.mean(0), ece.mean(0)
    alc_from_curve = brier @ WEIGHTS[: len(budgets)]
    # The official ALC summary is retained for comparison; use the curve for plots.
    with (out / "aggregate_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"pairs={len(pairs)}\n")
        f.write("budgets=" + repr(budgets) + "\n")
        f.write("mean_brier=" + repr(mean_brier.tolist()) + "\n")
        f.write("mean_ece=" + repr(mean_ece.tolist()) + "\n")
        f.write("mean_curve_alc=" + repr(float(alc_from_curve.mean())) + "\n")
        f.write("submitted_brier=0.1486051828\nsubmitted_ece=0.1357642624\n")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    x = np.arange(len(budgets))
    for row in brier:
        ax[0, 0].plot(budgets, row, color="#377eb8", alpha=.22, lw=1)
    ax[0, 0].plot(budgets, mean_brier, "o-", color="#111111", lw=2.5, label="mean")
    ax[0, 0].set(title="Brier learning curves", xlabel="labels acquired", ylabel="Brier ↓")
    ax[0, 0].legend()
    for row in ece:
        ax[0, 1].plot(budgets, row, color="#e41a1c", alpha=.22, lw=1)
    ax[0, 1].plot(budgets, mean_ece, "o-", color="#111111", lw=2.5, label="mean")
    ax[0, 1].set(title="Calibration error (ECE)", xlabel="labels acquired", ylabel="ECE ↓")
    ax[0, 1].legend()
    im = ax[1, 0].imshow(brier, aspect="auto", cmap="viridis", vmin=0, vmax=max(.25, brier.max()))
    ax[1, 0].set(title="Pair × budget Brier", xlabel="budget", ylabel="anonymous pair")
    ax[1, 0].set_xticks(x, budgets); ax[1, 0].set_yticks(np.arange(len(pairs)), pairs, fontsize=7)
    fig.colorbar(im, ax=ax[1, 0], fraction=.046)
    alc_vals = np.array([alc[p][1] for p in pairs])
    order = np.argsort(alc_vals)
    ax[1, 1].barh(np.arange(len(pairs)), alc_vals[order], color="#4daf4a")
    ax[1, 1].set(yticks=np.arange(len(pairs)), yticklabels=np.array(pairs)[order],
                 title="Official pair-level Brier ALC", xlabel="Brier ALC ↓")
    ax[1, 1].axvline(float(alc_vals.mean()), color="black", ls="--", lw=1, label="pair mean")
    ax[1, 1].legend()
    fig.suptitle("Codabench submission 1005155 — formative scoring", fontsize=15)
    fig.savefig(out / "codabench_1005155_diagnostics.png", dpi=180)
    plt.close(fig)

    # A simpler view of between-pair variability. This is an empirical spread
    # across evaluated subject-benchmark pairs, not posterior uncertainty.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, values, color, title, ylabel in (
        (ax[0], brier, "#377eb8", "Brier across label budgets", "Brier ↓"),
        (ax[1], ece, "#e41a1c", "Calibration error across label budgets", "ECE ↓"),
    ):
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        axis.fill_between(budgets, np.maximum(0, mean - sd), mean + sd,
                          color=color, alpha=.20, label="mean ± 1 SD across pairs")
        axis.plot(budgets, mean, "o-", color=color, lw=2.5, label="mean")
        axis.set(title=title, xlabel="labels acquired", ylabel=ylabel)
        axis.legend()
    fig.suptitle("Codabench submission 1005155 — mean and between-pair spread")
    fig.savefig(out / "codabench_1005155_mean_sd.png", dpi=180)
    plt.close(fig)

    print(f"Wrote {out / 'codabench_1005155_diagnostics.png'}")
    print(f"Wrote {out / 'codabench_1005155_mean_sd.png'}")
    print(f"Wrote {out / 'parsed_scores.csv'}")
    print("mean Brier by budget:", ", ".join(f"{b}:{v:.4f}" for b, v in zip(budgets, mean_brier)))
    print("mean ECE by budget:", ", ".join(f"{b}:{v:.4f}" for b, v in zip(budgets, mean_ece)))
    print(f"mean pair ALC: {alc_vals.mean():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
