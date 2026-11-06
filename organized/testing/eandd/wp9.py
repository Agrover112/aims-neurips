"""Rebuild WP9 tables and figures from the saved Person A handoff only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from analysis import compare
from plot_replay import build
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from plotstyle import CYCLE, TEXTWIDTH, style

PRED_KEYS = ["model", "draw", "task_id", "order_seed", "budget"]
DRAW_KEYS = ["shape", "role", "draw", "task_id", "order_seed"]
LOBO = Path(__file__).resolve().parents[3] / "paper/eandd/data/wp10-lobo-explained.csv"


def verify_draws(score, handoff):
    observed = score[DRAW_KEYS].drop_duplicates()
    for (shape, role), group in observed.groupby(["shape", "role"]):
        path = handoff / ("draws-id.parquet" if role == "id" else "draws.parquet")
        frozen = pd.read_parquet(path)
        expected = frozen.loc[frozen["shape"].eq(shape) & frozen.role.eq(role), DRAW_KEYS]
        if set(group.itertuples(index=False, name=None)) != set(expected.itertuples(index=False, name=None)):
            raise ValueError(f"Scores do not match the frozen {shape}/{role} draws")


def prediction_metrics(path, scores):
    rows = []
    files = sorted(path.glob("draw-*.parquet"))
    if not files:
        raise FileNotFoundError(path)
    for file in files:
        p = pd.read_parquet(file, columns=PRED_KEYS + ["content_hash", "p", "y"])
        if p.duplicated(PRED_KEYS + ["content_hash"]).any() or not p.p.between(0, 1).all() or not p.y.isin([0, 1]).all():
            raise ValueError(f"Invalid prediction rows in {file}")
        for key, g in p.groupby(PRED_KEYS, observed=True, sort=False):
            probability = g.p.to_numpy(float)
            truth = g.y.to_numpy(float)
            err = probability-truth
            bins = np.minimum((10*probability).astype(int), 9)
            rows.append((*key, np.mean(err**2),
                         roc_auc_score(truth, probability) if np.unique(truth).size == 2 else np.nan,
                         np.abs(np.bincount(bins, weights=err, minlength=10)).sum()/len(g)))
    metrics = pd.DataFrame(rows, columns=PRED_KEYS + ["brier_from_preds", "auc", "ece"])
    joined = scores[PRED_KEYS + ["brier"]].merge(metrics, on=PRED_KEYS, how="outer", validate="one_to_one")
    delta = (joined.brier-joined.brier_from_preds).abs()
    if delta.isna().any() or delta.max() > 1e-7:
        raise ValueError(f"Saved predictions and scores disagree: max error {delta.max()}")
    task = joined.groupby(["model", "task_id", "budget"], observed=True, as_index=False).agg(
        within_task_auc=("auc", "mean"), within_task_ece_10_equal_width_bins=("ece", "mean"))
    macro = task.groupby(["model", "budget"], observed=True, as_index=False).agg(
        within_task_auc=("within_task_auc", "mean"), auc_eligible_tasks=("within_task_auc", "count"),
        within_task_ece_10_equal_width_bins=("within_task_ece_10_equal_width_bins", "mean"),
        tasks=("task_id", "size"))
    return macro, {"prediction_files": len(files), "score_groups": len(joined),
                   "max_brier_reconstruction_error": float(delta.max())}


def save_figure(fig, out):
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext),
                    bbox_inches=Bbox.from_bounds(0, 0, *fig.get_size_inches()), dpi=180)
    plt.close(fig)


def prior_source_plot(summary, out):
    t = summary.loc[summary.run.eq("wp5-competition-test") & summary.budget.ne("ALC")].copy()
    t["budget"] = t.budget.astype(int)
    budgets = [0, 1, 3, 7, 15, 31]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.52))
    for j, (model, name) in enumerate([("jeffreys", "Jeffreys"),
                                       ("bb-tuned", "Validation tuned"),
                                       ("bb-eb", "Training fitted")]):
        rows = t.loc[t.model.eq(model)].set_index("budget").loc[budgets]
        for ax, metric in zip(axes, ("brier", "delta")):
            ax.plot(range(len(budgets)), rows[metric], marker=["o", "s", "^"][j],
                    ms=3.5, color=CYCLE[j], label=name, lw=1.25)
    axes[0].set_ylabel("Brier score")
    axes[0].set_ylim(0, .30)
    axes[1].set_ylabel("Brier difference to Jeffreys")
    axes[1].set_yticks([-.04, -.02, 0, .02, .04])
    for ax in axes:
        ax.set_xticks(range(len(budgets)), budgets)
        ax.set_xlabel("Label budget")
        style(ax)
    for ax, letter in zip(axes, ("(a)", "(b)")):
        ax.text(0, 1.04, letter, transform=ax.transAxes, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(left=.11, right=.98, top=.79, bottom=.21, wspace=.50)
    save_figure(fig, out)


def per_benchmark_plot(table, out):
    t = table.loc[table.budget.eq("ALC") & table.model.eq("bb-eb") & table.role.isin(["test", "validation"])]
    t = t.sort_values(["role", "benchmark"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.5))
    for j, (role, group) in enumerate(t.groupby("role", sort=True)):
        ax.errorbar(group.delta, group.index, xerr=[group.delta-group.ci_low, group.ci_high-group.delta],
                    fmt="o", color=CYCLE[j], capsize=2, label=role.title())
    ax.axvline(0, color=".55", lw=.8)
    ax.set_yticks(t.index, t.benchmark)
    ax.invert_yaxis()
    ax.set_xlabel("Training-fitted Beta–Binomial minus Jeffreys (Brier ALC)")
    style(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(left=.24, right=.97, top=.83, bottom=.21)
    save_figure(fig, out)


def lobo_plot(table, out):
    if table.held_out.duplicated().any():
        raise ValueError("Expected one row per held-out benchmark")
    x = table["mean_abs_acc_minus_0.52"]
    y = table.bb_eb_diff
    rho = spearmanr(x, y).statistic
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.85))
    ax.axhline(0, color=".55", lw=.8)
    ax.errorbar(x, y, yerr=[y-table.lo, table.hi-y], fmt="o", ms=4,
                color=CYCLE[0], capsize=2)
    ax.set_xlabel("Mean absolute distance of task accuracy from prior mean (0.52)")
    ax.set_ylabel("Brier ALC difference\n(fitted Beta–Binomial − Jeffreys)")
    ax.text(.04, .91, rf"Spearman $\rho={rho:.2f}$; {len(table)} benchmarks",
            transform=ax.transAxes, va="top")
    style(ax)
    fig.subplots_adjust(left=.16, right=.97, top=.93, bottom=.21)
    save_figure(fig, out)


def render_figures(summary, per_benchmark, lobo, out):
    prior_source_plot(summary, out / "figure1-prior-source")
    build(summary.loc[summary.run.isin(["wp6-competition-id", "wp6-competition-test"])], "competition",
          [("ID", "id"), ("OOD test", "test")],
          [("Offline fitted values kept", "offline-kept"), ("Online Rasch, fitted prior", "rasch-fitted")],
          out / "figure2-offline-online")
    per_benchmark_plot(per_benchmark.loc[per_benchmark.run.isin(["wp5-competition-val", "wp5-competition-test"])],
                       out / "figure3-benchmark-breakdown")
    lobo_plot(pd.read_csv(lobo), out / "figure4-lobo-explained")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--handoff", type=Path)
    source.add_argument("--figures-from", type=Path, help="WP9 directory containing paired-scores.csv and per-benchmark.csv")
    p.add_argument("--lobo", default=LOBO, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.figures_from:
        render_figures(pd.read_csv(args.figures_from / "paired-scores.csv"),
                       pd.read_csv(args.figures_from / "per-benchmark.csv"), args.lobo, out)
        return
    tables, breakdowns, audits = [], [], {}
    for path in sorted((args.handoff / "scores").glob("wp[568]-*.parquet")):
        score = pd.read_parquet(path)
        if not score.run.eq(path.stem).all():
            raise ValueError(f"Run name mismatch in {path}")
        verify_draws(score, args.handoff)
        tables.append(compare(score, "jeffreys").assign(run=path.stem))
        benchmarks = score.task_id.str.rsplit("::", n=1).str[-1]
        for b, rows in score.groupby(benchmarks):
            breakdowns.append(compare(rows, "jeffreys").assign(run=path.stem, benchmark=b))
        audits[path.stem] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = pd.concat(tables, ignore_index=True)
    per_benchmark = pd.concat(breakdowns, ignore_index=True)
    summary.to_csv(out / "paired-scores.csv", index=False)
    per_benchmark.to_csv(out / "per-benchmark.csv", index=False)
    test = summary.loc[summary.run.eq("wp5-competition-test") & summary.budget.eq("ALC")]
    matrix = {
        "fixed": ("jeffreys", "rasch-fixed", "kernel-fixed"),
        "validation_tuned": ("bb-tuned", "rasch-retuned", "kernel-old"),
        "training_fitted": ("bb-eb", "rasch-fitted", "kernel-fitted"),
    }
    table_rows = []
    for prior, models in matrix.items():
        for family, model in zip(("beta", "rasch", "kernel"), models):
            r = test.loc[test.model.eq(model)].iloc[0]
            table_rows.append(dict(prior=prior, family=family, model=model, alc=r.brier, delta=r.delta,
                                   ci_low=r.ci_low, ci_high=r.ci_high))
    pd.DataFrame(table_rows).to_csv(out / "table1.csv", index=False)
    pred, audit = prediction_metrics(args.handoff / "preds/wp5-competition-test",
                                    pd.read_parquet(args.handoff / "scores/wp5-competition-test.parquet"))
    pred.to_csv(out / "competition-test-auc-calibration.csv", index=False)
    render_figures(summary, per_benchmark, args.lobo, out)
    (out / "audit.json").write_text(json.dumps({"predictions": audit, "score_hashes": audits}, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
