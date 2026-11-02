#!/usr/bin/env python3
"""Post-hoc analyses of the saved benchmark-holdout replays, as report pages.

Everything here recombines predictions the online simulation already saved under random
acquisition, whose label order does not depend on the predictor; no model is
rerun. Four questions:

    combinations   can the Beta-Binomial rescue the ANP: fallback, ensemble, residual correction
    budgets        where the ANP loses: per-budget gap, seen vs unseen benchmark
    decomposition  which part fails: the ANP's ability component vs its difficulty component
    ability        does the loss grow with subject ability

``--build`` computes the summary tables from the run outputs and caches them
under ``organized/testing/online/results/``; without it the pages are drawn
from that cache, which is how ``plot_online_final_results.py`` includes them.

    python organized/utils/plot_online_posthoc.py --build \\
        --runs <dir of downloaded run outputs> --panel-items items.parquet \\
        --responses filtered-responses.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_names import NAME  # noqa: E402
from plotstyle import CYCLE, INK, MUTED, SUBTLE, TEXTWIDTH, style  # noqa: E402

CACHE = ROOT / "organized" / "testing" / "online" / "results"
TABLES = ("combinations", "budgets", "decomposition", "ability")
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = dict(zip(BUDGETS, (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)))
SEEDS = {"validation": "s3", "test": "s10"}
HOLDOUT = {k: NAME[k] for k in ("w64-filtered", "w64-sam", "w64-dense")}
RANDOM = {k: NAME[k] for k in ("w64-contam", "w128")}
SEEN = "editbench"  # the one test benchmark the random-split models trained on
KEY = ["task_id", "order_seed", "requested_budget", "item_uid"]
COLOR = {**{k: CYCLE[i] for i, k in enumerate(HOLDOUT)}, **{k: CYCLE[i + 3] for i, k in enumerate(RANDOM)}}


def exact_page(fig) -> Bbox:
    return Bbox.from_bounds(0, 0, *fig.get_size_inches())


# ------------------------------------------------------------ computation --
def _ci(x: np.ndarray, rng) -> tuple[float, float, float]:
    """Mean and 95% task-bootstrap interval; the task is the independent unit."""
    boots = [rng.choice(x, len(x)).mean() for _ in range(3000)]
    return float(x.mean()), *map(float, np.percentile(boots, [2.5, 97.5]))


def _predictions(runs: Path, role: str, model: str) -> pd.DataFrame:
    return pd.read_parquet(runs / f"{role}-{model}-random_prefix-{SEEDS[role]}-v2" / "predictions.parquet")


def _shift(runs: Path, role: str, model: str) -> pd.DataFrame:
    """Mean residual of the first k labels, each against the prediction made
    before that label was revealed -- the AIPW-style correction term."""
    d = pd.read_parquet(runs / f"{role}-{model}-random_prefix-{SEEDS[role]}-v2" / "acquisition_decisions.parquet")
    d = d[d.accepted].sort_values(["task_id", "order_seed", "labels_acquired_before"])
    d["res"] = d.revealed_label - d.probability_before_decision
    rows = []
    for (task, seed), g in d.groupby(["task_id", "order_seed"]):
        c = g.res.cumsum().to_numpy()
        rows += [(task, seed, k, c[k - 1] / k if k else 0.0) for k in BUDGETS]
    return pd.DataFrame(rows, columns=KEY[:3] + ["shift"])


def _task_budget(f: pd.DataFrame, col: str) -> pd.DataFrame:
    """Task x budget Brier, averaged over candidate orders."""
    se = (f[col] - f.label) ** 2
    return se.groupby([f.task_id, f.order_seed, f.requested_budget]).mean() \
        .groupby(level=[0, 2]).mean().unstack()


def _alc(tb: pd.DataFrame) -> pd.Series:
    return sum(w * tb[b] for b, w in WEIGHTS.items())


def build(runs: Path, panel_items: Path, responses: Path) -> dict:
    rng = np.random.default_rng(0)
    beta = {r: _predictions(runs, r, "beta")[KEY + ["probability"]].rename(columns={"probability": "beta"})
            for r in SEEDS}
    combos, budgets, decomp, ability_rows = [], [], [], []
    grid = np.round(np.linspace(0, 1, 11), 1)

    # Beta-Binomial with the AIPW-style shift, on both roles.
    for role in SEEDS:
        f = _predictions(runs, role, "beta").merge(_shift(runs, role, "beta"), on=KEY[:3])
        f["aipw"] = (f.probability + f["shift"]).clip(1e-4, 1 - 1e-4)
        diff = _alc(_task_budget(f, "aipw")) - _alc(_task_budget(f, "probability"))
        combos.append((role, "beta", "AIPW-style shift", *_ci(diff.to_numpy(), rng)))

    frames = {}
    for role in SEEDS:
        for m in HOLDOUT:
            f = _predictions(runs, role, m).rename(columns={"probability": "anp"}) \
                .merge(beta[role], on=KEY, validate="one_to_one") \
                .merge(_shift(runs, role, m), on=KEY[:3])
            f["fallback"] = np.where(f.requested_budget < 2, f.beta, f.anp)
            f["aipw"] = (f.anp + f["shift"]).clip(1e-4, 1 - 1e-4)
            frames[role, m] = f
    for m in HOLDOUT:
        V = frames["validation", m]
        # Blend weights fitted on validation only, then frozen for test.
        w_global = min(grid, key=lambda w: _alc(_task_budget(V.assign(p=w * V.anp + (1 - w) * V.beta), "p")).mean())
        w_budget = {b: min(grid, key=lambda w: ((w * V.anp + (1 - w) * V.beta - V.label)[V.requested_budget == b] ** 2)
                           .groupby(V.task_id).mean().mean()) for b in BUDGETS}
        for role in SEEDS:
            f = frames[role, m]
            f["ensemble"] = w_global * f.anp + (1 - w_global) * f.beta
            f["ensemble_b"] = f.requested_budget.map(w_budget) * f.anp + (1 - f.requested_budget.map(w_budget)) * f.beta
            base = _alc(_task_budget(f, "beta"))
            for col, name in (("anp", "ANP alone"), ("fallback", NAME["fallback"]),
                              ("ensemble", NAME["ensemble"]), ("ensemble_b", NAME["ensemble-budget"]),
                              ("aipw", "AIPW-style shift")):
                if role == "validation" and col.startswith("ensemble"):
                    continue  # fitted there; an in-sample score is not a result
                combos.append((role, m, name, *_ci((_alc(_task_budget(f, col)) - base).to_numpy(), rng)))

    # Per-budget gap and level/ordering decomposition, test only.
    for m in list(HOLDOUT) + list(RANDOM):
        f = (frames["test", m] if m in HOLDOUT else
             _predictions(runs, "test", m).rename(columns={"probability": "anp"}).merge(beta["test"], on=KEY))
        mean = f.groupby(KEY[:3]).anp.transform("mean")
        f = f.assign(level=mean, order=(f.beta + f.anp - mean).clip(0.001, 0.999))
        for subset, mask in (("seen", f.benchmark_id == SEEN), ("unseen", f.benchmark_id != SEEN)):
            if m in HOLDOUT and subset == "seen":
                continue  # holdout models saw no test benchmark
            g = f[mask]
            base = _task_budget(g, "beta")
            gap = _task_budget(g, "anp") - base
            for b in BUDGETS:
                budgets.append((m, subset, b, *_ci(gap[b].to_numpy(), rng)))
            for col, name in (("anp", NAME["full"]), ("level", NAME["level"]), ("order", NAME["order"])):
                d = _task_budget(g, col) - base
                for phase, cols, wts in (("0-1 labels", [0, 1], [.1, .2]), ("3-31 labels", [3, 7, 15, 31], [1, 1, 1, 1])):
                    v = sum(w * d[c] for c, w in zip(cols, wts)) / sum(wts)
                    decomp.append((m, subset, phase, name, *_ci(v.to_numpy(), rng)))

    # Ability: accuracy on the task's acquisition pool, never scored.
    items = pd.read_parquet(panel_items)
    acc = items[items.pool == "acquisition"].groupby("task_id").label.mean()
    r = pd.read_parquet(responses, columns=["subject_uid", "benchmark_id", "benchmark_split", "label"])
    train_acc = r[r.benchmark_split == "train"].groupby(["subject_uid", "benchmark_id"]).label.mean()
    for role in SEEDS:
        for m in HOLDOUT:
            f = frames[role, m]
            gap = _alc(_task_budget(f, "anp")) - _alc(_task_budget(f, "beta"))
            for task, v in gap.items():
                ability_rows.append((role, m, task, float(acc[task]), float(v)))
    out = {
        "combinations": pd.DataFrame(combos, columns=["role", "model", "method", "mean", "lo", "hi"]),
        "budgets": pd.DataFrame(budgets, columns=["model", "subset", "budget", "mean", "lo", "hi"]),
        "decomposition": pd.DataFrame(decomp, columns=["model", "subset", "phase", "variant", "mean", "lo", "hi"]),
        "ability": pd.DataFrame(ability_rows, columns=["role", "model", "task_id", "ability", "gap"]),
    }
    out["ability"].attrs["train_range"] = tuple(np.percentile(train_acc, [5, 95]))
    for name, frame in out.items():
        frame.to_parquet(CACHE / f"posthoc-{name}.parquet", index=False)
    pd.Series(np.percentile(train_acc, [5, 95]), index=["p5", "p95"]).to_frame("training_task_accuracy") \
        .to_parquet(CACHE / "posthoc-train-range.parquet")
    return out


def load() -> dict | None:
    paths = {n: CACHE / f"posthoc-{n}.parquet" for n in TABLES}
    if not all(p.exists() for p in paths.values()):
        return None
    out = {n: pd.read_parquet(p) for n, p in paths.items()}
    out["train_range"] = pd.read_parquet(CACHE / "posthoc-train-range.parquet")["training_task_accuracy"].to_numpy()
    return out


# ------------------------------------------------------------------ pages ---
def header(fig, title: str, subtitle: str) -> None:
    fig.text(0.018, 0.985, title, fontsize=9, fontweight="bold", va="top", ha="left")
    fig.text(0.018, 0.94, subtitle, fontsize=6.3, color=SUBTLE, va="top", ha="left", linespacing=1.3)


def _dot(ax, y, row, colour, marker="o"):
    ax.errorbar(row["mean"], y, xerr=[[row["mean"] - row.lo], [row.hi - row["mean"]]], fmt=marker,
                color=colour, ms=3.2, lw=0.9, capsize=1.6)


def combinations_page(pdf, t: dict):
    c = t["combinations"]
    methods = ["ANP alone", NAME["fallback"], NAME["ensemble"], NAME["ensemble-budget"],
               "AIPW-style shift"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.0), sharey=True)
    fig.subplots_adjust(left=0.27, right=0.985, top=0.78, bottom=0.2, wspace=0.08)
    header(fig, "Can Beta-Binomial rescue the ANP?",
           "Macro Brier ALC minus Beta-Binomial, random acquisition; bars: 95% task bootstrap.\n"
           "Negative = better than beta. Blend weights fitted on validation, so they appear on test only.")
    rows = methods + ["beta + AIPW-style shift"]
    for ax, role in zip(axes, ("validation", "test")):
        for j, m in enumerate(HOLDOUT):
            for i, meth in enumerate(methods):
                r = c[(c.role == role) & (c.model == m) & (c.method == meth)]
                if len(r):
                    _dot(ax, i + (j - 1) * 0.22, r.iloc[0], COLOR[m])
        r = c[(c.role == role) & (c.model == "beta")]
        _dot(ax, len(methods), r.iloc[0], INK, marker="s")
        ax.axvline(0, color=INK, lw=0.6, ls=(0, (4, 2)))
        ax.set_title(f"{role} ({82 if role == 'validation' else 45} tasks)", fontsize=7, pad=3)
        ax.set_xlabel("ALC minus beta", fontsize=6.5)
        ax.tick_params(labelsize=6)
        style(ax, grid_axis="x")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(rows, fontsize=6.2)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=COLOR[m], marker="o", ls="none", ms=3.2) for m in HOLDOUT]
    fig.legend(handles, list(HOLDOUT.values()), loc="lower center", ncol=3, fontsize=6, frameon=False,
               bbox_to_anchor=(0.6, 0.0))
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def budgets_page(pdf, t: dict):
    b = t["budgets"]
    x = np.arange(len(BUDGETS))
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.78, bottom=0.27, wspace=0.08)
    header(fig, "Where the ANP loses: Brier minus Beta-Binomial at each budget (test)",
           "Left: benchmark-holdout ANPs on the three OOD test benchmarks. Right: random-subject-split ANPs on\n"
           "the ID benchmark editbench (solid) and the OOD benchmarks algotune + cmmlu (dashed).")
    for ax, models in zip(axes, (HOLDOUT, RANDOM)):
        for m, name in models.items():
            for subset, ls in (("seen", "-"), ("unseen", "--" if m in RANDOM else "-")):
                r = b[(b.model == m) & (b.subset == subset)].sort_values("budget")
                if r.empty:
                    continue
                ax.fill_between(x, r.lo, r.hi, color=COLOR[m], alpha=0.15, lw=0)
                ax.plot(x, r["mean"], color=COLOR[m], ls=ls, marker="o", ms=2.6, lw=1.0,
                        label=name if m in HOLDOUT else f"{name.split(',')[0].replace('ANP ', '')} {'ID' if subset == 'seen' else 'OOD'}")
        ax.axhline(0, color=INK, lw=0.6, ls=(0, (4, 2)))
        ax.set_xticks(x)
        ax.set_xticklabels(BUDGETS)
        ax.set_xlabel("labels acquired", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5.4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
        style(ax, grid_axis="both")
    axes[0].set_ylabel("Brier minus Beta-Binomial", fontsize=6.5)
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def decomposition_page(pdf, t: dict):
    d = t["decomposition"]
    variants = [NAME["full"], NAME["level"], NAME["order"]]
    cases = [("w64-contam", "seen"), ("w64-contam", "unseen"), ("w128", "seen"), ("w128", "unseen"),
             ("w64-filtered", "unseen"), ("w64-sam", "unseen"), ("w64-dense", "unseen")]
    labels = [f"{(RANDOM | HOLDOUT)[m]}, {'ID (editbench)' if s == 'seen' else 'OOD'}" for m, s in cases]
    vcol = {NAME["full"]: INK, NAME["level"]: MUTED["teal"], NAME["order"]: MUTED["rose"]}
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.7), sharey=True)
    fig.subplots_adjust(left=0.45, right=0.985, top=0.76, bottom=0.25, wspace=0.08)
    header(fig, "Which part of the ANP fails on a new benchmark: ability or difficulty? (test)",
           "Ability component: the mean of the ANP's predictions for the task.\n"
           "Difficulty component: each item's offset from that mean.\n"
           "Brier minus Beta-Binomial; negative = better. Bars: 95% task bootstrap.")
    for ax, phase in zip(axes, ("0-1 labels", "3-31 labels")):
        for i, (m, s) in enumerate(cases):
            for j, v in enumerate(variants):
                r = d[(d.model == m) & (d.subset == s) & (d.phase == phase) & (d.variant == v)]
                _dot(ax, i + (j - 1) * 0.24, r.iloc[0], vcol[v])
        ax.axvline(0, color=INK, lw=0.6, ls=(0, (4, 2)))
        ax.set_title(phase, fontsize=7, pad=3)
        ax.set_xlabel("Brier minus Beta-Binomial", fontsize=6.5)
        ax.tick_params(labelsize=6)
        style(ax, grid_axis="x")
    axes[0].set_yticks(range(len(cases)))
    axes[0].set_yticklabels(labels, fontsize=5.6)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=vcol[v], marker="o", ls="none", ms=3.2) for v in variants]
    fig.legend(handles, variants, loc="lower center", ncol=1, fontsize=6, frameon=False, bbox_to_anchor=(0.6, 0.0))
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def ability_page(pdf, t: dict):
    a = t["ability"]
    lo, hi = t["train_range"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.8), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.78, bottom=0.24, wspace=0.08)
    header(fig, "Does the ANP's loss grow with subject ability?",
           "One point per task. Ability: the subject's accuracy on the task's acquisition pool (never scored).\n"
           f"Shaded: 5-95% of training-task accuracy ({lo:.2f}-{hi:.2f}). No test or validation subject appears in training.")
    for ax, role in zip(axes, ("validation", "test")):
        ax.axvspan(lo, hi, color=SUBTLE, alpha=0.08, lw=0)
        for m, name in HOLDOUT.items():
            r = a[(a.role == role) & (a.model == m)]
            ax.scatter(r.ability, r.gap, s=6, color=COLOR[m], alpha=0.7, lw=0, label=name)
            rho = r.ability.corr(r.gap, method="spearman")
            ax.text(0.02, 0.95 - 0.07 * list(HOLDOUT).index(m), f"{name}: $\\rho$ = {rho:+.2f}",
                    transform=ax.transAxes, fontsize=5.6, color=COLOR[m], va="top")
        ax.axhline(0, color=INK, lw=0.6, ls=(0, (4, 2)))
        ax.set_xlim(0, 1)
        ax.set_title(role, fontsize=7, pad=3)
        ax.set_xlabel("subject ability (acquisition-pool accuracy)", fontsize=6.5)
        ax.tick_params(labelsize=6)
        style(ax, grid_axis="both")
    axes[0].set_ylabel("ANP ALC minus Beta-Binomial", fontsize=6.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.0),
               markerscale=1.8)
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def pages(pdf, t: dict) -> list:
    # The Beta-Binomial combinations now have their own ablation page in
    # plot_online_main.py, so they are not drawn twice.
    return [("posthoc-2-budgets", budgets_page(pdf, t)),
            ("posthoc-3-decomposition", decomposition_page(pdf, t)),
            ("posthoc-4-ability", ability_page(pdf, t))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--panel-items", type=Path)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="standalone PDF of these pages only")
    parser.add_argument("--png-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.build:
        build(args.runs, args.panel_items, args.responses)
    t = load()
    if t is None:
        parser.error("no cached tables; run with --build")
    if args.out or args.png_dir:
        out = args.out or (args.png_dir / "posthoc.pdf")
        with PdfPages(out) as pdf:
            figs = pages(pdf, t)
        if args.png_dir:
            args.png_dir.mkdir(parents=True, exist_ok=True)
            for name, fig in figs:
                fig.savefig(args.png_dir / f"{name}.png", dpi=120, bbox_inches=exact_page(fig))
    print("post-hoc tables:", {k: len(v) for k, v in t.items() if k != "train_range"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
