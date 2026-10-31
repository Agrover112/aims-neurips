#!/usr/bin/env python3
"""Main-model curves and the ablation table, for the online final-results PDF.

Two protocols, never mixed on one axis because they put beta at different
levels:

    single-pair  each subject-benchmark task replayed alone on its own labels
                 (the protocol of every other page in the report)
    multi-pair   CodaBench-like draws of one or two subjects per benchmark,
                 where a predictor may use every pair's labels. Models that
                 ignore other pairs are placed on the same draws from their
                 single-pair scores, so the comparison is paired.

Every curve is mean Brier per budget with a 95% task-bootstrap band; the task
is the independent unit. ``--build`` assembles one tidy task x budget table
from the saved replay outputs and caches it; pages are drawn from the cache.

    python organized/utils/plot_online_main.py --build --cache /home/agrov/aims-replay-cache
"""

from __future__ import annotations

import argparse
import sys
import textwrap
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
import plot_online_posthoc as posthoc  # noqa: E402
from model_names import NAME  # noqa: E402
from plotstyle import CYCLE, INK, MUTED, SUBTLE, TEXTWIDTH, style  # noqa: E402

RESULTS = ROOT / "organized" / "testing" / "online" / "results"
PANEL = "benchmark-holdout-online-v1-s20261018-n15-m80-cap100000"
CACHE = RESULTS / "main-task-budget.parquet"
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = dict(zip(BUDGETS, (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)))
ROLES = ("validation", "test")

# (key, label, colour) for the main curves, per protocol.
MAIN = {
    "single-pair": [("const-mean", NAME["const-mean"], SUBTLE),
                    ("beta", NAME["beta"], INK),
                    ("w64-filtered", NAME["w64-filtered"], MUTED["indigo"])],
    "multi-pair": [("constant", NAME["constant"], SUBTLE),
                   ("jeffreys", NAME["jeffreys"], INK),
                   ("w64-filtered", NAME["w64-filtered"], MUTED["indigo"]),
                   ("pooled-rasch", NAME["pooled-rasch"], MUTED["rose"])],
}
BASELINE = {"single-pair": "beta", "multi-pair": "jeffreys"}

# Ablation groups: (title, protocol, [(key, label)]).
ABLATIONS = [
    ("No training recipe fixes the ANP; having seen the benchmark does", "single-pair",
     [(k, NAME[k]) for k in ("w64-sam", "w64-dense", "w64-contam", "w128")]),
    ("The beta-binomial can limit the ANP's damage, but not turn it into a gain", "single-pair",
     [(k, NAME[k]) for k in ("w64-filtered", "gate", "blend", "blend-budget", "aipw", "beta-aipw", "emp-mean")]),
    ("Subject metadata barely helps, and release date hurts", "single-pair",
     [(k, NAME[k]) for k in ("hier, no metadata", "hier + provider", "hier + name + provider",
                             "hier + provider + date", "hier + name + provider + date")]),
    ("Other subjects' labels help; a provider prior undoes it", "multi-pair",
     [(k, NAME[k]) for k in ("pooled-rasch", "pooled-rasch-no-item", "combined-provider")]),
]
TABLE_LABEL = {
    "No training recipe fixes the ANP; having seen the benchmark does": "ANP training recipe and split",
    "The beta-binomial can limit the ANP's damage, but not turn it into a gain": "Combining the ANP with the beta-binomial",
    "Subject metadata barely helps, and release date hurts": "Hierarchical beta-binomial: subject-metadata prior",
    "Other subjects' labels help; a provider prior undoes it": "Pooling all pairs' labels (multi-pair)",
}
CAPTION = {
    "No training recipe fixes the ANP; having seen the benchmark does": (
        "SAM and dense-prefix training of the benchmark-holdout ANP stay above the beta-binomial on test, by a wide "
        "margin at 0-1 labels and slightly from 3 labels on. The random-split ANPs trained on editbench, one of the three test benchmarks; their small "
        "test difference averages a large win on editbench with a loss on algotune and cmmlu."),
    "The beta-binomial can limit the ANP's damage, but not turn it into a gain": (
        "The ANP combined with the beta-binomial: the gate uses the beta-binomial below two labels; blend weights are fitted on "
        "validation and frozen for test (the validation curves for the blends are in-sample). The residual-correction (AIPW-style) "
        "variants shift each prediction by the mean residual of the labels seen so far; after one label "
        "that shift is about +/-0.5, which is why they spike at budget 1. The raw hit rate shows the same "
        "spike for the same reason: no shrinkage."),
    "Subject metadata barely helps, and release date hurts": (
        "A beta-binomial with a prior fitted on training benchmarks, whose mean depends on the subject's "
        "provider, model name and release date as sent by CodaBench. Test models are newer than any training "
        "model, so the release-date term extrapolates and hurts at budget 0. A leave-one-benchmark-out check "
        "on the 14 non-test benchmarks did not confirm the small gains of the other arms."),
    "Other subjects' labels help; a provider prior undoes it": (
        "Multi-pair draws (one or two subjects per benchmark). Pooled Rasch fits benchmark rate, subject "
        "ability and item difficulty from every pair's labels; removing the item term halves the gain. "
        "Centring ability on the subject's provider was chosen on validation but loses on test, at budget 0, "
        "because test models are stronger than training models from the same providers."),
}
NOT_REPLAYED = [(NAME["w64-ladder"], "trained, never replayed"),
                (NAME["w64-anchor"], "trained, never replayed"),
                (NAME["w64-sam-ladder"], "trained, never replayed"),
                (NAME["w64-noitem"], "training; step-400 checkpoint only")]


def exact_page(fig) -> Bbox:
    return Bbox.from_bounds(0, 0, *fig.get_size_inches())


# ------------------------------------------------------------ computation --
def _long(tb: pd.DataFrame, protocol: str, role: str, model: str) -> pd.DataFrame:
    out = tb.stack().rename("brier").reset_index()
    out.columns = ["task_id", "budget", "brier"]
    out[["protocol", "role", "model"]] = protocol, role, model
    return out


def build(cache: Path) -> pd.DataFrame:
    frames = []
    runs = cache / "runs"

    # Single-pair: the replay grid, random acquisition.
    single = {}
    for role in ROLES:
        d = pd.read_parquet(RESULTS / f"budget-scores-{PANEL}-{role}.parquet")
        d = d[d.policy == "random_prefix"]
        single[role] = d
        for m, g in d.groupby("model"):
            tb = g.groupby(["task_id", "requested_budget"]).brier.mean().unstack()
            frames.append(_long(tb, "single-pair", role, m))

    # Single-pair: ANP + beta combinations, recomputed from saved predictions.
    grid = np.round(np.linspace(0, 1, 11), 1)
    pred = {}
    for role in ROLES:
        beta = posthoc._predictions(runs, role, "beta")
        f = beta.merge(posthoc._shift(runs, role, "beta"), on=posthoc.KEY[:3])
        f["aipw"] = (f.probability + f["shift"]).clip(1e-4, 1 - 1e-4)
        frames.append(_long(posthoc._task_budget(f, "aipw"), "single-pair", role, "beta-aipw"))
        a = (posthoc._predictions(runs, role, "w64-filtered").rename(columns={"probability": "anp"})
             .merge(beta[posthoc.KEY + ["probability"]].rename(columns={"probability": "beta"}),
                    on=posthoc.KEY, validate="one_to_one")
             .merge(posthoc._shift(runs, role, "w64-filtered"), on=posthoc.KEY[:3]))
        a["gate"] = np.where(a.requested_budget < 2, a.beta, a.anp)
        a["aipw"] = (a.anp + a["shift"]).clip(1e-4, 1 - 1e-4)
        pred[role] = a
    V = pred["validation"]
    w_one = min(grid, key=lambda w: posthoc._alc(posthoc._task_budget(V.assign(p=w * V.anp + (1 - w) * V.beta), "p")).mean())
    w_b = {b: min(grid, key=lambda w: ((w * V.anp + (1 - w) * V.beta - V.label)[V.requested_budget == b] ** 2)
                  .groupby(V.task_id).mean().mean()) for b in BUDGETS}
    for role, a in pred.items():
        a["blend"] = w_one * a.anp + (1 - w_one) * a.beta
        wb = a.requested_budget.map(w_b)
        a["blend-budget"] = wb * a.anp + (1 - wb) * a.beta
        for col in ("gate", "aipw", "blend", "blend-budget"):
            frames.append(_long(posthoc._task_budget(a, col), "single-pair", role, col))

    # Single-pair: hierarchical beta arms (Modal run, same orders as the replay).
    h = pd.read_parquet(cache / "all-arms-task-budget.parquet")
    h = h[h.arm != "jeffreys"].rename(columns={"arm": "model"})
    h["protocol"] = "single-pair"
    frames.append(h[["task_id", "budget", "brier", "protocol", "role", "model"]])

    # Multi-pair: pooled models from the Modal run; the ANP placed on the same draws.
    c = pd.read_parquet(cache / "curves.parquet")
    for role in ROLES:
        cr = c[c.role == role]
        s = single[role].set_index(["task_id", "order_seed", "requested_budget", "model"]).brier
        keys = cr[cr.model == "jeffreys"][["draw", "task_id", "order_seed", "budget"]]
        check = s.xs("beta", level="model").reindex(pd.MultiIndex.from_frame(keys[["task_id", "order_seed", "budget"]]))
        gap = np.abs(check.to_numpy() - cr[cr.model == "jeffreys"].brier.to_numpy()).max()
        assert gap < 1e-9, f"multi-pair Jeffreys does not match single-pair beta ({gap})"
        anp = keys.assign(model="w64-filtered", brier=s.xs("w64-filtered", level="model").reindex(
            pd.MultiIndex.from_frame(keys[["task_id", "order_seed", "budget"]])).to_numpy())
        both = pd.concat([cr[["draw", "task_id", "order_seed", "budget", "model", "brier"]], anp])
        # A task can appear in many draws; average its occurrences first.
        tb = both.groupby(["model", "task_id", "budget"]).brier.mean().reset_index()
        tb[["protocol", "role"]] = "multi-pair", role
        frames.append(tb)

    out = pd.concat(frames, ignore_index=True)[["protocol", "role", "model", "task_id", "budget", "brier"]]
    out["budget"] = out["budget"].astype(int)
    out.to_parquet(CACHE, index=False)
    return out


def load() -> pd.DataFrame | None:
    return pd.read_parquet(CACHE) if CACHE.exists() else None


def _boot(x: np.ndarray, rng, n: int = 2000) -> tuple[float, float, float]:
    boots = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _matrix(t: pd.DataFrame, protocol: str, role: str, model: str) -> pd.DataFrame:
    g = t[(t.protocol == protocol) & (t.role == role) & (t.model == model)]
    return g.pivot(index="task_id", columns="budget", values="brier")[BUDGETS]


def _alc(m: pd.DataFrame) -> pd.Series:
    return sum(w * m[b] for b, w in WEIGHTS.items())


# ------------------------------------------------------------------ pages ---
def header(fig, title: str) -> None:
    fig.text(0.018, 0.985, title, fontsize=9, fontweight="bold", va="top", ha="left")


def caption(fig, label: str, text: str, width: int = 116) -> None:
    """Self-contained caption under the figure, bottom-aligned to the page."""
    body = textwrap.fill(f"{label} {text}", width=width)
    fig.text(0.018, 0.012, body, fontsize=6.1, va="bottom", ha="left", linespacing=1.35)


def shared_legend(fig, axes, y: float, ncol: int = 4) -> None:
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", ncol=ncol, fontsize=5.9, frameon=False,
               bbox_to_anchor=(0.5, y))


def main_page(pdf, t: pd.DataFrame, protocol: str):
    rng = np.random.default_rng(0)
    x = np.arange(len(BUDGETS))
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.55), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.40, wspace=0.08)
    header(fig, "The ANP loses to the beta-binomial on unseen benchmarks, at 0-1 labels" if protocol == "single-pair"
           else "Pooling other subjects' labels is the only thing that beats the beta-binomial")
    alcs = {}
    for ax, role in zip(axes, ROLES):
        for key, label, colour in MAIN[protocol]:
            m = _matrix(t, protocol, role, key)
            if m.empty:
                continue
            stats = np.array([_boot(m[b].to_numpy(), rng) for b in BUDGETS])
            ax.fill_between(x, stats[:, 1], stats[:, 2], color=colour, alpha=0.15, lw=0)
            ax.plot(x, stats[:, 0], color=colour, marker="o", ms=2.6, lw=1.1, label=label)
            alcs.setdefault(label, {})[role] = _alc(m).mean()
        n = t[(t.protocol == protocol) & (t.role == role)].task_id.nunique()
        ax.set_title(f"{role} ({n} tasks)", fontsize=7, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(BUDGETS)
        ax.set_xlabel("labels acquired", fontsize=6.5)
        ax.tick_params(labelsize=6)
        style(ax, grid_axis="both")
    axes[0].set_ylabel("Brier (lower is better)", fontsize=6.5)
    shared_legend(fig, axes, 0.25)
    alc_text = "; ".join(f"{k} {v['validation']:.4f} / {v['test']:.4f}" for k, v in alcs.items())
    if protocol == "single-pair":
        body = ("Mean Brier on each task's held-out evaluation items after 0-31 labels, each subject-benchmark "
                "task replayed alone on its own labels with random acquisition (validation: 3 candidate orders, "
                "test: 10; benchmarks never seen in training). Bands: 95% bootstrap over tasks. The ANP's whole "
                "deficit sits at 0 and 1 labels; from 3 labels on it matches the beta-binomial. The constant ignores labels "
                f"and sets the floor. Brier-ALC (val / test): {alc_text}. Pooled Rasch needs other pairs' labels "
                "and is shown on the multi-pair page.")
    else:
        body = ("Same models on CodaBench-like draws: one or two subjects per benchmark, labels acquired in the "
                "replay's frozen orders, and every pair's labels available to the predictor (60 validation and "
                "150 test draws). Models that ignore other pairs are placed on the same draws from their "
                "single-pair scores. Pooled Rasch fits benchmark rate, subject ability and item difficulty from "
                "all pairs' labels and is the only model below the beta-binomial on test at every budget. Bands: 95% "
                f"bootstrap over tasks. Brier-ALC (val / test): {alc_text}.")
    caption(fig, f"Figure ({protocol}).", body)
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def ablation_table_page(pdf, t: pd.DataFrame):
    rng = np.random.default_rng(1)
    rows, bold = [], set()
    for title, protocol, models in ABLATIONS:
        rows.append([TABLE_LABEL[title], "", "", "", ""])
        bold.add(len(rows) - 1)
        base = {r: _matrix(t, protocol, r, BASELINE[protocol]) for r in ROLES}
        for key, label in models:
            cells = []
            for role in ROLES:
                m = _matrix(t, protocol, role, key)
                cells.append(f"{_alc(m).mean():.4f}" if not m.empty else "-")
            m = _matrix(t, protocol, "test", key)
            d = (_alc(m) - _alc(base["test"]).reindex(m.index)).dropna().to_numpy()
            mean, lo, hi = _boot(d, rng)
            star = "*" if lo > 0 or hi < 0 else ""
            rows.append([f"  {label}", protocol, cells[0], cells[1], f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}"])
    rows.append(["Not replayed", "", "", "", ""])
    bold.add(len(rows) - 1)
    for label, why in NOT_REPLAYED:
        rows.append([f"  {label}", "-", "-", "-", why])

    height = 1.55 + 0.15 * len(rows)
    fig = plt.figure(figsize=(TEXTWIDTH, height))
    header(fig, "Only pooling other subjects' labels beats the beta-binomial on unseen benchmarks")
    ax = fig.add_axes((0.015, 1.0 / height, 0.97, 1 - 1.38 / height))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["model", "protocol", "val ALC", "test ALC", "test minus beta-binomial [95% CI]"],
                     colWidths=[0.36, 0.11, 0.1, 0.1, 0.33], bbox=[0, 0, 1, 1], cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(5.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.4)
        if c == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02
        if r == 0 or (r - 1) in bold:
            cell.set_text_props(weight="bold")
        if r > 0 and (r - 1) in bold:
            cell.set_facecolor("#F4F4F4")
    caption(fig, "Table.", "Macro Brier-ALC (lower is better) on validation and on the untouched test benchmarks "
            "(algotune, cmmlu, editbench), random acquisition. The last column is the paired test difference to "
            "the beta-binomial (Jeffreys prior) under the same protocol, with a 95% bootstrap interval over tasks; * marks intervals "
            "that exclude 0. Single-pair beta-binomial: 0.1717 / 0.2494 (val / test); multi-pair: 0.1685 / 0.2497. "
            "Validation selected the blend weights, the metadata L2 and the pooled-Rasch priors. The "
            "random-split ANPs trained on editbench. Rows under 'Not replayed' were trained but have no "
            "replay result.")
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def ablation_curves_page(pdf, t: pd.DataFrame, title: str, protocol: str, models: list):
    rng = np.random.default_rng(2)
    x = np.arange(len(BUDGETS))
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.65), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.42, wspace=0.08)
    header(fig, title)
    for ax, role in zip(axes, ROLES):
        base = _matrix(t, protocol, role, BASELINE[protocol])
        for i, (key, label) in enumerate(models):
            m = _matrix(t, protocol, role, key)
            if m.empty:
                continue
            d = (m - base.reindex(m.index)).dropna()
            stats = np.array([_boot(d[b].to_numpy(), rng) for b in BUDGETS])
            colour = CYCLE[i % len(CYCLE)]
            ax.fill_between(x, stats[:, 1], stats[:, 2], color=colour, alpha=0.13, lw=0)
            ax.plot(x, stats[:, 0], color=colour, marker="o", ms=2.4, lw=1.0, label=label)
        ax.axhline(0, color=INK, lw=0.6, ls=(0, (4, 2)))
        ax.set_title(role, fontsize=7, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(BUDGETS)
        ax.set_xlabel("labels acquired", fontsize=6.5)
        ax.tick_params(labelsize=6)
        style(ax, grid_axis="both")
    axes[0].set_ylabel("Brier minus beta-binomial (negative = better)", fontsize=6.5)
    shared_legend(fig, axes, 0.265, ncol=3)
    caption(fig, f"Ablation ({protocol}).",
            f"Paired Brier difference to the beta-binomial (Jeffreys prior) at each label budget, validation (left) and test (right); "
            f"bands: 95% bootstrap over tasks; the dashed line is the beta-binomial. {CAPTION[title]}")
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def pages(pdf, t: pd.DataFrame) -> list:
    out = [("main-1-single-pair", main_page(pdf, t, "single-pair")),
           ("main-2-multi-pair", main_page(pdf, t, "multi-pair")),
           ("main-3-ablation-table", ablation_table_page(pdf, t))]
    for i, (title, protocol, models) in enumerate(ABLATIONS, start=4):
        out.append((f"main-{i}-ablation", ablation_curves_page(pdf, t, title, protocol, models)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path.home() / "aims-replay-cache")
    parser.add_argument("--png-dir", type=Path, default=None)
    args = parser.parse_args()
    t = build(args.cache) if args.build else load()
    if t is None:
        parser.error(f"no cache at {CACHE}; run with --build")
    print(t.groupby(["protocol", "role"]).model.nunique().to_dict())
    if args.png_dir:
        args.png_dir.mkdir(parents=True, exist_ok=True)
        with PdfPages(args.png_dir / "main.pdf") as pdf:
            for name, fig in pages(pdf, t):
                fig.savefig(args.png_dir / f"{name}.png", dpi=120, bbox_inches=exact_page(fig))
    return 0


if __name__ == "__main__":
    sys.exit(main())
