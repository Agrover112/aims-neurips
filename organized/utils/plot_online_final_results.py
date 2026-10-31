#!/usr/bin/env python3
"""Final online-replay results: ALC tables and Brier curves, validation and test.

Reads the per-task, per-seed, per-budget scores that
``organized/testing/online/run_sweep.py --collect --role <role>`` saves under
``organized/testing/online/results/``, so the PDF is regenerated from the
volume's runs rather than from hand-copied numbers.

Each split gets three pages:

1. macro-task Brier ALC for every model x acquisition policy, and the paired
   difference from beta-binomial under the same policy with a 95% CI;
2. Brier against the number of labels acquired, one panel per policy;
3. the same curves as a paired difference from beta-binomial, which is where
   per-budget effects are actually visible -- the raw curves overlap almost
   everywhere because subject difficulty dominates the absolute level.

Test comes first because it is the split to cite: no model trained on or was
selected with its benchmarks. Validation benchmarks were used to pick every
ANP checkpoint, which flatters the ANPs there.

Every page is exactly TEXTWIDTH wide. Tables are drawn into a fixed bounding
box so added rows shrink the row height instead of spilling into the next
table, and titles are short enough not to widen the page.

    python organized/utils/plot_online_final_results.py
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
from plotstyle import CYCLE, INK, SUBTLE, TEXTWIDTH, style  # noqa: E402
import plot_online_posthoc as posthoc  # noqa: E402
import plot_online_main as mainpages  # noqa: E402

PANEL = "benchmark-holdout-online-v1-s20261018-n15-m80-cap100000"
SCORES = ROOT / "organized" / "testing" / "online" / "results"
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = dict(zip(BUDGETS, (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)))

# Closed-form baselines (dotted), holdout-trained ANPs (solid), then the
# random-subject-split pair, which trained on some evaluation benchmarks
# (dashed).
MODELS = [
    ("const-mean", "Constant (train mean)", ":"),
    ("emp-mean", "Empirical mean", ":"),
    ("beta", "Beta-binomial", "-"),
    ("w64-filtered", "ANP w64", "-"),
    ("w64-dense", "ANP w64 dense-prefix", "-"),
    ("w64-sam", "ANP w64 SAM", "-"),
    ("w64-contam", "ANP w64, random split*", "--"),
    ("w128", "ANP w128, random split*", "--"),
]
RANDOM_SPLIT = {"w64-contam", "w128"}
POLICIES = [
    ("random_prefix", "random"),
    ("uncertainty", "uncertainty"),
    ("confidence_extremes", "conf. extremes"),
    ("bernoulli_variance", "Bernoulli var."),
    ("expected_brier_reduction", "exp. Brier red."),
]
ROLE_TEXT = {
    "test": ("Test",
             "algotune, cmmlu, editbench: no model trained on or was selected "
             "with these. 45 tasks x 10 order seeds."),
    "validation": ("Validation",
                   "ai2d_test, bird_sql, financebench. ANP checkpoints were selected "
                   "here, which flatters them. 82 tasks x 3 seeds."),
}
FOOTNOTE = ("* random-subject split: trained on ai2d_test, bird_sql (validation) and "
            "editbench (test). Other ANPs used a benchmark holdout.")
COLORS = {key: CYCLE[i % len(CYCLE)] for i, (key, _, _) in enumerate(MODELS)}


def task_budget_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Tasks x budgets, averaging order seeds within a task first.

    Each task then counts once, which is the macro-task aggregation the ALC
    uses; a task with more seeds or items does not get more weight.
    """
    return (frame.groupby(["task_id", "requested_budget"])["brier"].mean()
            .unstack().reindex(columns=BUDGETS))


def alc(matrix: pd.DataFrame) -> pd.Series:
    return sum(WEIGHTS[b] * matrix[b] for b in BUDGETS)


def paired(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    """Mean of the per-task difference and its 95% half-width."""
    d = (a - b).dropna()
    if len(d) < 3:
        return float("nan"), float("nan"), len(d)
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d))), len(d)


def load(role: str) -> pd.DataFrame | None:
    path = SCORES / f"budget-scores-{PANEL}-{role}.parquet"
    if not path.exists() and role == "validation":
        path = SCORES / f"budget-scores-{PANEL}.parquet"  # pre role-split collect
    return pd.read_parquet(path) if path.exists() else None


def matrices(frame: pd.DataFrame) -> dict:
    return {key: task_budget_matrix(group)
            for key, group in frame.groupby(["model", "policy"])}


def exact_page(fig) -> Bbox:
    """The figure's own rectangle, for ``savefig(bbox_inches=...)``.

    ``bbox_inches=None`` does not mean "no cropping": it falls back to
    rcParams["savefig.bbox"], which plotstyle sets to "tight", and a tight box
    grows to swallow any text that overflows. A page then silently becomes
    wider than TEXTWIDTH instead of showing the overflow. Pinning the box makes
    every page exactly the declared size, so an overflow is visible and fixed.
    """
    return Bbox.from_bounds(0, 0, *fig.get_size_inches())


def page_header(fig, title: str, subtitle: str) -> None:
    fig.text(0.018, 0.985, title, fontsize=9, fontweight="bold", va="top", ha="left")
    fig.text(0.018, 0.945, subtitle, fontsize=6.4, color=SUBTLE, va="top", ha="left")


def draw_table(ax, header: list[str], rows: list[list[str]], bold: set,
               caption: str, fontsize: float = 5.9, label_width: float = 0.27) -> None:
    """A table filling ``ax`` exactly; labels are the first column, not rowLabels.

    rowLabels are drawn outside the cell box and are what spilled past the left
    edge; carrying them as an ordinary column keeps the whole table inside its
    axes, so it can be laid out like any other panel.
    """
    ax.axis("off")
    widths = [label_width] + [(1 - label_width) / (len(header) - 1)] * (len(header) - 1)
    table = ax.table(cellText=rows, colLabels=header, colWidths=widths,
                     cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.4)
        if c == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.03
        if r == 0 or (r - 1, c) in bold:
            cell.set_text_props(weight="bold")
    ax.set_title(caption, fontsize=6.6, loc="left", color=INK, pad=3)


def table_page(pdf, role: str, mats: dict):
    title, subtitle = ROLE_TEXT[role]
    fig = plt.figure(figsize=(TEXTWIDTH, 6.4))
    page_header(fig, f"{title}: Brier ALC by model and acquisition policy", subtitle)
    header = ["model"] + [label for _, label in POLICIES]

    rows, bold = [], set()
    for key, label, _ in MODELS:
        line = [label]
        for policy, _ in POLICIES:
            m = mats.get((key, policy))
            line.append(f"{alc(m).mean():.4f}" if m is not None else "pending")
        rows.append(line)
    for c, (policy, _) in enumerate(POLICIES, start=1):
        scored = [(alc(mats[(k, policy)]).mean(), r) for r, (k, _, _) in enumerate(MODELS)
                  if k not in RANDOM_SPLIT and (k, policy) in mats]
        if scored:
            bold.add((min(scored)[1], c))
    draw_table(fig.add_axes([0.01, 0.52, 0.98, 0.36]), header, rows, bold,
               "Macro-task Brier ALC, lower is better. Bold: lowest among rows that "
               "never trained on these benchmarks.")

    others = [m for m in MODELS if m[0] != "beta"]
    rows, bold = [], set()
    for r, (key, label, _) in enumerate(others):
        line = [label]
        for c, (policy, _) in enumerate(POLICIES, start=1):
            m, ref = mats.get((key, policy)), mats.get(("beta", policy))
            if m is None or ref is None:
                line.append("pending")
                continue
            mean, half, _ = paired(alc(m), alc(ref))
            line.append(f"{mean:+.4f} ±{half:.4f}")
            if abs(mean) > half:
                bold.add((r, c))
        rows.append(line)
    draw_table(fig.add_axes([0.01, 0.07, 0.98, 0.34]), header, rows, bold,
               "ALC minus beta-binomial under the same policy, paired over tasks: "
               "mean ± 95% CI. Bold: CI excludes 0.", fontsize=5.2, label_width=0.23)
    fig.text(0.018, 0.012, FOOTNOTE, fontsize=5.6, color=SUBTLE)
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def series(mats: dict, key: str, policy: str, difference: bool):
    """(mean, band) per budget for one model and policy, or None if not run.

    Raw curves use ± 1 SE over tasks; difference curves use the paired 95% CI
    against beta-binomial under the same policy.
    """
    m = mats.get((key, policy))
    ref = mats.get(("beta", policy))
    if m is None or (difference and (key == "beta" or ref is None)):
        return None
    if difference:
        stats = [paired(m[b], ref[b]) for b in BUDGETS]
        return np.array([s[0] for s in stats]), np.array([s[1] for s in stats])
    return m.mean().to_numpy(), (m.std(ddof=1) / np.sqrt(m.count())).to_numpy()


def y_limits(mats: dict, difference: bool) -> tuple[float, float]:
    """One y-range for every policy of a split, so separate pages compare."""
    values = []
    for policy, _ in POLICIES:
        for key, _, _ in MODELS:
            s = series(mats, key, policy, difference)
            if s is not None:
                values += list(s[0] + s[1]) + list(s[0] - s[1])
    values = [v for v in values if np.isfinite(v)]
    top = max(values) * 1.06
    if not difference:
        return 0.0, top  # Brier's floor is 0; keep it rather than zoom in
    span = top - min(values)
    return min(values) - 0.04 * span, top


def single_plot(role: str, mats: dict, policy: str, difference: bool):
    """One policy's curves on one page, with no title or footnote.

    For the paper: LaTeX supplies the caption, so anything printed here would
    duplicate it. Full text width, legend beside the axes.
    """
    fig = plt.figure(figsize=(TEXTWIDTH, 2.5))
    # Wide enough for a rotated label beside negative tick labels ("−0.05");
    # 0.09 clipped the label's first letters at the page edge.
    ax = fig.add_axes([0.125, 0.17, 0.545, 0.79])
    x = np.arange(len(BUDGETS))
    for key, label, ls in MODELS:
        s = series(mats, key, policy, difference)
        if s is None:
            continue
        mean, band = s
        ax.plot(x, mean, ls, marker="o", ms=2.4, color=COLORS[key], label=label)
        ax.fill_between(x, mean - band, mean + band, color=COLORS[key], alpha=0.15, lw=0)
    if difference:
        ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(x, [str(b) for b in BUDGETS])
    ax.set_ylim(*y_limits(mats, difference))
    ax.set_xlabel("labels acquired")
    ax.set_ylabel("Brier − beta" if difference else "Brier score")
    style(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.2, frameon=False)
    return fig


def curve_page(pdf, role: str, mats: dict, difference: bool):
    title, subtitle = ROLE_TEXT[role]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 4.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.84, bottom=0.13,
                        wspace=0.12, hspace=0.34)
    what = ("Brier minus beta-binomial (paired, 95% CI)" if difference
            else "Brier by labels acquired (mean ± 1 SE)")
    page_header(fig, f"{title}: {what}", subtitle)
    x = np.arange(len(BUDGETS))
    for ax, (policy, plabel) in zip(axes.flat, POLICIES):
        for key, label, ls in MODELS:
            s = series(mats, key, policy, difference)
            if s is None:
                continue
            mean, band = s
            ax.plot(x, mean, ls, marker="o", ms=2.2, color=COLORS[key], label=label)
            ax.fill_between(x, mean - band, mean + band, color=COLORS[key],
                            alpha=0.15, lw=0)
        if difference:
            ax.axhline(0, color=INK, lw=0.6)
        ax.set_title(plabel, fontsize=7, pad=3)
        ax.set_xticks(x, [str(b) for b in BUDGETS])
        style(ax)
    axes.flat[0].set_ylim(*y_limits(mats, difference))
    for ax in axes[1]:
        ax.set_xlabel("labels acquired")
    for ax in axes[:, 0]:
        ax.set_ylabel("Brier − beta" if difference else "Brier")
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", fontsize=6, frameon=False)
    fig.text(0.018, 0.012, FOOTNOTE, fontsize=5.6, color=SUBTLE)
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "online-final-results.pdf")
    parser.add_argument("--plots-out", type=Path,
                        default=ROOT / "docs" / "online-final-plots.pdf",
                        help="one plot per page, no titles, for the paper")
    parser.add_argument("--png-dir", type=Path, default=None,
                        help="also write each page as a PNG, for checking the render")
    args = parser.parse_args()

    pages = []
    plot_pages, page_map = [], []
    with PdfPages(args.out) as pdf, PdfPages(args.plots_out) as plots:
        # Main-model curves and the ablation table lead the report; built by
        # plot_online_main.py --build, skipped until its cache exists.
        main_table = mainpages.load()
        if main_table is not None:
            pages += mainpages.pages(pdf, main_table)
        for role in ("test", "validation"):
            frame = load(role)
            if frame is None:
                print(f"no saved scores for {role}; run run_sweep.py --collect --role {role}",
                      file=sys.stderr)
                continue
            mats = matrices(frame)
            done = sum((k, p) in mats for k, _, _ in MODELS for p, _ in POLICIES)
            print(f"{role}: {done} of {len(MODELS) * len(POLICIES)} cells")
            pages.append((f"{role}-1-tables", table_page(pdf, role, mats)))
            pages.append((f"{role}-2-curves", curve_page(pdf, role, mats, difference=False)))
            pages.append((f"{role}-3-vs-beta", curve_page(pdf, role, mats, difference=True)))
            for difference in (False, True):
                for policy, plabel in POLICIES:
                    fig = single_plot(role, mats, policy, difference)
                    plots.savefig(fig, bbox_inches=exact_page(fig))
                    kind = "Brier minus beta-binomial" if difference else "Brier"
                    page_map.append(f"{len(page_map) + 1}: {role}, {kind}, {plabel}")
                    plot_pages.append((f"plot-{len(page_map):02d}", fig))
        # Post-hoc recombinations of the saved replays; built by
        # plot_online_posthoc.py --build, skipped until its cache exists.
        tables = posthoc.load()
        if tables is not None:
            pages += posthoc.pages(pdf, tables)
    if args.png_dir:
        args.png_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in pages + plot_pages:
            # The PNG is the page exactly as printed, so anything cut off here
            # is cut off in the PDF too.
            fig.savefig(args.png_dir / f"{name}.png", dpi=120,
                        bbox_inches=exact_page(fig))
    print(f"wrote {args.out} ({len(pages)} pages)")
    index = args.plots_out.with_suffix(".pages.txt")
    index.write_text("\\includegraphics[page=N]{" + args.plots_out.name + "}\n\n"
                     + "\n".join(page_map) + "\n")
    print(f"wrote {args.plots_out} ({len(page_map)} pages); page map {index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
