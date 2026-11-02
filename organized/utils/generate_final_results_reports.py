#!/usr/bin/env python3
"""Build protocol-separated final-results PDFs from committed local evidence.

This deliberately does not merge scores from incompatible experiments.  It
creates four reports in ``docs/``:

* ``protocol-final-results.pdf``: the faithful sequential evaluation diagram;
* ``data-analysis-final-results.pdf``: split shift, item-difficulty diagnostics,
  and the existing response/semantic-graph audit;
* ``offline-final-results.pdf``: the older three-seed BeaverTails comparisons,
  explicitly marked as historical offline evidence;
* ``training-final-results.pdf``: local ANP checkpoint-selection evidence and
  an inventory of what training histories are actually available.

The benchmark-held-out online test is intentionally not copied here: it is
already rendered by ``plot_online_final_results.py`` as
``docs/online-final-results.pdf``.

Usage:
    .venv/bin/python organized/utils/generate_final_results_reports.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from pypdf import PdfReader, PdfWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import (  # noqa: E402
    CYCLE,
    HIGH_CONTRAST,
    INK,
    MUTED,
    SUBTLE,
    TEXTWIDTH,
    header,
    style,
)

DOCS = ROOT / "docs"
BUDGETS = [0, 1, 3, 7, 15, 31]


def save_page(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig)
    plt.close(fig)


def append_pdf(destination: Path, additions: list[Path]) -> None:
    """Append existing evidence pages without rasterising vector figures."""
    writer = PdfWriter()
    for source in [destination, *additions]:
        reader = PdfReader(str(source))
        for page in reader.pages:
            writer.add_page(page)
    with destination.open("wb") as handle:
        writer.write(handle)


def table(ax, rows: list[list[str]], columns: list[str], widths=None, fontsize=6.3) -> None:
    ax.axis("off")
    artist = ax.table(cellText=rows, colLabels=columns, loc="upper left",
                      cellLoc="center", colWidths=widths)
    artist.auto_set_font_size(False)
    artist.set_fontsize(fontsize)
    artist.scale(1, 1.22)
    for (row, _), cell in artist.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F2F2F2")


def protocol_report(output: Path) -> None:
    with PdfPages(output) as pdf:
        fig = plt.figure(figsize=(TEXTWIDTH, 3.35))
        header(fig, "Faithful online evaluation under benchmark holdout",
               "A policy sees only the current candidate and previously acquired labels. "
               "Future candidates and the scored pool remain hidden.")
        ax = fig.add_axes([0.06, 0.24, 0.88, 0.58])
        ax.axis("off")
        boxes = [
            (0.00, 0.52, 0.19, 0.30, "Train benchmarks", "fit frozen predictor\nand policy knobs"),
            (0.27, 0.52, 0.19, 0.30, "Unseen benchmark", "one subject × benchmark\ntask at a time"),
            (0.54, 0.52, 0.19, 0.30, "Current candidate", "item text / embedding\n+ past accepted labels"),
            (0.81, 0.52, 0.19, 0.30, "Decision", "accept or reject\nbefore next item"),
            (0.54, 0.05, 0.19, 0.26, "If accepted", "reveal its binary label\nupdate context"),
            (0.81, 0.05, 0.19, 0.26, "Score", "predict disjoint held-out\nitems at B=0…31"),
        ]
        colours = [MUTED["grey"], MUTED["sand"], MUTED["cyan"], MUTED["rose"],
                   MUTED["teal"], MUTED["indigo"]]
        for (x, y, w, h, title, body), colour in zip(boxes, colours, strict=True):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                        facecolor=colour, edgecolor=INK, lw=0.55))
            ax.text(x + w / 2, y + h * .67, title, ha="center", va="center",
                    fontsize=7.3, weight="bold")
            ax.text(x + w / 2, y + h * .35, body, ha="center", va="center", fontsize=6.4)
        for start, end in [((.19, .67), (.27, .67)), ((.46, .67), (.54, .67)),
                           ((.73, .67), (.81, .67)), ((.905, .52), (.905, .31)),
                           ((.73, .18), (.81, .18))]:
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                         lw=.8, color=INK))
        ax.text(.065, .38, "Training ends here", color=SUBTLE, fontsize=6.2, ha="center")
        ax.text(.365, .38, "Evaluation starts here", color=SUBTLE, fontsize=6.2, ha="center")
        ax.text(.01, -.13,
                "Invalid shortcut: selecting the global best item from the entire remaining pool. "
                "That requires future candidates that the streaming policy has not received.",
                fontsize=6.4, color=MUTED["wine"])
        fig.text(.02, .04,
                 "Source: docs/benchmark-holdout-online-replay.md. The online final-results PDF "
                 "contains the matched validation and untouched-test scores.",
                 fontsize=6.2, color=SUBTLE)
        save_page(pdf, fig)


def _exact_save(pdf: PdfPages, fig: plt.Figure) -> None:
    """Save at the figure's own size: a "tight" box grows around overflowing
    text and silently widens the page past TEXTWIDTH."""
    pdf.savefig(fig, bbox_inches=Bbox.from_bounds(0, 0, *fig.get_size_inches()))
    plt.close(fig)


def _caption(fig: plt.Figure, text: str, width: int = 116) -> None:
    import textwrap

    fig.text(0.018, 0.012, textwrap.fill(text, width), fontsize=6.1, va="bottom", linespacing=1.35)


def _booktabs(ax, columns: list[str], rows: list[list[str]], xs: list[float], right: set[int],
              widths: list[float], fontsize: float = 6.4) -> None:
    """Rules-only table filling ``ax``: top, mid and bottom rules, no grid."""
    ax.axis("off")
    step = 1.0 / (len(rows) + 1.4)
    y = 1.0

    def put(cells, weight="normal"):
        for c, (x, text) in enumerate(zip(xs, cells)):
            ax.text(x + (widths[c] if c in right else 0), y - step / 2, text, transform=ax.transAxes,
                    ha="right" if c in right else "left", va="center", fontsize=fontsize, fontweight=weight)

    def rule(lw):
        ax.plot([0, 1], [y, y], color=INK, lw=lw, transform=ax.transAxes, clip_on=False)

    rule(1.0)
    put(columns, "bold")
    y -= step
    rule(0.5)
    for cells in rows:
        put(cells)
        y -= step
    rule(1.0)


def data_split_page(pdf: PdfPages) -> None:
    holdout = pd.read_csv(DOCS / "benchmark-holdout-split-distribution-summary.csv")
    random = pd.read_csv(DOCS / "non-benchmark-holdout-split-distribution-summary.csv")
    fig = plt.figure(figsize=(TEXTWIDTH, 5.0))
    header(fig, "A random-subject split is not a test of benchmark transfer", "")
    roles = ["train", "validation", "test"]
    x = np.arange(len(roles))
    configs = [(holdout, "Benchmark holdout", MUTED["rose"]), (random, "Random-subject split", MUTED["teal"])]
    axes = [fig.add_axes((0.105 + i * 0.31, 0.60, 0.235, 0.29)) for i in range(3)]
    for frame, label, colour in configs:
        f = frame.set_index("split").loc[roles]
        cosine_col = next(col for col in frame.columns if col.endswith("centroid_cosine_to_train"))
        for ax, col in zip(axes, ["mean_label", "item_pass_rate_std", cosine_col]):
            ax.plot(x, f[col], "o-", color=colour, label=label, ms=3, lw=1.0)
    for ax, title, ylabel in zip(axes, ["(a) Mean correct response", "(b) Item pass-rate spread",
                                        "(c) Shift in item text"],
                                 ["mean response", "SD across items", "cosine to train"]):
        ax.set_title(title, loc="left", weight="bold", fontsize=6.8)
        ax.set_xticks(x, ["train", "val.", "test"])
        ax.set_ylabel(ylabel, fontsize=6.2)
        ax.tick_params(labelsize=5.8)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", ncol=2, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, 0.505))
    fig.text(0.02, 0.455, "(d) Retained responses after filtering", fontsize=6.8, weight="bold")
    rows = []
    for name, frame in [("Benchmark holdout", holdout), ("Random-subject split", random)]:
        for role in roles:
            r = frame.set_index("split").loc[role]
            rows.append([name if role == "train" else "", role, f"{int(r['responses']):,}",
                         f"{int(r['tasks']):,}", f"{int(r['items']):,}"])
    ax = fig.add_axes((0.02, 0.165, 0.96, 0.27))
    _booktabs(ax, ["Split", "Role", "Responses", "Tasks", "Items"], rows,
              xs=[0.0, 0.30, 0.45, 0.66, 0.82], right={2, 3, 4}, widths=[0.3, 0.15, 0.17, 0.12, 0.14])
    _caption(fig, "Figure. The same filtered responses split two ways. The benchmark holdout keeps whole "
             "benchmarks out of training, so validation and test items are new text (c) with a different mean "
             "response (a). The random-subject split holds out subjects but keeps every benchmark in training, so "
             "its held-out items match training exactly (cosine 1.0): it measures new subjects, not new "
             "benchmarks, and is not a transfer baseline. The two splits use different frozen embedding models, "
             "so compare cosines within a split only. Sources: docs/benchmark-holdout-split-distribution-summary."
             "csv and docs/non-benchmark-holdout-split-distribution-summary.csv.")
    _exact_save(pdf, fig)


def data_split_schematic_page(pdf: PdfPages) -> None:
    """Schematic: how each split cuts the subject x item response matrix.

    Drawn to the data's real structure rather than its scale: items grouped
    into benchmarks of unequal size, and each subject answering one benchmark
    (occasionally two), so the observed cells form blocks.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    rng = np.random.default_rng(3)
    widths = [14, 9, 17, 7, 12, 8, 10, 6, 9]                     # items per benchmark
    roles = ["train"] * 5 + ["validation"] * 2 + ["test"] * 2    # benchmark-holdout role
    subjects_per = [9, 6, 11, 5, 8, 5, 6, 4, 6]
    edges = np.cumsum([0] + widths)
    rows = []
    for b, n in enumerate(subjects_per):
        for _ in range(n):
            row = np.zeros(edges[-1], dtype=bool)
            row[edges[b]:edges[b + 1]] = rng.random(widths[b]) < 0.8
            if rng.random() < 0.15:                                # a few subjects answer a second benchmark
                o = int(rng.integers(len(widths)))
                row[edges[o]:edges[o + 1]] |= rng.random(widths[o]) < 0.5
            rows.append((b, row))
    observed = np.array([r for _, r in rows])
    home = np.array([b for b, _ in rows])
    code = {"train": 1, "validation": 2, "test": 3}

    holdout = np.zeros(observed.shape, dtype=int)
    for b, role in enumerate(roles):
        holdout[:, edges[b]:edges[b + 1]] = code[role]
    random_split = np.zeros(observed.shape, dtype=int)
    row_role = rng.choice([1, 2, 3], size=len(rows), p=[0.7, 0.15, 0.15])
    for b in range(len(widths)):                                   # every benchmark gets some of each role
        idx = np.flatnonzero(home == b)
        row_role[idx[:max(1, len(idx) // 7)]] = 2
        row_role[idx[-max(1, len(idx) // 7):]] = 3
    random_split[:] = row_role[:, None]
    holdout[~observed] = 0
    random_split[~observed] = 0

    cmap = ListedColormap(["#FFFFFF", "#BFC4DD", MUTED["teal"], MUTED["rose"]])
    fig = plt.figure(figsize=(TEXTWIDTH, 5.6))
    fig.text(0.018, 0.985, "The two splits cut the response matrix in different directions", fontsize=9,
             fontweight="bold", va="top")
    panels = [(random_split, "(a) Random-subject split: hold out subjects"),
              (holdout, "(b) Benchmark holdout: hold out benchmarks")]
    for i, (grid, title) in enumerate(panels):
        ax = fig.add_axes((0.06 + i * 0.48, 0.45, 0.42, 0.45))
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
        for e in edges[1:-1]:
            ax.axvline(e - 0.5, color=INK, lw=0.4, alpha=0.5)
        ax.set_xticks((edges[:-1] + edges[1:]) / 2 - 0.5, [f"B{j + 1}" for j in range(len(widths))], fontsize=5.6)
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=6.8, fontweight="bold", pad=4)
        ax.set_xlabel("items, grouped by benchmark", fontsize=6.2)
        if i == 0:
            ax.set_ylabel("subjects", fontsize=6.2)
        if i == 1:   # one outline per held-out role, labelled above the matrix
            for role in ("validation", "test"):
                bs = [b for b, r in enumerate(roles) if r == role]
                x0, x1 = edges[bs[0]] - 0.5, edges[bs[-1] + 1] - 0.5
                ax.add_patch(Rectangle((x0, -0.5), x1 - x0, observed.shape[0], fill=False, ec=INK, lw=1.1))
                ax.text((x0 + x1) / 2, 1.5, role, ha="center", va="top", fontsize=5.6,
                        bbox=dict(fc="white", ec="none", pad=0.6))
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    # (c) inside one held-out task: acquisition vs evaluation pool.
    ax = fig.add_axes((0.06, 0.285, 0.9, 0.06))
    n = 40
    pool = np.array([1] * 20 + [0] * 20)
    rng.shuffle(pool)
    acquired = np.zeros(n, dtype=bool)
    acquired[np.flatnonzero(pool == 1)[:7]] = True
    colours = np.where(pool == 1, np.where(acquired, 3, 1), 2)
    ax.imshow(colours[None, :], cmap=ListedColormap(["#FFFFFF", "#E6E6E6", "#9A9A9A", MUTED["indigo"]]),
              vmin=0, vmax=3, aspect="auto", interpolation="nearest")
    for k in range(1, n):            # cell borders, so each item reads as one cell
        ax.axvline(k - 0.5, color="white", lw=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(c) Inside one held-out task (one subject on one benchmark): the online protocol",
                 loc="left", fontsize=6.8, fontweight="bold", pad=4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    handles = [Patch(fc="#BFC4DD", ec=INK, lw=0.3, label="training"),
               Patch(fc=MUTED["teal"], ec=INK, lw=0.3, label="validation"),
               Patch(fc=MUTED["rose"], ec=INK, lw=0.3, label="test"),
               Patch(fc="#FFFFFF", ec=INK, lw=0.3, label="not observed"),
               Patch(fc="#E6E6E6", ec=INK, lw=0.3, label="(c) acquisition pool"),
               Patch(fc=MUTED["indigo"], ec=INK, lw=0.3, label="(c) label bought"),
               Patch(fc="#9A9A9A", ec=INK, lw=0.3, label="(c) evaluation pool, scored")]
    fig.legend(handles=handles, loc="center", ncol=4, fontsize=5.9, frameon=False, bbox_to_anchor=(0.5, 0.225))
    _caption(fig, "Figure. Schematic of the subject x item response matrix (shape, not scale): items are grouped "
             "into benchmarks of unequal size, and each subject answers mostly one benchmark, so the observed cells "
             "form blocks. (a) The random-subject split holds out subjects inside every benchmark, so every "
             "benchmark is seen in training. (b) The benchmark holdout holds out whole benchmarks (outlined): "
             "validation and test items come from benchmarks never seen in training; ours uses 11 training, 3 "
             "validation and 3 test benchmarks. (c) Within each held-out task the items are split once, 50/50, "
             "into an acquisition pool, from which up to 31 labels are bought one at a time in stream order, and a "
             "fixed evaluation pool that is never revealed and is scored at every budget.")
    _exact_save(pdf, fig)


def data_difficulty_page(pdf: PdfPages) -> None:
    from model_names import NAME

    held = pd.read_csv(DOCS / "np-online-passrate-figure11-summary.csv")
    nonheld = pd.read_csv(DOCS / "non-benchmark-anp128-qwen-item-passrate-summary.csv")
    fig = plt.figure(figsize=(TEXTWIDTH, 4.6))
    header(fig, "The ANPs barely predict which new items are hard", "")
    names = ["ANP,\nrandom-subject\nsplit (width 128)", "ANP,\nbudget-ladder\ntraining", "ANP"]
    xx = np.arange(len(names))
    width = .36
    a0 = fig.add_axes((0.09, 0.60, 0.38, 0.28))
    a1 = fig.add_axes((0.59, 0.60, 0.38, 0.28))
    a0.bar(xx - width / 2, held["pass_rate_correlation"], width, color=MUTED["indigo"], label="all items")
    a0.bar(xx + width / 2, held["within_benchmark_correlation"], width, color=MUTED["teal"],
           label="within benchmark")
    a0.set_ylabel("correlation", fontsize=6.2)
    a0.set_title("(a) Predicted vs observed item pass rate", loc="left", weight="bold", fontsize=6.8)
    a1.bar(xx - width / 2, held["prediction_sd"], width, color=MUTED["rose"], label="predicted")
    a1.bar(xx + width / 2, held["observed_pass_rate_sd"], width, color=MUTED["grey"], label="observed")
    a1.set_ylabel("SD across items", fontsize=6.2)
    a1.set_title("(b) Predictions are compressed", loc="left", weight="bold", fontsize=6.8)
    for ax in (a0, a1):
        ax.set_xticks(xx, names)
        ax.tick_params(labelsize=5.6)
        ax.legend(loc="upper center", fontsize=5.6, frameon=False, ncol=2)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
        style(ax)
    fig.text(0.02, 0.38, "(c) Two protocols, test split", fontsize=6.8, weight="bold")
    row = nonheld.set_index("split").loc["test"]
    f = held.iloc[2]
    rows = [["ANP, random-subject split (width 128, Qwen3-8B embeddings)", f"{row.correlation:.3f}",
             f"{row.within_benchmark_correlation:.3f}", f"{row.prediction_sd:.3f}", f"{row.observed_pass_rate_sd:.3f}"],
            [NAME["w64-filtered"] + " (benchmark holdout, BGE-M3 embeddings)", f"{f.pass_rate_correlation:.3f}",
             f"{f.within_benchmark_correlation:.3f}", f"{f.prediction_sd:.3f}", f"{f.observed_pass_rate_sd:.3f}"]]
    ax = fig.add_axes((0.02, 0.215, 0.96, 0.15))
    _booktabs(ax, ["Model", "Corr.", "Within", "Pred. SD", "Obs. SD"], rows,
              xs=[0.0, 0.58, 0.68, 0.78, 0.89], right={1, 2, 3, 4}, widths=[0.58, 0.08, 0.08, 0.09, 0.09])
    _caption(fig, "Figure. Item-difficulty transfer on 297 held-out items, at zero labels. (a) Correlation between "
             "each item's predicted and observed pass rate, over all items and within each benchmark. (b) The "
             "spread of the predictions is a fraction of the observed spread: the models predict nearly the same "
             "probability for every item. (c) The same contrast for a random-subject-split model scored on its own "
             "test split. The two rows differ in split, encoder and evaluation set, so this is evidence of a "
             "transfer bottleneck, not an architecture ablation. Sources: docs/np-online-passrate-figure11-"
             "summary.csv and docs/non-benchmark-anp128-qwen-item-passrate-summary.csv.")
    _exact_save(pdf, fig)


def benchmark_difficulty_page(pdf: PdfPages) -> None:
    """Booktabs-style table: how hard each benchmark is, by split role.

    Rules only (top, mid, bottom, and a light rule between roles), no grid,
    as a typeset paper table would be. Numbers come from
    docs/benchmark-subject-difficulty-summary.csv.
    """
    import textwrap

    d = pd.read_csv(DOCS / "benchmark-subject-difficulty-summary.csv")
    roles = [("train", "Train"), ("validation", "Validation"), ("test", "Test")]
    bench = d[d.level == "benchmark"]
    subj = d[d.level == "subjects"].set_index("benchmark_split")
    cols = ["Role", "Benchmark", "Items", "Subjects", "Mean", "SD", "10%", "90%"]
    xs = [0.02, 0.15, 0.47, 0.57, 0.68, 0.76, 0.84, 0.92]   # left edge of each column
    right = {2, 3, 4, 5, 6, 7}                                # right-aligned numeric columns
    width = [0.13, 0.30, 0.08, 0.09, 0.07, 0.07, 0.07, 0.07]

    lines = []   # (kind, cells) with kind in {"row", "rule", "lightrule"}
    for i, (role, label) in enumerate(roles):
        g = bench[bench.benchmark_split == role].sort_values("mean")
        for j, r in enumerate(g.itertuples()):
            lines.append(("row", [label if j == 0 else "", r.benchmark_id,
                                  f"{int(r.items):,}", f"{int(r.subjects):,}", f"{r.mean:.2f}",
                                  f"{r.sd:.2f}", f"{r.p10:.2f}", f"{r.p90:.2f}"]))
        lines.append(("lightrule", None) if i < len(roles) - 1 else ("midrule", None))
    lines.append(("head2", ["", "All subjects in the role", "", "Subjects", "Mean", "SD", "10%", "90%"]))
    lines.append(("midrule", None))
    for role, label in roles:
        r = subj.loc[role]
        lines.append(("row", [label, "mean accuracy per subject", "", f"{int(r.subjects):,}", f"{r['mean']:.2f}",
                              f"{r['sd']:.2f}", f"{r.p10:.2f}", f"{r.p90:.2f}"]))
    lines.append(("bottomrule", None))

    row_h = 0.205    # inches per text row
    height = 0.75 + row_h * (sum(k in ("row", "head2") for k, _ in lines) + 1) + 1.35
    fig = plt.figure(figsize=(TEXTWIDTH, height))
    ax = fig.add_axes((0.02, 1.35 / height, 0.96, 1 - 2.1 / height))
    ax.axis("off")
    n_rows = sum(k in ("row", "head2") for k, _ in lines) + 1
    y = 1.0
    step = 1.0 / (n_rows + 0.6)

    def put_row(cells, weight="normal"):
        for c, (x, text) in enumerate(zip(xs, cells)):
            ax.text(x + (width[c] if c in right else 0), y - step / 2, text, ha="right" if c in right else "left",
                    va="center", fontsize=6.6, fontweight=weight, transform=ax.transAxes)

    def rule(lw):
        ax.plot([0, 1], [y, y], color=INK, lw=lw, transform=ax.transAxes, clip_on=False)

    rule(1.0)                                   # \toprule
    put_row(cols, weight="bold"); y -= step
    rule(0.5)                                   # \midrule
    for kind, cells in lines:
        if kind == "row":
            put_row(cells); y -= step
        elif kind == "head2":
            y -= step * 0.3
            put_row(cells, weight="bold"); y -= step
        elif kind == "lightrule":
            ax.plot([0, 1], [y, y], color=SUBTLE, lw=0.3, transform=ax.transAxes)
        else:
            rule(1.0 if kind == "bottomrule" else 0.5)

    fig.text(0.018, 1 - 0.12 / height, "The held-out benchmarks differ in difficulty more than the subjects do",
             fontsize=9, fontweight="bold", va="top")
    body = ("Table. Difficulty of every benchmark in the benchmark-holdout split, as subject accuracy on it: each "
            "subject's accuracy on the benchmark, then the mean, standard deviation and 10th/90th percentiles over "
            "subjects (lower means harder). The lower block pools each role's subjects. Validation holds the "
            "hardest benchmark (bird_sql, 0.15) and test holds three mid-to-easy ones (0.54-0.69), so test "
            "subjects look stronger than training subjects (0.61 against 0.47). Much of that gap is the training "
            "side: alignment_faking contributes 4,278 of the 6,013 training subjects (71%), many near zero, which "
            "pulls the per-subject training average down. With three benchmarks per held-out role, which "
            "benchmarks are held out moves these averages substantially.")
    fig.text(0.018, 0.012, textwrap.fill(body, 116), fontsize=6.1, va="bottom", linespacing=1.35)
    pdf.savefig(fig, bbox_inches=Bbox.from_bounds(0, 0, *fig.get_size_inches()))
    plt.close(fig)


def data_analysis_report(output: Path) -> None:
    with PdfPages(output) as pdf:
        data_split_page(pdf)
        data_split_schematic_page(pdf)
        data_difficulty_page(pdf)
        benchmark_difficulty_page(pdf)
    audit = DOCS / "data-structure-inductive-bias.pdf"
    if audit.exists():
        append_pdf(output, [audit])


def load_metrics(name: str) -> dict:
    return json.loads((ROOT / "results" / name / "metrics.json").read_text())


def offline_main_page(pdf: PdfPages) -> None:
    metrics = load_metrics("faithful-np-beavertails-3seed-v2-fixed")["aggregate"]
    display = {
        "online_airt_2pl": "Online AIRT 2PL",
        "tnpd_airt": "TNP-D AIRT",
        "anp": "ANP",
        "np": "latent NP",
        "stacked_mhsa_anp": "stacked ANP",
        "beta_binomial": "Beta-Binomial",
    }
    order = [key for key in display if key in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.9), gridspec_kw={"width_ratios": [1.45, 1]})
    header(fig, "Historical offline result: three-seed NP/IRT comparison",
           "BeaverTails protocol, 3 seeds, 12 held-out evaluations. This is not the final benchmark-held-out online panel.")
    for index, key in enumerate(order):
        curve = metrics[key]["by_budget"]
        values = np.array([curve[str(b)]["brier"] for b in BUDGETS])
        se = np.array([curve[str(b)].get("brier_se", 0.0) for b in BUDGETS])
        axes[0].plot(BUDGETS, values, marker="o", ms=2.7, color=CYCLE[index], label=display[key])
        axes[0].fill_between(BUDGETS, values - 1.96 * se, values + 1.96 * se,
                             color=CYCLE[index], alpha=.10, lw=0)
    axes[0].set_xlabel("labels acquired"); axes[0].set_ylabel("Brier score")
    axes[0].set_xticks(BUDGETS); axes[0].set_title("(a) Brier by budget", loc="left", weight="bold")
    axes[0].legend(fontsize=5.5, ncol=2, loc="upper right"); style(axes[0])
    ranked = sorted(order, key=lambda key: metrics[key]["brier_alc"])
    values = [metrics[key]["brier_alc"] for key in ranked]
    errors = [metrics[key].get("brier_alc_seed_std", 0.0) for key in ranked]
    bars = axes[1].barh([display[key] for key in ranked], values, xerr=errors,
                         color=[CYCLE[order.index(key)] for key in ranked], capsize=1.8)
    axes[1].bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2, fontsize=5.7)
    axes[1].invert_yaxis(); axes[1].set_xlabel("macro Brier ALC")
    axes[1].set_title("(b) ALC ± seed SD", loc="left", weight="bold"); style(axes[1], "x")
    fig.text(.02, .015,
             "Source: results/faithful-np-beavertails-3seed-v2-fixed/metrics.json. "
             "It documents a useful earlier model comparison, but must not be combined numerically with the final unseen-benchmark online simulation.",
             fontsize=5.9, color=MUTED["wine"])
    fig.tight_layout(rect=(0, .08, 1, .84), pad=.55)
    save_page(pdf, fig)


def offline_sensitivity_page(pdf: PdfPages) -> None:
    versions = [
        ("v1", load_metrics("faithful-np-beavertails-3seed-v1")["aggregate"]),
        ("v2 fixed", load_metrics("faithful-np-beavertails-3seed-v2-fixed")["aggregate"]),
        ("v3 residual", load_metrics("faithful-np-beavertails-3seed-v3-residual")["aggregate"]),
    ]
    keys = ["beta_binomial", "online_airt_2pl", "anp", "np", "stacked_mhsa_anp", "tnpd_airt"]
    labels = {"beta_binomial": "Beta", "online_airt_2pl": "AIRT", "anp": "ANP",
              "np": "NP", "stacked_mhsa_anp": "Stacked ANP", "tnpd_airt": "TNP-D"}
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 3.1))
    header(fig, "Historical offline sensitivity across implementation variants",
           "Same broad BeaverTails family; variants change more than one implementation detail. "
           "Read this as robustness context, not a clean ablation.")
    x = np.arange(len(keys)); width = .23
    for offset, (name, metrics) in zip((-width, 0, width), versions, strict=True):
        vals = [metrics.get(key, {}).get("brier_alc", np.nan) for key in keys]
        ax.bar(x + offset, vals, width, label=name)
    ax.set_xticks(x, [labels[key] for key in keys], rotation=20, ha="right")
    ax.set_ylabel("macro Brier ALC"); ax.set_title("ALC varies materially across protocol/implementation versions", loc="left", weight="bold")
    ax.legend(fontsize=6.4, ncol=3); style(ax)
    fig.text(.02, .025,
             "Sources: results/faithful-np-beavertails-3seed-v{1,2-fixed,3-residual}/metrics.json. "
             "The key lesson is not a winner: conclusions from a single legacy protocol are fragile, motivating the frozen benchmark-holdout online simulation.",
             fontsize=6.1, color=SUBTLE)
    fig.tight_layout(rect=(0, .10, 1, .86), pad=.6)
    save_page(pdf, fig)


def offline_report(output: Path) -> None:
    with PdfPages(output) as pdf:
        offline_main_page(pdf)
        offline_sensitivity_page(pdf)


def checkpoint_rows() -> list[dict]:
    assets = ROOT / "organized" / "submission" / "assets"
    rows = []
    for path in sorted(assets.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "selection" not in payload:
            continue
        config, selection = payload.get("config", {}), payload.get("selection", {})
        split = "benchmark holdout" if config.get("split_protocol") else "random subject split"
        rows.append({
            "file": path.name,
            "model": config.get("model_family", "unknown"),
            "arm": config.get("arm", "unknown"),
            "width": config.get("width", np.nan),
            "blocks": config.get("n_blocks", np.nan),
            "latent": config.get("use_latent", False),
            "split": split,
            "selected_step": selection.get("step", np.nan),
            "selected_brier": selection.get("value", np.nan),
        })
    return rows


def training_selection_page(pdf: PdfPages) -> None:
    frame = pd.DataFrame(checkpoint_rows()).sort_values("selected_brier")
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.2), gridspec_kw={"width_ratios": [1.1, 1.25]})
    header(fig, "Available ANP checkpoint-selection evidence",
           "These are fixed-bank validation Brier values saved inside checkpoints. They are selection diagnostics, not final test scores or comparable across split protocols.")
    labels = [f"{row.arm}\nw{int(row.width)}, sa{int(row.blocks)}" for _, row in frame.iterrows()]
    colors = [MUTED["rose"] if row.split == "benchmark holdout" else MUTED["teal"] for _, row in frame.iterrows()]
    axes[0].barh(labels, frame.selected_brier, color=colors, edgecolor=INK, lw=.3)
    axes[0].invert_yaxis(); axes[0].set_xlabel("fixed-bank validation Brier")
    axes[0].set_title("(a) Saved selection metric", loc="left", weight="bold"); style(axes[0], "x")
    axes[0].text(.98, .02, "rose = benchmark holdout\nteal = random subject split", ha="right", va="bottom",
                 transform=axes[0].transAxes, fontsize=5.8, color=SUBTLE)
    selected = frame[["split", "arm", "width", "blocks", "latent", "selected_step", "selected_brier"]].copy()
    selected["width"] = selected["width"].astype(int)
    selected["blocks"] = selected["blocks"].astype(int)
    selected["latent"] = selected["latent"].map({True: "yes", False: "no"})
    selected["selected_step"] = selected["selected_step"].astype(int)
    rows = [[row.split.replace("random subject split", "random split"), row.arm, str(row.width),
             str(row.blocks), row.latent, f"{row.selected_step:,}", f"{row.selected_brier:.4f}"]
            for _, row in selected.iterrows()]
    table(axes[1], rows, ["split", "model", "w", "SA", "z", "step", "Brier"],
          widths=[.24, .18, .07, .08, .06, .16, .16], fontsize=5.2)
    axes[1].set_title("(b) Checkpoint metadata", loc="left", weight="bold", pad=5)
    fig.text(.02, .015,
             "Source: organized/submission/assets/*.pt (config + selection fields). The raw per-step loss histories are not stored in these checkpoints; this report intentionally does not invent a training curve.",
             fontsize=5.8, color=MUTED["wine"])
    fig.tight_layout(rect=(0, .08, 1, .85), pad=.55)
    save_page(pdf, fig)


def training_inventory_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(TEXTWIDTH, 3.0))
    header(fig, "What training evidence is actually recoverable locally",
           "Only plot a loss curve when its sampled history, split, and metric are preserved together.")
    ax = fig.add_axes([.08, .20, .84, .55])
    rows = [
        ["Current benchmark-holdout ANP", "checkpoint config + best fixed-bank metric", "no", "selection only; test is in online-final-results.pdf"],
        ["Current random-split ANP/CNP", "checkpoint config + best fixed-bank metric", "no", "diagnostic only; benchmark overlap"],
        ["Faithful NP/IRT experiment", "3-seed offline Brier/ECE summaries", "no", "historical offline comparison"],
        ["Demo AIRT / CNP-AIRT weights", "state_dict only", "no", "not sufficient for training dynamics"],
        ["Local W&B exports", "final summary + image for older runs", "no", "no raw step histories preserved"],
    ]
    table(ax, rows, ["source", "available", "raw step history", "safe use"],
          widths=[.24, .25, .14, .37], fontsize=6.0)
    fig.text(.08, .08,
             "To add real ANP loss curves later: export W&B histories for the exact run IDs, including "
             "training negative ELBO/NLL, fixed-bank target NLL/Brier, KL, and coverage; save them as a CSV next to the checkpoint. "
             "Do not reconstruct them from checkpoint-selection metadata.",
             fontsize=6.5, color=SUBTLE)
    save_page(pdf, fig)


def training_report(output: Path) -> None:
    with PdfPages(output) as pdf:
        training_selection_page(pdf)
        training_inventory_page(pdf)


def main() -> None:
    protocol_report(DOCS / "protocol-final-results.pdf")
    data_analysis_report(DOCS / "data-analysis-final-results.pdf")
    offline_report(DOCS / "offline-final-results.pdf")
    training_report(DOCS / "training-final-results.pdf")
    for name in ["protocol", "data-analysis", "offline", "training", "online"]:
        path = DOCS / f"{name}-final-results.pdf"
        if path.exists():
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
