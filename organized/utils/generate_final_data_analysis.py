#!/usr/bin/env python3
"""Regenerate ``docs/final-data-analysis.pdf`` in the repository plot style.

This report intentionally does *not* append pages from earlier PDFs. Every new
page uses ``organized/utils/plotstyle.py`` so typography, palette, and page
width are consistent. It is a descriptive data report: it keeps
response-derived quantities (training-only) separate from text-derived ones.

Example (the audit artefacts used in the repository worktree):

    .venv/bin/python organized/utils/generate_final_data_analysis.py \
      --filter-retention /tmp/aims-data-audit-Kur9U6/filter-retention.parquet \
      --item-passrate /tmp/aims-figure4-data/item-passrate.parquet \
      --classic-irt-dir /tmp/.../scratchpad/da \
      --graph-audit-dir /tmp/aims-laplacian-final-WRWvGe \
      --prediction-dir /tmp/aims-figure4-data \
      --bhold-responses /tmp/bhold-filtered-responses.parquet

The graph-audit pages are rendered again from their parquet inputs by
``plot_data_structure_audit.py``; no legacy PDF artwork is copied in.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from pypdf import PdfReader, PdfWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_data_structure_audit import render as render_graph_audit  # noqa: E402
from plotstyle import INK, MUTED, SUBTLE, TEXTWIDTH, header, style  # noqa: E402

# ``plotstyle`` uses a tight bounding box for standalone paper figures.  This
# is a multi-page report: fixed page boxes make pages align in a PDF viewer and
# prevent an annotation from silently changing a page's physical width.
matplotlib.rcParams["savefig.bbox"] = None


def append_pdf(destination: Path, source: Path) -> None:
    """Append already house-styled vector pages to a PDF."""
    writer = PdfWriter()
    for page in PdfReader(str(destination)).pages:
        writer.add_page(page)
    for page in PdfReader(str(source)).pages:
        writer.add_page(page)
    with destination.open("wb") as handle:
        writer.write(handle)


def _text(ax, x: float, y: float, text: str, *, size: float = 7.0, **kwargs) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, va="top", **kwargs)


def _footer(fig, text: str, *, y: float = .035) -> None:
    """Keep report footnotes inside the fixed-width page box."""
    fig.text(.07, y, textwrap.fill(text, width=105), fontsize=6.15, color=SUBTLE, va="bottom")


def _correlation(values: pd.DataFrame) -> tuple[float, float]:
    """Return global and benchmark-centred Pearson correlations."""
    r = np.corrcoef(values["prediction"], values["pass_rate"])[0, 1]
    centred = values.assign(
        prediction=values["prediction"] - values.groupby("benchmark_id")["prediction"].transform("mean"),
        pass_rate=values["pass_rate"] - values.groupby("benchmark_id")["pass_rate"].transform("mean"),
    )
    rw = np.corrcoef(centred["prediction"], centred["pass_rate"])[0, 1]
    return float(r), float(rw)


def guide_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(TEXTWIDTH, 3.45))
    header(
        fig,
        "Final data analysis",
        "Support, response geometry, language coverage, and item-difficulty diagnostics.",
    )
    ax = fig.add_axes([.07, .13, .87, .70])
    ax.axis("off")
    blocks = [
        ("1  Support matters", "Sparse item responses create exact 0/1 pass rates that are not reliable difficulty estimates. The first two figures show what the support filter removes."),
        ("2  Text and responses are different signals", "Language and content length come from item text. Response matrices, empirical pass rates, and classic IRT fits use observed labels and are training-only."),
        ("3  Geometry is descriptive", "The Qwen kNN graph and its Laplacian describe content structure. They motivate a graph ablation, but do not by themselves prove an online-prediction gain."),
        ("4  Transfer is the open problem", "The classic-IRT diagnostic checks recoverability when item responses are available. The archival neural diagnostic tests whether an embedding-only prior recovered comparable item variation."),
    ]
    y = .98
    for title, body in blocks:
        _text(ax, .00, y, title, size=8.1, fontweight="bold")
        _text(ax, .035, y - .105, body, size=7.2, color=SUBTLE, wrap=True)
        y -= .235
    fig.text(
        .07,
        .045,
        "Scope: no leaderboard or acquisition-policy claim here; see the frozen sequential replay.",
        fontsize=6.6,
        color=MUTED["wine"],
    )
    pdf.savefig(fig)
    plt.close(fig)


def matrix_and_filter_page(pdf: PdfPages, matrix_sample: Path, retention_path: Path) -> None:
    sample = np.load(matrix_sample)
    raw, filtered = sample["raw"], sample["filtered"]
    metadata = json.loads(str(sample["metadata"]))
    retention = pd.read_parquet(retention_path)
    survived = retention[retention["items_after"] > 0].copy()
    cmap = ListedColormap([MUTED["rose"], MUTED["indigo"]])
    cmap.set_bad("white")

    fig = plt.figure(figsize=(TEXTWIDTH, 5.15))
    header(
        fig,
        "Response matrix before and after support filtering",
        "AfriEval samples: blue = correct, rose = incorrect, white = unobserved.",
    )
    panels = [
        ("(a) Raw observations", raw, "highest-support rows and columns before filtering"),
        ("(b) n≥30 / n≥80 core", filtered, "item responders ≥30; retained subjects answer ≥80 items"),
    ]
    for index, (title, matrix, note) in enumerate(panels):
        ax = fig.add_axes([.075 + .49 * index, .57, .405, .26])
        ax.imshow(np.ma.masked_where(matrix < 0, matrix), cmap=cmap, vmin=0, vmax=1,
                  aspect="auto", interpolation="nearest")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(f"{matrix.shape[1]} displayed items  hard → easy")
        ax.set_ylabel(f"{matrix.shape[0]} displayed subjects  weak → strong")
        ax.set_xticks([])
        ax.set_yticks([])
        _text(ax, .99, 1.03, f"{(matrix >= 0).mean():.0%} observed", size=6.2, ha="right", color=SUBTLE)
        _text(ax, .01, -.16, note, size=5.8, color=SUBTLE)

    ax = fig.add_axes([.09, .20, .37, .21])
    retained_colour = np.where(retention["items_after"] > 0, MUTED["teal"], MUTED["grey"])
    ax.scatter(retention["items_before"], retention["items_after"], s=12, c=retained_colour,
               edgecolor="none", alpha=.88)
    limit = max(retention[["items_before", "items_after"]].to_numpy().max(), 1)
    ax.plot([1, limit], [1, limit], color=INK, lw=.6, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("items before filtering")
    ax.set_ylabel("items after filtering")
    ax.set_title("(c) Benchmark-level item retention", loc="left", fontweight="bold")
    style(ax, "both")

    ax = fig.add_axes([.59, .20, .33, .21])
    shown = survived.nlargest(7, "items_after").sort_values("items_after")
    y = np.arange(len(shown))
    ax.barh(y, shown["items_before"], color=MUTED["grey"], label="raw")
    ax.barh(y, shown["items_after"], color=MUTED["teal"], label="core")
    ax.set_yticks(y, shown["benchmark_id"], fontsize=6.1)
    ax.set_xscale("log")
    ax.set_xlabel("items (log scale)")
    ax.set_title("(d) Seven retained benchmark cores", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=6)
    style(ax, "x")

    _footer(
        fig,
        f"AfriEval: {metadata['raw_rows']:,} responses → {metadata['filtered_rows']:,}; "
        f"{metadata['raw_items']:,} items → {metadata['filtered_items']:,}; "
        f"{metadata['raw_subjects']:,} subjects → {metadata['filtered_subjects']:,}. "
        f"Corpus core: {len(retention)} → {len(survived)} benchmarks; "
        f"{retention['items_before'].sum():,} → {retention['items_after'].sum():,} items. "
        "Panels are descriptive samples, not a cellwise matched comparison.",
        y=.025,
    )
    pdf.savefig(fig)
    plt.close(fig)


def support_and_difficulty_page(pdf: PdfPages, items: pd.DataFrame) -> None:
    stable = items.loc[items["n_responses"] >= 5].copy()
    fig = plt.figure(figsize=(TEXTWIDTH, 4.8))
    header(
        fig,
        "Response support and empirical item difficulty",
        "Pass rate is the observed share correct; one response forces a 0/1 estimate.",
    )
    ax = fig.add_axes([.09, .57, .37, .25])
    counts = items["n_responses"].clip(upper=500)
    bins = np.unique(np.r_[np.arange(1, 21), np.geomspace(20, 500, 25)]).astype(float)
    ax.hist(counts, bins=bins, color=MUTED["grey"])
    ax.set_xscale("log")
    ax.set_xlabel("responses per item (capped at 500)")
    ax.set_ylabel("items")
    ax.set_title("(a) Item response support", loc="left", fontweight="bold")
    style(ax, "y")

    ax = fig.add_axes([.58, .57, .34, .25])
    for threshold, colour in [(1, MUTED["grey"]), (5, MUTED["teal"]), (30, MUTED["indigo"])]:
        subset = items.loc[items["n_responses"] >= threshold, "pass_rate"]
        ax.hist(subset, bins=35, range=(0, 1), density=True, histtype="step", lw=1.3,
                color=colour, label=f"n≥{threshold}  ({len(subset):,})")
    ax.set_xlabel("observed item pass rate")
    ax.set_ylabel("density")
    ax.set_title("(b) Support removes 0/1 artefacts", loc="left", fontweight="bold")
    ax.legend(loc="upper center", fontsize=5.8)
    style(ax, "y")

    ax = fig.add_axes([.09, .20, .37, .22])
    thresholds = np.array([1, 2, 5, 10, 15, 30, 50, 100])
    shares = np.array([(items["n_responses"] >= n).mean() for n in thresholds])
    ax.plot(thresholds, shares, marker="o", color=MUTED["teal"])
    ax.set_xscale("log")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("minimum responses per item")
    ax.set_ylabel("share of all items retained")
    ax.set_title("(c) Coverage–support trade-off", loc="left", fontweight="bold")
    style(ax, "y")

    ax = fig.add_axes([.58, .20, .34, .22])
    summary = stable.groupby("benchmark_id")["pass_rate"].agg(["median", "size"]).sort_values("median")
    marker_size = 8 + 32 * np.sqrt(summary["size"] / summary["size"].max())
    ax.scatter(summary["median"], np.arange(len(summary)), s=marker_size, color=MUTED["indigo"], alpha=.85)
    ax.set_yticks([])
    ax.set_xlim(-.02, 1.02)
    ax.set_xlabel("median item pass rate within benchmark")
    ax.set_title("(d) Difficulty varies across benchmarks", loc="left", fontweight="bold")
    style(ax, "x")
    hardest, easiest = summary.index[0], summary.index[-1]
    _text(ax, .02, .98, f"hardest median: {hardest}\neasiest median: {easiest}\npoint area ∝ qualifying items", size=6.0, color=SUBTLE)

    _footer(
        fig,
        f"{len(items):,} items total; {len(stable):,} have ≥5 responses; {(items['n_responses'] >= 30).sum():,} have ≥30. "
        "The support threshold is a measurement-quality decision, not a test-time assumption.",
        y=.025,
    )
    pdf.savefig(fig)
    plt.close(fig)


def language_and_length_page(pdf: PdfPages, items: pd.DataFrame) -> None:
    stable = items.loc[items["n_responses"] >= 5].copy()
    summary = (stable.groupby("language", as_index=False)
               .agg(items=("pass_rate", "size"), median_pass_rate=("pass_rate", "median"),
                    median_chars=("content_chars", "median"))
               .sort_values("items", ascending=False))
    top = summary.head(12).sort_values("items")
    fig = plt.figure(figsize=(TEXTWIDTH, 4.8))
    header(
        fig,
        "Language and content-length coverage",
        "Language is heuristic metadata: a coverage audit, not a causal difficulty label.",
    )
    ax = fig.add_axes([.18, .57, .28, .25])
    ax.barh(top["language"], top["items"], color=MUTED["teal"])
    ax.set_xscale("log")
    ax.set_xlabel("items with ≥5 responses (log scale)")
    ax.set_title("(a) Largest language groups", loc="left", fontweight="bold")
    style(ax, "x")

    ax = fig.add_axes([.60, .57, .32, .25])
    qualified = summary.loc[summary["items"] >= 500].sort_values("median_pass_rate")
    size = 8 + 32 * np.sqrt(qualified["items"] / qualified["items"].max())
    ax.scatter(qualified["median_pass_rate"], np.arange(len(qualified)), s=size,
               color=MUTED["indigo"], alpha=.9)
    ax.set_yticks(np.arange(len(qualified)), qualified["language"], fontsize=6)
    ax.set_xlim(-.02, 1.02)
    ax.set_xlabel("median item pass rate")
    ax.set_title("(b) Difficulty differs by language", loc="left", fontweight="bold")
    style(ax, "x")

    ax = fig.add_axes([.18, .20, .28, .22])
    chars = stable["content_chars"].clip(lower=1, upper=20_000)
    bins = np.geomspace(1, 20_000, 36)
    ax.hist(chars, bins=bins, color=MUTED["grey"])
    ax.set_xscale("log")
    ax.set_xlabel("item content characters (capped at 20k)")
    ax.set_ylabel("items")
    ax.set_title("(c) Item content has a long tail", loc="left", fontweight="bold")
    style(ax, "y")

    ax = fig.add_axes([.60, .20, .32, .22])
    shown = summary.head(12).sort_values("median_chars")
    ax.scatter(shown["median_chars"], np.arange(len(shown)), s=25, color=MUTED["rose"])
    ax.set_xscale("log")
    ax.set_yticks(np.arange(len(shown)), shown["language"], fontsize=6)
    ax.set_xlabel("median content characters")
    ax.set_title("(d) Text length differs by language", loc="left", fontweight="bold")
    style(ax, "x")

    _footer(
        fig,
        "Implication: multilingual, long-context item encoders are sensible inputs. Coverage differences alone do not establish cross-benchmark difficulty transfer.",
        y=.025,
    )
    pdf.savefig(fig)
    plt.close(fig)


def function_realisations_page(pdf: PdfPages) -> None:
    """Explain the response-function unit without treating embeddings as 1D IRT difficulty."""
    fig = plt.figure(figsize=(TEXTWIDTH, 4.8))
    header(
        fig,
        "What is a function in the two evaluation protocols?",
        "One function is one subject--benchmark task. Curves are conceptual, not fitted IRT estimates.",
    )

    ax = fig.add_axes([.07, .70, .86, .16])
    ax.axis("off")
    _text(
        ax,
        .00,
        .93,
        r"Task $\tau=(s,b)$: $D_{s,b}=\{(e_i,y_{s,b,i})\}_i$ is a finite sample from "
        r"$f_{s,b}(e_i)=\Pr(Y_{s,b,i}=1\mid e_i,s,b)$.",
        size=8.0,
    )
    _text(
        ax,
        .00,
        .36,
        "The NP receives a few labelled context items from this task and predicts held-out items. "
        "At test time it changes its conditional prediction through the context, not its network weights.",
        size=7.0,
        color=SUBTLE,
        wrap=True,
    )

    coordinate = np.linspace(0, 1, 300)

    def response_curve(ability: float, *, slope: float = 8.0, bend: float = 0.0) -> np.ndarray:
        logit = slope * (coordinate - (1.0 - ability)) + bend * np.sin(2 * np.pi * coordinate)
        return 1.0 / (1.0 + np.exp(-logit))

    def task_points(ax, curve: np.ndarray, *, show_legend: bool) -> None:
        context_x = np.array([.13, .42, .76])
        context_y = np.array([0, 0, 1])
        target_x = np.array([.27, .61, .90])
        target_y = np.interp(target_x, coordinate, curve)
        ax.scatter(context_x, context_y, s=20, c=[MUTED["rose"], MUTED["rose"], MUTED["teal"]],
                   edgecolor="white", linewidth=.45, zorder=4,
                   label="acquired binary labels")
        ax.scatter(target_x, target_y, s=19, facecolor="white", edgecolor=INK, linewidth=.6,
                   zorder=4, label="items to predict")
        if show_legend:
            ax.legend(loc="lower right", fontsize=5.6)

    ax = fig.add_axes([.10, .27, .34, .31])
    for ability in [.31, .48, .66]:
        ax.plot(coordinate, response_curve(ability), color=MUTED["grey"], lw=.85, alpha=.9)
    test_curve = response_curve(.52)
    ax.plot(coordinate, test_curve, color=MUTED["teal"], lw=1.6, label=r"test $f_{s^*,b}$")
    task_points(ax, test_curve, show_legend=True)
    ax.text(.02, .95, r"$b$ is represented in training", transform=ax.transAxes,
            fontsize=6.25, va="top", color=SUBTLE)
    ax.set_title("(a) Non-benchmark holdout", loc="left", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-.06, 1.06)
    ax.set_yticks([0, .5, 1], ["0", ".5", "1"])
    ax.set_xlabel("conceptual difficulty-like coordinate: hard → easy", fontsize=6.5)
    ax.set_ylabel("probability correct", fontsize=6.5)
    style(ax, "both")

    ax = fig.add_axes([.57, .27, .34, .31])
    for ability in [.31, .48, .66]:
        ax.plot(coordinate, response_curve(ability), color=MUTED["grey"], lw=.85, alpha=.8)
    heldout_curve = response_curve(.47, slope=5.9, bend=.34)
    ax.plot(coordinate, heldout_curve, color=MUTED["indigo"], lw=1.6, label=r"test $f_{s^*,b_{\rm test}}$")
    task_points(ax, heldout_curve, show_legend=False)
    ax.text(.02, .95, r"$b_{\rm test}$ never appears in training", transform=ax.transAxes,
            fontsize=6.25, va="top", color=SUBTLE)
    ax.text(.02, .84, r"grey: seen $b_{\rm train}$ functions", transform=ax.transAxes,
            fontsize=5.9, va="top", color=SUBTLE)
    ax.set_title("(b) Benchmark holdout", loc="left", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-.06, 1.06)
    ax.set_yticks([0, .5, 1], ["0", ".5", "1"])
    ax.set_xlabel("conceptual difficulty-like coordinate: hard → easy", fontsize=6.5)
    ax.set_ylabel("probability correct", fontsize=6.5)
    style(ax, "both")

    _footer(
        fig,
        "Grey curves sketch functions from a familiar training benchmark. In (a), the test task shares that benchmark's item family and response structure. "
        "In (b), the test function belongs to an unseen benchmark, so the conditional prior must transfer through item content embeddings. "
        "The horizontal coordinate is only an explanatory slice: the evaluated NP uses high-dimensional item embeddings, not a known scalar difficulty.",
        y=.035,
    )
    pdf.savefig(fig)
    plt.close(fig)


def observed_benchmark_holdout_curves_page(pdf: PdfPages, responses_path: Path) -> None:
    """Plot actual held-out-task trajectories against a post-hoc LOO difficulty proxy.

    The horizontal coordinate uses labels from the other subjects on the same
    held-out benchmark.  This is deliberately an analysis-only diagnostic: it
    makes the observed response functions visible, but is unavailable to a
    submission when it first encounters an unseen benchmark.
    """
    columns = ["benchmark_id", "benchmark_split", "subject_uid", "item_uid", "label"]
    responses = pd.read_parquet(
        responses_path,
        columns=columns,
        filters=[("benchmark_split", "=", "test")],
    )
    if responses.empty:
        raise ValueError("benchmark-holdout response file contains no test rows")

    fig = plt.figure(figsize=(TEXTWIDTH, 4.8))
    header(
        fig,
        "Observed subject response functions on held-out benchmarks",
        "Post-hoc only: the x-axis uses labels from other test subjects.",
    )
    colours = {"low": MUTED["rose"], "median": MUTED["indigo"], "high": MUTED["teal"]}
    display_names = {"algotune": "AlgoTune", "cmmlu": "C-MMLU", "editbench": "EditBench"}

    def representative_subjects(means: pd.Series) -> dict[str, str]:
        chosen: set[str] = set()
        selected: dict[str, str] = {}
        for name, quantile in [("low", .10), ("median", .50), ("high", .90)]:
            centre = means.quantile(quantile)
            for subject in (means - centre).abs().sort_values().index:
                if subject not in chosen:
                    selected[name] = str(subject)
                    chosen.add(str(subject))
                    break
        return selected

    for col, benchmark in enumerate(sorted(responses["benchmark_id"].unique())):
        group = responses.loc[responses["benchmark_id"].eq(benchmark)].copy()
        item_stats = group.groupby("item_uid")["label"].agg(item_sum="sum", item_n="size")
        group = group.join(item_stats, on="item_uid")
        group = group.loc[group["item_n"] > 1].copy()
        group["loo_pass_rate"] = (group["item_sum"] - group["label"]) / (group["item_n"] - 1)
        subject_means = group.groupby("subject_uid")["label"].mean().sort_values()
        choices = representative_subjects(subject_means)
        ax = fig.add_axes([.08 + .31 * col, .26, .245, .34])
        rhos: list[float] = []
        curves: dict[str, pd.DataFrame] = {}

        for subject, one in group.groupby("subject_uid", sort=False):
            one = one.copy()
            # Ranking makes eight equal-count bins even when empirical pass
            # rates contain ties. The line is an observed binned mean, not a fit.
            one["bin"] = pd.qcut(one["loo_pass_rate"].rank(method="first"), 8, labels=False)
            curve = one.groupby("bin").agg(
                observed=("label", "mean"),
                item_pass_rate=("loo_pass_rate", "mean"),
            )
            ax.plot(curve["item_pass_rate"], curve["observed"], color=MUTED["grey"],
                    lw=.55, alpha=.18, zorder=1)
            rhos.append(float(one["label"].corr(one["loo_pass_rate"], method="spearman")))
            if str(subject) in set(choices.values()):
                curves[str(subject)] = curve

        for name, subject in choices.items():
            curve = curves[subject]
            ax.plot(curve["item_pass_rate"], curve["observed"], marker="o", ms=2.6,
                    color=colours[name], lw=1.25, zorder=3,
                    label=f"{name} subject" if col == 0 else None)

        ax.set_title(f"({chr(97 + col)}) {display_names.get(benchmark, benchmark)}", loc="left", fontweight="bold")
        ax.set_xlim(-.02, 1.02)
        ax.set_ylim(-.05, 1.05)
        ax.set_xticks([0, .5, 1])
        ax.set_yticks([0, .5, 1])
        ax.set_xlabel("LOO item pass rate\nhard ←   → easy", fontsize=6.25)
        if col == 0:
            ax.set_ylabel("observed correctness", fontsize=6.5)
            ax.legend(loc="lower right", fontsize=5.4)
        else:
            ax.set_yticklabels([])
        ax.text(
            .03,
            .97,
            f"{group['subject_uid'].nunique()} subjects × {group['item_uid'].nunique():,} items\n"
            f"median $\\rho_s$ = {np.nanmedian(rhos):.2f}",
            transform=ax.transAxes,
            fontsize=5.7,
            va="top",
            color=SUBTLE,
        )
        style(ax, "both")

    _footer(
        fig,
        "Faint traces are all held-out test tasks. Coloured curves select the subjects nearest the 10th, 50th, and 90th percentiles of observed task accuracy. "
        "Each point averages that subject's binary responses in one of eight equal-count bins of leave-one-subject-out item pass rate. "
        "These reveal real response-function variation, but use held-out benchmark labels and are strictly post-hoc.",
        y=.035,
    )
    pdf.savefig(fig)
    plt.close(fig)


def classic_irt_page(pdf: PdfPages, classic_dir: Path) -> None:
    files15 = sorted(classic_dir.glob("classic-irt-filter-test-seed2026????.parquet"))
    files30 = sorted(classic_dir.glob("classic-irt-filter-test-seed*-n30.parquet"))
    if not files15 or not files30:
        raise FileNotFoundError("classic IRT parquet files are missing")
    n15 = pd.concat([pd.read_parquet(path) for path in files15], ignore_index=True)
    n30 = pd.concat([pd.read_parquet(path) for path in files30], ignore_index=True)
    n15 = n15[n15["stage"].isin(["before", "after"])].copy()
    n30 = n30[n30["stage"] == "after_n30"].copy()
    item_fit = pd.read_parquet(classic_dir / "filter-2pl-items-avg.parquet")
    n15_bench = n15.loc[n15["stage"] == "after", "benchmark_id"].nunique()
    n30_bench = n30["benchmark_id"].nunique()
    item_fit["prediction"] = item_fit["pred_pass_rate"]
    item_fit["pass_rate"] = item_fit["obs_pass_rate"]
    correlation, within = _correlation(item_fit)

    fig = plt.figure(figsize=(TEXTWIDTH, 4.9))
    header(
        fig,
        "Filtering and within-benchmark item difficulty",
        "Classic IRT fits observed response matrices and scores masked cells; it cannot cold-start a new item.",
    )
    stages = [
        ("raw", n15[n15["stage"] == "before"], MUTED["grey"]),
        ("n≥15", n15[n15["stage"] == "after"], MUTED["teal"]),
        ("n≥30", n30, MUTED["indigo"]),
    ]
    models = ["1pl", "2pl", "3pl"]
    x = np.arange(len(models))
    for metric, xpos, ylabel, title in [
        ("auc", [.08, .54, .38, .27], "masked-cell AUC", "(a) Discrimination"),
        ("brier", [.58, .54, .34, .27], "masked-cell Brier (↓)", "(b) Calibration"),
    ]:
        ax = fig.add_axes(xpos)
        width = .23
        for i, (name, values, colour) in enumerate(stages):
            stat = values.groupby("model")[metric].agg(["mean", "std"])
            y = [stat.loc[model, "mean"] for model in models]
            error = [stat.loc[model, "std"] for model in models]
            ax.bar(x + (i - 1) * width, y, width, yerr=error, capsize=1.8, color=colour, label=name)
        ax.set_xticks(x, [model.upper() for model in models])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        if metric == "auc":
            ax.set_ylim(.45, 1.0)
        ax.legend(loc="best", fontsize=5.8)
        style(ax, "y")

    ax = fig.add_axes([.08, .11, .37, .28])
    draw = item_fit.sample(min(40_000, len(item_fit)), random_state=0)
    ax.hexbin(draw["prediction"], draw["pass_rate"], gridsize=35, bins="log", mincnt=1,
              extent=(0, 1, 0, 1), cmap="viridis", rasterized=True)
    ax.plot([0, 1], [0, 1], color=MUTED["rose"], lw=.9, ls="--")
    ax.set_xlabel("2PL fitted item pass rate")
    ax.set_ylabel("observed item pass rate")
    ax.set_title("(c) Full-data 2PL item fit after n≥15", loc="left", fontweight="bold")
    _text(ax, .03, .97, f"global r = {correlation:.2f}\nwithin-benchmark r = {within:.2f}", size=6.2)
    style(ax, "both")

    ax = fig.add_axes([.58, .11, .34, .28])
    ax.axis("off")
    note = (
        "What this establishes\n"
        "• Stable response matrices contain learnable item-difficulty signal.\n\n"
        "What it does not establish\n"
        "• Text embeddings predict difficulty for a cold item.\n"
        "• A test-time submission can use these fitted parameters.\n\n"
        f"Coverage: n≥15 retains {n15_bench} benchmarks; n≥30 retains {n30_bench}.\n"
        "The stricter filter is a quality–coverage trade-off."
    )
    _text(ax, .00, 1.0, note, size=7.0, color=SUBTLE)
    pdf.savefig(fig)
    plt.close(fig)


def archival_neural_prior_page(pdf: PdfPages, item_passrate_path: Path, prediction_dir: Path) -> None:
    """Restyle the earlier non-holdout item-prior diagnostic, keeping scope explicit."""
    item_rates = pd.read_parquet(item_passrate_path)
    stable = item_rates[item_rates["n_responses"] >= 5][["item_uid", "benchmark_id", "pass_rate"]]
    arms = [
        ("cnp_gelu", "CNP + GELU", MUTED["sand"]),
        ("cnp_ln_gelu", "CNP + LayerNorm + GELU", MUTED["teal"]),
        ("bcirt_d4_gelu_all", "Bayesian CNP-IRT, D=4", MUTED["indigo"]),
    ]
    rows = []
    merged: dict[str, pd.DataFrame] = {}
    for stem, name, colour in arms:
        predictions = pd.read_parquet(prediction_dir / f"{stem}-items.parquet")
        aggregated = (predictions[(predictions["split"] == "test") & (predictions["budget"] == 0)]
                      .groupby(["item_uid", "benchmark_id"], as_index=False)["p"].mean()
                      .rename(columns={"p": "prediction"}))
        values = aggregated.merge(stable, on=["item_uid", "benchmark_id"], how="inner")
        r, rw = _correlation(values)
        rows.append((name, r, rw, values["prediction"].std(), values["pass_rate"].std(), len(values)))
        merged[stem] = values

    fig = plt.figure(figsize=(TEXTWIDTH, 4.8))
    header(
        fig,
        "Archival neural item-prior diagnostic",
        "Historical non-holdout split, test items at budget 0; not the final benchmark-holdout result.",
    )
    for col, (stem, name, colour) in enumerate(arms):
        values = merged[stem]
        ax = fig.add_axes([.07 + col * .31, .54, .26, .27])
        ax.hist(values["pass_rate"], bins=28, range=(0, 1), density=True, color=MUTED["grey"], alpha=.55, label="observed")
        ax.hist(values["prediction"], bins=28, range=(0, 1), density=True, histtype="step", lw=1.35, color=colour, label="predicted")
        record = next(row for row in rows if row[0] == name)
        _text(ax, .03, .97, f"r={record[1]:.2f}; within={record[2]:.2f}\nσpred={record[3]:.2f}; σobs={record[4]:.2f}", size=5.6)
        ax.set_title(name, loc="left", fontweight="bold", fontsize=6.8)
        ax.set_xlabel("pass rate / predicted p")
        if col == 0:
            ax.set_ylabel("density")
        if col == 2:
            ax.legend(loc="lower left", fontsize=5.3)
        style(ax, "y")

    ax = fig.add_axes([.09, .12, .83, .25])
    labels = [row[0] for row in rows]
    all_r = [row[1] for row in rows]
    within_r = [row[2] for row in rows]
    positions = np.arange(len(labels))
    width = .31
    ax.bar(positions - width / 2, all_r, width, color=MUTED["grey"], label="across items")
    ax.bar(positions + width / 2, within_r, width, color=MUTED["indigo"], label="after benchmark centring")
    ax.axhline(0, color=INK, lw=.55)
    ax.set_ylim(-.1, 1.02)
    ax.set_xticks(positions, labels, fontsize=6.5)
    ax.set_ylabel("correlation with observed item pass rate")
    ax.set_title("(d) Benchmark centring removes most apparent difficulty signal", loc="left", fontweight="bold")
    ax.legend(loc="upper right")
    style(ax, "y")
    _footer(
        fig,
        "The narrow predicted distributions and near-zero within-benchmark correlations motivate the filtered, benchmark-centred holdout work. These configurations did not transfer item difficulty reliably.",
    )
    pdf.savefig(fig)
    plt.close(fig)


def build(args: argparse.Namespace) -> None:
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    items = pd.read_parquet(args.item_passrate)
    needed = {"n_responses", "pass_rate", "benchmark_id", "content_chars", "language"}
    missing = needed.difference(items.columns)
    if missing:
        raise ValueError(f"item pass-rate table lacks columns: {sorted(missing)}")
    with PdfPages(output) as pdf:
        guide_page(pdf)
        matrix_and_filter_page(pdf, args.matrix_sample, args.filter_retention)
        support_and_difficulty_page(pdf, items)
        language_and_length_page(pdf, items)
        classic_irt_page(pdf, args.classic_irt_dir)
        archival_neural_prior_page(pdf, args.item_passrate, args.prediction_dir)

    # These pages are rendered afresh with the same TEXTWIDTH, palette, and serif style.
    with tempfile.TemporaryDirectory(prefix="aims-graph-audit-") as directory:
        graph_pdf = Path(directory) / "graph-audit.pdf"
        functions_pdf = Path(directory) / "function-realisations.pdf"
        observed_functions_pdf = Path(directory) / "observed-function-realisations.pdf"
        render_graph_audit(args.graph_audit_dir, graph_pdf)
        append_pdf(output, graph_pdf)
        # Keep the conceptual sketch next to the observed held-out diagnostic.
        with PdfPages(functions_pdf) as pdf:
            function_realisations_page(pdf)
        append_pdf(output, functions_pdf)
        with PdfPages(observed_functions_pdf) as pdf:
            observed_benchmark_holdout_curves_page(pdf, args.bhold_responses)
        append_pdf(output, observed_functions_pdf)

    print(output)
    print("pages", len(PdfReader(str(output)).pages))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DOCS / "final-data-analysis.pdf")
    parser.add_argument("--matrix-sample", type=Path, default=DOCS / "response-matrix-before-after.npz")
    parser.add_argument("--filter-retention", type=Path, required=True)
    parser.add_argument("--item-passrate", type=Path, required=True)
    parser.add_argument("--classic-irt-dir", type=Path, required=True)
    parser.add_argument("--graph-audit-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True,
                        help="directory with cnp_gelu-items.parquet, cnp_ln_gelu-items.parquet, and bcirt_d4_gelu_all-items.parquet")
    parser.add_argument("--bhold-responses", type=Path, required=True,
                        help="filtered benchmark-holdout response parquet; used only for the post-hoc curve diagnostic")
    args = parser.parse_args()
    for path in (args.matrix_sample, args.filter_retention, args.item_passrate, args.classic_irt_dir, args.graph_audit_dir, args.prediction_dir, args.bhold_responses):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    build(args)


if __name__ == "__main__":
    main()
