"""Render the response-spectrum and Qwen semantic-graph appendix PDF.

Usage:
    python organized/utils/plot_data_structure_audit.py <audit_dir> <out.pdf>
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import INK, MUTED, SEQUENTIAL, SUBTLE, TEXTWIDTH, style  # noqa: E402


def _header(fig, title, subtitle):
    """A slightly roomier report header than the default plotstyle helper."""
    fig.suptitle(title, fontsize=10, fontweight="bold", x=.02, ha="left", y=.985)
    # This is a fixed-width report rather than a crop-to-content paper figure.
    # Wrapping avoids the otherwise subtle right-edge truncation in a PDF viewer.
    fig.text(.02, .935, textwrap.fill(subtitle, width=112), fontsize=7.2, color=SUBTLE,
             va="top", linespacing=1.15)


def _heatmap(ax, values, labels, title, value_format, vmin=None, vmax=None):
    image = ax.imshow(values, cmap=SEQUENTIAL, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    ax.set_xlabel("neighbour benchmark")
    ax.set_ylabel("source benchmark")
    ax.set_title(title, loc="left", fontweight="bold")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(col, row, value_format.format(value), ha="center", va="center", fontsize=5.2,
                    color="white" if image.norm(value) > .56 else INK)
    return image


def _page_data(pdf, table, audit):
    fig = plt.figure(figsize=(TEXTWIDTH, 3.55))
    _header(fig, "What is analysed in this appendix",
            "Two different objects are kept separate: observed binary responses and frozen item-text embeddings.")
    ax = fig.add_axes([.10, .19, .42, .58])
    shown = table.sort_values("responses", ascending=True)
    size = 12 + 8 * np.sqrt(shown["responses"] / shown["responses"].max())
    scatter = ax.scatter(shown["subjects"], shown["items"], s=size**2, c=shown["pass_rate"], cmap=SEQUENTIAL,
                         edgecolor=INK, linewidth=.35)
    offsets = {"alignment_faking": (-76, 3), "bridging_gap": (3, 3), "afrieval": (3, 3),
               "ai2d_test": (3, 3), "hallusionbench": (3, 3), "bird_sql": (3, 3), "editbench": (3, 3)}
    for _, row in shown.iterrows():
        ax.annotate(row["benchmark_id"], (row["subjects"], row["items"]), xytext=offsets[row["benchmark_id"]],
                    textcoords="offset points", fontsize=5.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("subjects in benchmark"); ax.set_ylabel("items in benchmark")
    ax.set_title("(a) Filtered response matrices", loc="left", fontweight="bold")
    fig.colorbar(scatter, ax=ax, fraction=.047, pad=.02, label="mean correct response")
    style(ax, "both")
    ax = fig.add_axes([.66, .18, .30, .59]); ax.axis("off")
    blocks = [
        ("Response graph", "A bipartite edge joins a subject to an item only when that binary response was observed. Missing responses are not imputed."),
        ("Qwen item graph", "Each filtered item connects to its 20 nearest 1024-d Qwen vectors by cosine similarity. It uses question text only."),
        ("Why separate them", "Response structure needs labels and is training-only. Qwen edges can be built for a new item before its label is known."),
    ]
    y = .98
    for title, body in blocks:
        ax.text(0, y, title, fontsize=8, weight="bold", va="top")
        ax.text(0, y - .075, textwrap.fill(body, width=36), fontsize=7.0, va="top")
        y -= .33
    fig.text(.06, .045, f"Filtering: {audit['raw_rows']:,} raw rows to {audit['final_rows']:,}; item support >= {audit['min_subjects_per_item']}, subject-task responses >= {audit['min_responses_per_subject_task']}.",
             fontsize=6.4, color=SUBTLE)
    pdf.savefig(fig); plt.close(fig)


def _page_semantic_graph(pdf, items, pairs):
    labels = sorted(items["benchmark_id"].unique())
    prob = pairs.pivot(index="source_benchmark", columns="target_benchmark", values="neighbour_probability").reindex(index=labels, columns=labels).fillna(0)
    cosine = pairs.pivot(index="source_benchmark", columns="target_benchmark", values="mean_cosine").reindex(index=labels, columns=labels).fillna(0)
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 5.25), gridspec_kw={"height_ratios": [1, .92]})
    _header(fig, "Qwen3 item similarity forms a content-only graph",
            "Each row contains the 20 nearest text-embedding neighbours of items from that benchmark. It is directed because each item chooses its own neighbours.")
    image = _heatmap(axes[0, 0], prob.to_numpy(), labels, "(a) Neighbour destination probability", "{:.0%}", 0, max(.01, prob.to_numpy().max()))
    fig.colorbar(image, ax=axes[0, 0], fraction=.045, pad=.02, label="share of 20 neighbours")
    image = _heatmap(axes[0, 1], cosine.to_numpy(), labels, "(b) Mean cosine similarity", "{:.2f}", cosine.to_numpy().min(), cosine.to_numpy().max())
    fig.colorbar(image, ax=axes[0, 1], fraction=.045, pad=.02, label="cosine")
    ax = axes[1, 0]
    summary = items.groupby("benchmark_id", as_index=False).agg(cross=("cross_benchmark_neighbour_fraction", "mean")).sort_values("cross")
    ax.barh(summary["benchmark_id"], 100 * summary["cross"], color=MUTED["teal"], edgecolor=INK, lw=.35)
    ax.set_xlabel("neighbours from another benchmark (%)")
    ax.set_title("(c) Cross-benchmark\nsemantic connectivity", loc="left", fontweight="bold", fontsize=7)
    style(ax, "x")
    ax = axes[1, 1]
    pairs = pairs.copy(); pairs["gap_reduction"] = pairs["random_pass_rate_gap"] - pairs["mean_pass_rate_gap"]
    gain = pairs.groupby("source_benchmark", as_index=False)["gap_reduction"].mean().sort_values("gap_reduction")
    ax.barh(gain["source_benchmark"], gain["gap_reduction"], color=[MUTED["green"] if x >= 0 else MUTED["rose"] for x in gain["gap_reduction"]], edgecolor=INK, lw=.35)
    ax.axvline(0, color=INK, lw=.6)
    ax.set_xlabel("random item gap minus semantic-neighbour gap")
    ax.set_title("(d) Are semantic neighbours closer\nin pass rate?", loc="left", fontweight="bold", fontsize=7)
    style(ax, "x")
    fig.text(.02, .012, "Panel (d): positive means Qwen neighbours have more similar observed pass rates than random items, after matching source and target benchmarks. It is descriptive, not a test-time score.", fontsize=6.35, color=SUBTLE)
    fig.tight_layout(rect=(0, .045, 1, .88), pad=.55)
    pdf.savefig(fig); plt.close(fig)


def _ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    return values, np.arange(1, len(values) + 1) / len(values)


def _page_graph_construction_and_degrees(pdf, response_degrees, items, graph_meta):
    fig = plt.figure(figsize=(TEXTWIDTH, 5.1))
    _header(fig, "How the two graphs are constructed", "The response graph and the Qwen kNN graph have different edges, degree distributions, and test-time status.")
    ax = fig.add_axes([.07, .63, .86, .20]); ax.axis("off")
    boxes = [
        (.01, "item text", "Qwen3-8B\nMRL 1024-d"),
        (.34, "cosine search", "all filtered\nitems"),
        (.67, "directed edge", r"$i \rightarrow j$ if $j \in \mathrm{TopK}_{20}(i)$"),
    ]
    for x, title, detail in boxes:
        ax.add_patch(FancyBboxPatch((x, .16), .24, .64, boxstyle="round,pad=.02", facecolor=MUTED["grey"], edgecolor=INK, lw=.6))
        ax.text(x + .12, .62, title, ha="center", va="center", weight="bold", fontsize=8)
        ax.text(x + .12, .34, detail, ha="center", va="center", fontsize=7)
    for x in (.27, .60):
        ax.add_patch(FancyArrowPatch((x, .48), (x + .065, .48), arrowstyle="-|>", mutation_scale=10, lw=.8, color=INK))
    ax.text(.01, -.12, r"Direction comes from the source item's own Top-K list. It need not be mutual: $i \rightarrow j$ can exist while $j \nrightarrow i$ does not.", fontsize=6.8, color=SUBTLE)

    ax = fig.add_axes([.10, .12, .35, .32])
    for node_type, color in (("subject", MUTED["teal"]), ("item", MUTED["rose"])):
        values = response_degrees.loc[response_degrees.node_type == node_type, "degree"]
        x, y = _ecdf(values)
        ax.step(x, y, where="post", label=node_type, color=color)
    ax.set_xscale("log"); ax.set_xlabel("response-graph degree")
    ax.set_ylabel("ECDF of nodes")
    ax.set_title("(a) Response graph degree", loc="left", fontweight="bold")
    ax.legend(loc="lower right"); style(ax)

    ax = fig.add_axes([.58, .12, .35, .32])
    x, y = _ecdf(items["in_degree"])
    ax.step(x, y, where="post", color=MUTED["indigo"], label="in-degree")
    ax.axvline(graph_meta["neighbours_per_item"], color=MUTED["sand"], ls="--", label="out-degree = 20")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Qwen kNN degree")
    ax.set_ylabel("ECDF of items")
    ax.set_title("(b) Directed Qwen graph degree", loc="left", fontweight="bold")
    ax.legend(loc="lower right"); style(ax)
    fig.text(.07, .035, f"Qwen graph: {graph_meta['directed_edges']:,} directed edges; {graph_meta['reciprocal_edge_fraction']:.1%} are reciprocal; {graph_meta['weak_components']} weakly connected component(s), {graph_meta['strong_components']} strongly connected component(s).", fontsize=6.5, color=SUBTLE)
    pdf.savefig(fig); plt.close(fig)


def _spectral_cut(frame, max_rank=16):
    shown = frame.sort_values("rank").iloc[:-1].copy()
    shown = shown[shown["rank"] <= max_rank]
    ratios = shown["eigenvalue"].to_numpy() / frame.sort_values("rank")["eigenvalue"].to_numpy()[1:len(shown) + 1]
    index = int(np.argmax(ratios))
    return int(shown.iloc[index]["rank"]), float(ratios[index])


def _spectrum_axis(ax, spectra, scope, show_cut=True):
    shown = spectra[spectra["scope"] == scope].sort_values("rank")
    ax.plot(shown["rank"], shown["eigenvalue"], marker="o", ms=2.1, color=MUTED["indigo"])
    cut, ratio = _spectral_cut(shown)
    if show_cut:
        ax.axvline(cut, color=MUTED["rose"], lw=.8, ls="--")
        ax.text(cut + .35, shown["eigenvalue"].max() * .88, rf"$k^*={cut}$", fontsize=6, color=MUTED["wine"])
    ax.set_yscale("log"); ax.set_xlabel(r"component rank $k$"); ax.set_ylabel(r"eigenvalue $\lambda_k=\sigma_k^2$")
    ax.set_title(scope, loc="left", fontweight="bold", fontsize=7)
    style(ax)
    return cut, ratio


def _page_spectra(pdf, spectra):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.8))
    _header(fig, "Response-matrix spectrum and its eigengap",
            r"For a rectangular response operator $R$, $\lambda_k=\sigma_k^2$ are eigenvalues of $R^\top R$. An eigengap is a candidate latent-rank cut, not a cluster count.")
    cut, ratio = _spectrum_axis(axes[0], spectra, "all benchmarks")
    ax = axes[1]
    shown = spectra[spectra["scope"] == "all benchmarks"].sort_values("rank").copy()
    ratio_values = shown["eigenvalue"].to_numpy()[:-1] / shown["eigenvalue"].to_numpy()[1:]
    ax.plot(shown["rank"].to_numpy()[:-1], ratio_values, marker="o", ms=2.1, color=MUTED["green"])
    ax.axvline(cut, color=MUTED["rose"], lw=.8, ls="--", label=rf"largest early drop: $k^*={cut}$")
    ax.set_xlabel(r"candidate cut $k$"); ax.set_ylabel(r"relative eigengap $\lambda_k/\lambda_{k+1}$")
    ax.set_title("(b) Relative eigengap", loc="left", fontweight="bold"); ax.legend(loc="upper right"); style(ax)
    fig.text(.02, .025, rf"Global largest relative drop among the first 16 values: $k^*={cut}$, $\lambda_{{k^*}}/\lambda_{{k^*+1}}={ratio:.2f}$. This is weak evidence for a 3-factor response approximation, not evidence of three graph clusters.", fontsize=6.5, color=SUBTLE)
    fig.tight_layout(rect=(0, .10, 1, .84), pad=.55)
    pdf.savefig(fig); plt.close(fig)
    scopes = [scope for scope in spectra["scope"].unique() if scope != "all benchmarks"]
    fig, axes = plt.subplots(3, 3, figsize=(TEXTWIDTH, 5.1))
    _header(fig, "Response-matrix spectrum by benchmark",
            "Every panel uses the same observed-only, item-centred, degree-normalised operator. The y-axis is logarithmic so spectral decay remains visible.")
    for ax, scope in zip(axes.flat, scopes, strict=False): _spectrum_axis(ax, spectra, scope)
    for ax in axes.flat[len(scopes):]: ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, .88), pad=.7)
    pdf.savefig(fig); plt.close(fig)


def _page_laplacian(pdf, laplacian, graph_meta):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.85))
    _header(fig, "Laplacian spectrum of the symmetrised Qwen kNN graph",
            "The graph symmetrises directed Top-20 cosine edges. Near-zero values count components; an eigengap is only a clustering heuristic.")
    values = laplacian.sort_values("rank")["eigenvalue"].to_numpy()
    ranks = np.arange(1, len(values) + 1)
    gaps = np.diff(values)
    cut = int(np.argmax(gaps) + 1)
    axes[0].plot(ranks, values, marker="o", ms=2.2, color=MUTED["indigo"])
    axes[0].axvline(cut, color=MUTED["rose"], lw=.8, ls="--")
    axes[0].set_xlabel(r"eigenvalue index $k$"); axes[0].set_ylabel(r"Laplacian eigenvalue $\mu_k$")
    axes[0].set_title("(a) Smallest Laplacian eigenvalues", loc="left", fontweight="bold"); style(axes[0])
    axes[1].bar(ranks[:-1], gaps, color=MUTED["teal"], edgecolor=INK, lw=.25)
    axes[1].axvline(cut, color=MUTED["rose"], lw=.8, ls="--", label=rf"largest gap after $k={cut}$")
    axes[1].set_xlabel(r"gap after index $k$"); axes[1].set_ylabel(r"$\mu_{k+1}-\mu_k$")
    axes[1].set_title("(b) Consecutive eigengaps", loc="left", fontweight="bold"); axes[1].legend(loc="upper right"); style(axes[1])
    fig.text(.02, .015, f"There are {graph_meta['weak_components']} weak components, so the first {graph_meta['weak_components']} eigenvalues should be near zero. The largest displayed gap is after k={cut}. This is a heuristic, not proof that k clusters are stable.", fontsize=6.5, color=SUBTLE)
    fig.tight_layout(rect=(0, .05, 1, .84), pad=.55)
    pdf.savefig(fig); plt.close(fig)


def _page_fiedler(pdf, fiedler, graph_meta):
    frame = fiedler.dropna(subset=["fiedler_value"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.0), gridspec_kw={"width_ratios": [1.08, .92]})
    _header(fig, "What the Fiedler vector can, and cannot, show",
            "It is the second-smallest-eigenvalue vector, computed inside the largest weak component; the disconnected global graph has no unique Fiedler vector.")
    labels = sorted(frame["benchmark_id"].unique())
    data = [frame.loc[frame.benchmark_id == label, "fiedler_value"].to_numpy() for label in labels]
    violin = axes[0].violinplot(data, showextrema=False, showmedians=True)
    for body in violin["bodies"]:
        body.set_facecolor(MUTED["teal"]); body.set_edgecolor(INK); body.set_alpha(.75)
    violin["cmedians"].set_color(INK)
    axes[0].set_xticks(np.arange(1, len(labels) + 1), labels, rotation=35, ha="right", fontsize=6)
    axes[0].set_ylabel(r"Fiedler coordinate $v_2(i)$")
    axes[0].set_title("(a) Fiedler values by benchmark", loc="left", fontweight="bold"); style(axes[0])
    axes[1].scatter(frame["fiedler_value"], frame["item_pass_rate"], s=2, alpha=.08,
                    color=MUTED["indigo"], rasterized=True)
    axes[1].set_xlabel(r"Fiedler coordinate $v_2(i)$")
    axes[1].set_ylabel("observed item pass rate")
    axes[1].set_title("(b) Geometry versus observed difficulty", loc="left", fontweight="bold"); style(axes[1])
    fig.text(.02, .015, f"Largest weak component: {graph_meta['largest_weak_component_items']:,} items. The sign of $v_2$ is arbitrary. Separation by benchmark means the graph retains benchmark structure; a smooth relation to pass rate would suggest a difficulty-relevant semantic direction.", fontsize=6.5, color=SUBTLE)
    fig.tight_layout(rect=(0, .05, 1, .84), pad=.55)
    pdf.savefig(fig); plt.close(fig)


def _page_reading_guide(pdf, items, pairs, spectra):
    fig = plt.figure(figsize=(TEXTWIDTH, 3.65))
    _header(fig, "How to read this before choosing an inductive bias", "This is a structural audit. It does not show that a graph model will improve the competition policy.")
    average_cross = items["cross_benchmark_neighbour_fraction"].mean()
    average_gain = (pairs["random_pass_rate_gap"] - pairs["mean_pass_rate_gap"]).mean()
    top_energy = spectra[(spectra["scope"] == "all benchmarks") & (spectra["rank"] == 1)]["energy_fraction"].iloc[0]
    rows = [
        ("Response spectrum", f"The top component has only {top_energy:.1%} of observed-residual energy.", "Do not make a global low-rank response matrix the main bias. Structure is likely benchmark- or task-specific."),
        ("Qwen graph", f"Only {average_cross:.1%} of nearest neighbours cross a benchmark boundary on average.", "The content graph has little cross-benchmark mixing. It needs a strict item-cold-start test."),
        ("Pass-rate check", f"Semantic neighbours reduce the matched random pass-rate gap by {average_gain:.3f} on average.", "This is a weak within-data association, not evidence of unseen-benchmark difficulty transfer."),
        ("Fair comparison", "Compare a set CNP to the same CNP plus frozen Qwen kNN message passing.", "Keep split, context budget, and acquisition fixed. Do not use response-derived edges at test time."),
    ]
    positions = [(0.06, .51), (.53, .51), (.06, .14), (.53, .14)]
    for (title, finding, action), (x, y) in zip(rows, positions, strict=True):
        ax = fig.add_axes([x, y, .40, .26]); ax.axis("off")
        ax.text(0, 1, title, fontsize=8, weight="bold", va="top")
        ax.text(0, .69, textwrap.fill(finding, width=44), fontsize=7, va="top")
        ax.text(0, .28, textwrap.fill("Action: " + action, width=44), fontsize=7, va="top", color=SUBTLE)
    pdf.savefig(fig); plt.close(fig)


def render(data_dir: Path, output: Path) -> None:
    table = pd.read_parquet(data_dir / "benchmarks.parquet")
    spectra = pd.read_parquet(data_dir / "response_spectra.parquet")
    items = pd.read_parquet(data_dir / "qwen_item_knn.parquet")
    pairs = pd.read_parquet(data_dir / "qwen_knn_by_benchmark_pair.parquet")
    response_degrees = pd.read_parquet(data_dir / "response_degrees.parquet")
    laplacian = pd.read_parquet(data_dir / "qwen_laplacian_spectrum.parquet")
    fiedler = pd.read_parquet(data_dir / "qwen_fiedler.parquet")
    audit = json.loads((data_dir / "audit.json").read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        _page_data(pdf, table, audit)
        _page_graph_construction_and_degrees(pdf, response_degrees, items, audit["qwen_graph"])
        _page_semantic_graph(pdf, items, pairs)
        _page_laplacian(pdf, laplacian, audit["qwen_graph"])
        _page_fiedler(pdf, fiedler, audit["qwen_graph"])


if __name__ == "__main__":
    if len(sys.argv) != 3: raise SystemExit("usage: plot_data_structure_audit.py <audit_dir> <out.pdf>")
    render(Path(sys.argv[1]), Path(sys.argv[2]))
