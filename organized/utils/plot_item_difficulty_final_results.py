#!/usr/bin/env python3
"""Item difficulty: observed pass rates by split, and what each ANP believes.

Observed difficulty is an item's pass rate: the share of subjects answering it
correctly, kept for items with at least ``MIN_N`` responses. A model's belief
about an item is its prediction with an empty context (budget 0), when nothing
about the subject is known and the item embedding is the only input. That is
the cleanest readout of learned item difficulty; it is a diagnostic, not a
calibrated forecast for any one subject.

Every model here is a benchmark-holdout ANP (w64, bge_m3_long): validation and
test benchmarks were never trained on. If a model learned difficulty that
transfers, its predictions should spread like the observed pass rates and
correlate with them on validation and test, not only on train.

Histograms are filled at low opacity so overlapping distributions stay legible.
The computed table is cached; pass the data paths once, or ``--refresh``::

    python organized/utils/plot_item_difficulty_final_results.py \\
      --responses   <export>/filtered-responses.parquet \\
      --embeddings  <export>/datasets/full/allbench-embed-bge_m3_long-v2 \\
      --checkpoints <dir with the six .pt files>
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
from plotstyle import CYCLE, INK, SUBTLE, TEXTWIDTH, style  # noqa: E402

CACHE = ROOT / "organized" / "training" / "results" / "item-difficulty-b0.parquet"
MIN_N = 5
BINS = np.linspace(0, 1, 26)
SPLITS = [("train", "train (11 benchmarks)"), ("validation", "validation"), ("test", "test")]
HELD_OUT = ["ai2d_test", "bird_sql", "financebench", "algotune", "cmmlu", "editbench"]
TRAIN_GREY, SPLIT_BLUE = "#9A9A9A", "#4477AA"

# (substring of the checkpoint file, display name)
MODELS = [
    ("filtered-benchmarkholdout", NAME["w64-filtered"]),
    ("bhold-budgetladder", NAME["w64-ladder"]),
    ("bhold-sam005-unweighted", NAME["w64-sam"]),
    ("bhold-anchor-unweighted", NAME["w64-anchor"]),
    ("sam005-alcweighted", NAME["w64-sam-ladder"]),
    ("denseprefix-unweighted-best-step4550", NAME["w64-dense"]),
]
# Same architecture family trained on the random-subject split: their
# validation subjects come from benchmarks they also trained on. Both trained
# on exactly these seven benchmarks (read from their evaluation outputs), so an
# item here is "seen" if its benchmark is in the set, whatever its panel split.
RANDOM_MODELS = [
    ("anp-sa1-w64-bgem3-20261028", NAME["w64-contam"]),
    ("anp128-bgem3-l40s-100k", NAME["w128"]),
]
SEEN_BY_RANDOM = {"afrieval", "ai2d_test", "alignment_faking", "bird_sql",
                  "bridging_gap", "editbench", "hallusionbench"}
ALL_MODELS = MODELS + RANDOM_MODELS
COLOR = {name: CYCLE[(i + 2) % len(CYCLE)] for i, (_, name) in enumerate(ALL_MODELS)}


def exact_page(fig) -> Bbox:
    """Pin the saved box to the figure: plotstyle's default "tight" box grows
    around overflowing text and silently widens a page past TEXTWIDTH."""
    return Bbox.from_bounds(0, 0, *fig.get_size_inches())


# ------------------------------------------------------------ computation --
def build_table(responses: Path, embeddings: Path, checkpoints: Path) -> pd.DataFrame:
    import torch

    sys.path.insert(0, str(ROOT / "organized" / "training"))
    import anp_crossattn_blocks as blocks

    d = pd.read_parquet(responses, columns=["item_uid", "benchmark_id", "benchmark_split", "label"])
    items = (d.groupby(["item_uid", "benchmark_id", "benchmark_split"], as_index=False)
             .agg(observed=("label", "mean"), n=("label", "size")))
    if items.item_uid.duplicated().any():
        raise RuntimeError("an item appears in two splits; difficulty would be double-counted")
    items = items[items.n >= MIN_N].reset_index(drop=True)

    ids = pd.read_parquet(embeddings / "items.parquet")["item_uid"].astype(str).to_numpy()
    store = np.load(embeddings / "item_embeddings.npy", mmap_mode="r")
    position = {uid: i for i, uid in enumerate(ids)}
    rows = np.array([position[u] for u in items.item_uid.astype(str)])
    x = torch.as_tensor(np.asarray(store[rows], dtype=np.float32))
    empty_x, empty_y = torch.zeros((0, x.shape[1])), torch.zeros(0)

    for key, name in ALL_MODELS:
        match = sorted(checkpoints.glob(f"*{key}*.pt"))
        match = [m for m in match if not m.name.endswith(".resume.pt")]
        if len(match) != 1:
            raise FileNotFoundError(f"{name}: expected one checkpoint matching {key}, got {match}")
        ck = torch.load(match[0], map_location="cpu", weights_only=False)
        c = ck["config"]
        model = blocks.AttentiveNeuralProcess(
            x.shape[1], width=int(c["width"]), self_attention_blocks=int(c["n_blocks"]),
            cross_attention_heads=int(c["cross_attention_heads"]),
            use_latent=bool(c["use_latent"]), latent_dim=int(c["latent_dim"]),
            residual_proj=bool(c.get("residual_proj", False)),
            dropout=float(c.get("dropout", 0.0)),
            # Without this the anchored model would drop its prior offset and
            # report a different item belief than the one it was trained with.
            prior_anchor=bool(c.get("prior_anchor", False)),
        )
        model.load_state_dict(ck["state_dict"], strict=True)
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(x), 8192):
                out.append(torch.sigmoid(model.logits(empty_x, empty_y, x[start:start + 8192])))
        items[name] = torch.cat(out).numpy().astype(np.float64)
        print(f"  {name:<26} step {ck.get('selection', {}).get('step')}  "
              f"mean {items[name].mean():.3f}  sd {items[name].std():.3f}")
    items["seen_by_random_split"] = items.benchmark_id.isin(SEEN_BY_RANDOM)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    items.to_parquet(CACHE, index=False)
    return items


# ------------------------------------------------------------------ pages --
def header(fig, title: str, subtitle: str) -> None:
    fig.text(0.018, 0.985, title, fontsize=9, fontweight="bold", va="top")
    fig.text(0.018, 0.945, subtitle, fontsize=6.3, color=SUBTLE, va="top")


def filled(ax, values, color, label, alpha=0.35):
    """Low-opacity fill plus a crisp outline, so overlaps read as overlaps."""
    ax.hist(values, bins=BINS, density=True, histtype="stepfilled", color=color,
            alpha=alpha, lw=0, label=label)
    ax.hist(values, bins=BINS, density=True, histtype="step", color=color, lw=0.8)


def splits_page(pdf, t: pd.DataFrame):
    fig = plt.figure(figsize=(TEXTWIDTH, 5.4))
    header(fig, "Observed item difficulty by split",
           f"Pass rate = share of subjects answering the item correctly; items with ≥{MIN_N} "
           "responses. Lower means harder.")
    ax = fig.add_axes([0.08, 0.56, 0.90, 0.32])
    palette = {"train": TRAIN_GREY, "validation": SPLIT_BLUE, "test": CYCLE[1]}
    for split, label in SPLITS:
        v = t[t.benchmark_split == split].observed
        filled(ax, v, palette[split], f"{split}: {len(v):,} items, mean {v.mean():.2f}")
    ax.set(xlim=(0, 1), xlabel="observed pass rate", ylabel="density")
    style(ax)
    ax.legend(loc="upper left", fontsize=6, frameon=False)

    train = t[t.benchmark_split == "train"].observed
    for i, bench in enumerate(HELD_OUT):
        ax = fig.add_axes([0.08 + (i % 3) * 0.31, 0.28 - (i // 3) * 0.215, 0.26, 0.145])
        v = t[t.benchmark_id == bench].observed
        split = t[t.benchmark_id == bench].benchmark_split.iloc[0]
        filled(ax, train, TRAIN_GREY, "train")
        filled(ax, v, palette[split], bench)
        ax.set_title(f"{bench} ({split}), {len(v):,} items", fontsize=6.3, pad=2)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=5.5)
        style(ax)
    fig.text(0.08, 0.475, "Each held-out benchmark against the pooled training distribution (grey).",
             fontsize=6.3, color=INK)
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def summary_page(pdf, t: pd.DataFrame):
    fig = plt.figure(figsize=(TEXTWIDTH, 6.6))
    header(fig, "Does the model's item belief track observed difficulty?",
           "Budget-0 prediction (empty context) vs observed pass rate, per item. "
           "r Pearson, ρ Spearman, sd = spread.")
    cols = ["model", "split", "items", "r", "ρ", "mean pred", "mean obs", "sd pred", "sd obs"]
    rows = []
    for _, name in ALL_MODELS:
        for split, _ in SPLITS:
            s = t[t.benchmark_split == split]
            rows.append([name, split, f"{len(s):,}",
                         f"{s[name].corr(s.observed):+.3f}",
                         f"{s[name].corr(s.observed, method='spearman'):+.3f}",
                         f"{s[name].mean():.3f}", f"{s.observed.mean():.3f}",
                         f"{s[name].std():.3f}", f"{s.observed.std():.3f}"])
    ax = fig.add_axes([0.01, 0.10, 0.98, 0.80])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, bbox=[0, 0, 1, 1], cellLoc="center",
                     colWidths=[0.24, 0.11, 0.08, 0.08, 0.08, 0.105, 0.105, 0.10, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(5.0)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(weight="bold")
        if c == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.03
        if r > 0 and (r - 1) % 3 == 0:
            cell.set_facecolor("#F4F4F4")
        if r > 3 * len(MODELS):  # random-split rows set apart
            cell.set_text_props(style="italic")
    import textwrap

    note = ("A model that learned transferable difficulty would keep r and its spread on "
            "validation and test; one that memorised its training benchmarks keeps them only "
            "on train. Italic rows: random-subject split, which also trained on ai2d_test and "
            "bird_sql (validation) and editbench (test); see the seen-vs-unseen page.")
    for i, line in enumerate(textwrap.wrap(note, 125)):
        fig.text(0.018, 0.055 - i * 0.025, line, fontsize=5.8, color=SUBTLE)
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def model_page(pdf, t: pd.DataFrame, name: str, random_split: bool = False):
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.85), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.965, top=0.70, bottom=0.30, wspace=0.12)
    header(fig, f"{name}: predicted vs observed item difficulty",
           ("Random-subject split: trained on 7 benchmarks incl. ai2d_test, bird_sql, editbench. "
            if random_split else "") +
           "Grey: train obs. Blue: this split obs. Colour: budget-0 prediction.")
    train = t[t.benchmark_split == "train"].observed
    for ax, (split, label) in zip(axes, SPLITS):
        s = t[t.benchmark_split == split]
        filled(ax, train, TRAIN_GREY, "observed, train")
        if split != "train":
            # One legend serves all three panels, so the label must hold on each.
            filled(ax, s.observed, SPLIT_BLUE, "observed, this split")
        filled(ax, s[name], COLOR[name], "model prediction", alpha=0.45)
        ax.set_title(f"{label}   r = {s[name].corr(s.observed):+.2f}", fontsize=6.6, pad=3)
        ax.set(xlim=(0, 1), xlabel="pass rate / predicted p")
        ax.tick_params(labelsize=6)
        style(ax)
    axes[0].set_ylabel("density")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def overlay_page(pdf, t: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 3.4), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.965, top=0.80, bottom=0.30, wspace=0.12)
    header(fig, "All holdout models' item beliefs, by split",
           "Grey: observed pass rate on that split. Colours: each model's budget-0 "
           "prediction, filled at low opacity.")
    for ax, (split, label) in zip(axes, SPLITS):
        s = t[t.benchmark_split == split]
        filled(ax, s.observed, TRAIN_GREY, "observed")
        for _, name in MODELS:
            filled(ax, s[name], COLOR[name], name, alpha=0.22)
        ax.set_title(label, fontsize=6.8, pad=3)
        ax.set(xlim=(0, 1), xlabel="pass rate / predicted p")
        ax.tick_params(labelsize=6)
        style(ax)
    axes[0].set_ylabel("density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=5.8, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def seen_page(pdf, t: pd.DataFrame):
    """Random-split models on benchmarks they trained on vs ones they never saw.

    Same models, same items' observed difficulty; the only difference is
    whether the item's benchmark was in training. A gap in r here is the
    memorisation the benchmark holdout exists to remove.
    """
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 4.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.965, top=0.83, bottom=0.16, wspace=0.12, hspace=0.40)
    header(fig, "Random-split models: benchmarks seen in training vs never seen",
           "Grey: observed. Colour: budget-0 prediction. Seen = benchmark "
           "in the 7 trained on.")
    for row, (_, name) in enumerate(RANDOM_MODELS):
        for col, (seen, label) in enumerate(((True, "seen in training"), (False, "never seen"))):
            ax = axes[row][col]
            s = t[t.seen_by_random_split == seen]
            filled(ax, s.observed, TRAIN_GREY, "observed")
            filled(ax, s[name], COLOR[name], "model prediction", alpha=0.45)
            r = s[name].corr(s.observed)
            ax.set_title(f"{name}: {label}\n{len(s):,} items, r = {r:+.2f}, "
                         f"sd pred {s[name].std():.2f} vs obs {s.observed.std():.2f}",
                         fontsize=6.2, pad=3)
            ax.set_xlim(0, 1)
            ax.tick_params(labelsize=6)
            style(ax)
        axes[row][0].set_ylabel("density", fontsize=6.6)
    for ax in axes[1]:
        ax.set_xlabel("pass rate / predicted p", fontsize=6.6)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, ["observed", "model prediction"], loc="lower center", ncol=2,
               fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.0))
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "item_difficulty-final-results.pdf")
    parser.add_argument("--png-dir", type=Path, default=None)
    args = parser.parse_args()

    if CACHE.exists() and not args.refresh:
        table = pd.read_parquet(CACHE)
    else:
        if not (args.responses and args.embeddings and args.checkpoints):
            raise SystemExit("no cache yet: pass --responses, --embeddings and --checkpoints")
        table = build_table(args.responses, args.embeddings, args.checkpoints)
    print(f"{len(table):,} items: " + ", ".join(
        f"{s} {int((table.benchmark_split == s).sum()):,}" for s, _ in SPLITS))

    pages = []
    with PdfPages(args.out) as pdf:
        pages.append(("01-splits", splits_page(pdf, table)))
        pages.append(("02-summary", summary_page(pdf, table)))
        pages.append(("03-overlay", overlay_page(pdf, table)))
        for i, (_, name) in enumerate(MODELS, start=4):
            pages.append((f"{i:02d}-{name.replace(' ', '_').replace(',', '')}",
                          model_page(pdf, table, name)))
        pages.append((f"{4 + len(MODELS):02d}-random-split-seen-vs-unseen", seen_page(pdf, table)))
        for i, (_, name) in enumerate(RANDOM_MODELS, start=5 + len(MODELS)):
            pages.append((f"{i:02d}-{name.replace(' ', '_').replace(',', '')}",
                          model_page(pdf, table, name, random_split=True)))
    if args.png_dir:
        args.png_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in pages:
            fig.savefig(args.png_dir / f"{name}.png", dpi=120, bbox_inches=exact_page(fig))
    for _, fig in pages:
        plt.close(fig)
    print(f"wrote {args.out} ({len(pages)} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
