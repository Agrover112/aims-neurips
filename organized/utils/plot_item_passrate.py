"""Distribution of raw per-item pass rate, no splits.

Input is the table built by ``modal_item_passrate.py``. Pass rate is the share
of correct responses; lower means harder. An item answered once has a pass rate
of exactly 0 or 1, so the main panels use items with at least ``MIN_N``
responses and the single-response spike is shown on its own.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

MIN_N = 5
SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
t = pd.read_parquet(SRC)
good = t[t["n_responses"] >= MIN_N]
GREY, BLUE, RED = "#7f8c9b", "#2b6ca3", "#c0392b"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


pdf = PdfPages(OUT)

# ---- page 1: global --------------------------------------------------------
fig = plt.figure(figsize=(11, 8.5))
fig.suptitle("Item pass rate, all benchmarks together", fontsize=14, fontweight="bold",
             x=0.055, ha="left", y=0.965)
fig.text(0.055, 0.928, f"{len(t):,} items with at least one response. Pass rate = share of "
         "responses that were correct (low = hard). No splits used.", fontsize=8.5,
         color="#55606b")
ax = fig.add_axes([0.07, 0.56, 0.40, 0.31])
ax.hist(t["pass_rate"], bins=50, color=GREY)
ax.set_title("A. All items (spikes at 0 and 1 are single-response items)", fontsize=9.5,
             loc="left", fontweight="bold")
ax.set_xlabel("pass rate"); ax.set_ylabel("items"); style(ax)
ax = fig.add_axes([0.57, 0.56, 0.38, 0.31])
for k, c in ((5, BLUE), (10, RED)):
    ax.hist(t[t["n_responses"] >= k]["pass_rate"], bins=40, histtype="step", lw=2, color=c,
            label=f">= {k} responses  (n={int((t.n_responses >= k).sum()):,})", density=True)
ax.set_title("B. Items with enough responses to trust", fontsize=9.5, loc="left",
             fontweight="bold")
ax.set_xlabel("pass rate"); ax.set_ylabel("density"); ax.legend(fontsize=8, frameon=False)
style(ax)
ax = fig.add_axes([0.07, 0.10, 0.40, 0.31])
ks = [1, 2, 5, 10, 20, 50]
ax.bar([str(k) for k in ks], [(t.n_responses >= k).mean() for k in ks], color=GREY)
for i, k in enumerate(ks):
    ax.text(i, (t.n_responses >= k).mean() + 0.015, f"{(t.n_responses >= k).sum():,}",
            ha="center", fontsize=7.5)
ax.set_title("C. How many items have at least k responses", fontsize=9.5, loc="left",
             fontweight="bold")
ax.set_xlabel("minimum responses k"); ax.set_ylabel("share of items"); ax.set_ylim(0, 1.1)
style(ax)
ax = fig.add_axes([0.57, 0.10, 0.38, 0.31])
q = good["pass_rate"].quantile([.05, .25, .5, .75, .95])
bands = [("hard  (<0.2)", (good.pass_rate < .2).mean()),
         ("0.2-0.5", ((good.pass_rate >= .2) & (good.pass_rate < .5)).mean()),
         ("0.5-0.8", ((good.pass_rate >= .5) & (good.pass_rate < .8)).mean()),
         ("easy (>=0.8)", (good.pass_rate >= .8).mean())]
ax.barh([b[0] for b in bands][::-1], [b[1] for b in bands][::-1], color=BLUE)
for i, b in enumerate(bands[::-1]):
    ax.text(b[1] + 0.01, i, f"{b[1]:.0%}", va="center", fontsize=8)
ax.set_title(f"D. Difficulty bands (items with >= {MIN_N} responses)", fontsize=9.5,
             loc="left", fontweight="bold")
ax.set_xlabel("share of items"); ax.set_xlim(0, 0.6); style(ax)
fig.text(0.055, 0.03, f"Median pass rate {good.pass_rate.median():.2f}; middle 50% of items "
         f"between {q[.25]:.2f} and {q[.75]:.2f}; 5th-95th percentile {q[.05]:.2f}-{q[.95]:.2f}.",
         fontsize=8, color="#55606b")
pdf.savefig(fig); plt.close(fig)

# ---- page 2: per benchmark -------------------------------------------------
stats = (good.groupby("benchmark_id")["pass_rate"].agg(["median", "mean", "size"])
         .sort_values(["median", "mean"]))
order = list(stats.index)
fig = plt.figure(figsize=(11, 8.5))
fig.suptitle("Item pass rate per benchmark", fontsize=14, fontweight="bold", x=0.055,
             ha="left", y=0.965)
fig.text(0.055, 0.928, f"Items with >= {MIN_N} responses. Box = middle 50%, line = median, "
         "whiskers 5th-95th percentile. Label shows qualifying items.", fontsize=8.5,
         color="#55606b")
ax = fig.add_axes([0.24, 0.06, 0.72, 0.84])
data = [good[good.benchmark_id == b]["pass_rate"].to_numpy() for b in order]
bp = ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False, patch_artist=True,
                medianprops={"color": "white", "lw": 1.2}, widths=0.65)
for patch in bp["boxes"]:
    patch.set_facecolor(BLUE); patch.set_edgecolor("white")
ax.set_yticks(range(1, len(order) + 1))
ax.set_yticklabels([f"{b}  (n={int(stats.loc[b, 'size']):,})" for b in order], fontsize=6.3)
ax.set_xlim(-0.02, 1.02); style(ax)
ax.invert_yaxis()
ax.set_xlabel("pass rate   (top = hardest, bottom = easiest; sorted by median, ties by mean)")
pdf.savefig(fig); plt.close(fig)

# ---- page 3: per language --------------------------------------------------
lang = good.groupby("language")["pass_rate"].agg(["median", "mean", "size"])
lang = lang[lang["size"] >= 150].sort_values(["median", "mean"])
fig = plt.figure(figsize=(11, 8.5))
fig.suptitle("Item pass rate per language (all benchmarks pooled)", fontsize=14,
             fontweight="bold", x=0.055, ha="left", y=0.965)
fig.text(0.055, 0.928, f"Items with >= {MIN_N} responses; languages with >= 150 such items. "
         "Language is a heuristic: a lang= tag, else dominant non-Latin script, else "
         "English/Latin.", fontsize=8.5, color="#55606b")
ax = fig.add_axes([0.20, 0.07, 0.76, 0.82])
data = [good[good.language == l]["pass_rate"].to_numpy() for l in lang.index]
bp = ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False, patch_artist=True,
                medianprops={"color": "white", "lw": 1.2}, widths=0.65)
for patch, l in zip(bp["boxes"], lang.index):
    patch.set_facecolor(RED if l != "English/Latin" else GREY); patch.set_edgecolor("white")
ax.set_yticks(range(1, len(lang) + 1))
ax.set_yticklabels([f"{l}  (n={int(lang.loc[l, 'size']):,})" for l in lang.index], fontsize=8)
ax.set_xlim(-0.02, 1.02); style(ax)
ax.invert_yaxis()
ax.set_xlabel("pass rate   (top = hardest, bottom = easiest; sorted by median, ties by mean)")
pdf.savefig(fig); plt.close(fig)

# ---- page 4: language x benchmark -----------------------------------------
cell = (good.groupby(["language", "benchmark_id"])["pass_rate"]
        .agg(["median", "size"]).reset_index())
cell = cell[(cell["size"] >= 50) & (cell["language"] != "English/Latin")]
cell = cell.sort_values(["language", "median"])
fig = plt.figure(figsize=(11, 8.5))
fig.suptitle("Item pass rate per language per benchmark", fontsize=14, fontweight="bold",
             x=0.055, ha="left", y=0.965)
fig.text(0.055, 0.928, "Non-English cells with >= 50 items having >= 5 responses. Most languages "
         "live in a single benchmark, so language and benchmark are tangled.", fontsize=8.5,
         color="#55606b")
ax = fig.add_axes([0.30, 0.07, 0.66, 0.82])
labels = [f"{r.language} | {r.benchmark_id}  (n={int(r.size):,})" for r in cell.itertuples()]
data = [good[(good.language == r.language) & (good.benchmark_id == r.benchmark_id)]
        ["pass_rate"].to_numpy() for r in cell.itertuples()]
bp = ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False, patch_artist=True,
                medianprops={"color": "white", "lw": 1.2}, widths=0.65)
for patch in bp["boxes"]:
    patch.set_facecolor(RED); patch.set_edgecolor("white")
ax.set_yticks(range(1, len(labels) + 1)); ax.set_yticklabels(labels, fontsize=7)
ax.set_xlim(-0.02, 1.02); ax.set_xlabel("pass rate"); style(ax)
pdf.savefig(fig); plt.close(fig)
# ---- page 5: does the answerer mix explain pass rate? ----------------------
if len(sys.argv) > 3:
    a = pd.read_parquet(sys.argv[3])
    a = a[a["n_responses"] >= MIN_N].dropna(subset=["answerer_acc"]).copy()
    for c in ("pass_rate", "answerer_acc"):
        a[c + "_w"] = a[c] - a.groupby("benchmark_id")[c].transform("mean")
    r_all = np.corrcoef(a.pass_rate, a.answerer_acc)[0, 1]
    r_in = np.corrcoef(a.pass_rate_w, a.answerer_acc_w)[0, 1]
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("How much of an item's pass rate is just who answered it?", fontsize=14,
                 fontweight="bold", x=0.055, ha="left", y=0.965)
    fig.text(0.055, 0.928, "Answerer strength = average accuracy of the subjects that answered "
             "the item, measured on their OTHER items (the item cannot predict itself).\n"
             f"Items with >= {MIN_N} responses, n={len(a):,}. Colour in A = number of items, "
             "log scale.", fontsize=8.5, color="#55606b", va="top")
    ax = fig.add_axes([0.07, 0.55, 0.36, 0.30])
    hb = ax.hexbin(a.answerer_acc, a.pass_rate, gridsize=40, bins="log", cmap="Blues",
                   mincnt=1)
    ax.plot([0, 1], [0, 1], color=RED, lw=1, ls="--")
    ax.set_xlabel("answerer strength"); ax.set_ylabel("item pass rate")
    ax.set_title(f"A. All items   (correlation {r_all:.2f})", fontsize=9.5, loc="left",
                 fontweight="bold")
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02); style(ax)
    ax = fig.add_axes([0.62, 0.55, 0.33, 0.30])
    bins = pd.qcut(a.answerer_acc_w, 20, duplicates="drop")
    m = a.groupby(bins, observed=True)[["answerer_acc_w", "pass_rate_w"]].mean()
    ax.scatter(a.answerer_acc_w.sample(20000, random_state=0),
               a.pass_rate_w.sample(20000, random_state=0), s=2, color=GREY, alpha=0.25)
    ax.plot(m.answerer_acc_w, m.pass_rate_w, color=RED, lw=2, marker="o", ms=4,
            label="mean per bin")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("answerer strength minus its benchmark average")
    ax.set_ylabel("pass rate minus its benchmark average")
    ax.set_title(f"B. Within a benchmark   (correlation {r_in:.2f}, R^2 {r_in ** 2:.2f})",
                 fontsize=9.5, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False); style(ax)
    ax = fig.add_axes([0.07, 0.17, 0.88, 0.28])
    bm = a.groupby("benchmark_id")[["pass_rate", "answerer_acc"]].median()
    bm = bm.sort_values("pass_rate")
    xs = np.arange(len(bm))
    ax.bar(xs, bm.pass_rate, color=BLUE, width=0.8, label="median item pass rate")
    ax.scatter(xs, bm.answerer_acc, color=RED, s=14, zorder=3,
               label="median answerer strength")
    ax.set_xticks(xs); ax.set_xticklabels(bm.index, rotation=90, fontsize=6.5)
    ax.set_ylabel("rate"); ax.set_ylim(0, 1.02)
    ax.set_title("C. Per benchmark: where the red dot sits far from the bar, the answerer mix "
                 "and the items disagree", fontsize=9.5, loc="left", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper left"); style(ax)
    pdf.savefig(fig); plt.close(fig)
    print(f"answerer-strength: corr {r_all:.3f} overall, {r_in:.3f} within benchmark "
          f"(R^2 {r_in ** 2:.3f}); overall R^2 {r_all ** 2:.3f}")

pdf.close()

# console summary for the reply
print(f"good items {len(good):,} / {len(t):,}")
print("global median", round(good.pass_rate.median(), 3))
print("benchmarks hardest:", [(b, round(stats.loc[b, 'median'], 2)) for b in order[:3]])
print("benchmarks easiest:", [(b, round(stats.loc[b, 'median'], 2)) for b in order[-3:]])
print("languages (median):", {l: round(lang.loc[l, 'median'], 2) for l in lang.index})
print("spread of benchmark medians:", round(stats['median'].std(), 3),
      " within-benchmark IQR mean:",
      round(good.groupby('benchmark_id')['pass_rate'].apply(lambda s: s.quantile(.75) - s.quantile(.25)).mean(), 3))
