"""Pages 6+ of docs/model-results-visualization.pdf: model vs raw data.

Usage: plot_model_vs_data.py <data_dir> <base.pdf> <out.pdf>

``data_dir`` holds item-passrate.parquet, response-matrices.npz, the v4.1
``items.parquet``, ``eval_items.parquet`` and ``eval_orders.parquet`` files,
and, per model, ``<rung>-fig11.parquet`` and ``<rung>-items.parquet`` (from
modal_prediction_dump.py). The pages are appended to ``base.pdf``.
"""

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from pypdf import PdfReader, PdfWriter

DATA, BASE, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
RUNGS = {"cnp_ln_gelu": ("CNP + LayerNorm + GELU", "#2b6ca3"),
         "cnp_gelu": ("CNP + GELU  (selected)", "#d98324"),
         "bcirt_d4_gelu_all": ("Bayesian CNP-IRT, D=4", "#4b8b6f")}
GREY, DARK, RED = "#7f8c9b", "#33414e", "#c0392b"
VAL = {"cmmlu", "afrimedqa", "appworld", "cruxeval", "devbench"}
TEST = {"alpha_sql", "bertaqa", "causal_axioms", "cellverse", "dave", "disco"}
MATRIX_ROLES = [("train", ("ai2d_test", "hallusionbench")),
                ("validation", ("cmmlu", "afrimedqa")), ("test", ("bertaqa", "dave"))]


def role_of(bench):
    return "validation" if bench in VAL else ("test" if bench in TEST else "train")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def header(fig, title, sub):
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.055, ha="left", y=0.968)
    fig.text(0.055, 0.935, sub, fontsize=8.8, color="#44505c", va="top")


def note(fig, x, y, text, size=9, color=DARK, **kw):
    fig.text(x, y, text, fontsize=size, color=color, va="top", **kw)


items = pd.read_parquet(DATA / "item-passrate.parquet")
items["role"] = items["benchmark_id"].map(role_of)
good = items[items["n_responses"] >= 5]
buf = io.BytesIO()
pdf = PdfPages(buf)

# ------------------------------------------------ page 6: raw response matrices
mats = np.load(DATA / "response-matrices.npz")
fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
header(fig, "The raw response matrix, by role",
       "Each panel is one benchmark: rows = subjects (weakest at the top, strongest at the "
       "bottom), "
       "columns = items (hardest on the left). Blue = correct, red = wrong, white = not "
       "answered.\nSubjects with the most responses and items answered by the most subjects "
       "are shown. Roles are ours: train / validation / test benchmarks.")
cmap = ListedColormap(["#c0392b", "#2b6ca3"])
for col, (role, benches) in enumerate(MATRIX_ROLES):
    for row, bench in enumerate(benches):
        ax = axes[row, col]
        g = mats[bench]
        masked = np.ma.masked_where(g < 0, g)
        ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_facecolor("white")
        ax.set_title(f"{role.upper()}: {bench}", fontsize=9.5, fontweight="bold", loc="left")
        ax.set_xlabel(f"{g.shape[1]} items (hard -> easy)", fontsize=8)
        ax.set_ylabel(f"{g.shape[0]} subjects (weak -> strong)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.99, 1.02, f"pass {np.mean(g[g >= 0]):.0%}  missing {(g < 0).mean():.0%}",
                transform=ax.transAxes, ha="right", fontsize=7.5, color=DARK)
fig.subplots_adjust(left=0.05, right=0.97, top=0.84, bottom=0.07, hspace=0.32, wspace=0.12)
pdf.savefig(fig); plt.close(fig)

# --------------------------------------- page 7: item pass rate A, B (+ by role)
fig = plt.figure(figsize=(11, 8.5))
header(fig, "Item pass rate: A and B, and the same thing per role",
       "Pass rate = share of responses that were correct (low = hard). A and B are the global "
       "plots from the data analysis; C and D split the items by our train / validation / "
       "test roles.")
ax = fig.add_axes([0.07, 0.55, 0.40, 0.30])
ax.hist(items["pass_rate"], bins=50, color=GREY)
ax.set_title("A. All items (spikes = single-response items)", fontsize=9.5, loc="left",
             fontweight="bold")
ax.set_xlabel("pass rate"); ax.set_ylabel("items"); style(ax)
ax = fig.add_axes([0.57, 0.55, 0.38, 0.30])
for k, c in ((5, "#2b6ca3"), (10, RED)):
    ax.hist(items[items["n_responses"] >= k]["pass_rate"], bins=40, histtype="step", lw=2,
            color=c, density=True,
            label=f">= {k} responses (n={int((items.n_responses >= k).sum()):,})")
ax.set_title("B. Items with enough responses to trust", fontsize=9.5, loc="left",
             fontweight="bold")
ax.set_xlabel("pass rate"); ax.set_ylabel("density"); ax.legend(fontsize=8, frameon=False)
style(ax)
ax = fig.add_axes([0.07, 0.10, 0.40, 0.30])
for role, c in (("train", GREY), ("validation", "#d98324"), ("test", RED)):
    d = good[good["role"] == role]["pass_rate"]
    ax.hist(d, bins=30, histtype="step", lw=2.2, color=c, density=True,
            label=f"{role}  (n={len(d):,})")
ax.set_title("C. Items with >= 5 responses, by role", fontsize=9.5, loc="left",
             fontweight="bold")
ax.set_xlabel("pass rate"); ax.set_ylabel("density"); ax.legend(fontsize=8, frameon=False)
style(ax)
ax = fig.add_axes([0.57, 0.10, 0.38, 0.30])
rows = []
for role in ("train", "validation", "test"):
    a = items[items["role"] == role]
    b = a[a["n_responses"] >= 5]
    rows.append((role, len(a), len(b), a["n_responses"].eq(1).mean(), b["pass_rate"].mean(),
                 b["pass_rate"].std()))
ax.axis("off")
ax.text(0, 1, "D. Per role", fontsize=9.5, fontweight="bold", va="top")
ax.text(0, 0.86, f"{'role':<12}{'items':>10}{'>=5 resp':>10}{'single':>8}{'mean':>7}{'sd':>6}",
        fontsize=8.5, family="monospace", va="top")
for i, r in enumerate(rows):
    ax.text(0, 0.72 - i * 0.12,
            f"{r[0]:<12}{r[1]:>10,}{r[2]:>10,}{r[3]:>8.0%}{r[4]:>7.2f}{r[5]:>6.2f}",
            fontsize=8.5, family="monospace", va="top")
ax.text(0, 0.30, "single = share of items answered once. mean / sd are over items with\n"
        ">= 5 responses. Validation has only 5 benchmarks and test 6, so their curves\n"
        "are lumpy: one benchmark can dominate a role.", fontsize=8, color=DARK, va="top")
pdf.savefig(fig); plt.close(fig)

# --------------------------------------------- load model predictions
pred = {}
for rung in RUNGS:
    it = pd.read_parquet(DATA / f"{rung}-items.parquet")
    agg = (it.groupby(["split", "budget", "item_uid"], as_index=False)
           .agg(p=("p", "mean"), n_tasks=("p", "size")))
    agg = agg.merge(good[["item_uid", "pass_rate", "benchmark_id", "n_responses"]],
                    on="item_uid", how="inner")
    pred[rung] = agg


def stats(a):
    r = np.corrcoef(a["p"], a["pass_rate"])[0, 1]
    w = a.assign(pw=a["p"] - a.groupby("benchmark_id")["p"].transform("mean"),
                 ow=a["pass_rate"] - a.groupby("benchmark_id")["pass_rate"].transform("mean"))
    rw = np.corrcoef(w["pw"], w["ow"])[0, 1] if w["pw"].std() > 0 else float("nan")
    return r, rw, a["p"].std(), a["pass_rate"].std(), a["p"].mean(), a["pass_rate"].mean()


# ---------------------------------- page 8: predicted vs observed item pass rate
fig = plt.figure(figsize=(11, 8.5))
header(fig, "Is the model learning the item pass-rate distribution?",
       "Grey = observed pass rate of items that have >= 5 responses. Coloured line = what the "
       "model predicts for the same items with NO labels revealed (budget 0), averaged over "
       "the tasks that contain the item.\nIf the model had learned item difficulty, the "
       "coloured line would follow the grey shape. Predictions are raw, not calibrated.")
splits = [("train", 0, "TRAIN items (the model was fitted on these)"),
          ("validation", 0, "VALIDATION items"), ("test", 0, "TEST items")]
table = []
for r, (split, budget, label) in enumerate(splits):
    for c, (rung, (name, col)) in enumerate(RUNGS.items()):
        a = pred[rung]
        a = a[(a["split"] == split) & (a["budget"] == budget)]
        ax = fig.add_axes([0.07 + c * 0.315, 0.64 - r * 0.215, 0.27, 0.16])
        ax.hist(a["pass_rate"], bins=30, range=(0, 1), color="#c9d1d9", density=True)
        ax.hist(a["p"], bins=30, range=(0, 1), histtype="step", lw=2.2, color=col, density=True)
        s = stats(a)
        table.append((split, name, len(a), *s))
        ax.set_title(f"{name}" if r == 0 else "", fontsize=8.8, fontweight="bold", loc="left")
        ax.text(0.02, 0.97, f"corr {s[0]:.2f} | within-bench {s[1]:.2f}\n"
                f"sd pred {s[2]:.2f} vs obs {s[3]:.2f}", transform=ax.transAxes, va="top",
                fontsize=7.3, color=DARK)
        if c == 0:
            ax.set_ylabel(label.split(" (")[0].split(" items")[0], fontsize=8)
        if r == 2:
            ax.set_xlabel("pass rate / predicted probability", fontsize=8)
        style(ax)
note(fig, 0.055, 0.205, "How to read it", 10, DARK, fontweight="bold")
note(fig, 0.055, 0.185,
     "- corr = correlation between predicted and observed pass rate across items.  "
     "within-bench = the same after removing each benchmark's average,\n"
     "  i.e. can the model rank items INSIDE a benchmark?  sd pred vs sd obs = how spread out "
     "the predictions are compared with reality.\n"
     "- A model that only knows 'this benchmark is easy' gets a decent corr but a within-bench "
     "corr near 0, and predictions bunched up in the middle.",
     8.6)
pdf.savefig(fig); plt.close(fig)

# --------------------- page 9: scatter, budget 0 vs budget 31, validation + test
fig = plt.figure(figsize=(11, 8.5))
header(fig, "Predicted vs observed pass rate, item by item",
       "Each point is an item with >= 5 responses. The red dashed line is perfect agreement. "
       "Top row: no labels revealed. Bottom row: 31 labels revealed (order 0), which lets the "
       "model adapt to the subject.\nValidation and test items only (those are the roles with "
       "labels revealed).")
for r, budget in enumerate((0, 31)):
    for c, (rung, (name, col)) in enumerate(RUNGS.items()):
        a = pred[rung]
        a = a[(a["split"].isin(["validation", "test"])) & (a["budget"] == budget)]
        ax = fig.add_axes([0.07 + c * 0.315, 0.50 - r * 0.40 + 0.0, 0.27, 0.32])
        hb = ax.hexbin(a["p"], a["pass_rate"], gridsize=30, bins="log", cmap="Blues",
                       mincnt=1, extent=(0, 1, 0, 1))
        ax.plot([0, 1], [0, 1], color=RED, ls="--", lw=1)
        s = stats(a)
        ax.set_title(f"{name}  -  budget {budget}", fontsize=8.8, fontweight="bold",
                     loc="left")
        ax.text(0.03, 0.97, f"corr {s[0]:.2f}   within-bench {s[1]:.2f}",
                transform=ax.transAxes, va="top", fontsize=7.8, color=DARK)
        ax.set_xlabel("predicted probability (mean over tasks)", fontsize=8)
        if c == 0:
            ax.set_ylabel("observed item pass rate", fontsize=8)
        style(ax)
pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------- pages 10-11: Figure 11 analog
def fig11_page(split, benches, title_extra):
    f11 = {r: pd.read_parquet(DATA / f"{r}-fig11.parquet") for r in RUNGS}
    fig, axes = plt.subplots(len(benches), 3, figsize=(11, 8.5))
    header(fig, f"Figure 11 analog: how well does a few-label estimate recover the truth? "
                f"({title_extra})",
           "For each subject (task) and each random label order we estimate its accuracy two ways: "
           "CTT = the plain average of the labels revealed so far (grey);\nthe model = the mean "
           "probability it predicts over the whole evaluation pool. Plots show estimate MINUS "
           "the true evaluation-pool accuracy. A good estimator is centred on 0 and narrow.")
    bins = np.linspace(-0.7, 0.7, 57)
    for i, bench in enumerate(benches):
        for j, budget in enumerate((3, 7, 31)):
            ax = axes[i, j] if len(benches) > 1 else axes[j]
            lines = []
            first = f11["cnp_gelu"]
            d = first[(first["benchmark_id"] == bench) & (first["budget"] == budget)]
            err = d["ctt"] - d["truth"]
            ax.hist(err, bins=bins, histtype="stepfilled", color="#d5dbe1", density=True)
            lines.append(("CTT average", err.mean(), np.sqrt((err ** 2).mean())))
            for rung, (name, col) in RUNGS.items():
                d = f11[rung]
                d = d[(d["benchmark_id"] == bench) & (d["budget"] == budget)]
                e = d["model"] - d["truth"]
                ax.hist(e, bins=bins, histtype="step", lw=2, color=col, density=True)
                lines.append((name.split(" (")[0], e.mean(), np.sqrt((e ** 2).mean())))
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{bench} - {budget} labels  ({d['task_id'].nunique()} subjects)",
                         fontsize=8.6, fontweight="bold", loc="left")
            ax.text(0.98, 0.97, "\n".join(f"{n[:16]:<16} bias {b:+.2f} rmse {r_:.2f}"
                                          for n, b, r_ in lines),
                    transform=ax.transAxes, ha="right", va="top", fontsize=6.0,
                    family="monospace", color=DARK)
            if i == len(benches) - 1:
                ax.set_xlabel("estimate - truth", fontsize=8)
            style(ax)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.85, bottom=0.07, hspace=0.38, wspace=0.18)
    handles = [plt.Line2D([0], [0], color="#b9c2cb", lw=6, label="CTT average")] + [
        plt.Line2D([0], [0], color=col, lw=2, label=name) for name, col in RUNGS.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    pdf.savefig(fig); plt.close(fig)


fig11_page("validation", ["cmmlu", "afrimedqa", "appworld"], "validation benchmarks")
fig11_page("test", ["bertaqa", "dave", "disco"], "test benchmarks")


# --------------------------------------- page 12: paper Figure 4 analog
def rank_auc(y, score):
    """Binary ROC AUC from average ranks, without an sklearn dependency."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def finite_corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2 or x[keep].std() == 0 or y[keep].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[keep], y[keep])[0, 1])


def loo_easiness(frame):
    """Smoothed item logit after removing this subject's own response."""
    n = frame["n_responses"].to_numpy(dtype=float)
    y = frame["label"].to_numpy(dtype=float)
    successes_other = frame["pass_rate"].to_numpy(dtype=float) * n - y
    # Beta(1, 1) smoothing also gives unseen-after-LOO items a neutral logit.
    prob = (successes_other + 1.0) / (n - 1.0 + 2.0)
    return np.log(prob / (1.0 - prob))


def map_theta(item_easiness, labels):
    """Scalar Rasch MAP with theta ~ N(0, 1), solved by Newton updates."""
    theta = 0.0
    for _ in range(30):
        logits = np.clip(theta + item_easiness, -30.0, 30.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = float(np.sum(labels - probs) - theta)
        hess = float(-np.sum(probs * (1.0 - probs)) - 1.0)
        step = grad / hess
        theta -= step
        if abs(step) < 1e-9:
            break
    return theta


def traditional_budget31_predictions():
    """Response-calibrated Rasch oracle using the exact order-0 context."""
    eval_items = pd.read_parquet(DATA / "eval_items.parquet")
    orders = pd.read_parquet(DATA / "eval_orders.parquet")
    item_uids = pd.read_parquet(DATA / "items.parquet").reset_index(names="item_index")
    item_stats = items[["item_uid", "n_responses", "pass_rate"]]
    eval_items = (eval_items.merge(item_uids, on="item_index", how="left",
                                   validate="many_to_one")
                  .merge(item_stats, on="item_uid", how="left", validate="many_to_one"))

    context_order = orders[(orders["order_seed"] == 0)
                           & (orders["acquisition_rank"] <= 31)]
    context = context_order.merge(
        eval_items[["task_id", "item_index", "label", "n_responses", "pass_rate"]],
        on=["task_id", "item_index"], how="left", validate="one_to_one")
    if context[["label", "n_responses", "pass_rate"]].isna().any().any():
        raise ValueError("Figure 4 context rows did not match the compact evaluation store")

    theta = {}
    for task_id, rows in context.groupby("task_id", sort=False):
        theta[task_id] = map_theta(loo_easiness(rows), rows["label"].to_numpy(float))

    target = eval_items[eval_items["pool"] == "evaluation"].copy()
    if target[["n_responses", "pass_rate"]].isna().any().any():
        raise ValueError("Figure 4 target items are missing raw pass-rate statistics")
    target["theta_traditional"] = target["task_id"].map(theta)
    target["item_easiness_loo"] = loo_easiness(target)
    logit = np.clip(target["theta_traditional"] + target["item_easiness_loo"], -30, 30)
    target["p_traditional"] = 1.0 / (1.0 + np.exp(-logit))
    return target[["split", "benchmark_id", "task_id", "item_uid", "label",
                   "p_traditional"]].rename(columns={"label": "y"})


traditional = traditional_budget31_predictions()
fig4_rows = []
for rung in RUNGS:
    model_rows = pd.read_parquet(DATA / f"{rung}-items.parquet")
    model_rows = model_rows[(model_rows["budget"] == 31)
                            & model_rows["split"].isin(["validation", "test"])]
    joined = model_rows.merge(
        traditional,
        on=["split", "benchmark_id", "task_id", "item_uid", "y"],
        how="inner", validate="one_to_one")
    if len(joined) != len(model_rows):
        raise ValueError(f"Figure 4 merge lost {len(model_rows) - len(joined):,} rows for {rung}")
    for (split, benchmark), rows in joined.groupby(["split", "benchmark_id"], sort=True):
        by_task = rows.groupby("task_id", as_index=False).agg(
            model_avg=("p", "mean"), traditional_avg=("p_traditional", "mean"),
            observed_avg=("y", "mean"))
        enough_tasks = len(by_task) >= 3
        fig4_rows.append({
            "rung": rung, "split": split, "benchmark_id": benchmark,
            "n_tasks": len(by_task), "n_responses": len(rows),
            "model_auc": rank_auc(rows["y"], rows["p"]),
            "traditional_auc": rank_auc(rows["y"], rows["p_traditional"]),
            "model_avg_corr": (finite_corr(by_task["model_avg"], by_task["observed_avg"])
                               if enough_tasks else float("nan")),
            "traditional_avg_corr": (
                finite_corr(by_task["traditional_avg"], by_task["observed_avg"])
                if enough_tasks else float("nan")),
        })
fig4 = pd.DataFrame(fig4_rows)

fig, axes = plt.subplots(3, 2, figsize=(11, 8.5))
header(fig, "Figure 4 analog: neural predictions vs response-calibrated Rasch",
       "Each point is one held-out benchmark at budget 31. x = our model; y = a traditional "
       "scalar Rasch oracle whose item logits come from other subjects' raw responses. Both "
       "use the exact same 31 order-0 context labels.\nLeft: response AUC. Right: correlation "
       "across tasks between predicted and observed mean accuracy. Dashed line = equal fit.")
role_style = {"validation": ("#d98324", "o"), "test": ("#c0392b", "s")}
for row_i, (rung, (model_name, _)) in enumerate(RUNGS.items()):
    data = fig4[fig4["rung"] == rung]
    for col_i, (xcol, ycol, metric) in enumerate((
            ("model_auc", "traditional_auc", "Response AUC"),
            ("model_avg_corr", "traditional_avg_corr", "Task-average correlation"))):
        ax = axes[row_i, col_i]
        for split, (color, marker) in role_style.items():
            d = data[data["split"] == split].dropna(subset=[xcol, ycol])
            ax.scatter(d[xcol], d[ycol], s=38, color=color, marker=marker,
                       label=split.capitalize(), zorder=3)
            rho = finite_corr(d[xcol], d[ycol])
            rho_text = "n/a" if not np.isfinite(rho) else f"{rho:.2f}"
            ax.text(0.02, 0.97 - (0.09 if split == "test" else 0),
                    f"rho {split[:3]} = {rho_text}", transform=ax.transAxes,
                    fontsize=7.2, color=color, va="top")
            # Label only the largest disagreement in each role; labelling every point
            # makes these compact panels unreadable.
            highlighted = d.assign(gap=(d[ycol] - d[xcol]).abs()).nlargest(1, "gap")
            for point in highlighted.itertuples():
                ax.annotate(point.benchmark_id, (getattr(point, xcol), getattr(point, ycol)),
                            xytext=(3, 2), textcoords="offset points", fontsize=5.6,
                            color="#53606c")
        lo = 0.40 if col_i == 0 else -0.05
        ax.plot([lo, 1.01], [lo, 1.01], color="black", ls="--", lw=0.9)
        ax.set_xlim(lo, 1.01); ax.set_ylim(lo, 1.01)
        ax.set_title(f"{model_name} - {metric}", fontsize=8.8, fontweight="bold", loc="left")
        if row_i == 2:
            ax.set_xlabel("our model", fontsize=8)
        ax.set_ylabel("traditional Rasch oracle", fontsize=8)
        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=7.5, frameon=False, loc="lower right")
        style(ax)
fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.105, hspace=0.40, wspace=0.22)
note(fig, 0.055, 0.055,
     "Important: this is our competition-aligned analog, not the paper's identical split. "
     "Validation/test denote held-out benchmark roles. Item logits use leave-one-subject-out "
     "counts with Beta(1,1) smoothing; benchmarks with fewer than three subjects do not get "
     "a task-correlation point.", 7.7, GREY)
pdf.savefig(fig); plt.close(fig)
pdf.close()

# --------------------------------------------------------------- merge + print
buf.seek(0)
writer = PdfWriter()
for page in PdfReader(str(BASE)).pages:
    writer.add_page(page)
for page in PdfReader(buf).pages:
    writer.add_page(page)
with open(OUT, "wb") as handle:
    writer.write(handle)
print("pages:", len(writer.pages))
print(f"{'split':<11}{'model':<26}{'items':>8}{'corr':>7}{'within':>8}{'sd_p':>7}{'sd_obs':>8}"
      f"{'mean_p':>8}{'mean_obs':>9}")
for t in table:
    print(f"{t[0]:<11}{t[1]:<26}{t[2]:>8,}{t[3]:>7.2f}{t[4]:>8.2f}{t[5]:>7.2f}{t[6]:>8.2f}"
          f"{t[7]:>8.2f}{t[8]:>9.2f}")
print("\nFigure 4 analog (budget 31):")
print(fig4.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
