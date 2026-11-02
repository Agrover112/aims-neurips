#!/usr/bin/env python3
"""A trained ANP learning from context: the Garnelo et al. figures, real model.

``cnp_figures.py`` draws these with an exact 2PL posterior on a synthetic
world. This draws them with the random-split ANP w64 checkpoint on bird_sql,
a benchmark it trained on, so it shows the model working in distribution. It
is an illustration of the mechanism, not evidence of transfer: on held-out
benchmarks the same model gains far less (see the online results).

    page 1  Fig-2 analogue: one subject's predicted pass probability per item
            as the context grows, +-1 sd over latent samples.
    page 2  Fig-3a analogue: subject x item matrix -- context, mean, spread --
            with held-out Brier per budget, beside the unseen-benchmark model
            and the Jeffreys Beta-Binomial.

The model pass is cached under ``organized/training/results/`` so the training
report can include these pages without the data:

    python organized/utils/plot_np_context_illustration.py \\
        --responses filtered-responses.parquet --embeddings <store dir> \\
        --checkpoints <dir with both .pt files>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import HAIRLINE, INK, MUTED, SEQUENTIAL, SUBTLE, TEXTWIDTH, style  # noqa: E402

CACHE_DIR = ROOT / "organized" / "training" / "results"
RANDOM_SPLIT = "anp-sa1-w64-bgem3-20261028"            # trained on bird_sql, not algotune
HOLDOUT = "denseprefix-unweighted-best-step4550"       # trained on neither

# (benchmark, model drawn, comparison model, their labels, title tag).
# In distribution: the drawn model trained on the benchmark, the context comes
# from new subjects on it. Out of distribution: a test benchmark of the
# holdout split that neither model ever trained on.
VARIANTS = {
    "in-dist": dict(benchmark="bird_sql", main=RANDOM_SPLIT, other=HOLDOUT,
                    main_label="ANP, random-subject split (bird_sql: ID)",
                    other_label="ANP, prefix training (bird_sql: OOD)",
                    tag="ID: the benchmark was in training"),
    "ood": dict(benchmark="algotune", main=HOLDOUT, other=RANDOM_SPLIT,
                main_label="ANP, prefix training (algotune: OOD)",
                other_label="ANP, random-subject split (algotune: OOD)",
                tag="OOD: the benchmark was not in training"),
}
BUDGETS = [0, 1, 3, 7, 15, 31]
SHOWN = [0, 1, 7, 31]
N_LATENT = 64     # latent samples behind the +-1 sd band
N_ORDERS = 5      # random context orders averaged in the Brier panel
N_MATRIX = (24, 60)  # subjects x items drawn in the matrix page
SEED = 20261018


def exact_page(fig) -> Bbox:
    return Bbox.from_bounds(0, 0, *fig.get_size_inches())


# ------------------------------------------------------------ computation --
def load_model(checkpoints: Path, key: str, emb_dim: int):
    import torch

    sys.path.insert(0, str(ROOT / "organized" / "training"))
    import anp_crossattn_blocks as blocks

    match = [m for m in sorted(checkpoints.glob(f"*{key}*.pt")) if not m.name.endswith(".resume.pt")]
    if len(match) != 1:
        raise FileNotFoundError(f"expected one checkpoint matching {key}, got {match}")
    ck = torch.load(match[0], map_location="cpu", weights_only=False)
    c = ck["config"]
    model = blocks.AttentiveNeuralProcess(
        emb_dim, width=int(c["width"]), self_attention_blocks=int(c["n_blocks"]),
        cross_attention_heads=int(c["cross_attention_heads"]),
        use_latent=bool(c["use_latent"]), latent_dim=int(c["latent_dim"]),
        residual_proj=bool(c.get("residual_proj", False)),
        dropout=float(c.get("dropout", 0.0)),
        prior_anchor=bool(c.get("prior_anchor", False)),
    )
    model.load_state_dict(ck["state_dict"], strict=True)
    return model.eval()


def predict(model, x, y, ctx, n_latent=0, gen=None):
    """Mean p on every item given context indices `ctx`; with n_latent, also
    the sd of p over z ~ q(z | context), the model's own epistemic spread."""
    import torch

    x_c, y_c = x[ctx], y[ctx]
    with torch.no_grad():
        mean = torch.sigmoid(model.logits(x_c, y_c, x)).numpy()
        if not n_latent:
            return mean, None
        mu, logvar = model.latent_params(x_c, y_c)
        draws = [torch.sigmoid(model.logits(
            x_c, y_c, x, z=mu + torch.randn(mu.shape, generator=gen) * (0.5 * logvar).exp()))
            for _ in range(n_latent)]
    return mean, torch.stack(draws).std(0).numpy()


def cache_path(variant: str) -> Path:
    return CACHE_DIR / f"np-context-{variant}.npz"


def build(variant: str, responses: Path, embeddings: Path, checkpoints: Path) -> dict:
    v = VARIANTS[variant]
    import pandas as pd
    import torch

    d = pd.read_parquet(responses, columns=["benchmark_id", "subject_uid", "item_uid", "label"])
    d = d[d.benchmark_id == v["benchmark"]]
    resp = d.pivot_table(index="subject_uid", columns="item_uid", values="label", aggfunc="mean")
    resp = resp.dropna(axis=1)  # bird_sql is complete; guard anyway
    # Columns hard -> easy by pass rate over all subjects, rows low -> high.
    resp = resp[resp.mean(0).sort_values().index]
    resp = resp.loc[resp.mean(1).sort_values().index]
    Y = (resp.to_numpy() > 0.5).astype(np.float32)
    n_subj, n_item = Y.shape

    ids = pd.read_parquet(embeddings / "items.parquet")["item_uid"].astype(str).to_numpy()
    pos = {u: i for i, u in enumerate(ids)}
    store = np.load(embeddings / "item_embeddings.npy", mmap_mode="r")
    x = torch.as_tensor(np.asarray(store[[pos[u] for u in resp.columns]], dtype=np.float32))

    main_model = load_model(checkpoints, v["main"], x.shape[1])
    other_model = load_model(checkpoints, v["other"], x.shape[1])
    rng = np.random.default_rng(SEED)
    gen = torch.Generator().manual_seed(SEED)
    orders = np.stack([np.stack([rng.permutation(n_item) for _ in range(n_subj)])
                       for _ in range(N_ORDERS)])  # (order, subject, item)

    # Held-out Brier per budget: every subject, every order, cells not in context.
    brier = {"main": np.zeros((N_ORDERS, len(BUDGETS))), "other": np.zeros((N_ORDERS, len(BUDGETS))),
             "beta": np.zeros((N_ORDERS, len(BUDGETS)))}
    for o in range(N_ORDERS):
        for bi, k in enumerate(BUDGETS):
            acc = {m: [] for m in brier}
            for i in range(n_subj):
                ctx = orders[o, i, :k]
                held = np.setdiff1d(np.arange(n_item), ctx)
                yt = torch.as_tensor(Y[i])
                for name, model in (("main", main_model), ("other", other_model)):
                    p, _ = predict(model, x, yt, torch.as_tensor(ctx, dtype=torch.long))
                    acc[name].append(((p[held] - Y[i, held]) ** 2).mean())
                p_beta = (Y[i, ctx].sum() + 0.5) / (k + 1.0)
                acc["beta"].append(((p_beta - Y[i, held]) ** 2).mean())
            for m in brier:
                brier[m][o, bi] = np.mean(acc[m])

    # Matrix page: evenly spaced subjects and items, first order only.
    rows = np.unique(np.linspace(0, n_subj - 1, N_MATRIX[0]).round().astype(int))
    cols = np.unique(np.linspace(0, n_item - 1, N_MATRIX[1]).round().astype(int))
    mean = np.zeros((len(BUDGETS), n_subj, n_item), dtype=np.float32)
    spread = np.zeros_like(mean)
    for bi, k in enumerate(BUDGETS):
        for i in rows:
            ctx = torch.as_tensor(orders[0, i, :k], dtype=torch.long)
            mean[bi, i], spread[bi, i] = predict(main_model, x, torch.as_tensor(Y[i]), ctx, N_LATENT, gen)

    out = dict(Y=Y, orders=orders[0], rows=rows, cols=cols, mean=mean, spread=spread,
               **{f"brier_{m}": b for m, b in brier.items()})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path(variant), **out)
    return out | v


def load_cache(variant: str) -> dict | None:
    path = cache_path(variant)
    return dict(np.load(path)) | VARIANTS[variant] if path.exists() else None


# ------------------------------------------------------------------ pages ---
def header(fig, title: str, subtitle: str) -> None:
    fig.text(0.018, 0.985, title, fontsize=9, fontweight="bold", va="top")
    fig.text(0.018, 0.935, subtitle, fontsize=6.3, color=SUBTLE, va="top", linespacing=1.3)


def curves_page(pdf, c: dict):
    Y, orders, mean, spread, rows = c["Y"], c["orders"], c["mean"], c["spread"], c["rows"]
    n_item = Y.shape[1]
    # Subjects nearest 30% and 70% accuracy among the drawn rows; the weakest
    # rows can score 0 and make a flat, uninformative panel.
    acc = Y[rows].mean(1)
    subjects = [(rows[np.abs(acc - q).argmin()], name) for q, name in ((0.3, "weaker"), (0.7, "stronger"))]
    xs = np.arange(n_item)
    fig, axes = plt.subplots(2, len(SHOWN), figsize=(TEXTWIDTH, 3.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.79, bottom=0.2, wspace=0.08, hspace=0.18)
    header(fig, f"ANP learning one subject from context ({c['tag']})",
           f"{c['main_label']}. {c['benchmark']}, {n_item} items hard $\\rightarrow$ easy.\n"
           f"Curves are 25-item rolling means; band $\\pm1$ sd over {N_LATENT} latent draws.")
    kernel = np.ones(25) / 25

    def smooth(v):
        return np.convolve(np.pad(v, 12, mode="edge"), kernel, mode="valid")

    for r, (i, label) in enumerate(subjects):
        truth = smooth(Y[i])
        for col, k in enumerate(SHOWN):
            ax = axes[r][col]
            bi = BUDGETS.index(k)
            raw = mean[bi, i]
            m, s = smooth(raw), smooth(spread[bi, i])
            ax.plot(xs, raw, color=MUTED["indigo"], lw=0.3, alpha=0.3, zorder=2)
            ax.fill_between(xs, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                            color=MUTED["indigo"], alpha=0.22, lw=0, zorder=2)
            ax.plot(xs, m, color=MUTED["indigo"], lw=1.1, zorder=3)
            ax.plot(xs, truth, color=MUTED["rose"], lw=1.0, ls=(0, (4, 2)), zorder=4)
            ctx = orders[i, :k]
            ax.plot(ctx, Y[i, ctx], "o", ms=2.4, mfc=INK, mec=INK, zorder=5)
            ax.set_xlim(0, n_item - 1)
            ax.set_ylim(-0.06, 1.06)
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(f"$k={k}$", fontsize=7, fontweight="bold", pad=3)
            if col == 0:
                ax.set_ylabel("P(correct)", fontsize=6.3)
                ax.text(0.03, 0.9, f"{label} subject, acc {Y[i].mean():.2f}", transform=ax.transAxes,
                        fontsize=5.6, color=SUBTLE)
            if r == 1:
                ax.set_xlabel("item rank", fontsize=6.3)
            style(ax, grid_axis="both")
    handles = [plt.Line2D([], [], color=MUTED["indigo"], lw=1.1),
               plt.Line2D([], [], color=MUTED["indigo"], lw=5, alpha=0.22),
               plt.Line2D([], [], color=MUTED["rose"], lw=1.0, ls=(0, (4, 2))),
               plt.Line2D([], [], color=INK, marker="o", ms=2.4, ls="none")]
    fig.legend(handles, ["ANP mean (faint: per item)", "$\\pm1$ sd (latent)", "observed, smoothed", "context label"],
               loc="lower center", ncol=4, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.0))
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def matrix_page(pdf, c: dict):
    Y, orders, rows, cols = c["Y"], c["orders"], c["rows"], c["cols"]
    binary = ListedColormap([MUTED["rose"], MUTED["indigo"]])
    binary.set_bad(HAIRLINE)
    fig = plt.figure(figsize=(TEXTWIDTH, 4.6))
    gs = GridSpec(4, len(BUDGETS), figure=fig, height_ratios=[1, 1, 1, 1.1],
                  hspace=0.32, wspace=0.12, left=0.105, right=0.925, top=0.84, bottom=0.09)
    header(fig, f"Response-matrix completion by the ANP ({c['tag']})",
           f"{c['benchmark']}: {len(rows)} subjects (weak $\\rightarrow$ strong) x {len(cols)} items "
           "(hard $\\rightarrow$ easy). Random context order; grey = unrevealed.")
    for col, k in enumerate(BUDGETS):
        context = np.full(Y.shape, np.nan)
        for i in rows:
            context[i, orders[i, :k]] = Y[i, orders[i, :k]]
        panels = [(np.ma.masked_invalid(context[np.ix_(rows, cols)]), binary, 0, 1),
                  (c["mean"][col][np.ix_(rows, cols)], SEQUENTIAL, 0, 1),
                  (c["spread"][col][np.ix_(rows, cols)], "magma", 0, 0.3)]
        for r, (grid, cmap, lo, hi) in enumerate(panels):
            ax = fig.add_subplot(gs[r, col])
            im = ax.imshow(grid, cmap=cmap, vmin=lo, vmax=hi, aspect="auto", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"$k={k}$", fontsize=7, fontweight="bold", pad=3)
            if col == 0:
                ax.set_ylabel(["Context", "Mean", "Spread"][r], fontsize=6.5, fontweight="bold")
            if col == len(BUDGETS) - 1 and r > 0:
                cb = fig.colorbar(im, cax=ax.inset_axes([1.07, 0.0, 0.08, 1.0]))
                cb.ax.tick_params(labelsize=5, width=0.4, length=1.6, pad=1.5)
                cb.outline.set_linewidth(0.4)
            style(ax, grid_axis="none")

    ax = fig.add_subplot(gs[3, :])
    x = np.arange(len(BUDGETS))
    for key, label, colour, ls in (("main", c["main_label"], MUTED["indigo"], "-"),
                                   ("other", c["other_label"], MUTED["teal"], "-."),
                                   ("beta", "Beta-Binomial", INK, "--")):
        v = c[f"brier_{key}"]
        ax.errorbar(x, v.mean(0), yerr=v.std(0), color=colour, ls=ls, marker="o", ms=2.6,
                    lw=1.0, capsize=1.5, elinewidth=0.6, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(BUDGETS)
    ax.set_xlabel("context size $k$ (labels revealed per subject)", fontsize=6.5)
    ax.set_ylabel("Brier, held-out\n(all subjects)", fontsize=6.3)
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.6, loc="lower left", frameon=False, ncol=2)
    style(ax, grid_axis="both")
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def pages(pdf, contexts: dict) -> list:
    out = []
    for variant, c in contexts.items():
        out += [(f"np-context-{variant}-curves", curves_page(pdf, c)),
                (f"np-context-{variant}-matrix", matrix_page(pdf, c))]
    return out


def load_all() -> dict:
    """Every variant whose cache exists, in VARIANTS order."""
    return {v: c for v in VARIANTS if (c := load_cache(v)) is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "np-context-final.pdf")
    parser.add_argument("--png-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.responses:
        contexts = {v: build(v, args.responses, args.embeddings, args.checkpoints) for v in VARIANTS}
    else:
        contexts = load_all()
    if not contexts:
        parser.error(f"no cache in {CACHE_DIR}; pass --responses/--embeddings/--checkpoints")
    with PdfPages(args.out) as pdf:
        figs = pages(pdf, contexts)
    if args.png_dir:
        args.png_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs:
            fig.savefig(args.png_dir / f"{name}.png", dpi=120, bbox_inches=exact_page(fig))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
