#!/usr/bin/env python3
"""Training dynamics of the bge_m3_long ANP runs, from Weights & Biases.

Pulls every logged training and validation curve for the eight ANP runs and
lays each metric out as in-distribution (random-subject split: training and
validation share benchmarks) against out-of-distribution (benchmark holdout:
validation benchmarks were never trained on). The architectures are the same;
the split is the variable.

The W&B pull is cached under ``organized/training/results/`` so the PDF can be
rebuilt offline; ``--refresh`` pulls again.

Read the fixed-bank pages with care: the validation bank's context sizes follow
each run's own sampling scheme (uniform 0-31, the budget ladder, or every
prefix), and the in-distribution banks contain benchmarks those runs trained
on. A fixed-bank number is comparable only between runs with the same bank;
the replay columns on the overview page are the common yardstick.

    python organized/utils/plot_training_final_results.py
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
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
import plot_np_context_illustration as npctx  # noqa: E402

ENTITY_PROJECT = "agrgnn-kth-royal-institute-of-technology/aims-predictive-evaluation"
BASE_URL = "https://api.forge.coreweave.com"
CACHE = ROOT / "organized" / "training" / "results"
REPLAY = ROOT / "organized" / "testing" / "online" / "results"
PANEL = "benchmark-holdout-online-v1-s20261018-n15-m80-cap100000"

# (W&B id, short name, group, replay key on the 82/45-task panel or None)
RUNS = [
    ("np-470e0bbf02b0032be63c", NAME["w64-contam"], "in-dist", "w64-contam"),
    ("np-3f049245dbd45b5b563f", NAME["w128"], "in-dist", "w128"),
    ("np-a3923b0595e9dd6718ce", NAME["w64-filtered"], "OOD", "w64-filtered"),
    ("np-33e8824bdcf39224e35d", NAME["w64-ladder"], "OOD", None),
    ("np-8eb427f4a7ecfd897d04", NAME["w64-sam"], "OOD", "w64-sam"),
    ("np-d67e60f80a9c475947c2", NAME["w64-anchor"], "OOD", None),
    ("np-07013d1f3e4c31d20b7e", NAME["w64-sam-ladder"], "OOD", None),
    ("np-048f1586cd2fbaa02d48", NAME["w64-dense"], "OOD", "w64-dense"),
]
GROUPS = [("in-dist", "In-distribution: random-subject split (validation shares training benchmarks)"),
          ("OOD", "Out-of-distribution: benchmark holdout (validation benchmarks never trained on)")]
COLOR = {rid: CYCLE[i % len(CYCLE)] for i, (rid, *_rest) in enumerate(RUNS)}

KEYS = {
    "elbo_tr": "loss/negative_elbo_per_target/training",
    "elbo_va": "loss/negative_elbo_per_target/validation_resampled_episodes",
    "nll_tr": "loss/posterior_target_nll/training",
    "nll_va": "loss/posterior_target_nll/validation_resampled_episodes",
    "kl_tr": "loss/latent_kl_total/training",
    "kl_va": "loss/latent_kl_total/validation_resampled_episodes",
    "fb_brier": "brier_score/unseen_target/validation_fixed_bank",
    "fb_nll": "loss/unseen_target_nll/validation_fixed_bank",
    "fb_ctx": "loss/context_reconstruction_nll/validation_fixed_bank",
    "active": "latent/kl/active_dimensions_above_0.01_nat",
    "prior_var": "latent/prior/variance_mean",
    "post_var": "latent/posterior/variance_mean",
    "mu_shift": "latent/posterior_minus_prior/mu_abs_mean",
    "sh_brier_tr": "diagnostic/training_subjects/same_subject_split_half/brier_score",
    "sh_brier_ho": "diagnostic/heldout_subjects/same_subject_split_half/brier_score",
    "sh_auc_tr": "diagnostic/training_subjects/same_subject_split_half/roc_auc",
    "sh_auc_ho": "diagnostic/heldout_subjects/same_subject_split_half/roc_auc",
    "rh_brier_tr": "diagnostic/training_subjects/same_subject_response_holdout/brier_score",
    "rh_brier_ho": "diagnostic/heldout_subjects/same_subject_response_holdout/brier_score",
    "rh_auc_tr": "diagnostic/training_subjects/same_subject_response_holdout/roc_auc",
    "rh_auc_ho": "diagnostic/heldout_subjects/same_subject_response_holdout/roc_auc",
}
# Single-batch training losses swing by an order of magnitude between steps;
# these get an exponential moving average drawn over the faint raw trace.
NOISY = {"elbo_tr", "elbo_va", "nll_tr", "nll_va", "kl_tr", "kl_va"}

# One page per tuple: (page title, [(metric, panel title, y label)]).
PAGES = [
    ("Objective: negative ELBO per target",
     [("elbo_tr", "training", "neg. ELBO"), ("elbo_va", "validation (resampled)", "neg. ELBO")]),
    ("Posterior target NLL",
     [("nll_tr", "training", "NLL"), ("nll_va", "validation (resampled)", "NLL")]),
    ("Latent KL (total)",
     [("kl_tr", "training", "KL (nats)"), ("kl_va", "validation (resampled)", "KL (nats)")]),
    ("Fixed validation bank: checkpoint-selection metric",
     [("fb_brier", "unseen-target Brier", "Brier"), ("fb_nll", "unseen-target NLL", "NLL")]),
    ("Fixed validation bank: context fit and latent use",
     [("fb_ctx", "context reconstruction NLL", "NLL"),
      ("active", "active latent dims (KL > 0.01 nat)", "dims")]),
    ("Latent distribution",
     [("prior_var", "prior variance (mean)", "variance"),
      ("post_var", "posterior variance (mean)", "variance"),
      ("mu_shift", "|posterior − prior| mean", "|Δμ|")]),
    ("Generalisation: same-subject split-half Brier",
     [("sh_brier_tr", "training subjects", "Brier"), ("sh_brier_ho", "held-out subjects", "Brier")]),
    ("Generalisation: same-subject split-half AUC",
     [("sh_auc_tr", "training subjects", "ROC AUC"), ("sh_auc_ho", "held-out subjects", "ROC AUC")]),
    ("Generalisation: same-subject response-holdout Brier",
     [("rh_brier_tr", "training subjects", "Brier"), ("rh_brier_ho", "held-out subjects", "Brier")]),
    ("Generalisation: same-subject response-holdout AUC",
     [("rh_auc_tr", "training subjects", "ROC AUC"), ("rh_auc_ho", "held-out subjects", "ROC AUC")]),
]


def exact_page(fig) -> Bbox:
    """Pin the saved box to the figure; plotstyle's default "tight" box grows
    to swallow overflowing text, silently widening a page past TEXTWIDTH."""
    return Bbox.from_bounds(0, 0, *fig.get_size_inches())


# ---------------------------------------------------------------- W&B pull --
def pull(refresh: bool) -> tuple[pd.DataFrame, dict]:
    curves, meta_path = CACHE / "wandb-training-histories.parquet", CACHE / "wandb-training-runs.json"
    if curves.exists() and meta_path.exists() and not refresh:
        return pd.read_parquet(curves), json.loads(meta_path.read_text())
    import wandb

    api = wandb.Api(overrides={"base_url": BASE_URL}, timeout=60)
    meta, jobs = {}, []
    for rid, *_ in RUNS:
        run = api.run(f"{ENTITY_PROJECT}/{rid}")
        c, s = run.config, run.summary
        meta[rid] = {
            "state": run.state, "label": c.get("run_label"), "steps": c.get("steps"),
            "width": c.get("width"), "latent_dim": c.get("latent_dim"),
            "sampling": c.get("context_size_sampling") or "uniform_0_to_max_context",
            "sam_rho": c.get("sam_rho") or 0.0, "prior_anchor": bool(c.get("prior_anchor")),
            "dense_prefix": bool(c.get("dense_prefix_training")),
            "split": c.get("split_protocol"), "accelerator": c.get("requested_accelerator"),
            "best_step": s.get("checkpoint/best_step"),
            "best_brier": s.get("checkpoint/best_fixed_bank_brier_score"),
        }
        jobs += [(run, name, key) for name, key in KEYS.items()]

    # Pull each key on its own: scan_history drops any row missing one of the
    # requested keys, and these metrics are logged on different cadences.
    def fetch(job):
        run, name, key = job
        rows = run.scan_history(keys=["_step", key], page_size=5000)
        return [(run.id, name, r["_step"], r[key]) for r in rows
                if r.get(key) is not None and np.isfinite(r[key])]

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = [row for rows in pool.map(fetch, jobs) for row in rows]
    frame = pd.DataFrame(records, columns=["run", "metric", "step", "value"])
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(curves, index=False)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return frame, meta


# ------------------------------------------------------------------ pages ---
def header(fig, title: str, subtitle: str = "") -> None:
    fig.text(0.018, 0.985, title, fontsize=9, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.018, 0.948, subtitle, fontsize=6.3, color=SUBTLE, va="top")


def method(m: dict) -> str:
    sampling = {"uniform_0_to_max_context": "uniform, 0-31 labels",
                "alc_scored_budget_ladder": "budget ladder, ALC weights",
                "uniform_scored_budget_ladder": "budget ladder, equal weights",
                "all_stream_prefixes_0_to_max_context": "every prefix, 0-31"}.get(m["sampling"], m["sampling"])
    extras = [f"SAM {m['sam_rho']:g}" if m["sam_rho"] else "",
              "anchor" if m["prior_anchor"] else ""]
    return ", ".join([sampling] + [e for e in extras if e])


def replay_alc() -> dict:
    """random_prefix macro ALC per replay model, for validation and test."""
    weights = dict(zip((0, 1, 3, 7, 15, 31), (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)))
    out = {}
    for role in ("validation", "test"):
        path = REPLAY / f"budget-scores-{PANEL}-{role}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df[df.policy == "random_prefix"]
        tm = df.groupby(["model", "task_id", "requested_budget"])["brier"].mean().unstack()
        task_alc = sum(w * tm[b] for b, w in weights.items())
        out[role] = task_alc.groupby(level="model").mean().to_dict()
    return out


def overview_page(pdf, meta: dict):
    fig = plt.figure(figsize=(TEXTWIDTH, 4.6))
    header(fig, "Training runs: ANP with BGE-M3 item embeddings (cross-attention, 1 self-attention block, 4 heads)",
           "Same architecture throughout; the split is the variable. Replay: random "
           "acquisition, macro-task Brier ALC.")
    alc = replay_alc()
    cols = ["run", "split", "width /\nlatent", "context sizes, method", "steps",
            "best step", "bank Brier", "replay val", "replay test"]
    rows = []
    for rid, name, group, key in RUNS:
        m = meta[rid]
        rows.append([
            # Long names and methods break onto a second line rather than
            # spilling into the next column.
            name.replace(" (", "\n(").replace(", budget", ",\nbudget"),
            "random\nsubject" if group == "in-dist" else "benchmark\nholdout",
            f"{m['width']} / {m['latent_dim']}", method(m).replace(", SAM", ",\nSAM").replace(", anchor", ",\nanchor"),
            f"{m['steps']:,}",
            f"{m['best_step']:,}" if m["best_step"] is not None else "-",
            f"{m['best_brier']:.4f}" if m["best_brier"] is not None else "-",
            f"{alc.get('validation', {}).get(key, float('nan')):.4f}" if key else "not run",
            f"{alc.get('test', {}).get(key, float('nan')):.4f}" if key else "not run",
        ])
    ax = fig.add_axes([0.01, 0.30, 0.98, 0.56])
    ax.axis("off")
    widths = [0.20, 0.085, 0.075, 0.2, 0.075, 0.07, 0.085, 0.085, 0.085]
    table = ax.table(cellText=rows, colLabels=cols, colWidths=widths,
                     cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(5.1)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(weight="bold")
        if c in (0, 3):
            cell.set_text_props(ha="left")
            cell.PAD = 0.03
    import textwrap

    notes = (
        "bank Brier: best value on the run's own fixed validation bank, the metric that "
        "picked its checkpoint. That bank's context sizes follow the run's sampling "
        "scheme, and in-distribution banks hold benchmarks the run trained on, so bank "
        "Brier compares only runs with the same bank.\n"
        "replay val / test: random acquisition on the 82-task validation and 45-task test "
        "panels, the same protocol for every row. The random-split runs trained on "
        "ai2d_test and bird_sql (validation) and editbench (test).\n"
        "The dense-prefix run kept training after its replay and CodaBench checkpoint was frozen "
        "at step 4,550 (bank Brier 0.1700); its final best is the step shown. SAM ran to "
        "30,000 steps but its best checkpoint, step 5,450, did not change."
    )
    y = 0.255
    for paragraph in notes.split("\n"):
        for line in textwrap.wrap(paragraph, 128):
            fig.text(0.018, y, line, fontsize=5.6, color=SUBTLE)
            y -= 0.035
        y -= 0.012
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def ema(values: np.ndarray, span: int = 20) -> np.ndarray:
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()


def metric_page(pdf, frame: pd.DataFrame, meta: dict, title: str, panels: list):
    ncols = len(panels)
    fig, axes = plt.subplots(2, ncols, figsize=(TEXTWIDTH, 4.9), sharex=True, squeeze=False)
    fig.subplots_adjust(left=0.12, right=0.965, top=0.86, bottom=0.17,
                        wspace=0.30, hspace=0.42)
    smoothed = any(m in NOISY for m, _, _ in panels)
    starred = any(m == "fb_brier" for m, _, _ in panels)
    header(fig, title, "x: optimiser step (log scale). " + (
        "Faint: logged values; solid: exponential moving average (span 20 points)."
        if smoothed else "Star: the step each run's checkpoint was selected at."
        if starred else "Values as logged."))
    for row, (group, gtitle) in enumerate(GROUPS):
        for col, (metric, ptitle, ylabel) in enumerate(panels):
            ax = axes[row][col]
            for rid, name, g, _ in RUNS:
                if g != group:
                    continue
                # Step 0 cannot sit on a log axis; clipping it to 1 drew a flat
                # fake segment out to the first logged step, so it is dropped.
                d = frame[(frame.run == rid) & (frame.metric == metric)
                          & (frame.step > 0)].sort_values("step")
                if d.empty:
                    continue
                x = d.step.to_numpy()
                y = d.value.to_numpy()
                if metric in NOISY:
                    ax.plot(x, y, color=COLOR[rid], lw=0.4, alpha=0.18)
                    ax.plot(x, ema(y), color=COLOR[rid], lw=1.0, label=name)
                else:
                    ax.plot(x, y, color=COLOR[rid], lw=0.9, label=name)
                best = meta[rid].get("best_step")
                if metric == "fb_brier" and best is not None:
                    hit = d[d.step == best]
                    if not hit.empty:
                        ax.plot([best], [hit.value.iloc[0]], marker="*", ms=6,
                                color=COLOR[rid], markeredgecolor=INK, markeredgewidth=0.4)
            ax.set_xscale("log")
            style(ax)
            ax.set_title(f"{ptitle}", fontsize=6.8, pad=3)
            if col == 0:
                ax.set_ylabel(f"{'in-dist' if group == 'in-dist' else 'OOD'}\n{ylabel}",
                              fontsize=6.6, labelpad=4)
            if row == 1:
                ax.set_xlabel("step", fontsize=6.6)
            ax.tick_params(labelsize=6)
    # One legend for the OOD row and one for in-dist, below the panels.
    for row, (group, _) in enumerate(GROUPS):
        handles, labels = axes[row][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, [f"{'in-dist' if group == 'in-dist' else 'OOD'}: {l}" for l in labels],
                       loc="lower center", fontsize=5.6, frameon=False,
                       ncol=min(len(labels), 3 if group == "OOD" else 2),
                       bbox_to_anchor=(0.5, 0.06 if group == "in-dist" else 0.0))
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="pull W&B again")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "training-final-results.pdf")
    parser.add_argument("--png-dir", type=Path, default=None)
    args = parser.parse_args()

    frame, meta = pull(args.refresh)
    print(f"{frame.run.nunique()} runs, {frame.metric.nunique()} metrics, {len(frame):,} points")
    pages = []
    with PdfPages(args.out) as pdf:
        pages.append(("00-overview", overview_page(pdf, meta)))
        for i, (title, panels) in enumerate(PAGES, start=1):
            pages.append((f"{i:02d}-{panels[0][0]}", metric_page(pdf, frame, meta, title, panels)))
        # In-distribution illustration of the trained ANP learning from context;
        # built by plot_np_context_illustration.py, skipped until its cache exists.
        context = npctx.load_all()
        if context:
            pages += [(f"{len(pages) + j:02d}-{name}", fig)
                      for j, (name, fig) in enumerate(npctx.pages(pdf, context))]
    if args.png_dir:
        args.png_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in pages:
            fig.savefig(args.png_dir / f"{name}.png", dpi=120, bbox_inches=exact_page(fig))
    print(f"wrote {args.out} ({len(pages)} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
