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


def data_split_page(pdf: PdfPages) -> None:
    holdout = pd.read_csv(DOCS / "benchmark-holdout-split-distribution-summary.csv")
    random = pd.read_csv(DOCS / "non-benchmark-holdout-split-distribution-summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 4.45))
    header(fig, "Data split diagnostics: random subject split is not benchmark transfer",
           "Each value is descriptive. The two columns use different frozen embedding arms, so "
           "centroid cosine is interpreted within each split, not compared as an encoder ranking.")
    roles = ["train", "validation", "test"]
    x = np.arange(len(roles))
    configs = [(holdout, "benchmark holdout", MUTED["rose"]),
               (random, "random subject split", MUTED["teal"])]
    for frame, label, colour in configs:
        axes[0, 0].plot(x, frame.set_index("split").loc[roles, "mean_label"], "o-",
                        color=colour, label=label)
        axes[0, 1].plot(x, frame.set_index("split").loc[roles, "item_pass_rate_std"], "o-",
                        color=colour, label=label)
        cosine_col = next(col for col in frame.columns if col.endswith("centroid_cosine_to_train"))
        axes[1, 0].plot(x, frame.set_index("split").loc[roles, cosine_col], "o-",
                        color=colour, label=label)
    for ax, title, ylabel in [
        (axes[0, 0], "(a) Mean binary response", "mean correct response"),
        (axes[0, 1], "(b) Item pass-rate variation", "SD across items"),
        (axes[1, 0], "(c) Text-embedding shift", "centroid cosine to train"),
    ]:
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xticks(x, ["train", "validation", "test"])
        ax.set_ylabel(ylabel)
        style(ax)
    axes[0, 0].legend(loc="best", fontsize=6.3)
    ax = axes[1, 1]
    rows = []
    for name, frame in [("benchmark holdout", holdout), ("random subject", random)]:
        for role in roles:
            row = frame.set_index("split").loc[role]
            rows.append([name if role == "train" else "", role,
                         f"{int(row['responses']):,}", f"{int(row['tasks']):,}",
                         f"{int(row['items']):,}"])
    table(ax, rows, ["split", "role", "responses", "tasks", "items"],
          widths=[.29, .19, .19, .15, .15], fontsize=5.8)
    ax.set_title("(d) Retained filtered support", loc="left", weight="bold", pad=5)
    fig.text(.02, .02,
             "Sources: docs/benchmark-holdout-split-distribution-summary.csv and "
             "docs/non-benchmark-holdout-split-distribution-summary.csv. "
             "The random-subject split shares benchmark identity with train; it is a split diagnostic, not a transfer baseline.",
             fontsize=5.9, color=SUBTLE)
    fig.tight_layout(rect=(0, .07, 1, .90), pad=.55)
    save_page(pdf, fig)


def data_difficulty_page(pdf: PdfPages) -> None:
    held = pd.read_csv(DOCS / "np-online-passrate-figure11-summary.csv")
    nonheld = pd.read_csv(DOCS / "non-benchmark-anp128-qwen-item-passrate-summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.8))
    header(fig, "Item-difficulty transfer diagnostic",
           "A model can adapt to labels yet still fail to predict which new items are hard. "
           "This is a descriptive 297-item held-out diagnostic, not a causal decomposition.")
    names = ["non-holdout\nW128 BGE", "holdout\nladder", "holdout\nfiltered"]
    overall = held["pass_rate_correlation"].to_numpy()
    within = held["within_benchmark_correlation"].to_numpy()
    xx = np.arange(len(names)); width = .36
    axes[0].bar(xx - width / 2, overall, width, color=MUTED["indigo"], label="overall")
    axes[0].bar(xx + width / 2, within, width, color=MUTED["teal"], label="within benchmark")
    axes[0].set_xticks(xx, names); axes[0].set_ylabel("correlation")
    axes[0].set_title("(a) Held-out item pass-rate correlation", loc="left", weight="bold")
    axes[0].legend(loc="upper left", fontsize=5.8); style(axes[0])
    predicted = held["prediction_sd"].to_numpy()
    observed = held["observed_pass_rate_sd"].to_numpy()
    axes[1].bar(xx - width / 2, predicted, width, color=MUTED["rose"], label="prediction SD")
    axes[1].bar(xx + width / 2, observed, width, color=MUTED["grey"], label="observed SD")
    axes[1].set_xticks(xx, names); axes[1].set_ylabel("standard deviation")
    axes[1].set_title("(b) Predicted variation is compressed", loc="left", weight="bold")
    axes[1].legend(loc="upper left", fontsize=5.8); style(axes[1])
    ax = axes[2]
    row = nonheld.set_index("split").loc["test"]
    rows = [
        ["non-holdout Qwen test", f"{row.correlation:.3f}", f"{row.within_benchmark_correlation:.3f}", f"{row.prediction_sd:.3f}", f"{row.observed_pass_rate_sd:.3f}"],
        ["holdout W64 filtered", f"{held.iloc[2].pass_rate_correlation:.3f}", f"{held.iloc[2].within_benchmark_correlation:.3f}", f"{held.iloc[2].prediction_sd:.3f}", f"{held.iloc[2].observed_pass_rate_sd:.3f}"],
    ]
    table(ax, rows, ["setting", "corr.", "within", "pred SD", "obs SD"],
          widths=[.37, .14, .16, .16, .16], fontsize=5.7)
    ax.set_title("(c) Contrast in two different protocols", loc="left", weight="bold", pad=5)
    fig.text(.02, .015,
             "Sources: docs/np-online-passrate-figure11-summary.csv and "
             "docs/non-benchmark-anp128-qwen-item-passrate-summary.csv. The contrast is useful "
             "evidence for a transfer bottleneck, but differs in split, encoder, and evaluation set; do not treat it as an architecture ablation.",
             fontsize=5.8, color=SUBTLE)
    fig.tight_layout(rect=(0, .08, 1, .84), pad=.5)
    save_page(pdf, fig)


def data_analysis_report(output: Path) -> None:
    with PdfPages(output) as pdf:
        data_split_page(pdf)
        data_difficulty_page(pdf)
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
        "beta_binomial": "Beta-binomial",
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
             "It documents a useful earlier model comparison, but must not be combined numerically with the final unseen-benchmark replay.",
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
             "The key lesson is not a winner: conclusions from a single legacy protocol are fragile, motivating the frozen benchmark-holdout replay.",
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
