#!/usr/bin/env python3
"""Subject ability: what it looks like, and how each model estimates it.

Ability is shown on the probability scale, as the subject's accuracy on a
task (one subject on one benchmark); a reader can check it against a raw hit
rate, which a logit theta would hide. Each model's estimate of it is the mean
probability it assigns over the task's scored items:

    Beta-Binomial                (hits + 0.5) / (labels + 1), item-blind
    Pooled Rasch                 sigmoid(mu_b + theta_s - b_h), averaged over items;
                                 fitted on every pair's labels (multi-pair draws)
    ANP                          the network's mean prediction, its "level"
    ANP, random-subject split    same, for a model that trained on editbench

The truth is the subject's accuracy on the task's evaluation pool, the items
that are scored. ``--build`` computes the per-task estimates from the cached
online-simulation outputs and caches them; pages are drawn from the cache, which is how
``plot_item_difficulty_final_results.py`` includes them.

    python organized/utils/plot_ability_final.py --build --cache /home/agrov/aims-replay-cache
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_names import NAME  # noqa: E402
from plotstyle import CYCLE, INK, MUTED, SUBTLE, TEXTWIDTH, style  # noqa: E402

CACHE = ROOT / "organized" / "training" / "results" / "ability-levels.parquet"
TRAIN_CACHE = ROOT / "organized" / "training" / "results" / "ability-train-tasks.parquet"
PANEL = "benchmark-holdout-online-v1-s20261018-n15-m80-cap100000"
SEEDS = {"validation": "s3", "test": "s10"}
SHOWN = [0, 1, 7, 31]
SEEN = "editbench"
RANDOM_SPLIT_BENCHMARKS = {"afrieval", "ai2d_test", "alignment_faking", "bird_sql", "bridging_gap",
                           "editbench", "hallusionbench"}
ROWS = {"validation": ["beta", "pooled-rasch", "w64-filtered"],
        "test": ["beta", "pooled-rasch", "w64-filtered", "w64-contam"]}
COLOUR = {"beta": INK, "pooled-rasch": MUTED["rose"], "w64-filtered": MUTED["indigo"], "w64-contam": MUTED["teal"]}


def exact_page(fig) -> Bbox:
    return Bbox.from_bounds(0, 0, *fig.get_size_inches())


# ------------------------------------------------------------ computation --
def build(cache: Path, responses: Path, draws_per_role: int = 40) -> pd.DataFrame:
    items = pd.read_parquet(cache / "panel" / "items.parquet")
    ev = items[items["pool"] == "evaluation"]
    truth = ev.groupby("task_id").agg(observed=("label", "mean"), benchmark_id=("benchmark_id", "first"))
    rows = []

    # ANP and Beta-Binomial: mean prediction over the scored items, from the online simulation.
    for role, seeds in SEEDS.items():
        for model in ROWS[role]:
            if model == "pooled-rasch":
                continue
            run = cache / "runs" / f"{role}-{model}-random_prefix-{seeds}-v2" / "predictions.parquet"
            if not run.exists():
                continue
            p = pd.read_parquet(run, columns=["task_id", "order_seed", "requested_budget", "probability"])
            lv = (p.groupby(["task_id", "order_seed", "requested_budget"]).probability.mean()
                  .groupby(level=[0, 2]).mean().rename("estimate").reset_index())
            lv[["role", "model"]] = role, model
            rows.append(lv.rename(columns={"requested_budget": "budget"}))

    # Pooled Rasch: refit on simulated multi-pair draws, as in the online simulation.
    sys.path.insert(0, str(ROOT / "organized" / "testing" / "online"))
    import modal_pooled_rasch as pr

    pr.REPLAY = cache / "replayroot"
    r = pd.read_parquet(responses, columns=["subject_uid", "benchmark_id", "benchmark_split", "label"])
    m0 = pr._logit(float(r.loc[r["benchmark_split"] == "train", "label"].mean()))
    rng = np.random.default_rng(20261018)
    for role in SEEDS:
        streams, evaluation, meta, seeds = pr._load(role)
        acc = []
        for pairs in pr._draws(meta, seeds, draws_per_role, rng):
            for k in pr.BUDGETS:
                evidence = [(meta.loc[t, "benchmark_id"], meta.loc[t, "subject_uid"], h, y)
                            for t, seed in pairs for h, y in streams[(t, seed)][:k]]
                fit = pr._fit(evidence, m0, 2.0, 1.0, 1.0) if evidence else None
                for t, _ in pairs:
                    hashes, _ = evaluation[t]
                    if fit:
                        mu, th, bh = fit
                        z = (mu.get(meta.loc[t, "benchmark_id"], m0) + th.get(meta.loc[t, "subject_uid"], 0.0)
                             - np.array([bh.get(h, 0.0) for h in hashes]))
                    else:
                        z = np.full(len(hashes), m0)
                    acc.append((t, k, float((1 / (1 + np.exp(-z))).mean())))
        lv = pd.DataFrame(acc, columns=["task_id", "budget", "estimate"]).groupby(["task_id", "budget"]).estimate.mean()
        lv = lv.reset_index()
        lv[["role", "model"]] = role, "pooled-rasch"
        rows.append(lv)
        print(f"{role}: pooled Rasch levels for {lv.task_id.nunique()} tasks", flush=True)

    out = pd.concat(rows, ignore_index=True).merge(truth, left_on="task_id", right_index=True)
    out.to_parquet(CACHE, index=False)

    # Ability distributions of the tasks each model was trained on.
    tasks = r.groupby(["subject_uid", "benchmark_id", "benchmark_split"], as_index=False).label.mean()
    tasks.rename(columns={"label": "observed"}).to_parquet(TRAIN_CACHE, index=False)
    return out


def load():
    if not (CACHE.exists() and TRAIN_CACHE.exists()):
        return None
    return {"levels": pd.read_parquet(CACHE), "tasks": pd.read_parquet(TRAIN_CACHE)}


# ------------------------------------------------------------------ pages ---
def header(fig, title: str) -> None:
    fig.text(0.018, 0.985, title, fontsize=9, fontweight="bold", va="top", ha="left")


def caption(fig, label: str, text: str, width: int = 116) -> None:
    fig.text(0.018, 0.012, textwrap.fill(f"{label} {text}", width=width), fontsize=6.1,
             va="bottom", ha="left", linespacing=1.35)


def distribution_page(pdf, d: dict):
    tasks, levels = d["tasks"], d["levels"]
    panel = levels.drop_duplicates(["role", "task_id"])
    groups = [
        ("Benchmark-holdout training tasks", tasks[tasks.benchmark_split == "train"].observed, SUBTLE),
        ("Random-subject-split training benchmarks", tasks[tasks.benchmark_id.isin(RANDOM_SPLIT_BENCHMARKS)].observed,
         MUTED["teal"]),
        ("Validation tasks (online simulation)", panel[panel.role == "validation"].observed, MUTED["indigo"]),
        ("Test tasks (online simulation)", panel[panel.role == "test"].observed, MUTED["rose"]),
    ]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 3.9))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.89, bottom=0.52)
    header(fig, "Test subjects are stronger than the subjects the models trained on")
    bins = np.linspace(0, 1, 21)
    for label, values, colour in groups:
        ax.hist(values, bins=bins, density=True, histtype="step", lw=1.3, color=colour,
                label=f"{label}: n = {len(values):,}, mean {values.mean():.2f}")
        ax.axvline(values.mean(), color=colour, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlim(0, 1)
    ax.set_xlabel("subject accuracy on the task (ability)", fontsize=6.6)
    ax.set_ylabel("density", fontsize=6.6)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    style(ax, grid_axis="both")
    caption(fig, "Figure.", "Distribution of subject ability, measured as each subject's accuracy on one benchmark "
            "(a task); dashed lines mark the means. The benchmark-holdout models trained on the grey tasks; the "
            "random-subject-split models trained on the seven benchmarks in teal, with subjects split at random. "
            "Validation and test tasks are those of the online simulation, scored on their evaluation pools. Test subjects are "
            "on average about 0.15 more accurate than the training subjects (0.62 against 0.47), and validation "
            "subjects slightly less (0.40), so a prior centred on the training mean starts too low on test and too "
            "high on validation. None of the validation or test subjects appear in training.")
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def estimates_page(pdf, d: dict, role: str):
    lv = d["levels"][d["levels"].role == role]
    models = [m for m in ROWS[role] if (lv.model == m).any()]
    height = 1.25 * len(models) + 2.2
    fig, axes = plt.subplots(len(models), len(SHOWN), figsize=(TEXTWIDTH, height),
                             sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.2, right=0.975, top=1 - 0.42 / height, bottom=1.85 / height,
                        wspace=0.16, hspace=0.12)
    header(fig, "The Beta-Binomial tracks ability best; pooled Rasch shrinks it"
           if role == "test" else "On validation the ANP tracks ability about as well as the Beta-Binomial")
    for r, m in enumerate(models):
        g = lv[lv.model == m]
        for c, k in enumerate(SHOWN):
            ax = axes[r][c]
            s = g[g.budget == k]
            if m == "w64-contam":
                seen = s.benchmark_id == SEEN
                ax.scatter(s.observed[~seen], s.estimate[~seen], s=5, color=COLOUR[m], alpha=0.75, lw=0)
                ax.scatter(s.observed[seen], s.estimate[seen], s=7, color=MUTED["sand"], alpha=0.9, lw=0,
                           marker="D")
            else:
                ax.scatter(s.observed, s.estimate, s=5, color=COLOUR[m], alpha=0.75, lw=0)
            ax.plot([0, 1], [0, 1], color=SUBTLE, lw=0.6, ls=(0, (3, 2)))
            rmse = float(np.sqrt(((s.estimate - s.observed) ** 2).mean()))
            ax.text(0.04, 0.95, f"RMSE {rmse:.2f}", transform=ax.transAxes, fontsize=5.4, va="top")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
            ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
            ax.tick_params(labelsize=5.5)
            if r == 0:
                ax.set_title(f"{k} label{'s' if k != 1 else ''}", fontsize=6.6, pad=3)
            if c == 0:
                ax.set_ylabel(textwrap.fill(NAME[m], 16), fontsize=6, labelpad=2)
            if r == len(models) - 1:
                ax.set_xlabel("true accuracy", fontsize=6)
            style(ax, grid_axis="both")
    fig.text(0.012, (1.85 + (height - 0.42)) / 2 / height, "estimated accuracy", rotation=90, fontsize=6.4,
             va="center")
    seen_note = (" In the random-subject-split row, diamonds are editbench tasks, a benchmark that model trained "
                 "on; circles are algotune and cmmlu, which it never saw." if role == "test" else "")
    finding = {
        "test": ("The Beta-Binomial's estimate tracks the truth most closely from one label on; the random-subject-"
                 "split ANP matches it, and the benchmark-holdout ANP stalls at an error of about 0.09 from seven "
                 "labels on. Pooled Rasch shrinks ability harder (RMSE 0.18 after one label, 0.07 after 31): its "
                 "lower Brier comes from pooling the benchmark rate and item difficulty, not from a sharper ability "
                 "estimate."),
        "validation": ("The ANP's estimate is slightly closer than the Beta-Binomial's at one and three labels and "
                       "slightly further from seven on; pooled Rasch again shrinks ability the most. Pooled Rasch is "
                       "shown for the 72 validation tasks its 40 draws covered."),
    }[role]
    caption(fig, f"Figure ({role}).",
            f"Each point is one {role} task: its true accuracy on the scored items (x) against the model's estimate "
            "of it (y), the mean probability the model assigns over those items, after 0, 1, 7 and 31 labels "
            "(random acquisition, averaged over candidate orders; pooled Rasch over 40 multi-pair draws). The "
            "dashed diagonal is a perfect estimate and each panel gives the RMSE. With no labels every model "
            "predicts nearly one level for everyone. " + finding + seen_note)
    pdf.savefig(fig, bbox_inches=exact_page(fig))
    return fig


def pages(pdf, d: dict) -> list:
    return [("ability-1-distribution", distribution_page(pdf, d)),
            ("ability-2-test", estimates_page(pdf, d, "test")),
            ("ability-3-validation", estimates_page(pdf, d, "validation"))]


def main() -> int:
    from matplotlib.backends.backend_pdf import PdfPages

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path.home() / "aims-replay-cache")
    parser.add_argument("--responses", type=Path, default=Path.home() / "aims-noitem-run" / "filtered-responses.parquet")
    parser.add_argument("--png-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.build:
        build(args.cache, args.responses)
    d = load()
    if d is None:
        parser.error("no cache; run with --build")
    if args.png_dir:
        args.png_dir.mkdir(parents=True, exist_ok=True)
        with PdfPages(args.png_dir / "ability.pdf") as pdf:
            for name, fig in pages(pdf, d):
                fig.savefig(args.png_dir / f"{name}.png", dpi=120, bbox_inches=exact_page(fig))
    return 0


if __name__ == "__main__":
    sys.exit(main())
