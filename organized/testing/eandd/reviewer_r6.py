"""Join Person A's R6 fit scores to the sampled triples; preserve the fit protocol."""
import argparse
import json
from pathlib import Path

import pandas as pd

from split_sensitivity import ORIGINAL
from wp9 import CYCLE, TEXTWIDTH, plt, save_figure, style


def run(scores, triples, out):
    data = pd.read_csv(scores).set_index(
        ["excluded_training_benchmarks", "model", "benchmark"], verify_integrity=True)
    picked = [tuple(t[0] if len(t) == 1 and isinstance(t[0], list) else t)
              for t in json.loads(triples.read_text())]
    if len(picked) != 40 or len(set(picked)) != 40:
        raise ValueError("Expected the 40 distinct R6 sampled triples")
    train = set()
    for key in data.index.get_level_values(0).unique():
        if key != "none":
            train.update(key.split(","))
    result = []
    for triple in picked + ([ORIGINAL] if ORIGINAL not in picked else []):
        excluded = ",".join(sorted(set(triple) & train)) or "none"
        for model in ("bb-eb", "rasch-fitted"):
            matched = data.loc[[(excluded, model, b) for b in triple], "gain"]
            if len(matched) != 3 or matched.isna().any():
                raise ValueError(f"Incomplete R6 split: {triple} {model}")
            result.append((*triple, excluded, model, matched.mean(),
                           triple in picked, triple == ORIGINAL))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(result, columns=["benchmark_1", "benchmark_2", "benchmark_3",
                         "excluded_training_benchmarks", "model", "macro_delta",
                         "in_random_sample", "original_test_triple"])
    frame.to_csv(out / "r8-r6-split-sensitivity.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.65), sharey=True)
    for ax, (model, g) in zip(axes, frame.groupby("model")):
        vals = g.loc[g.in_random_sample, "macro_delta"]
        orig = g.loc[g.original_test_triple, "macro_delta"].item()
        ax.hist(vals, bins=12, color=CYCLE[0], alpha=.72)
        ax.axvline(0, color=".45", lw=1, ls=":", label="No change")
        ax.axvline(orig, color=CYCLE[1], lw=1.5, label="Original test trio")
        ax.set_xlabel({"bb-eb": "Beta-Binomial", "rasch-fitted": "Rasch"}[model])
        ax.text(.97, .93, f"{(vals > 0).mean():.1%} worse", transform=ax.transAxes,
                va="top", ha="right")
        style(ax)
        print(model, f"mean={vals.mean():.5f}; harmful={(vals > 0).mean():.1%};",
              f"original={orig:.5f}; p05/p95={vals.quantile([.05, .95]).tolist()}")
    axes[0].set_ylabel("Sampled test triples")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center")
    fig.supxlabel("Fitted minus Jeffreys (macro Brier ALC)", fontsize=8)
    fig.subplots_adjust(left=.11, right=.98, top=.83, bottom=.25, wspace=.20)
    save_figure(fig, out / "figure-r8-r6-split-sensitivity")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scores", type=Path)
    p.add_argument("triples", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    run(a.scores, a.triples, a.out)
