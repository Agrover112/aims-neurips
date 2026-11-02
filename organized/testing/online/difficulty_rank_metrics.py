#!/usr/bin/env python3
"""Item-difficulty rank metrics, in-distribution and on unseen benchmarks.

Brier ALC is dominated by a task's overall accuracy, so it says little about
whether a model knows which items are harder. These metrics isolate that:

    Spearman rank correlation  between a model's predicted probability of a
                               correct answer for each item and the item's
                               observed pass rate (fraction of subjects correct),
                               within one benchmark;
    pairwise accuracy          over pairs of items of one benchmark with different
                               observed pass rates, how often the model orders the
                               pair correctly (a coin flip scores 0.5).

Both are computed per benchmark and averaged over benchmarks. Predictions are
the ANPs' budget-0 outputs (no labels), cached by
plot_item_difficulty_final_results.py, and a ridge probe from BGE-M3 item
embeddings:

    ID   benchmarks the model trained on (probe: a random 80/20 split of that
         benchmark's items, fitted on the 80%);
    OOD  benchmarks the model never saw (probe: fitted on the 11 training
         benchmarks, scored on the 6 held-out ones).

    python organized/testing/online/difficulty_rank_metrics.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "organized" / "training" / "results" / "item-difficulty-b0.parquet"
OUT = ROOT / "organized" / "testing" / "online" / "results"
EMB = Path.home() / "aims-noitem-run/datasets/full/allbench-embed-bge_m3_long-v2"
RANDOM_SPLIT_MODELS = ("ANP, random-subject split", "ANP, random-subject split (width 128)")
SEEN_BY_RANDOM = {"afrieval", "ai2d_test", "alignment_faking", "bird_sql", "bridging_gap", "editbench", "hallusionbench"}


def pairwise_accuracy(pred: np.ndarray, obs: np.ndarray, rng, max_pairs: int = 200_000) -> float:
    i = rng.integers(0, len(pred), max_pairs)
    j = rng.integers(0, len(pred), max_pairs)
    keep = (i != j) & (obs[i] != obs[j])
    dp, do = pred[i[keep]] - pred[j[keep]], obs[i[keep]] - obs[j[keep]]
    return float(np.mean(np.where(dp == 0, 0.5, np.sign(dp) == np.sign(do))))


def score(frame: pd.DataFrame, pred_col: str, rng) -> pd.DataFrame:
    rows = []
    for bench, g in frame.groupby("benchmark_id"):
        pred, obs = g[pred_col].to_numpy(float), g["observed"].to_numpy(float)
        rho = spearmanr(pred, obs).statistic
        rows.append((bench, len(g), float(rho) if np.isfinite(rho) else 0.0, pairwise_accuracy(pred, obs, rng)))
    return pd.DataFrame(rows, columns=["benchmark_id", "items", "rho", "pairwise_acc"])


def main() -> int:
    rng = np.random.default_rng(20261102)
    t = pd.read_parquet(CACHE)
    models = [c for c in t.columns if c.startswith("ANP")]
    rows = []
    for m in models:
        if m in RANDOM_SPLIT_MODELS:
            id_mask, ood_mask = t.benchmark_id.isin(SEEN_BY_RANDOM), ~t.benchmark_id.isin(SEEN_BY_RANDOM)
        else:
            id_mask, ood_mask = t.benchmark_split == "train", t.benchmark_split != "train"
        for setting, mask in (("ID", id_mask), ("OOD", ood_mask)):
            s = score(t[mask], m, rng)
            s[["model", "setting"]] = m, setting
            rows.append(s)

    # The probe: ridge regression from item embeddings to centred observed pass rate.
    ids = pd.read_parquet(EMB / "items.parquet")["item_uid"].astype(str).to_numpy()
    pos = {u: i for i, u in enumerate(ids)}
    store = np.load(EMB / "item_embeddings.npy", mmap_mode="r")
    X = np.asarray(store[[pos[u] for u in t.item_uid.astype(str)]], dtype=np.float32)
    y = (t["observed"] - t.groupby("benchmark_id")["observed"].transform("mean")).to_numpy()
    alphas = [0.01, 0.1, 1, 10, 100]
    # ID: within each training benchmark, fit on a random 80% of its items, score on the other 20%.
    probe_id = []
    for bench, g in t[t.benchmark_split == "train"].groupby("benchmark_id"):
        idx = rng.permutation(g.index.to_numpy())
        cut = int(0.8 * len(idx))
        m = RidgeCV(alphas=alphas).fit(X[idx[:cut]], y[idx[:cut]])
        test = t.loc[idx[cut:]].assign(probe=m.predict(X[idx[cut:]]))
        probe_id.append(score(test, "probe", rng))
    s = pd.concat(probe_id)
    s[["model", "setting"]] = "Probe", "ID"
    rows.append(s)
    # OOD: fit on (up to 800 items of) each training benchmark, score on the unseen benchmarks.
    train_idx = np.concatenate([rng.choice(g.index.to_numpy(), min(800, len(g)), replace=False)
                                for _, g in t[t.benchmark_split == "train"].groupby("benchmark_id")])
    m = RidgeCV(alphas=alphas).fit(X[train_idx], y[train_idx])
    unseen = t[t.benchmark_split != "train"].copy()
    unseen["probe"] = m.predict(X[unseen.index.to_numpy()])
    s = score(unseen, "probe", rng)
    s[["model", "setting"]] = "Probe", "OOD"
    rows.append(s)

    per_bench = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    per_bench.to_parquet(OUT / "difficulty-rank-metrics.parquet", index=False)
    summary = (per_bench.groupby(["model", "setting"])
               .agg(benchmarks=("rho", "size"), rho=("rho", "mean"), pairwise_acc=("pairwise_acc", "mean")).round(3))
    print(summary.unstack("setting").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
