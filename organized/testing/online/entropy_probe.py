#!/usr/bin/env python3
"""Probe from LM entropy statistics vs the item-embedding probe, on the same items (ID and OOD).

    python organized/testing/online/entropy_probe.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
sys.path.insert(0, str(Path(__file__).parent))
from difficulty_rank_metrics import score, EMB

OUT = Path(__file__).parent / "results"
FEATS = ["ent_mean", "ent_std", "ent_q10", "ent_q25", "ent_q50", "ent_q75", "ent_q90", "ent_topk"]
ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]


def main() -> int:
    rng = np.random.default_rng(20261102)
    t = pd.read_parquet(OUT / "lm-entropy-pilot-3282.parquet").dropna(subset=FEATS).reset_index(drop=True)
    ids = pd.read_parquet(EMB / "items.parquet")["item_uid"].astype(str).to_numpy()
    pos = {u: i for i, u in enumerate(ids)}
    store = np.load(EMB / "item_embeddings.npy", mmap_mode="r")
    emb = np.asarray(store[[pos[u] for u in t.item_uid.astype(str)]], dtype=np.float32)
    ent_raw = t[FEATS].to_numpy(float)
    g = t.groupby("benchmark_id")[FEATS]
    ent_std = ((t[FEATS] - g.transform("mean")) / (g.transform("std") + 1e-9)).to_numpy(float)
    y = (t["observed"] - t.groupby("benchmark_id")["observed"].transform("mean")).to_numpy()
    train = (t.benchmark_split == "train").to_numpy()
    lt = np.log1p(t[["n_tokens"]].to_numpy(float))
    lt = (lt - t.assign(l=lt[:, 0]).groupby("benchmark_id")["l"].transform("mean").to_numpy()[:, None])
    z = lambda X: (X - X[train].mean(0)) / (X[train].std(0) + 1e-9)
    sets = {"Probe (item embeddings)": emb, "Entropy probe, raw": ent_raw, "Entropy probe, standardized per benchmark": ent_std,
            "Token count": lt, "Entropy + token count": np.hstack([z(ent_std), z(lt)]),
            "Embeddings + token count": np.hstack([z(emb), 10 * z(lt)]),
            "Entropy + embeddings": np.hstack([z(emb), 10 * z(ent_std)]),
            "Entropy + embeddings + token count": np.hstack([z(emb), 10 * z(ent_std), 10 * z(lt)])}
    rows = []
    for name, X in sets.items():
        if name.startswith("Entropy probe"):
            X = z(X)
        for b, gi in t[train].groupby("benchmark_id"):
            idx = rng.permutation(gi.index.to_numpy()); cut = int(0.8 * len(idx))
            m = RidgeCV(alphas=ALPHAS).fit(X[idx[:cut]], y[idx[:cut]])
            s = score(t.loc[idx[cut:]].assign(p=m.predict(X[idx[cut:]])), "p", rng)
            rows.append(s.assign(model=name, setting="ID"))
        m = RidgeCV(alphas=ALPHAS).fit(X[train], y[train])
        un = t[~train].assign(p=m.predict(X[~train]))
        rows.append(score(un, "p", rng).assign(model=name, setting="OOD"))
    pb = pd.concat(rows, ignore_index=True)
    pb.to_parquet(OUT / "entropy-probe-metrics.parquet", index=False)
    print(pb.groupby(["model", "setting"]).agg(benchmarks=("rho", "size"), rho=("rho", "mean"),
          pairwise_acc=("pairwise_acc", "mean"), pos=("rho", lambda r: int((r > 0).sum()))).round(3).to_string())
    print(pb[pb.setting == "OOD"].pivot(index="benchmark_id", columns="model", values="rho").round(3).to_string())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
