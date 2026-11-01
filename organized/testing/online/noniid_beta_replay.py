#!/usr/bin/env python3
"""Are a subject's answers IID within a task, and does it matter?

The beta-binomial treats a subject's answers on one benchmark as IID draws
with one rate, so every revealed label says the same about every unseen item.
Two checks against that, on the frozen replay panel:

1. Diagnostic. After removing each task's mean, do a subject's answers on two
   text-similar items (nearest neighbour in BGE-M3 space, same task) agree
   more than on two random items of the task? Under IID they agree equally.

2. Model. A group-weighted beta-binomial lets labels on similar items count
   more for a target item:

       g    = (hits + 0.5) / (labels + 1)                 the task estimate
       w_ij = exp((cos(i, j) - 1) / T)                     similarity weight
       p_i  = (kappa * g + sum_j w_ij * y_j) / (kappa + sum_j w_ij)

   kappa sets how much the task estimate counts against local evidence; T how
   sharply similarity decays. With kappa -> infinity it is the beta-binomial.
   Both are chosen on validation and scored once on test, with the replay's
   random-prefix acquisition orders, so it is paired with the beta-binomial.

Per-task x budget scores are written for the report's figures.

    python organized/testing/online/noniid_beta_replay.py \\
        --cache /home/agrov/aims-replay-cache \\
        --embeddings /home/agrov/aims-noitem-run/datasets/full/allbench-embed-bge_m3_long-v2
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "organized" / "testing" / "online" / "results"
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
SEEDS = {"validation": "s3", "test": "s10"}


def load(cache: Path, embeddings: Path):
    items = pd.read_parquet(cache / "panel" / "items.parquet")
    ids = pd.read_parquet(embeddings / "items.parquet")["item_uid"].astype(str).to_numpy()
    pos = {u: i for i, u in enumerate(ids)}
    store = np.load(embeddings / "item_embeddings.npy", mmap_mode="r")
    return items, pos, store


def task_vectors(group: pd.DataFrame, pos: dict, store) -> np.ndarray:
    v = np.asarray(store[[pos[str(u)] for u in group["item_uid"]]], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-8)


def diagnostic(items: pd.DataFrame, pos: dict, store, rng) -> pd.DataFrame:
    """Residual agreement for nearest-neighbour vs random item pairs, per task."""
    rows = []
    for (role, task), g in items.groupby(["benchmark_split", "task_id"]):
        g = g.drop_duplicates("item_uid")
        if len(g) < 20:
            continue
        y = g["label"].to_numpy(float)
        r = y - y.mean()
        if r.std() == 0:
            continue
        v = task_vectors(g, pos, store)
        sim = v @ v.T
        np.fill_diagonal(sim, -np.inf)
        nn = sim.argmax(1)
        rand = rng.integers(0, len(g) - 1, len(g))
        rand = rand + (rand >= np.arange(len(g)))                 # never pair an item with itself
        rows.append((role, task, g["benchmark_id"].iloc[0], len(g),
                     float(np.mean(r * r[nn]) / r.var()), float(np.mean(r * r[rand]) / r.var()),
                     float(np.mean(sim[np.arange(len(g)), nn]))))
    return pd.DataFrame(rows, columns=["role", "task_id", "benchmark_id", "items", "nn_corr", "random_corr", "nn_cos"])


def replay(items: pd.DataFrame, pos: dict, store, cache: Path, role: str, grid: list[tuple]):
    """Task x budget Brier for the beta-binomial and each (kappa, T)."""
    run = cache / "runs" / f"{role}-beta-random_prefix-{SEEDS[role]}-v2" / "acquisition_decisions.parquet"
    d = pd.read_parquet(run)
    d = d[d["accepted"]].sort_values(["task_id", "order_seed", "labels_acquired_before"])
    sub = items[items["benchmark_split"] == role]
    rows = []
    for task, g in sub.groupby("task_id"):
        g = g.drop_duplicates("item_uid").reset_index(drop=True)
        v = task_vectors(g, pos, store)
        index = {u: i for i, u in enumerate(g["item_uid"].astype(str))}
        ev = g.index[g["pool"] == "evaluation"].to_numpy()
        y_ev = g.loc[ev, "label"].to_numpy(float)
        sim = v[ev] @ v.T                                         # evaluation items x all task items
        for seed, s in d[d["task_id"] == task].groupby("order_seed"):
            acq = np.array([index[str(u)] for u in s["item_uid"]])
            lab = s["revealed_label"].to_numpy(float)
            for b, k in enumerate(BUDGETS):
                ctx, yc = acq[:k], lab[:k]
                g_hat = (yc.sum() + 0.5) / (k + 1)
                row = {"task_id": task, "order_seed": int(seed), "budget": k,
                       "beta": float(np.mean((g_hat - y_ev) ** 2))}
                for kappa, temp in grid:
                    if k == 0:
                        p = np.full(len(ev), g_hat)
                    else:
                        w = np.exp((sim[:, ctx] - 1.0) / temp)
                        p = (kappa * g_hat + w @ yc) / (kappa + w.sum(1))
                    row[f"k{kappa}_T{temp}"] = float(np.mean((p - y_ev) ** 2))
                rows.append(row)
    f = pd.DataFrame(rows)
    return f.groupby(["task_id", "budget"]).mean(numeric_only=True).drop(columns="order_seed")


def alc(tb: pd.DataFrame, col: str) -> pd.Series:
    m = tb[col].unstack()[BUDGETS]
    return m @ WEIGHTS


def boot(x: np.ndarray, rng, n: int = 4000):
    b = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path.home() / "aims-replay-cache")
    ap.add_argument("--embeddings", type=Path,
                    default=Path.home() / "aims-noitem-run/datasets/full/allbench-embed-bge_m3_long-v2")
    a = ap.parse_args()
    rng = np.random.default_rng(20261101)
    items, pos, store = load(a.cache, a.embeddings)

    diag = diagnostic(items, pos, store, rng)
    print("== 1. Residual agreement within a task (0 = IID)")
    for role, g in diag.groupby("role"):
        nn, rd = boot(g.nn_corr.to_numpy(), rng), boot(g.random_corr.to_numpy(), rng)
        print(f"  {role:<10} {len(g):3d} tasks | nearest neighbour {nn[0]:+.3f} [{nn[1]:+.3f},{nn[2]:+.3f}]"
              f" | random pair {rd[0]:+.3f} [{rd[1]:+.3f},{rd[2]:+.3f}] | mean NN cosine {g.nn_cos.mean():.2f}")
    print(diag.groupby(["role", "benchmark_id"])[["nn_corr", "random_corr"]].mean().round(3).to_string())

    grid = list(itertools.product((0.5, 1, 2, 5, 10, 30), (0.02, 0.05, 0.1)))
    val = replay(items, pos, store, a.cache, "validation", grid)
    val_alc = {c: float(alc(val, c).mean()) for c in val.columns}
    best = min((c for c in val.columns if c != "beta"), key=val_alc.get)
    print(f"\n== 2. Group-weighted beta-binomial; validation ALC beta {val_alc['beta']:.4f}, best {best} {val_alc[best]:.4f}")
    test = replay(items, pos, store, a.cache, "test", grid)
    out = {"diagnostic": diag.groupby("role")[["nn_corr", "random_corr"]].mean().to_dict(),
           "chosen": best, "validation_alc": val_alc, "test": {}}
    for role, tb in (("validation", val), ("test", test)):
        d_alc = (alc(tb, best) - alc(tb, "beta")).to_numpy()
        m, lo, hi = boot(d_alc, rng)
        per_b = (tb[best] - tb["beta"]).groupby(level="budget").mean().round(4).to_dict()
        out["test" if role == "test" else "validation_diff"] = {"alc": float(alc(tb, best).mean()),
            "beta_alc": float(alc(tb, "beta").mean()), "diff": [m, lo, hi], "per_budget": per_b}
        print(f"  {role:<10} beta {alc(tb, 'beta').mean():.4f}  group-weighted {alc(tb, best).mean():.4f}"
              f"  diff {m:+.4f} [{lo:+.4f},{hi:+.4f}]  per budget {per_b}")
    OUT.mkdir(parents=True, exist_ok=True)
    long = []
    for role, tb in (("validation", val), ("test", test)):
        for col, model in (("beta", "beta"), (best, "group-weighted-beta")):
            x = tb[col].rename("brier").reset_index()
            x[["role", "model"]] = role, model
            long.append(x)
    pd.concat(long).to_parquet(OUT / "noniid-beta-task-budget.parquet", index=False)
    diag.to_parquet(OUT / "noniid-diagnostic.parquet", index=False)
    (OUT / "noniid-beta-summary.json").write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
