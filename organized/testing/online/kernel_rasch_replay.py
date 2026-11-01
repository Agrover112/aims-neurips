#!/usr/bin/env python3
"""Non-IID items: duplicate check, clustered and kernel Rasch, diverse acquisition.

Pooled Rasch, ``logit p(s, i) = mu_b + theta_s - b_i``, gives every item an
independent difficulty prior, so a label informs only that exact item. Within
a task, answers on text-similar items are correlated (noniid_beta_replay.py),
so here the difficulties share a prior covariance within each benchmark:

    b ~ N(0, K),  K = tau^2 [(1 - rho) I + rho S]

    rasch     S = I                         (rho has no effect; pooled Rasch)
    cluster   S_ij = 1[i, j in the same k-means cluster of item embeddings]
    kernel    S_ij = exp((cos(x_i, x_j) - 1) / T)

All are MAP fits, refit at every checkpoint on every pair's revealed labels.
An item nobody has labelled gets the kriging estimate b_i = K_iL K_LL^-1 b_L,
so a label on a similar item moves it; with S = I that is 0, which reproduces
pooled Rasch exactly.

Steps:

    1. duplicate check: is the within-task agreement of similar items driven by
       near-copies (cosine > 0.97) or spread across similarities?
    2. settings for cluster and kernel chosen on the validation draws;
    3. scored on the same test draws as the pooled-Rasch result, paired;
    4. diverse acquisition: each pair buys labels so its item clusters stay
       evenly covered (streaming: accept a candidate whose cluster is among the
       least covered, with the usual reserve rule), against random acquisition,
       for the beta-binomial and the best model.

The multi-pair draws are rebuilt from pooled-rasch/curves.parquet, so every
row is paired with the existing report numbers.

    python organized/testing/online/kernel_rasch_replay.py --cache /home/agrov/aims-replay-cache
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "organized" / "testing" / "online" / "results"
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
M0 = float(np.log(0.512 / 0.488))
TAU_B, TAU_S = 2.0, 1.0

# Filled once per worker process by _init.
G: dict = {}


def _init(cache: str, embeddings: str, n_clusters: tuple) -> None:
    from sklearn.cluster import KMeans

    cache, embeddings = Path(cache), Path(embeddings)
    items = pd.read_parquet(cache / "panel" / "items.parquet")
    ids = pd.read_parquet(embeddings / "items.parquet")["item_uid"].astype(str).to_numpy()
    pos = {u: i for i, u in enumerate(ids)}
    store = np.load(embeddings / "item_embeddings.npy", mmap_mode="r")
    uniq = items.drop_duplicates("content_hash")
    vec = np.asarray(store[[pos[str(u)] for u in uniq["item_uid"]]], dtype=np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True).clip(1e-8)
    G["vec"] = dict(zip(uniq["content_hash"], vec))
    # Clusters per benchmark, from every panel item of that benchmark.
    clusters = {}
    for b, g in uniq.groupby("benchmark_id"):
        v = np.stack([G["vec"][h] for h in g["content_hash"]])
        for k in n_clusters:
            lab = KMeans(n_clusters=min(k, len(v)), n_init=4, random_state=0).fit_predict(v)
            clusters.update({(k, h): int(c) for h, c in zip(g["content_hash"], lab)})
    G["cluster"] = clusters
    G["items"] = items
    meta = items.drop_duplicates("task_id").set_index("task_id")[["benchmark_id", "subject_uid"]]
    G["meta"] = meta
    ev = items[items["pool"] == "evaluation"]
    G["eval"] = {t: (g["content_hash"].to_numpy(), g["label"].to_numpy(float)) for t, g in ev.groupby("task_id")}
    label = items.set_index(["task_id", "item_uid"])["label"]
    hashes = items.set_index(["task_id", "item_uid"])["content_hash"]
    orders = pd.read_parquet(cache / "panel" / "candidate_orders.parquet").sort_values(
        ["task_id", "order_seed", "stream_position"])
    G["stream"] = {(t, int(s)): [(hashes[(t, u)], int(label[(t, u)])) for u in g["item_uid"]]
                   for (t, s), g in orders.groupby(["task_id", "order_seed"])}


def _similarity(kind: str, param, hs: list[str]) -> np.ndarray:
    if kind == "rasch" or len(hs) == 0:
        return np.eye(len(hs))
    if kind == "cluster":
        c = np.array([G["cluster"][(param, h)] for h in hs])
        return (c[:, None] == c[None, :]).astype(float)
    v = np.stack([G["vec"][h] for h in hs])
    return np.exp((v @ v.T - 1.0) / param)


def _cross(kind: str, param, targets: list[str], hs: list[str]) -> np.ndarray:
    if kind == "rasch" or not hs:
        return np.zeros((len(targets), len(hs)))
    if kind == "cluster":
        ct = np.array([G["cluster"][(param, h)] for h in targets])
        cl = np.array([G["cluster"][(param, h)] for h in hs])
        return (ct[:, None] == cl[None, :]).astype(float)
    vt = np.stack([G["vec"][h] for h in targets])
    vl = np.stack([G["vec"][h] for h in hs])
    return np.exp((vt @ vl.T - 1.0) / param)


def fit_predict(evidence, targets_by_task, cfg) -> dict:
    """MAP fit on evidence rows (bench, subject, hash, y); predictions per task."""
    kind, param, rho, tau = cfg
    benches = sorted({e[0] for e in evidence})
    subjects = sorted({e[1] for e in evidence})
    items = sorted({(e[0], e[2]) for e in evidence})
    nb, ns, ni = len(benches), len(subjects), len(items)
    bi = {b: j for j, b in enumerate(benches)}
    si = {s: nb + j for j, s in enumerate(subjects)}
    ii = {it: nb + ns + j for j, it in enumerate(items)}
    n = nb + ns + ni
    X = np.zeros((len(evidence), n))
    for r, (b, s, h, _) in enumerate(evidence):
        X[r, bi[b]] = X[r, si[s]] = 1.0
        X[r, ii[(b, h)]] = -1.0
    y = np.array([e[3] for e in evidence], float)
    prec = np.zeros((n, n))
    prec[np.arange(nb), np.arange(nb)] = TAU_B ** -2
    prec[np.arange(nb, nb + ns), np.arange(nb, nb + ns)] = TAU_S ** -2
    inv_blocks = {}
    for b in benches:   # block-diagonal item prior: similarity only within a benchmark
        hs = [h for (bb, h) in items if bb == b]
        idx = np.array([ii[(b, h)] for h in hs])
        K = tau ** 2 * ((1 - rho) * np.eye(len(hs)) + rho * _similarity(kind, param, hs)) + 1e-6 * np.eye(len(hs))
        Kinv = np.linalg.inv(K)
        prec[np.ix_(idx, idx)] = Kinv
        inv_blocks[b] = (hs, idx, Kinv)
    mean = np.concatenate([np.full(nb, M0), np.zeros(ns + ni)])
    v = mean.copy()
    for _ in range(30):
        p = 1 / (1 + np.exp(-(X @ v)))
        grad = X.T @ (p - y) + prec @ (v - mean)
        hess = (X * (p * (1 - p))[:, None]).T @ X + prec
        step = np.linalg.solve(hess, grad)
        v -= step
        if np.abs(step).max() < 1e-6:
            break
    out = {}
    for t, (b, s, hashes) in targets_by_task.items():
        mu = v[bi[b]] if b in bi else M0
        th = v[si[s]] if s in si else 0.0
        diff = np.zeros(len(hashes))
        if b in inv_blocks:
            hs, idx, Kinv = inv_blocks[b]
            bL = v[idx]
            known = {h: v[ii[(b, h)]] for h in hs}
            cross = tau ** 2 * rho * _cross(kind, param, list(hashes), hs)
            diff = cross @ (Kinv @ bL)
            for j, h in enumerate(hashes):
                if h in known:
                    diff[j] = known[h]
        out[t] = 1 / (1 + np.exp(-(mu + th - diff)))
    return out


def _acquired(t: int, seed: int, mode: str, k_div: int = 8) -> list:
    stream = G["stream"][(t, seed)]
    if mode == "random":
        return stream[:31]
    present = {G["cluster"][(k_div, hh)] for hh, _ in stream}   # clusters this stream can offer
    counts, taken = {}, []
    for pos, (h, y) in enumerate(stream):
        if len(taken) == 31:
            break
        c = G["cluster"][(k_div, h)]
        least = min(counts.get(cc, 0) for cc in present)
        remaining = len(stream) - pos
        if counts.get(c, 0) <= least or remaining <= 31 - len(taken):
            taken.append((h, y))
            counts[c] = counts.get(c, 0) + 1
    return taken


def score_draw(args) -> list:
    """Brier per (task, budget, config) for one draw."""
    draw_id, pairs, configs, mode = args
    meta, ev = G["meta"], G["eval"]
    acquired = {(t, s): _acquired(t, s, mode) for t, s in pairs}
    rows = []
    targets = {t: (meta.loc[t, "benchmark_id"], meta.loc[t, "subject_uid"], ev[t][0]) for t, _ in pairs}
    for k in BUDGETS:
        evidence = [(meta.loc[t, "benchmark_id"], meta.loc[t, "subject_uid"], h, y)
                    for t, s in pairs for h, y in acquired[(t, s)][:k]]
        for t, s in pairs:
            own = [y for _, y in acquired[(t, s)][:k]]
            p = (sum(own) + 0.5) / (k + 1)
            rows.append((draw_id, t, s, k, "beta", float(np.mean((p - ev[t][1]) ** 2))))
        for name, cfg in configs.items():
            if not evidence:
                preds = {t: np.full(len(ev[t][1]), 1 / (1 + np.exp(-M0))) for t, _ in pairs}
            else:
                preds = fit_predict(evidence, targets, cfg)
            for t, s in pairs:
                rows.append((draw_id, t, s, k, name, float(np.mean((preds[t] - ev[t][1]) ** 2))))
    return rows


def run(draws: dict, configs: dict, mode: str, workers: int, init_args) -> pd.DataFrame:
    jobs = [(d, pairs, configs, mode) for d, pairs in draws.items()]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=init_args) as ex:
        rows = [r for chunk in ex.map(score_draw, jobs, chunksize=2) for r in chunk]
    return pd.DataFrame(rows, columns=["draw", "task_id", "order_seed", "budget", "model", "brier"])


def task_alc(f: pd.DataFrame) -> pd.DataFrame:
    tb = f.groupby(["model", "task_id", "budget"]).brier.mean().unstack("budget")[BUDGETS]
    return (tb @ WEIGHTS).unstack("model")


def boot(x, rng, n=4000):
    b = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def duplicate_check(cache: Path, embeddings: Path) -> pd.DataFrame:
    """Residual agreement of nearest-neighbour item pairs, by their cosine."""
    _init(str(cache), str(embeddings), ())
    rows = []
    for (role, task), g in G["items"].groupby(["benchmark_split", "task_id"]):
        g = g.drop_duplicates("content_hash")
        if len(g) < 20:
            continue
        y = g["label"].to_numpy(float)
        r = y - y.mean()
        if r.std() == 0:
            continue
        v = np.stack([G["vec"][h] for h in g["content_hash"]])
        sim = v @ v.T
        np.fill_diagonal(sim, -np.inf)
        nn = sim.argmax(1)
        c = sim[np.arange(len(g)), nn]
        for j in range(len(g)):
            rows.append((role, g["benchmark_id"].iloc[0], c[j], r[j] * r[nn[j]] / r.var()))
    return pd.DataFrame(rows, columns=["role", "benchmark_id", "nn_cos", "agreement"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path.home() / "aims-replay-cache")
    ap.add_argument("--embeddings", type=Path,
                    default=Path.home() / "aims-noitem-run/datasets/full/allbench-embed-bge_m3_long-v2")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rng = np.random.default_rng(20261101)

    # 1. Duplicate check.
    dup = duplicate_check(a.cache, a.embeddings)
    dup["bin"] = pd.cut(dup.nn_cos, [0, 0.8, 0.9, 0.95, 0.97, 1.0001])
    print("== 1. Residual agreement of a nearest-neighbour pair, by its cosine")
    print(dup.groupby(["role", "bin"], observed=True).agreement.agg(["size", "mean"]).round(3).to_string())
    print(dup.assign(near_copy=dup.nn_cos > 0.97).groupby(["role", "benchmark_id"]).agg(
        near_copy_share=("near_copy", "mean"), agreement=("agreement", "mean")).round(3).to_string())

    # Draws exactly as in the pooled-Rasch curves run.
    c = pd.read_parquet(a.cache / "curves.parquet")
    c = c[c.model == "jeffreys"].drop_duplicates(["role", "draw", "task_id"])
    draws = {role: {d: list(zip(g.task_id, g.order_seed.astype(int))) for d, g in c[c.role == role].groupby("draw")}
             for role in ("validation", "test")}
    n_clusters = (4, 8, 16)
    init = (str(a.cache), str(a.embeddings), n_clusters)

    # 2. Settings on validation.
    configs = {"rasch": ("rasch", None, 0.0, 1.0)}
    for tau in (1.0, 1.4):
        configs[f"rasch tau={tau}"] = ("rasch", None, 0.0, tau)
        for rho, k in itertools.product((0.5, 0.8), n_clusters):
            configs[f"cluster k={k} rho={rho} tau={tau}"] = ("cluster", k, rho, tau)
        for rho, T in itertools.product((0.5, 0.8), (0.05, 0.1, 0.2)):
            configs[f"kernel T={T} rho={rho} tau={tau}"] = ("kernel", T, rho, tau)
    val = run(draws["validation"], configs, "random", a.workers, init)
    va = task_alc(val).mean().sort_values()
    print("\n== 2. Validation macro ALC (multi-pair, random acquisition)")
    print(va.round(4).to_string())
    best = {kind: min((m for m in va.index if m.startswith(kind)), key=va.get) for kind in ("cluster", "kernel")}
    chosen = {"Pooled Rasch": configs["rasch"], "Clustered Rasch": configs[best["cluster"]],
              "Kernel Rasch": configs[best["kernel"]]}
    print("chosen:", best)

    # 3. Test, random acquisition; 4. diverse acquisition.
    results, frames = {}, []
    for mode in ("random", "diverse"):
        f = run(draws["test"], chosen, mode, a.workers, init)
        f["acquisition"] = mode
        frames.append(f)
        ta = task_alc(f)
        print(f"\n== {'3' if mode == 'random' else '4'}. Test, {mode} acquisition: ALC and difference vs beta-binomial")
        for m in ta.columns:
            d = boot((ta[m] - ta["beta"]).to_numpy(), rng)
            per_b = (f[f.model == m].groupby("budget").brier.mean() - f[f.model == "beta"].groupby("budget").brier.mean()).round(4).to_dict()
            results[f"{mode}/{m}"] = {"alc": float(ta[m].mean()), "diff_vs_beta": d, "per_budget": per_b}
            print(f"  {m:<16} {ta[m].mean():.4f}  {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]  per budget {per_b}")
        if mode == "random":
            for m in ("Clustered Rasch", "Kernel Rasch"):
                d = boot((ta[m] - ta["Pooled Rasch"]).to_numpy(), rng)
                results[f"random/{m} minus pooled"] = d
                print(f"  {m} minus Pooled Rasch: {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]")
    both = pd.concat(frames)
    tr, td = task_alc(both[both.acquisition == "random"]), task_alc(both[both.acquisition == "diverse"])
    for m in tr.columns:
        d = boot((td[m] - tr[m]).to_numpy(), rng)
        results[f"diverse minus random/{m}"] = d
        print(f"  diverse minus random acquisition, {m}: {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]")

    OUT.mkdir(parents=True, exist_ok=True)
    val.assign(acquisition="random", role="validation").to_parquet(OUT / "kernel-rasch-validation.parquet", index=False)
    both.assign(role="test").to_parquet(OUT / "kernel-rasch-test.parquet", index=False)
    dup.drop(columns="bin").to_parquet(OUT / "kernel-rasch-duplicates.parquet", index=False)
    (OUT / "kernel-rasch-summary.json").write_text(json.dumps(
        {"validation_alc": va.to_dict(), "chosen": best, "test": results}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
