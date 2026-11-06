#!/usr/bin/env python3
"""Run any set of models through the online simulation on frozen draws, saving scores and every prediction.

    python organized/testing/eandd/run_replay.py --run wp3-repro-comp --shape competition --role test \\
        --models jeffreys,rasch-old,rasch-retuned,kernel-old

A model is a config: ("beta", alpha, beta), or the kernel_rasch_replay config
(kind, param, rho, tau_item, tau_b, tau_s, m0). Nothing here depends on how a config was chosen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "online"))
import common as c  # noqa: E402
import kernel_rasch_replay as k  # noqa: E402

M0 = k.M0
MODELS = {
    "jeffreys": ("beta", 0.5, 0.5),
    # Fixed in advance: nothing learned.
    "rasch-fixed": ("rasch", None, 0.0, 1.0, 1.0, 1.0, 0.0),
    "kernel-fixed": ("kernel", 0.1, 0.5, 1.0, 1.0, 1.0, 0.0),
    # Tuned on validation (the existing settings).
    "rasch-old": ("rasch", None, 0.0, 1.0, 2.0, 1.0, M0),
    "rasch-retuned": ("rasch", None, 0.0, 1.0, 2.0, 0.5, M0),
    "kernel-old": ("kernel", 0.2, 0.8, 1.4, 2.0, 1.0, M0),
}


_OFFLINE: dict = {}


def _offline(run: str) -> dict:
    """Fitted values of an offline Rasch run: benchmark rates, item difficulties keyed (benchmark, content_hash)."""
    if run not in _OFFLINE:
        f = pd.read_parquet(c.DATA / "fits" / f"{run}.parquet")
        h = dict(zip(f[f.kind == "hyper"].key, f[f.kind == "hyper"].value))
        mu = dict(zip(f[f.kind == "mu"].key, f[f.kind == "mu"].value))
        bi = dict(zip(f[f.kind == "b"].key, f[f.kind == "b"].value))
        items = k.G["items"].drop_duplicates(["benchmark_id", "content_hash"])
        b = {(bm, ch): bi[u] for bm, ch, u in zip(items.benchmark_id, items.content_hash, items.item_uid) if u in bi}
        _OFFLINE[run] = {"h": h, "mu": mu, "b": b}
    return _OFFLINE[run]


def offline_predict(run: str, bench: str, hashes, own: list, theta_prior: float | None = None) -> np.ndarray:
    """Offline Rasch with fitted mu_b and b_i kept; only theta is fit online (1-d MAP on the task's own labels).
    A benchmark or item the offline fit never saw falls back to m0 and 0."""
    o = _offline(run)
    tau_s = o["h"]["tau_s"] if theta_prior is None else theta_prior
    mu = o["mu"].get(bench, o["h"]["m0"])
    offs = lambda hs: np.array([mu - o["b"].get((bench, h), 0.0) for h in hs])
    theta = 0.0
    if own:
        off = offs([h for h, _ in own]); y = np.array([y for _, y in own], float)
        for _ in range(20):
            p = 1 / (1 + np.exp(-(off + theta)))
            g = (p - y).sum() + theta / tau_s ** 2
            hess = (p * (1 - p)).sum() + 1 / tau_s ** 2
            theta -= g / hess
    return 1 / (1 + np.exp(-(offs(hashes) + theta)))


def _write_preds(path: Path, rows: list) -> None:
    table = pa.table({
        "model": pa.array([r[0] for r in rows for _ in r[5]]).dictionary_encode(),
        "draw": pa.array(np.concatenate([np.full(len(r[5]), r[1], np.int32) for r in rows])),
        "task_id": pa.array([r[2] for r in rows for _ in r[5]]).dictionary_encode(),
        "order_seed": pa.array(np.concatenate([np.full(len(r[5]), r[3], np.int32) for r in rows])),
        "budget": pa.array(np.concatenate([np.full(len(r[5]), r[4], np.int8) for r in rows])),
        "content_hash": pa.array([h for r in rows for h in r[5]]).dictionary_encode(),
        "p": pa.array(np.concatenate([r[6] for r in rows]).astype(np.float32)),
        "y": pa.array(np.concatenate([r[7] for r in rows]).astype(np.int8)),
    })
    pq.write_table(table, path, compression="zstd")


def score_draw(args) -> list:
    draw, pairs, configs, preds_dir = args
    meta, ev = k.G["meta"], k.G["eval"]
    acquired = {(t, s): k._acquired(t, s, "random") for t, s in pairs}
    targets = {t: (meta.loc[t, "benchmark_id"], meta.loc[t, "subject_uid"], ev[t][0]) for t, _ in pairs}
    rows, keep = [], []
    for budget in c.BUDGETS:
        evidence = [(meta.loc[t, "benchmark_id"], meta.loc[t, "subject_uid"], h, y)
                    for t, s in pairs for h, y in acquired[(t, s)][:budget]]
        for name, cfg in configs.items():
            if cfg[0] == "offline":
                preds = {t: offline_predict(cfg[1], meta.loc[t, "benchmark_id"], ev[t][0], acquired[(t, s)][:budget])
                         for t, s in pairs}
            elif cfg[0] == "beta":
                preds = {}
                for t, s in pairs:
                    own = [y for _, y in acquired[(t, s)][:budget]]
                    preds[t] = np.full(len(ev[t][1]), (sum(own) + cfg[1]) / (len(own) + cfg[1] + cfg[2]))
            elif not evidence:
                m0 = cfg[6] if len(cfg) > 6 else M0
                preds = {t: np.full(len(ev[t][1]), 1 / (1 + np.exp(-m0))) for t, _ in pairs}
            else:
                preds = k.fit_predict(evidence, targets, cfg)
            for t, s in pairs:
                rows.append((name, draw, t, s, budget, float(np.mean((preds[t] - ev[t][1]) ** 2))))
                keep.append((name, draw, t, s, budget, list(ev[t][0]), preds[t], ev[t][1]))
    if preds_dir:
        _write_preds(Path(preds_dir) / f"draw-{draw:05d}.parquet", keep)
    return rows


def run(draws: dict, configs: dict, workers: int, preds_dir: Path | None, cache=c.CACHE, emb=c.EMB) -> pd.DataFrame:
    jobs = [(d, pairs, configs, str(preds_dir) if preds_dir else None) for d, pairs in draws.items()]
    with ProcessPoolExecutor(workers, initializer=k._init, initargs=(str(cache), str(emb), ())) as ex:
        rows = [r for chunk in ex.map(score_draw, jobs, chunksize=2) for r in chunk]
    return pd.DataFrame(rows, columns=["model", "draw", "task_id", "order_seed", "budget", "brier"])


def alc_table(scores: pd.DataFrame, base: str = "jeffreys", seed: int = 20261101) -> pd.DataFrame:
    """Per-task ALC, then paired bootstrap over tasks against the base model."""
    w = pd.Series(c.WEIGHTS)
    tb = scores.groupby(["model", "task_id", "budget"]).brier.mean().unstack("budget")[c.BUDGETS]
    ta = (tb * w).sum(axis=1).unstack("model")
    rng, out = np.random.default_rng(seed), []
    for m in ta.columns:
        est, lo, hi = k.boot((ta[m] - ta[base]).to_numpy(), rng)
        out.append((m, ta[m].mean(), est, lo, hi))
    return pd.DataFrame(out, columns=["model", "alc", "diff", "lo", "hi"]).sort_values("alc")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--shape", required=True, choices=c.SHAPES)
    ap.add_argument("--role", required=True, choices=c.ROLES)
    ap.add_argument("--models", required=True, help="comma-separated names from MODELS or --extra")
    ap.add_argument("--extra", type=Path, help="JSON file of further {name: config}")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-preds", action="store_true")
    a = ap.parse_args()
    known = dict(MODELS)
    if a.extra:
        known.update({n: tuple(v) for n, v in json.loads(a.extra.read_text()).items()})
    configs = {m: known[m] for m in a.models.split(",")}
    started = time.time()
    preds_dir = None if a.no_preds else c.DATA / "preds" / a.run
    if preds_dir:
        preds_dir.mkdir(parents=True, exist_ok=False)
    cache = c.ID_PANEL if a.role == "id" else c.CACHE     # the replay reads <cache>/panel/{items,candidate_orders}.parquet
    scores = run(c.load_draws(a.shape, a.role), configs, a.workers, preds_dir, cache=cache).assign(shape=a.shape, role=a.role)
    c.save_run(a.run, {"shape": a.shape, "role": a.role, "models": configs, "workers": a.workers}, started, scores=scores)
    print(alc_table(scores).round(4).to_string(index=False))
    print(f"runtime {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
