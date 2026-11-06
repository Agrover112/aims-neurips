#!/usr/bin/env python3
"""Leave-one-benchmark-out for fitted priors, on the 11 training benchmarks.

For each training benchmark K: fit the offline Rasch hyperparameters and the Beta-Binomial empirical-Bayes prior on the
other 10 benchmarks (every row of K removed), then score the held-out-subject tasks of K from the ID panel with
(i) Jeffreys, (ii) the Beta-Binomial prior fitted without K, (iii) the Rasch prior fitted without K, (iv) the fixed
Rasch prior. K's base rate relative to the other ten benchmarks' range is recorded.

    python organized/testing/eandd/lobo_prior.py --run wp10-lobo-v1
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
from scipy.optimize import minimize
from scipy.special import betaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "online"))
import common as c  # noqa: E402
import offline_rasch as o  # noqa: E402
import run_replay as r  # noqa: E402
import kernel_rasch_replay as k  # noqa: E402


def eb_beta(counts: pd.DataFrame, weight: str) -> tuple[float, float]:
    """Beta-Binomial marginal-likelihood fit of (mean, strength) to task-level (hits, n)."""
    h, n = counts.hits.to_numpy(float), counts.n.to_numpy(float)
    w = np.ones(len(counts)) if weight == "task" else (1.0 / counts.groupby("benchmark_id").hits.transform("size")).to_numpy()
    w = w / w.sum() * len(counts)

    def nll(x):
        m, s = 1 / (1 + np.exp(-x[0])), np.exp(x[1])
        a, b = m * s, (1 - m) * s
        return -(w * (betaln(h + a, n - h + b) - betaln(a, b))).sum()
    res = minimize(nll, [0.0, 1.0], method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-8})
    return float(1 / (1 + np.exp(-res.x[0]))), float(np.exp(res.x[1]))


def fold(args):
    bench, rounds = args
    t0 = time.time()
    resp = pd.read_parquet(c.RESPONSES, columns=["benchmark_id", "subject_uid", "item_uid", "label", "benchmark_split"])
    resp = resp[resp.benchmark_split == "train"].drop(columns="benchmark_split")
    counts = resp.groupby(["benchmark_id", "subject_uid"]).label.agg(hits="sum", n="size").reset_index()
    if bench is not None:
        resp, counts = resp[resp.benchmark_id != bench], counts[counts.benchmark_id != bench]
    out = {"bench": bench, "eb_task": eb_beta(counts, "task"), "eb_bench": eb_beta(counts, "bench")}
    b, s, i, bu, su, iu, sb, ib = o.index(resp.reset_index(drop=True))
    y = resp.label.to_numpy(float)
    nb, ns, ni = len(bu), len(su), len(iu)
    m0, tb, ts, ti, x = float(np.log(0.512 / 0.488)), 2.0, 1.0, 1.0, None
    for _ in range(rounds):
        x, res = o.fit(b, s, i, y, nb, ns, ni, m0, tb, ts, ti, x0=x, maxiter=300)
        mu, th, bi = x[:nb], x[nb:nb + ns], x[nb + ns:]
        vb, vs, vi = o.post_var(x, b, s, i, nb, ns, ni, tb, ts, ti)
        new = (float(mu.mean()), float(np.sqrt(max(np.var(mu) + vb.mean(), 1e-4))),
               float(np.sqrt(np.mean([np.mean(th[sb == q] ** 2 + vs[sb == q]) for q in range(nb)]))),
               float(np.sqrt(np.mean([np.mean(bi[ib == q] ** 2 + vi[ib == q]) for q in range(nb)]))))
        done = max(abs(a - b_) for a, b_ in zip(new, (m0, tb, ts, ti))) < 0.01
        m0, tb, ts, ti = new
        if done:
            break
    out.update(m0=m0, tau_b=tb, tau_s=ts, tau_i=ti, rates=dict(zip(bu, map(float, mu))), seconds=round(time.time() - t0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--full-check-only", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.full_check_only:     # does my Beta-Binomial EB reproduce Person-B's (0.5216, 3.6755)?
        resp = pd.read_parquet(c.RESPONSES, columns=["benchmark_id", "subject_uid", "label", "benchmark_split"])
        resp = resp[resp.benchmark_split == "train"]
        cn = resp.groupby(["benchmark_id", "subject_uid"]).label.agg(hits="sum", n="size").reset_index()
        print("task-weighted:", eb_beta(cn, "task"), "| benchmark-weighted:", eb_beta(cn, "bench"))
        return 0
    bench = sorted(pd.read_parquet(c.ID_PANEL / "panel" / "items.parquet").benchmark_id.unique())
    with ProcessPoolExecutor(a.workers) as ex:
        folds = list(ex.map(fold, [(b, a.rounds) for b in bench]))
    print(f"folds fitted in {time.time() - t0:.0f}s", flush=True)
    # Score: tasks of benchmark K, one task per draw, every frozen order.
    items = pd.read_parquet(c.ID_PANEL / "panel" / "items.parquet", columns=["task_id", "benchmark_id"]).drop_duplicates("task_id")
    orders = pd.read_parquet(c.ID_PANEL / "panel" / "candidate_orders.parquet", columns=["order_seed"]).order_seed.unique()
    rows = []
    for f in folds:
        K = f["bench"]
        cfgs = {"jeffreys": ("beta", 0.5, 0.5),
                "bb-eb-without-K": ("beta", f["eb_bench"][0] * f["eb_bench"][1], (1 - f["eb_bench"][0]) * f["eb_bench"][1]),
                "rasch-fitted-without-K": ("rasch", None, 0.0, f["tau_i"], f["tau_b"], f["tau_s"], f["m0"]),
                "rasch-fixed": ("rasch", None, 0.0, 1.0, 1.0, 1.0, 0.0)}
        tasks = items[items.benchmark_id == K].task_id.tolist()
        draws = {j: [(t, int(sd))] for j, (t, sd) in enumerate((t, sd) for t in tasks for sd in sorted(orders))}
        sc = r.run(draws, cfgs, a.workers, None, cache=c.ID_PANEL).assign(held_out=K)
        rows.append(sc)
        print(K, "done", flush=True)
    scores = pd.concat(rows, ignore_index=True)
    c.save_run(a.run, {"folds": folds}, t0, scores=scores)
    w = pd.Series(c.WEIGHTS)
    tb = scores.groupby(["held_out", "model", "task_id", "budget"]).brier.mean().unstack("budget")
    alc = (tb * w).sum(axis=1).unstack("model")
    g = alc.groupby("held_out").mean()
    d = g.sub(g["jeffreys"], axis=0).drop(columns="jeffreys")
    rates = {f["bench"]: f["rates"] for f in folds}
    meta = []
    for f in folds:
        others = [v for kk, v in f["rates"].items()]
        meta.append((f["bench"], f["eb_bench"][0], f["eb_bench"][1], f["m0"], f["tau_b"], min(others), max(others), f["seconds"]))
    meta = pd.DataFrame(meta, columns=["held_out", "eb_mean", "eb_strength", "m0", "tau_b", "lowest_rate_logit", "highest_rate_logit", "fit_seconds"]).set_index("held_out")
    full = o.load  # noqa: F841  (rates of the held-out benchmark come from the full fit)
    fit = pd.read_parquet(c.DATA / "fits" / "wp4b-offline-rasch-v1.parquet")
    mu_full = dict(zip(fit[fit.kind == "mu"].key, fit[fit.kind == "mu"].value))
    d.insert(0, "jeffreys_alc", g["jeffreys"]); d["tasks"] = alc.groupby("held_out").size()
    d["held_out_rate_logit"] = pd.Series(mu_full)
    out = d.join(meta)
    out["outside_range"] = (out.held_out_rate_logit < out.lowest_rate_logit) | (out.held_out_rate_logit > out.highest_rate_logit)
    c.SUMMARY.mkdir(parents=True, exist_ok=True)
    out.round(5).to_csv(c.SUMMARY / f"{a.run}.csv")
    print(out.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
