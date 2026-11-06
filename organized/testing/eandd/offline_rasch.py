#!/usr/bin/env python3
"""Offline Rasch fit on the training benchmarks (WP4b).

    logit P(y_si = 1) = mu_b + theta_s - b_i,   mu_b ~ N(m0, tau_b^2), theta_s ~ N(0, tau_s^2), b_i ~ N(0, tau_i^2)

Joint MAP over every training response except the held-out subjects (Person-B's list), with the hyperparameters
(m0, tau_b, tau_s, tau_i) set by empirical Bayes and averaged per benchmark so the benchmark with most subjects
(alignment_faking) does not dominate. Saves every fitted value and the hyperparameters (fits/<run>.parquet) and writes
online configs to paper/eandd/configs/wp4b-fitted.json.

    python organized/testing/eandd/offline_rasch.py --run wp4b-offline-rasch-v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

CONFIGS = c.ROOT / "paper" / "eandd" / "configs" / "wp4b-fitted.json"


def load(heldout: pd.DataFrame) -> pd.DataFrame:
    r = pd.read_parquet(c.RESPONSES, columns=["benchmark_id", "subject_uid", "item_uid", "label", "benchmark_split"])
    r = r[r.benchmark_split == "train"].drop(columns="benchmark_split")
    out = set(zip(heldout.benchmark_id, heldout.subject_uid))
    keep = ~pd.Series(list(zip(r.benchmark_id, r.subject_uid)), index=r.index).isin(out)
    return r[keep].reset_index(drop=True)


def index(r: pd.DataFrame):
    b, bu = pd.factorize(r.benchmark_id)
    s, su = pd.factorize(r.subject_uid)
    i, iu = pd.factorize(r.item_uid)
    # Benchmark of every subject and item (each belongs to one in the filtered data).
    sb = pd.Series(b).groupby(s).first().to_numpy()
    ib = pd.Series(b).groupby(i).first().to_numpy()
    return b, s, i, bu, su, iu, sb, ib


def fit(b, s, i, y, nb, ns, ni, m0, tb, ts, ti, x0=None, maxiter=400):
    sign = 2.0 * y - 1.0

    def f(x):
        mu, th, bi = x[:nb], x[nb:nb + ns], x[nb + ns:]
        z = mu[b] + th[s] - bi[i]
        m = -sign * z
        loss = np.logaddexp(0.0, m).sum()
        g = -sign / (1.0 + np.exp(-m * -1.0)) if False else None
        p = 1.0 / (1.0 + np.exp(-z))
        r = p - y
        gmu = np.bincount(b, r, nb) + (mu - m0) / tb ** 2
        gth = np.bincount(s, r, ns) + th / ts ** 2
        gbi = -np.bincount(i, r, ni) + bi / ti ** 2
        pen = 0.5 * (((mu - m0) / tb) ** 2).sum() + 0.5 * ((th / ts) ** 2).sum() + 0.5 * ((bi / ti) ** 2).sum()
        return loss + pen, np.concatenate([gmu, gth, gbi])

    x = np.concatenate([np.full(nb, m0), np.zeros(ns + ni)]) if x0 is None else x0
    res = minimize(f, x, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxcor": 20})
    return res.x, res


def post_var(x, b, s, i, nb, ns, ni, tb, ts, ti):
    """Diagonal Laplace approximation of each parameter's posterior variance."""
    mu, th, bi = x[:nb], x[nb:nb + ns], x[nb + ns:]
    p = 1.0 / (1.0 + np.exp(-(mu[b] + th[s] - bi[i])))
    w = p * (1 - p)
    return (1.0 / (np.bincount(b, w, nb) + 1 / tb ** 2),
            1.0 / (np.bincount(s, w, ns) + 1 / ts ** 2),
            1.0 / (np.bincount(i, w, ni) + 1 / ti ** 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--heldout", type=Path, default=c.ID_PANEL / "heldout_subjects.parquet")
    a = ap.parse_args()
    t0 = time.time()
    r = load(pd.read_parquet(a.heldout))
    b, s, i, bu, su, iu, sb, ib = index(r)
    y = r.label.to_numpy(float)
    nb, ns, ni = len(bu), len(su), len(iu)
    print(f"{len(r):,} responses, {nb} benchmarks, {ns:,} subjects, {ni:,} items; load {time.time() - t0:.0f}s", flush=True)
    m0, tb, ts, ti = float(np.log(0.512 / 0.488)), 2.0, 1.0, 1.0
    x, history = None, []
    for rnd in range(a.rounds):
        x, res = fit(b, s, i, y, nb, ns, ni, m0, tb, ts, ti, x0=x)
        mu, th, bi = x[:nb], x[nb:nb + ns], x[nb + ns:]
        vb, vs, vi = post_var(x, b, s, i, nb, ns, ni, tb, ts, ti)
        # Empirical Bayes: moment updates with the posterior-variance correction, averaged per benchmark.
        m0n = float(mu.mean())
        tbn = float(np.sqrt(max(np.var(mu) + vb.mean(), 1e-4)))
        tsn = float(np.sqrt(np.mean([np.mean(th[sb == k] ** 2 + vs[sb == k]) for k in range(nb)])))
        tin = float(np.sqrt(np.mean([np.mean(bi[ib == k] ** 2 + vi[ib == k]) for k in range(nb)])))
        history.append((rnd, m0, tb, ts, ti, float(res.fun), int(res.nit)))
        print(f"round {rnd}: m0 {m0:+.3f} tau_b {tb:.3f} tau_s {ts:.3f} tau_i {ti:.3f}  (iters {res.nit}, {time.time() - t0:.0f}s)", flush=True)
        done = max(abs(m0n - m0), abs(tbn - tb), abs(tsn - ts), abs(tin - ti)) < 0.01
        m0, tb, ts, ti = m0n, tbn, tsn, tin
        if done:
            break
    x, _ = fit(b, s, i, y, nb, ns, ni, m0, tb, ts, ti, x0=x)
    mu, th, bi = x[:nb], x[nb:nb + ns], x[nb + ns:]
    fits = pd.concat([
        pd.DataFrame({"kind": "mu", "key": bu, "value": mu}),
        pd.DataFrame({"kind": "theta", "key": su, "value": th}),
        pd.DataFrame({"kind": "b", "key": iu, "value": bi}),
        pd.DataFrame({"kind": "hyper", "key": ["m0", "tau_b", "tau_s", "tau_i"], "value": [m0, tb, ts, ti]})], ignore_index=True)
    cfg = {"rasch-fitted": ["rasch", None, 0.0, ti, tb, ts, m0], "kernel-fitted": ["kernel", 0.2, 0.8, ti, tb, ts, m0]}
    CONFIGS.parent.mkdir(parents=True, exist_ok=True)
    CONFIGS.write_text(json.dumps(cfg, indent=1))
    c.save_run(a.run, {"rounds": history, "hyper": dict(m0=m0, tau_b=tb, tau_s=ts, tau_i=ti), "n_rows": len(r),
                       "heldout_file": str(a.heldout), "averaging": "per-benchmark mean of subject and item moments"},
               t0, fits=fits)
    print(f"final: m0 {m0:+.3f} tau_b {tb:.3f} tau_s {ts:.3f} tau_i {ti:.3f}; benchmark rates:")
    print(pd.Series(mu, index=bu).round(2).sort_values().to_string())
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
