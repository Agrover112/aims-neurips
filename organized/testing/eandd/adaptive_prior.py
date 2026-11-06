#!/usr/bin/env python3
"""R3 (E1) and R4 (E3): priors that adapt to the first labels, and the pilot switch rule.

Models (all closed form, nothing trained at test time):
  bb-eb      the fitted single Beta prior (reference)
  mix2, mix3 Beta mixtures fitted by EM on training tasks; weights update by marginal likelihood on the revealed labels
  bma-s      Bayesian averaging over prior strength at the fitted mean
  bma-ms     Bayesian averaging over a grid of means and strengths (nothing learned)
  switch-K   fitted prior unless the pilot accuracy (first K labels) is more than d from the prior mean, then Jeffreys;
             d chosen on validation only

Every LOBO fold refits all priors without any row of the held-out benchmark.

    python organized/testing/eandd/adaptive_prior.py --run r3-adaptive-v1
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "online"))
import common as c, run_replay as r, priors as pr, lobo_prior as lp

MEANS = [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
STRENGTHS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]


def bbl(h, n, a, b):
    return betaln(h + a, n - h + b) - betaln(a, b)


def fit_mixture(h, n, w, K, iters=40):
    acc = (h + 0.5) / (n + 1)
    srt = np.sort(acc)
    means = np.array([srt[int(q * (len(srt) - 1))] for q in np.linspace(0.15, 0.85, K)])
    s, pi = np.full(K, 4.0), np.full(K, 1.0 / K)
    for _ in range(iters):
        a, b = means * s, (1 - means) * s
        logp = np.stack([np.log(pi[k]) + bbl(h, n, a[k], b[k]) for k in range(K)], 1)
        resp = np.exp(logp - logsumexp(logp, 1, keepdims=True))
        pi = (w[:, None] * resp).sum(0) / w.sum()
        for k in range(K):
            wk = w * resp[:, k]

            def nll(x):
                m, ss = 1 / (1 + np.exp(-x[0])), np.exp(np.clip(x[1], np.log(0.3), np.log(300)))
                return -(wk * bbl(h, n, m * ss, (1 - m) * ss)).sum()
            res = minimize(nll, [np.log(means[k] / (1 - means[k])), np.log(s[k])], method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 300})
            means[k], s[k] = 1 / (1 + np.exp(-res.x[0])), float(np.exp(np.clip(res.x[1], np.log(0.3), np.log(300))))
    a, b = means * s, (1 - means) * s
    return [[float(pi[k]), float(a[k]), float(b[k])] for k in range(K)]


def fit_all(counts: pd.DataFrame) -> dict:
    h, n = counts.hits.to_numpy(float), counts.n.to_numpy(float)
    w = (1.0 / counts.groupby("benchmark_id").hits.transform("size")).to_numpy()
    w = w / w.sum() * len(w)
    m, s = lp.eb_beta(counts, "bench")
    return {"eb": (m, s), "mix2": fit_mixture(h, n, w, 2), "mix3": fit_mixture(h, n, w, 3)}


def configs(f: dict, d: dict) -> dict:
    m, s = f["eb"]
    out = {"jeffreys": ("beta", 0.5, 0.5), "bb-eb": ("beta", m * s, (1 - m) * s),
           "mix2": ("mixture", f["mix2"]), "mix3": ("mixture", f["mix3"]),
           "bma-s": ("mixture", pr.grid_components([m], STRENGTHS)),
           "bma-ms": ("mixture", pr.grid_components(MEANS, STRENGTHS))}
    for kmin, dd in d.items():
        out[f"switch-{kmin}"] = ("switch", m, s, dd, kmin)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    resp = pd.read_parquet(c.RESPONSES, columns=["benchmark_id", "subject_uid", "label", "benchmark_split"])
    resp = resp[resp.benchmark_split == "train"]
    counts = resp.groupby(["benchmark_id", "subject_uid"]).label.agg(hits="sum", n="size").reset_index()
    ho = pd.read_parquet(c.ID_PANEL / "heldout_subjects.parquet")
    out = set(zip(ho.benchmark_id, ho.subject_uid))
    clean = counts[~pd.Series(list(zip(counts.benchmark_id, counts.subject_uid))).isin(out).to_numpy()]
    full_fit, clean_fit = fit_all(counts), fit_all(clean)
    print("full fit: eb", [round(x, 3) for x in full_fit["eb"]], "mix2", [[round(v, 2) for v in comp] for comp in full_fit["mix2"]],
          "mix3", [[round(v, 2) for v in comp] for comp in full_fit["mix3"]], f"({time.time() - t0:.0f}s)", flush=True)

    # d for the switch rule: validation only.
    grid = {f"v{kmin}-{d}": ("switch", *full_fit["eb"], d, kmin) for kmin in (3, 7) for d in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)}
    base = dict(configs(full_fit, {}))
    val = r.run(c.load_draws("competition", "validation"), {**base, **grid}, a.workers, None)
    alc = r.alc_table(val)
    sel = {}
    for kmin in (3, 7):
        best = alc[alc.model.str.startswith(f"v{kmin}-")].sort_values("alc").iloc[0]
        sel[kmin] = float(best.model.split("-")[1])
        print(f"switch kmin={kmin}: d={sel[kmin]} chosen on validation (ALC {best.alc:.4f})", flush=True)
    cfg_full, cfg_clean = configs(full_fit, sel), configs(clean_fit, sel)

    rows, per_bench = [], []
    items = pd.read_parquet(c.CACHE / "panel" / "items.parquet", columns=["task_id", "benchmark_id"]).drop_duplicates("task_id")
    bmap = dict(zip(items.task_id, items.benchmark_id))
    W = pd.Series(c.WEIGHTS)
    for role, cfgs, cache in (("validation", cfg_full, c.CACHE), ("test", cfg_full, c.CACHE), ("id", cfg_clean, c.ID_PANEL)):
        s = r.run(c.load_draws("competition", role), cfgs, a.workers, None, cache=cache).assign(shape="competition", role=role)
        c.save_run(f"{a.run}-{role}", {"role": role, "models": dict(cfgs)}, t0, scores=s)
        t = r.alc_table(s)
        t.insert(0, "role", role)
        rows.append(t)
        if role != "id":
            tb = s.groupby(["model", "task_id", "budget"]).brier.mean().unstack("budget")
            al = (tb[c.BUDGETS] * W).sum(axis=1).unstack("model")
            al["benchmark"] = al.index.map(bmap)
            g = al.groupby("benchmark").mean(numeric_only=True)
            for b in g.index:
                for m in g.columns:
                    per_bench.append((b, role, m, float(g.loc[b, m] - g.loc[b, "jeffreys"])))
    summary = pd.concat(rows)
    print(summary.pivot(index="model", columns="role", values="diff").round(4).to_string(), flush=True)

    # Leave-one-benchmark-out: every prior refit without any row of K.
    items_id = pd.read_parquet(c.ID_PANEL / "panel" / "items.parquet", columns=["task_id", "benchmark_id"]).drop_duplicates("task_id")
    seeds = sorted(pd.read_parquet(c.ID_PANEL / "panel" / "candidate_orders.parquet", columns=["order_seed"]).order_seed.unique())
    lobo = []
    for K in sorted(items_id.benchmark_id.unique()):
        fk = fit_all(counts[counts.benchmark_id != K])
        tasks = items_id[items_id.benchmark_id == K].task_id.tolist()
        draws = {j: [(t, int(sd))] for j, (t, sd) in enumerate((t, sd) for t in tasks for sd in seeds)}
        lobo.append(r.run(draws, configs(fk, sel), a.workers, None, cache=c.ID_PANEL).assign(held_out=K))
        print("LOBO", K, f"({time.time() - t0:.0f}s)", flush=True)
    ls = pd.concat(lobo, ignore_index=True)
    c.save_run(f"{a.run}-lobo", {"models": list(cfg_full)}, t0, scores=ls)
    tb = ls.groupby(["held_out", "model", "task_id", "budget"]).brier.mean().unstack("budget")
    al = (tb[c.BUDGETS] * W).sum(axis=1).unstack("model")
    g = al.groupby("held_out").mean()
    for b in g.index:
        for m in g.columns:
            per_bench.append((b, "lobo", m, float(g.loc[b, m] - g.loc[b, "jeffreys"])))
    pb = pd.DataFrame(per_bench, columns=["benchmark", "source", "model", "gain"])
    pb.round(5).to_csv(c.SUMMARY / "r3-adaptive-per-benchmark.csv", index=False)
    summary.round(5).to_csv(c.SUMMARY / "r3-adaptive-summary.csv", index=False)
    json.dump({"full": full_fit, "clean": clean_fit, "d": sel}, open(c.ROOT / "paper/eandd/configs/r3-priors.json", "w"), indent=1)
    seventeen = pb[pb.source.isin(["lobo", "validation", "test"])]
    res = seventeen.groupby("model").gain.agg(macro="mean", worst="max", best="min")
    res["test_gain"] = summary[summary.role == "test"].set_index("model")["diff"]
    print("\n17-benchmark view (gain over Jeffreys; negative is better):")
    print(res.round(4).to_string())
    res.round(5).to_csv(c.SUMMARY / "r3-adaptive-17-benchmarks.csv")
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
