#!/usr/bin/env python3
"""R6 (E4, Person-A part): fitted priors for random sets of 3 test benchmarks, scored per benchmark.

For 40 random triples of the 17 benchmarks, the prior is refit without the triple's *training* benchmarks (validation and
test benchmarks are never in the fit). Each distinct set of excluded training benchmarks is fitted once (Beta prior by
marginal likelihood, Rasch hyperparameters by empirical Bayes, as in offline_rasch.py), then every benchmark in the set
(ID-panel tasks, held-out subjects) and the 6 validation/test benchmarks are scored. Person-B combines the per-benchmark
gains into the distribution of the test gain over splits.

    python organized/testing/eandd/split_fits.py --run r6-split-fits-v1 --triples 40
"""
from __future__ import annotations
import argparse, itertools, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "online"))
import common as c, offline_rasch as o, run_replay as r, lobo_prior as lp

TRAIN = ["afrieval", "afrimedqa", "agi_elo", "alignment_faking", "appworld", "brace", "bridging_gap", "cruxeval",
         "hallusionbench", "harmbench", "helm_afr"]


def fit_excluding(args):
    excl, rounds = args
    t0 = time.time()
    resp = pd.read_parquet(c.RESPONSES, columns=["benchmark_id", "subject_uid", "item_uid", "label", "benchmark_split"])
    resp = resp[(resp.benchmark_split == "train") & (~resp.benchmark_id.isin(excl))].drop(columns="benchmark_split")
    counts = resp.groupby(["benchmark_id", "subject_uid"]).label.agg(hits="sum", n="size").reset_index()
    out = {"excl": list(excl), "eb": lp.eb_beta(counts, "bench")}
    ho = pd.read_parquet(c.ID_PANEL / "heldout_subjects.parquet")
    gone = set(zip(ho.benchmark_id, ho.subject_uid))
    keep = ~pd.Series(list(zip(resp.benchmark_id, resp.subject_uid)), index=resp.index).isin(gone)
    b, s, i, bu, su, iu, sb, ib = o.index(resp[keep].reset_index(drop=True))
    y = resp[keep].label.to_numpy(float)
    nb, ns, ni = len(bu), len(su), len(iu)
    m0, tb, ts, ti, x = float(np.log(0.512 / 0.488)), 2.0, 1.0, 1.0, None
    for _ in range(rounds):
        x, res = o.fit(b, s, i, y, nb, ns, ni, m0, tb, ts, ti, x0=x, maxiter=600)
        mu, th, bi = x[:nb], x[nb:nb + ns], x[nb + ns:]
        vb, vs, vi = o.post_var(x, b, s, i, nb, ns, ni, tb, ts, ti)
        new = (float(mu.mean()), float(np.sqrt(max(np.var(mu) + vb.mean(), 1e-4))),
               float(np.sqrt(np.mean([np.mean(th[sb == q] ** 2 + vs[sb == q]) for q in range(nb)]))),
               float(np.sqrt(np.mean([np.mean(bi[ib == q] ** 2 + vi[ib == q]) for q in range(nb)]))))
        done = max(abs(p - q) for p, q in zip(new, (m0, tb, ts, ti))) < 0.01
        m0, tb, ts, ti = new
        if done:
            break
    out.update(m0=m0, tau_b=tb, tau_s=ts, tau_i=ti, seconds=round(time.time() - t0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--triples", type=int, default=40)
    ap.add_argument("--fit-workers", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    allb = sorted(TRAIN + ["algotune", "cmmlu", "editbench", "ai2d_test", "bird_sql", "financebench"])
    triples = list(itertools.combinations(allb, 3))
    rng = np.random.default_rng(20261107)
    pick = [triples[i] for i in rng.choice(len(triples), a.triples, replace=False)]
    sets = sorted({frozenset(b for b in t if b in TRAIN) for t in pick}, key=lambda s: (len(s), sorted(s)))
    print(f"{len(pick)} triples, {len(sets)} distinct excluded sets (incl. the empty set)", flush=True)

    todo = [(tuple(sorted(s)), a.rounds) for s in sets if s]
    with ProcessPoolExecutor(a.fit_workers) as ex:
        fits = list(ex.map(fit_excluding, todo))
    # The empty set: the full fit, with the tightly converged hyperparameters from R1.
    tight = pd.read_parquet(c.DATA / "fits" / "r1-offline-rasch-tight.parquet")
    h = dict(zip(tight[tight.kind == "hyper"].key, tight[tight.kind == "hyper"].value))
    full_counts = pd.read_parquet(c.RESPONSES, columns=["benchmark_id", "subject_uid", "label", "benchmark_split"])
    full_counts = full_counts[full_counts.benchmark_split == "train"].groupby(["benchmark_id", "subject_uid"]).label.agg(hits="sum", n="size").reset_index()
    fits.append({"excl": [], "eb": lp.eb_beta(full_counts, "bench"), "m0": h["m0"], "tau_b": h["tau_b"], "tau_s": h["tau_s"], "tau_i": h["tau_i"], "seconds": 0})
    print(f"fits done in {time.time() - t0:.0f}s", flush=True)

    ood = ["algotune", "cmmlu", "editbench", "ai2d_test", "bird_sql", "financebench"]
    id_items = pd.read_parquet(c.ID_PANEL / "panel" / "items.parquet", columns=["task_id", "benchmark_id"]).drop_duplicates("task_id")
    id_seeds = sorted(pd.read_parquet(c.ID_PANEL / "panel" / "candidate_orders.parquet", columns=["order_seed"]).order_seed.unique())
    ood_items = pd.read_parquet(c.CACHE / "panel" / "items.parquet", columns=["task_id", "benchmark_id", "benchmark_split"]).drop_duplicates("task_id")
    ood_seeds = {"validation": 3, "test": 10}
    allseeds = sorted(pd.read_parquet(c.CACHE / "panel" / "candidate_orders.parquet", columns=["order_seed"]).order_seed.unique())
    W = pd.Series(c.WEIGHTS)
    rows = []
    for f in fits:
        m, s = f["eb"]
        cfgs = {"jeffreys": ("beta", 0.5, 0.5), "bb-eb": ("beta", m * s, (1 - m) * s),
                "rasch-fitted": ("rasch", None, 0.0, f["tau_i"], f["tau_b"], f["tau_s"], f["m0"])}
        # (a) the excluded training benchmarks, on ID-panel tasks
        for cache, items, seeds in ((c.ID_PANEL, id_items[id_items.benchmark_id.isin(f["excl"])], id_seeds),
                                    (c.CACHE, ood_items, None)):
            if len(items) == 0:
                continue
            draws = {}
            for t, split in zip(items.task_id, items.get("benchmark_split", pd.Series(["id"] * len(items), index=items.index))):
                sd = seeds if seeds is not None else allseeds[: ood_seeds.get(split, 10)]
                for orders in sd:
                    draws[len(draws)] = [(t, int(orders))]
            sc = r.run(draws, cfgs, a.workers, None, cache=cache)
            bmap = dict(zip(items.task_id, items.benchmark_id))
            tb = sc.groupby(["model", "task_id", "budget"]).brier.mean().unstack("budget")
            al = (tb[c.BUDGETS] * W).sum(axis=1).unstack("model")
            al["benchmark"] = al.index.map(bmap)
            g = al.groupby("benchmark").mean(numeric_only=True)
            for bname in g.index:
                for mname in ("bb-eb", "rasch-fitted"):
                    rows.append((",".join(f["excl"]) or "none", bname, mname, float(g.loc[bname, mname] - g.loc[bname, "jeffreys"]),
                                 int(al[al.benchmark == bname].shape[0])))
        print("scored excluded set", f["excl"] or "none", f"({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows, columns=["excluded_training_benchmarks", "benchmark", "model", "gain", "tasks"])
    res.round(5).to_csv(c.SUMMARY / f"{a.run}-per-benchmark.csv", index=False)
    pd.DataFrame([{**f, "excl": ",".join(f["excl"]) or "none", "eb_mean": f["eb"][0], "eb_strength": f["eb"][1]} for f in fits]).drop(columns="eb").round(5).to_csv(c.SUMMARY / f"{a.run}-hyperparameters.csv", index=False)
    json.dump([[list(t)] for t in pick], open(c.ROOT / "paper/eandd/configs/r6-triples.json", "w"))
    c.save_run(a.run, {"triples": [list(t) for t in pick], "n_fits": len(fits)}, t0, scores=None)
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
