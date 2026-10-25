#!/usr/bin/env python3
"""Recompute the paired statistics quoted in docs/report/README.md.

Reads result files pulled from the ``aims-neurips-data`` Volume (JSON for the
r_C runs, row-level parquet for the FLOPs-prior and MIRT-acquisition runs) and
prints, for every comparison: the mean paired task difference, a 95% task
bootstrap interval, a 95% benchmark-cluster bootstrap interval, and task wins.

Pull the inputs first (small files only):

    modal volume get aims-neurips-data /results/r-c-ablation/<file>.json ...

then run:

    python scripts/report_paired_stats.py --results /path/to/pulled/results
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np
import pandas as pd

WEIGHTS = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
N_BOOT = 10_000
N_CLUSTER_BOOT = 2_000


def bootstrap(diff: np.ndarray, bench: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    task = diff[rng.integers(0, n, (N_BOOT, n))].mean(1)
    names = np.unique(bench)
    members = {b: np.where(bench == b)[0] for b in names}
    draws = rng.integers(0, len(names), (N_CLUSTER_BOOT, len(names)))
    cluster = np.array([diff[np.concatenate([members[names[j]] for j in row])].mean() for row in draws])
    return (
        diff.mean(),
        np.quantile(task, [0.025, 0.975]),
        np.quantile(cluster, [0.025, 0.975]),
        int((diff < 0).sum()),
    )


def line(name: str, diff: np.ndarray, bench: np.ndarray, extra: str = "") -> None:
    mean, task, cluster, wins = bootstrap(diff, bench)
    print(
        f"{name:52} d={mean:+.6f} task[{task[0]:+.6f},{task[1]:+.6f}] "
        f"bench[{cluster[0]:+.6f},{cluster[1]:+.6f}] wins={wins}/{len(diff)} {extra}"
    )


def task_alc(rows: pd.DataFrame) -> pd.DataFrame:
    """Weighted ALC per (arm, task), averaging orders/streams within budget."""
    g = rows.groupby(["arm", "task_id", "budget"]).brier.mean().reset_index()
    g["wb"] = g.budget.map(WEIGHTS) * g.brier
    return g.groupby(["arm", "task_id"]).wb.sum().unstack(0)


def load_rc(root: str):
    runs = {}
    for path in glob.glob(os.path.join(root, "r-c-ablation", "*.json")):
        name = os.path.basename(path)
        d = json.load(open(path))
        if d.get("split") != "validation" or d.get("status") != "complete":
            continue
        store = d.get("manifest", {}).get("arm", d.get("arm"))
        sampler = "v41" if name.endswith("-v41e1.json") else "random"
        kind = "score" if name.startswith("score-") else "rc"
        values = collections.defaultdict(dict)
        for r in d["task_alc"]:
            values[r["arm"]][r["task_id"]] = r["weighted"]
        arm = next(a for a in values if a != "beta")
        runs[(kind, d["rung"], store, sampler, d["seed"])] = (values[arm], values["beta"])
    return runs


def rc_section(root: str) -> None:
    runs = load_rc(root)
    tasks = sorted(next(iter(runs.values()))[1])
    bench = np.array([t.split("::")[1] for t in tasks])
    beta = np.array([next(iter(runs.values()))[1][t] for t in tasks])
    groups: dict = collections.defaultdict(dict)
    for (kind, rung, store, sampler, seed), (model, ref) in runs.items():
        assert abs(np.mean([ref[t] for t in tasks]) - beta.mean()) < 1e-9, "Beta reference differs"
        groups[(kind, rung, store, sampler)][seed] = np.array([model[t] for t in tasks])

    def mean_over(key, seeds=None):
        s = groups[key]
        use = sorted(s) if seeds is None else seeds
        return np.mean([s[x] for x in use], 0), use

    print(f"Beta reference: {beta.mean():.6f}  tasks={len(tasks)}  benchmarks={len(set(bench))}\n")
    print("== every r_C run vs Beta (seed-mean per task)")
    for key in sorted(groups):
        m, seeds = mean_over(key)
        per = [groups[key][s].mean() for s in seeds]
        line(f"{key[1]} | {key[2]} | {key[3]} | {key[0]}", m - beta, bench,
             f"seeds={len(seeds)} mean={m.mean():.6f} spread={max(per) - min(per):.6f}")

    def contrast(title, a, b):
        if a not in groups or b not in groups:
            return
        common = sorted(set(groups[a]) & set(groups[b]))
        if not common:
            return
        ma, _ = mean_over(a, common)
        mb, _ = mean_over(b, common)
        per = ["%+.6f" % (groups[a][s] - groups[b][s]).mean() for s in common]
        line(title, ma - mb, bench, f"seeds={len(common)} per-seed={per}")

    print("\n== encoder/store: me5 content-only vs minilm512 (random sampler)")
    for r in ["scalar", "direct_loop0", "additive_loop3", "cnp", "anp_attn"]:
        contrast(f"{r}: me5 - minilm512", ("rc", r, "me5_instruct", "random"), ("rc", r, "minilm512", "random"))
    print("\n== encoder: bge_m3_long vs me5 (random sampler)")
    for r in ["scalar", "direct_loop0", "additive_loop3", "mirt_direct_loop0_d4", "mirt_scalar_d4"]:
        contrast(f"{r}: bge_m3_long - me5", ("rc", r, "bge_m3_long", "random"), ("rc", r, "me5_instruct", "random"))
    print("\n== sampler: v4.1 - random (me5)")
    for r in ["cnp", "anp_attn", "scalar", "direct_loop0", "additive_loop3", "mirt_scalar_d4", "mirt_direct_loop0_d4"]:
        contrast(f"{r}: v41 - random", ("rc", r, "me5_instruct", "v41"), ("rc", r, "me5_instruct", "random"))
    print("\n== multidimensional ability (me5, random sampler)")
    for a, b in [("mirt_scalar_d4", "scalar"), ("mirt_direct_loop0_d4", "direct_loop0"),
                 ("mirt_additive_loop3_d4", "additive_loop3"), ("mirt_scalar_d4", "mirt_scalar_d1"),
                 ("mirt_direct_loop0_d4", "mirt_direct_loop0_d1"), ("mirt_additive_loop3_d4", "mirt_additive_loop3_d1")]:
        contrast(f"{a} - {b}", ("rc", a, "me5_instruct", "random"), ("rc", b, "me5_instruct", "random"))
    print("\n== v4.1 best arm vs every other v4.1 arm (me5, seed 20261018)")
    best = ("rc", "anp_attn", "me5_instruct", "v41")
    for key in sorted(groups):
        if key[2] == "me5_instruct" and key[3] == "v41" and key != best:
            contrast(f"anp_attn - {key[1]}", best, key)

    print("\n== per-benchmark mean delta vs Beta")
    table = {}
    for label, key in [("me5 scalar", ("rc", "scalar", "me5_instruct", "random")),
                       ("me5 direct_loop0", ("rc", "direct_loop0", "me5_instruct", "random")),
                       ("minilm512 scalar", ("rc", "scalar", "minilm512", "random")),
                       ("anp_attn v41", best)]:
        if key in groups:
            table[label] = pd.Series(mean_over(key)[0] - beta).groupby(bench).mean()
    out = pd.DataFrame(table)
    out["tasks"] = pd.Series(bench).value_counts()
    print(out.round(5).to_string())


def parquet_section(root: str) -> None:
    flops = os.path.join(root, "flops-prior-2pl", "rows-calibrated.parquet")
    if os.path.exists(flops):
        t = task_alc(pd.read_parquet(flops))
        bench = np.array([i.split("::")[1] for i in t.index])
        print("\n== ability-prior arms (2pl_tau4 checkpoint, calibrated)")
        print(t.mean().sort_values().round(6).to_string())
        for a, b in [("flops", "zero"), ("flops_tau1", "zero_tau1"), ("model", "zero"),
                     ("flops_model", "zero"), ("zero_tau1", "zero"), ("zero_tau1", "beta")]:
            line(f"{a} - {b}", (t[a] - t[b]).values, bench)
    acq = glob.glob(os.path.join(root, "mirt-acquisition", "*-6arms.rows.parquet"))
    for path in acq:
        rows = pd.read_parquet(path)
        for cal in ["oof", "raw"]:
            t = task_alc(rows[rows.calibration == cal])
            bench = np.array([i.split("::")[1] for i in t.index])
            print(f"\n== MIRT acquisition ({cal}) {os.path.basename(path)}")
            print(t.mean().sort_values().round(6).to_string())
            for a in [c for c in t.columns if c != "A-R"]:
                line(f"{a} - A-R", (t[a] - t["A-R"]).values, bench)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", required=True, help="directory holding the pulled results/ tree")
    args = parser.parse_args()
    rc_section(args.results)
    parquet_section(args.results)


if __name__ == "__main__":
    main()
