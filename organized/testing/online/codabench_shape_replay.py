#!/usr/bin/env python3
"""Re-tune the Rasch priors on draws shaped like CodaBench's.

CodaBench scores about nine subject-benchmark pairs spread over about seven
benchmarks, so most benchmarks have a single subject and nothing to pool with
on that benchmark. The earlier multi-pair draws had one or two subjects on
each of three benchmarks. Here every draw takes exactly one subject per
benchmark. The prior widths (benchmark rate tau_b, ability tau_s, item tau_i)
are chosen on validation draws of that shape and scored once on test draws of
the same shape, plus on the earlier draws to check nothing regresses.

    python organized/testing/online/codabench_shape_replay.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_rasch_replay as k  # noqa: E402

CACHE = Path.home() / "aims-replay-cache"
EMB = Path.home() / "aims-noitem-run/datasets/full/allbench-embed-bge_m3_long-v2"


def one_per_benchmark(role: str, n: int, rng) -> dict:
    items = pd.read_parquet(CACHE / "panel" / "items.parquet")
    tasks = items[items.benchmark_split == role].drop_duplicates("task_id")
    orders = pd.read_parquet(CACHE / "panel" / "candidate_orders.parquet")
    seeds = sorted(orders.order_seed.unique())[: 3 if role == "validation" else 10]
    by_bench = {b: g.task_id.tolist() for b, g in tasks.groupby("benchmark_id")}
    return {d: [(str(rng.choice(ts)), int(rng.choice(seeds))) for ts in by_bench.values()] for d in range(n)}


def main() -> int:
    rng = np.random.default_rng(20261101)
    init = (str(CACHE), str(EMB), (8,))
    configs = {}
    for tb, ts, ti in itertools.product((0.5, 1.0, 2.0), (0.5, 1.0), (1.0, 1.4)):
        configs[f"rasch tb={tb} ts={ts} ti={ti}"] = ("rasch", None, 0.0, ti, tb, ts)
        configs[f"kernel tb={tb} ts={ts} ti={ti}"] = ("kernel", 0.2, 0.8, ti, tb, ts)
    val = k.run(one_per_benchmark("validation", 120, rng), configs, "random", 4, init)
    va = k.task_alc(val).mean().sort_values()
    print("== Validation, one subject per benchmark: macro ALC")
    print(va.round(4).to_string())
    best = {kind: min((m for m in va.index if m.startswith(kind)), key=va.get) for kind in ("rasch", "kernel")}
    print("chosen:", best)
    chosen = {"Pooled Rasch (old priors)": ("rasch", None, 0.0, 1.0, 2.0, 1.0),
              "Kernel Rasch (old priors)": ("kernel", 0.2, 0.8, 1.4, 2.0, 1.0),
              "Pooled Rasch (re-tuned)": configs[best["rasch"]],
              "Kernel Rasch (re-tuned)": configs[best["kernel"]]}
    curves = pd.read_parquet(CACHE / "curves.parquet")
    curves = curves[(curves.role == "test") & (curves.model == "jeffreys")].drop_duplicates(["draw", "task_id"])
    shapes = {"one subject per benchmark": one_per_benchmark("test", 150, rng),
              "one or two per benchmark (earlier draws)": {d: list(zip(g.task_id, g.order_seed.astype(int)))
                                                         for d, g in curves.groupby("draw")}}
    out = {"validation_alc": va.to_dict(), "chosen": best, "test": {}}
    frames = []
    for shape, draws in shapes.items():
        f = k.run(draws, chosen, "random", 4, init)
        f["shape"] = shape
        frames.append(f)
        ta = k.task_alc(f)
        print(f"\n== Test, {shape}: ALC and difference vs beta-binomial")
        for m in ta.columns:
            d = k.boot((ta[m] - ta["beta"]).to_numpy(), rng)
            per_b = (f[f.model == m].groupby("budget").brier.mean()
                     - f[f.model == "beta"].groupby("budget").brier.mean()).round(4).to_dict()
            out["test"][f"{shape}/{m}"] = {"alc": float(ta[m].mean()), "diff": d, "per_budget": per_b}
            print(f"  {m:<28} {ta[m].mean():.4f}  {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]  per budget {per_b}")
    pd.concat(frames).to_parquet(k.OUT / "codabench-shape-test.parquet", index=False)
    val.to_parquet(k.OUT / "codabench-shape-validation.parquet", index=False)
    (k.OUT / "codabench-shape-summary.json").write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
