#!/usr/bin/env python3
"""Draw-seed robustness: the prior-source matrix on fresh competition-shaped draws.

Frozen draws are never touched. Each extra seed samples a new set of draws (one task per benchmark, a random order
seed) from the same panel, and runs the same models.

    python organized/testing/eandd/robustness.py --seeds 20261201 20261202
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "online"))
import common as c, run_replay as r

MODELS = "jeffreys,bb-tuned,bb-eb,rasch-fixed,rasch-retuned,rasch-fitted,kernel-fixed,kernel-old,kernel-fitted".split(",")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    known = dict(r.MODELS)
    known.update({n: tuple(v) for n, v in json.loads(Path("paper/eandd/configs/wp5-all.json").read_text()).items()})
    configs = {m: known[m] for m in MODELS}
    items = pd.read_parquet(c.CACHE / "panel" / "items.parquet")
    orders = pd.read_parquet(c.CACHE / "panel" / "candidate_orders.parquet")
    out = []
    for seed in a.seeds:
        for role in ("validation", "test"):
            t0 = time.time()
            rng = np.random.default_rng(seed + (0 if role == "validation" else 1))
            d = c.one_per_benchmark(role, 150, rng, items, orders)
            draws = {int(i): list(zip(g.task_id, g.order_seed.astype(int))) for i, g in d.groupby("draw")}
            s = r.run(draws, configs, a.workers, None).assign(shape="competition", role=role, seed=seed)
            c.save_run(f"wp12-seed{seed}-{role}", {"seed": seed, "role": role, "models": configs}, t0, scores=s)
            t = r.alc_table(s); t.insert(0, "seed", seed); t.insert(1, "role", role)
            out.append(t)
            print(seed, role, f"{time.time() - t0:.0f}s", flush=True)
    res = pd.concat(out)
    res.round(5).to_csv(c.SUMMARY / "wp12-draw-seed-robustness.csv", index=False)
    piv = res.pivot_table(index=["role", "model"], columns="seed", values="diff")
    print(piv.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
