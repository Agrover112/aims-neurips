#!/usr/bin/env python3
"""R5: does a probe's predicted item difficulty help as the prior mean of the Rasch item deviation delta_j?

Item prior N(gamma * z_j, sigma_delta^2) with z_j the probe's difficulty centred within its benchmark. The probes were trained on
the 11 training benchmarks only (Person-B's R9 exports). gamma is chosen on validation only; gamma = 0 is the fitted Rasch.

    python organized/testing/eandd/r5_prior_mean.py --run r5-prior-mean-v1
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "online"))
import common as c, run_replay as r

ARMS = ["pass_rate_mlp", "rasch_difficulty_mlp"]
GAMMAS = [0.0, 0.25, 0.5, 1.0, 1.5]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    tight = pd.read_parquet(c.DATA / "fits" / "r1-offline-rasch-tight.parquet")
    h = dict(zip(tight[tight.kind == "hyper"].key, tight[tight.kind == "hyper"].value))
    base = ("rasch", None, 0.0, h["tau_i"], h["tau_b"], h["tau_s"], h["m0"])
    cfgs = {"jeffreys": ("beta", 0.5, 0.5), "rasch-fitted": base}
    for arm in ARMS:
        for g in GAMMAS[1:]:
            cfgs[f"pm-{arm}-{g}"] = base + ((arm, g),)
    out = []
    for role in ("validation", "test"):
        s = r.run(c.load_draws("competition", role), cfgs, a.workers, None).assign(shape="competition", role=role)
        c.save_run(f"{a.run}-{role}", {"role": role, "models": {k: [str(x) for x in v] for k, v in cfgs.items()}}, t0, scores=s)
        t = r.alc_table(s); t.insert(0, "role", role); out.append(t)
        print(role, f"({time.time() - t0:.0f}s)", flush=True)
    res = pd.concat(out)
    print(res.pivot(index="model", columns="role", values="diff").round(4).to_string())
    res.round(5).to_csv(c.SUMMARY / f"{a.run}-summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
