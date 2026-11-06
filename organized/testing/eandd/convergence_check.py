#!/usr/bin/env python3
"""How often does the online Newton solver hit its 30-step limit, and how large is the last step? (R1, E6)

    python organized/testing/eandd/convergence_check.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "online"))
import common as c
import kernel_rasch_replay as k
import run_replay as r


def main() -> int:
    k._init(str(c.CACHE), str(c.EMB), ())
    meta, ev = k.G["meta"], k.G["eval"]
    known = dict(r.MODELS)
    known.update({n: tuple(v) for n, v in json.loads(Path("paper/eandd/configs/wp5-all.json").read_text()).items()})
    rows = []
    for shape, role in (("competition", "test"), ("competition", "validation"), ("multi-pair", "test")):
        draws = c.load_draws(shape, role)
        for name in ("rasch-fitted", "kernel-fitted", "rasch-old"):
            cfg = known[name]
            for d, pairs in draws.items():
                acq = {(t, s): k._acquired(t, s, "random") for t, s in pairs}
                tg = {t: (meta.loc[t, "benchmark_id"], meta.loc[t, "subject_uid"], ev[t][0]) for t, _ in pairs}
                for b in (1, 3, 7, 15, 31):
                    evid = [(meta.loc[t, "benchmark_id"], meta.loc[t, "subject_uid"], h, y) for t, s in pairs for h, y in acq[(t, s)][:b]]
                    k.fit_predict(evid, tg, cfg)
                    rows.append((shape, role, name, b, k.LAST_NEWTON["steps"], k.LAST_NEWTON["max_step"], k.LAST_NEWTON["converged"]))
    df = pd.DataFrame(rows, columns=["shape", "role", "model", "budget", "steps", "last_step", "converged"])
    df.to_csv(c.SUMMARY / "r1-newton-convergence.csv", index=False)
    g = df.groupby(["shape", "role", "model"]).agg(fits=("steps", "size"), hit_limit=("converged", lambda x: int((~x).sum())),
                                                    max_steps=("steps", "max"), mean_steps=("steps", "mean"), worst_last_step=("last_step", "max"))
    print(g.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
