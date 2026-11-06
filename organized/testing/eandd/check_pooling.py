#!/usr/bin/env python3
"""Does the joint Rasch fit differ from fitting each task alone, in competition-shaped draws?

With one subject per benchmark the evidence splits into disconnected blocks (a benchmark, its subject and its
items share no parameter with any other task), so the joint MAP should equal the per-task MAPs.

    python organized/testing/eandd/check_pooling.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "online"))
import common as c
import kernel_rasch_replay as k
import run_replay as r

def main() -> int:
    k._init(str(c.CACHE), str(c.EMB), ())
    meta, ev = k.G["meta"], k.G["eval"]
    worst = {}
    for shape in ("competition", "multi-pair"):
        draws = c.load_draws(shape, "test")
        for name in ("rasch-old", "kernel-old"):
            cfg, diffs = r.MODELS[name], []
            for d, pairs in list(draws.items())[:40]:
                acq = {(t, s): k._acquired(t, s, "random") for t, s in pairs}
                tg = {t: (meta.loc[t, "benchmark_id"], meta.loc[t, "subject_uid"], ev[t][0]) for t, _ in pairs}
                for b in (1, 3, 7, 15, 31):
                    evid = [(meta.loc[t, "benchmark_id"], meta.loc[t, "subject_uid"], h, y) for t, s in pairs for h, y in acq[(t, s)][:b]]
                    joint = k.fit_predict(evid, tg, cfg)
                    for t, s in pairs:
                        alone = [(meta.loc[t, "benchmark_id"], meta.loc[t, "subject_uid"], h, y) for h, y in acq[(t, s)][:b]]
                        solo = k.fit_predict(alone, {t: tg[t]}, cfg)[t]
                        diffs.append(float(np.max(np.abs(joint[t] - solo))))
            worst[(shape, name)] = (max(diffs), float(np.mean(diffs)))
            print(f"{shape:12s} {name:11s} joint vs alone: max |dp| {max(diffs):.2e}  mean {np.mean(diffs):.2e}  ({len(diffs)} task-budget cells)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
