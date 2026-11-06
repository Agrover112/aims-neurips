#!/usr/bin/env python3
"""Paired intervals between prior sources, and the label budget after which they stop differing (saved scores only)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

W = pd.Series(c.WEIGHTS)


def boot(x, rng, n=5000):
    b = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    rng = np.random.default_rng(20261106)
    runs = {"test": "wp5-competition-test", "validation": "wp5-competition-val", "id": "wp5b-competition-id-clean"}
    pairs = [("rasch-fitted", "rasch-fixed"), ("kernel-fitted", "rasch-fitted"), ("rasch-fitted", "bb-eb"),
             ("bb-eb", "jeffreys"), ("rasch-fitted", "jeffreys"), ("bb-tuned", "jeffreys")]
    rows, cross = [], []
    for role, run in runs.items():
        s = pd.read_parquet(c.DATA / "scores" / f"{run}.parquet")
        if role == "id":
            s = s.assign(model=s.model.replace({"bb-eb": "bb-eb-leaky", "bb-eb-clean": "bb-eb"}))
        tb = s.groupby(["model", "task_id", "budget"]).brier.mean().unstack("budget")
        alc = (tb[c.BUDGETS] * W).sum(axis=1).unstack("model")
        have = set(alc.columns)
        for a, b in pairs:
            if a in have and b in have:
                e, lo, hi = boot((alc[a] - alc[b]).to_numpy(), rng)
                rows.append((role, a, b, e, lo, hi, len(alc)))
        for m in ("bb-eb", "rasch-fitted", "bb-tuned"):
            if m not in have:
                continue
            for bud in c.BUDGETS:
                d = (tb[bud].unstack("model")[m] - tb[bud].unstack("model")["jeffreys"]).to_numpy() if False else None
            x = tb.reset_index().pivot(index="task_id", columns="model", values=list(c.BUDGETS))
            for bud in c.BUDGETS:
                d = (x[(bud, m)] - x[(bud, "jeffreys")]).to_numpy()
                e, lo, hi = boot(d, rng)
                cross.append((role, m, bud, e, lo, hi, "differs" if (lo > 0 or hi < 0) else "n.s."))
    p = pd.DataFrame(rows, columns=["role", "model_a", "model_b", "a_minus_b", "lo", "hi", "tasks"])
    q = pd.DataFrame(cross, columns=["role", "model", "budget", "diff_vs_jeffreys", "lo", "hi", "ci_excludes_zero"])
    p.round(5).to_csv(c.SUMMARY / "wp12-paired-prior-sources.csv", index=False)
    q.round(5).to_csv(c.SUMMARY / "wp12-crossover-by-budget.csv", index=False)
    print(p.round(4).to_string(index=False)); print()
    print(q[q.role == "test"].pivot(index="model", columns="budget", values="diff_vs_jeffreys").round(4).to_string())
    print(q[q.role == "test"].pivot(index="model", columns="budget", values="ci_excludes_zero").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
