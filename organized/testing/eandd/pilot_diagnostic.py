#!/usr/bin/env python3
"""R4 (E3): can the first labels warn that the fitted prior will hurt this task?

For every (task, order) of the 17 benchmarks: realised gain = ALC(fitted Beta prior) - ALC(Jeffreys); "hurt" = gain > 0.
Pilot distance = |hits/k - prior mean| from the first k labels of the same order (k = 3, 7). AUC of the distance for
predicting "hurt" (pooled, and the mean over benchmarks). Training benchmarks use the leave-one-benchmark-out fits, the
validation and test benchmarks the full fit (runs r3-adaptive-v1-*).

    python organized/testing/eandd/pilot_diagnostic.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

M = 0.5216


def auc(score, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    r = rankdata(score)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def pilot(items, orders, k):
    lab = items[items.pool == "acquisition"][["task_id", "item_uid", "label"]]
    o = orders.merge(lab, on=["task_id", "item_uid"])
    o = o[o.stream_position < k].groupby(["task_id", "order_seed"]).label.agg(hits="sum", n="size").reset_index()
    return o[o.n == k]


def task_gain(scores, model_a="bb-eb", model_b="jeffreys"):
    w = pd.Series(c.WEIGHTS)
    tb = scores[scores.model.isin([model_a, model_b])].groupby(["model", "task_id", "order_seed", "budget"]).brier.mean().unstack("budget")
    alc = (tb[c.BUDGETS] * w).sum(axis=1).unstack("model")
    return (alc[model_a] - alc[model_b]).rename("gain").reset_index()


def main() -> int:
    D = c.DATA / "scores"
    parts = []
    for run, panel in (("r3-adaptive-v1-validation", c.CACHE), ("r3-adaptive-v1-test", c.CACHE), ("r3-adaptive-v1-lobo", c.ID_PANEL)):
        s = pd.read_parquet(D / f"{run}.parquet")
        g = task_gain(s)
        items = pd.read_parquet(panel / "panel" / "items.parquet", columns=["task_id", "benchmark_id", "item_uid", "label", "pool"])
        orders = pd.read_parquet(panel / "panel" / "candidate_orders.parquet")
        bmap = dict(zip(items.task_id, items.benchmark_id))
        g["benchmark"] = g.task_id.map(bmap)
        for k in (3, 7):
            p = pilot(items, orders, k)
            p["dist"] = (p.hits / p.n - M).abs()
            parts.append(g.merge(p[["task_id", "order_seed", "dist"]], on=["task_id", "order_seed"]).assign(k=k, source=run.split("-")[-1]))
    d = pd.concat(parts, ignore_index=True)
    d["hurt"] = d.gain > 0
    rows = []
    for k in (3, 7):
        x = d[d.k == k]
        per = x.groupby("benchmark").apply(lambda g: auc(g.dist, g.hurt), include_groups=False)
        rows.append((k, len(x), float(x.hurt.mean()), auc(x.dist, x.hurt), float(per.mean()), int(per.notna().sum())))
        print(f"k={k}: {len(x):,} task-orders, {x.hurt.mean():.0%} hurt; pooled AUC {auc(x.dist, x.hurt):.3f}; mean per-benchmark AUC {per.mean():.3f} over {int(per.notna().sum())} benchmarks")
    pd.DataFrame(rows, columns=["pilot_labels", "task_orders", "share_hurt", "auc_pooled", "auc_mean_per_benchmark", "benchmarks"]).round(4).to_csv(c.SUMMARY / "r4-pilot-auc.csv", index=False)
    # Gain of the fitted prior by pilot distance (the mechanism)
    x = d[d.k == 7].copy()
    x["bin"] = pd.cut(x.dist, [-0.01, 0.1, 0.2, 0.3, 0.4, 0.6])
    t = x.groupby("bin", observed=True).agg(task_orders=("gain", "size"), mean_gain=("gain", "mean"), share_hurt=("hurt", "mean"))
    t.round(4).to_csv(c.SUMMARY / "r4-gain-by-pilot-distance-k7.csv"); print(t.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def switch_table() -> None:
    """Switch rule scored from saved per-budget Brier: Jeffreys replaces the fitted prior at budgets >= k when the pilot is far.
    d is chosen on the 11 training benchmarks (leave-one-out fits) and scored on the 6 unseen benchmarks."""
    D, w = c.DATA / "scores", pd.Series(c.WEIGHTS)
    out = []
    for run, panel in (("r3-adaptive-v1-validation", c.CACHE), ("r3-adaptive-v1-test", c.CACHE), ("r3-adaptive-v1-lobo", c.ID_PANEL)):
        s = pd.read_parquet(D / f"{run}.parquet")
        items = pd.read_parquet(panel / "panel" / "items.parquet", columns=["task_id", "benchmark_id", "item_uid", "label", "pool"])
        orders = pd.read_parquet(panel / "panel" / "candidate_orders.parquet")
        bmap = dict(zip(items.task_id, items.benchmark_id))
        tb = s[s.model.isin(["bb-eb", "jeffreys"])].groupby(["model", "task_id", "order_seed", "budget"]).brier.mean().unstack("budget")
        for k in (3, 7):
            p = pilot(items, orders, k)
            p["dist"] = (p.hits / p.n - M).abs()
            fit, jef = tb.loc["bb-eb"][c.BUDGETS], tb.loc["jeffreys"][c.BUDGETS]
            df = fit.join(jef, lsuffix="_fit", rsuffix="_jef").reset_index().merge(p[["task_id", "order_seed", "dist"]], on=["task_id", "order_seed"])
            for d in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 9.0):
                far = df.dist > d
                alc = 0.0
                for b in c.BUDGETS:
                    use_j = far & (b >= k)
                    alc = alc + w[b] * np.where(use_j, df[f"{b}_jef"], df[f"{b}_fit"])
                alc_j = sum(w[b] * df[f"{b}_jef"] for b in c.BUDGETS)
                gain = pd.Series(alc - alc_j.to_numpy(), index=df.index)
                g = pd.DataFrame({"task_id": df.task_id, "gain": gain}).groupby("task_id").gain.mean()
                bench = pd.Series(g.index.map(bmap), index=g.index)
                for bname, v in g.groupby(bench).mean().items():
                    out.append((bname, "lobo" if run.endswith("lobo") else run.split("-")[-1], k, d, float(v)))
    t = pd.DataFrame(out, columns=["benchmark", "source", "k", "d", "gain"])
    t.round(5).to_csv(c.SUMMARY / "r4-switch-per-benchmark.csv", index=False)
    rows = []
    for (k, d), g in t.groupby(["k", "d"]):
        tr, un = g[g.source == "lobo"], g[g.source != "lobo"]
        rows.append((k, d, tr.gain.mean(), tr.gain.max(), un.gain.mean(), un.gain.max(), g.gain.mean(), g.gain.max(), g[g.source == "test"].gain.mean()))
    r = pd.DataFrame(rows, columns=["k", "d", "macro_11_train", "worst_11_train", "macro_6_unseen", "worst_6_unseen", "macro_17", "worst_17", "test_gain"])
    r.round(4).to_csv(c.SUMMARY / "r4-switch-sweep.csv", index=False)
    print(r.round(4).to_string(index=False))
    for k in (3, 7):
        best = r[r.k == k].sort_values("macro_11_train").iloc[0]
        print(f"\nk={k}: d chosen on the 11 training benchmarks = {best.d} -> unseen macro {best.macro_6_unseen:+.4f}, unseen worst {best.worst_6_unseen:+.4f}, 17-benchmark macro {best.macro_17:+.4f}, worst {best.worst_17:+.4f}, test {best.test_gain:+.4f}")


if __name__ == "__main__":
    switch_table()
