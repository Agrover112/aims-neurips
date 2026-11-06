"""R8: rescore frozen Beta-Binomial trajectories for all three-benchmark holdouts."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

import common as c
from lobo_prior import eb_beta
from wp9 import verify_draws

ORIGINAL = ("algotune", "cmmlu", "editbench")
KEYS = ["draw", "task_id", "order_seed"]


def load_statistics(handoff):
    frames = []
    for run in ("wp5-competition-test", "wp5-competition-val", "wp10-lobo-v1"):
        config = json.loads((handoff / "runs" / f"{run}.json").read_text())["config"]
        sc = pd.read_parquet(handoff / "scores" / f"{run}.parquet")
        if run == "wp10-lobo-v1":
            for fold in config["folds"]:
                bench = fold["bench"]
                mean, strength = fold["eb_bench"]
                frames.append(decode(sc.loc[sc.held_out.eq(bench)], "bb-eb-without-K",
                                     mean * strength, (1 - mean) * strength))
        else:
            verify_draws(sc, handoff)
            _, a, b = config["models"]["bb-eb"]
            frames.append(decode(sc, "bb-eb", a, b))
    return pd.concat(frames, ignore_index=True)


def brier(p, q):
    return p * p + q * (1 - 2 * p)


def decode(sc, fitted, a, b):
    # For a constant prediction p, Brier = p^2 + q(1-2p), with q the
    # evaluation success rate. Budget zero uniquely identifies q since the
    # fitted mean differs from .5. Then enumerate the integer prefix hits.
    t = (sc.loc[sc.model.isin([fitted, "jeffreys"])]
         .pivot(index=KEYS + ["budget"], columns="model", values="brier")
         .reset_index().sort_values(KEYS + ["budget"]))
    p0 = a / (a + b)
    if abs(1 - 2 * p0) < 1e-7:
        raise ValueError("Cannot decode a symmetric prior")
    q = t.loc[t.budget.eq(0), KEYS + [fitted]].copy()
    q["q"] = (q.pop(fitted) - p0 * p0) / (1 - 2 * p0)
    t = t.merge(q, on=KEYS, validate="many_to_one")
    if not t.q.between(-1e-7, 1 + 1e-7).all():
        raise ValueError("Decoded evaluation rate outside [0, 1]")
    h = np.arange(max(c.BUDGETS) + 1)[None, :]
    k = t.budget.to_numpy()[:, None]
    q = t.q.to_numpy()[:, None]
    err = np.maximum(abs(brier((h+a)/(k+a+b), q) - t[fitted].to_numpy()[:, None]),
                     abs(brier((h+.5)/(k+1), q) - t.jeffreys.to_numpy()[:, None]))
    match = (h <= k) & (err < 1e-7)
    if not (match.sum(axis=1) == 1).all():
        raise ValueError("Saved scores do not uniquely identify acquisition hits")
    t["hits"] = match.argmax(axis=1)
    d = t.groupby(KEYS)[["hits", "budget"]].diff().dropna()
    if not d.hits.between(0, d.budget).all():
        raise ValueError("Decoded hits violate nested acquisition")
    return t.assign(benchmark=t.task_id.str.rsplit("::", n=1).str[-1],
                    historical_delta=t[fitted] - t.jeffreys).drop(columns=fitted)


def by_task(t, column):
    curve = t.groupby(["benchmark", "task_id", "budget"])[column].mean().unstack("budget")
    curve = curve.reindex(columns=c.BUDGETS)
    if curve.isna().any().any():
        raise ValueError("Incomplete budget ladder")
    return curve @ pd.Series(c.WEIGHTS)


def run(handoff, responses, output):
    t = load_statistics(handoff)
    observed = by_task(t, "historical_delta")
    historical = observed.groupby("benchmark").mean()
    test = t.loc[t.benchmark.isin(ORIGINAL)]
    test_scores = pd.read_parquet(handoff / "scores/wp5-competition-test.parquet")
    references = {"bb-eb": -.0138, "rasch-fitted": -.0119, "rasch-retuned": -.0032,
                  "kernel-old": -.0024, "jeffreys": .2499}
    base = by_task(test, "jeffreys").mean()
    for model, expected in references.items():
        sc = test_scores.loc[test_scores.model.eq(model)].copy()
        sc["benchmark"] = sc.task_id.str.rsplit("::", n=1).str[-1]
        got = by_task(sc, "brier").mean() - (base if model != "jeffreys" else 0)
        if round(got, 4) != expected:
            raise ValueError(f"Stored reference did not reproduce: {model}: {got}")
    counts = pd.read_parquet(responses, columns=["benchmark_id", "subject_uid", "benchmark_split", "label"])
    train = counts.benchmark_split.eq("train")
    cn = counts.groupby(["benchmark_id", "subject_uid"]).label.agg(hits="sum", n="size").reset_index()
    train_cn = counts.loc[train].groupby(["benchmark_id", "subject_uid"]).label.agg(hits="sum", n="size").reset_index()
    original_fit = eb_beta(train_cn, "bench")
    if not np.allclose(original_fit, [.5216, 3.6755], atol=[1e-5, .002], rtol=0):
        raise ValueError(f"Canonical training prior did not reproduce: {original_fit}")
    print(f"Original training-only refit: mean={original_fit[0]:.8f}, strength={original_fit[1]:.8f}",
          flush=True)
    if len(historical) != 17 or set(historical.index) != set(cn.benchmark_id):
        raise ValueError("Expected the same 17 benchmarks in scores and response counts")
    rows = []
    groups = {bench: g.copy() for bench, g in t.groupby("benchmark")}
    for split in combinations(sorted(groups), 3):
        mean, strength = eb_beta(cn.loc[~cn.benchmark_id.isin(split)], "bench")
        fold = pd.concat([groups[b] for b in split], ignore_index=True)
        p = (fold.hits + mean * strength) / (fold.budget + strength)
        fold["delta"] = brier(p, fold.q) - fold.jeffreys
        per_task = by_task(fold, "delta")
        macro = per_task.groupby("benchmark").mean().mean()
        rows.append((*split, split == ORIGINAL, mean, strength, macro, per_task.mean(),
                     historical.loc[list(split)].mean()))
    result = pd.DataFrame(rows, columns=["benchmark_1", "benchmark_2", "benchmark_3", "original_test_triple",
                                         "prior_mean", "prior_strength", "refit_macro_delta", "refit_task_delta",
                                         "historical_macro_delta"])
    output.mkdir(parents=True, exist_ok=False)
    result.to_csv(output / "wp13-split-sensitivity.csv", index=False)
    rng = np.random.default_rng(20261106)
    rates = t.loc[t.budget.eq(0)].groupby(["benchmark", "task_id"]).q.mean()
    explanation = []
    for bench, task_delta in observed.groupby("benchmark"):
        values = task_delta.to_numpy()
        low, high = np.quantile(rng.choice(values, size=(2000, len(values))).mean(axis=1), [.025, .975])
        distance = abs(rates.loc[bench] - .52).mean()
        explanation.append((bench, len(values), values.mean(), low, high, distance))
    pd.DataFrame(explanation, columns=["benchmark", "tasks", "delta", "ci_low", "ci_high",
                                     "mean_abs_query_acc_minus_0.52"]).to_csv(
                                         output / "wp13-benchmark-explanation.csv", index=False)
    for col in ["historical_macro_delta", "refit_macro_delta", "refit_task_delta"]:
        v = result[col]
        orig = v.loc[result.original_test_triple].item()
        print(f"{col}: original={orig:.5f}; percentile={(v <= orig).mean():.2%}; "
              f"mean={v.mean():.5f}; p05/p95={v.quantile([.05, .95]).tolist()}; "
              f"harm={(v > 0).mean():.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--handoff", required=True, type=Path)
    ap.add_argument("--responses", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    run(args.handoff, args.responses, args.output)
