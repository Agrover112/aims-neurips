"""Fit and select Beta-Binomial priors without opening test replay rows."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, expit, logit

BUDGETS = np.array([0, 1, 3, 7, 15, 31])
WEIGHTS = np.array([.1, .2, .2, .2, .2, .1])
GRID = tuple(itertools.product((.2, .3, .4, .45, .5, .55, .6, .7, .8),
                               (.25, .5, 1, 2, 4, 8, 16, 32, 64)))


def fit_empirical_bayes(responses):
    if set(responses.benchmark_split) != {"train"}:
        raise ValueError("Empirical Bayes must receive training rows only")
    if not responses.label.isin([0, 1]).all():
        raise ValueError("Expected binary labels")
    counts = responses.groupby(["benchmark_id", "subject_uid"]).label.agg(["sum", "count"]).reset_index()
    if counts.empty:
        raise ValueError("No training tasks")
    weights = 1 / counts.groupby("benchmark_id").subject_uid.transform("size").to_numpy()
    weights /= weights.sum()
    s, n = counts["sum"].to_numpy(), counts["count"].to_numpy()

    def objective(z):
        mean, strength = expit(z[0]), np.exp(z[1])
        a, b = mean * strength, (1 - mean) * strength
        return -float(weights @ (betaln(a+s, b+n-s) - betaln(a, b)))

    opt = minimize(objective, [0., 0.], method="L-BFGS-B",
                   bounds=[(logit(.01), logit(.99)), (np.log(.05), np.log(256))])
    if not opt.success:
        raise RuntimeError(opt.message)
    return float(expit(opt.x[0])), float(np.exp(opt.x[1]))


def replay_curves(draws, items, orders):
    keys = ["shape", "role", "draw", "task_id", "order_seed"]
    if draws.empty or draws.duplicated(keys).any():
        raise ValueError("Expected nonempty, unique frozen draw rows")
    item_keys = ["task_id", "item_uid"]
    if items.duplicated(item_keys).any() or not items.label.isin([0, 1]).all():
        raise ValueError("Expected unique binary task/item rows")
    if set(items.benchmark_split) != set(draws.role):
        raise ValueError("Panel and draws have different roles")
    stream = orders.merge(items.loc[items.pool.eq("acquisition"), item_keys + ["label"]],
                          on=item_keys, how="left", validate="many_to_one")
    if stream.label.isna().any() or stream.duplicated(["task_id", "order_seed", "item_uid"]).any():
        raise ValueError("Duplicate stream item or stream item outside acquisition pool")
    query = items.loc[items.pool.eq("evaluation")].groupby("task_id").label.mean()
    prefix = {}
    for key, group in stream.groupby(["task_id", "order_seed"]):
        group = group.sort_values("stream_position")
        if len(group) < 31 or not np.array_equal(group.stream_position, np.arange(len(group))):
            raise ValueError(f"Invalid candidate order for {key}")
        prefix[key] = np.r_[0, group.label.to_numpy()[:31].cumsum()][BUDGETS]
    successes = np.stack([prefix[t, s] for t, s in zip(draws.task_id, draws.order_seed)])
    rates = query.loc[draws.task_id].to_numpy()[:, None]
    return successes, rates


def curve(mean, strength, successes, rates):
    p = (mean * strength + successes) / (strength + BUDGETS)
    return rates * (1-p)**2 + (1-rates) * p**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("responses", "panel", "draws", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    train = pd.read_parquet(args.responses, columns=["benchmark_id", "subject_uid", "label", "benchmark_split"],
                            filters=[("benchmark_split", "==", "train")])
    empirical = fit_empirical_bayes(train)
    draws = pd.read_parquet(args.draws, filters=[("role", "==", "validation")])
    items = pd.read_parquet(args.panel / "items.parquet", filters=[("benchmark_split", "==", "validation")])
    orders = pd.read_parquet(args.panel / "candidate_orders.parquet",
                             filters=[("task_id", "in", sorted(set(draws.task_id)))])
    if set(draws.role) != {"validation"}:
        raise ValueError("Selection must use validation only")
    result = {}
    grid_rows = []
    for shape, group in draws.groupby("shape", sort=True):
        success, rate = replay_curves(group, items, orders)
        def score(params):
            alc = curve(*params, success, rate) @ WEIGHTS
            return float(pd.Series(alc).groupby(group.task_id.to_numpy()).mean().mean())
        scored = sorted((score(params), params) for params in GRID)
        result[str(shape)] = dict(tuned=dict(zip(("mean", "strength"), scored[0][1])),
                                  tuned_validation_alc=scored[0][0], jeffreys_validation_alc=score((.5, 1)),
                                  empirical_bayes_validation_alc=score(empirical))
        grid_rows.extend(dict(shape=shape, mean=m, strength=k, validation_alc=alc)
                         for alc, (m, k) in scored)
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(grid_rows).to_csv(args.output / "validation-grid.csv", index=False)
    (args.output / "priors.json").write_text(json.dumps(dict(
        empirical_bayes=dict(zip(("mean", "strength"), empirical)),
        shapes=result, averaging="equal task means; average repeated appearances within task",
        train_rows=len(train), test_read=False,
        input_bytes={str(p): p.stat().st_size for p in
                     (args.responses, args.draws, args.panel/"items.parquet", args.panel/"candidate_orders.parquet")}),
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
