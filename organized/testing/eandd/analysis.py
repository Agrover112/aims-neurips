"""Score-only paired summaries; never calls replay or fits a model."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = np.array([.1, .2, .2, .2, .2, .1])
KEYS = ["shape", "role", "draw", "task_id", "order_seed"]


def compare(scores, reference, seed=20261106, replicates=4000):
    if scores.empty or not scores.brier.between(0, 1).all():
        raise ValueError("Expected nonempty finite Brier scores in [0, 1]")
    if scores.duplicated(["model", *KEYS, "budget"]).any():
        raise ValueError("Duplicate score keys; select one configuration per model")
    rows = []
    rng = np.random.default_rng(seed)
    for (shape, role), group in scores.groupby(["shape", "role"], sort=True):
        curves = {}
        for name, arm in group.groupby("model", sort=True):
            curve = arm.pivot(index=KEYS, columns="budget", values="brier").sort_index()
            if tuple(curve.columns) != BUDGETS or curve.isna().any().any():
                raise ValueError(f"Incomplete budget ladder for {name}")
            curves[name] = curve
        control = curves[reference]
        for name, curve in curves.items():
            if not curve.index.equals(control.index):
                raise ValueError(f"Unpaired trajectories for {name}")
            task = curve.index.get_level_values("task_id")
            weight = 1 / pd.Series(task).map(pd.Series(task).value_counts()).to_numpy()
            values = np.column_stack([curve.to_numpy(), curve.to_numpy() @ WEIGHTS])
            delta = np.column_stack([(curve-control).to_numpy(), (curve-control).to_numpy() @ WEIGHTS])
            # Resample whole tasks, keeping all their draws, budgets, and order seeds together.
            cluster = pd.DataFrame(delta * weight[:, None]).groupby(task).sum().to_numpy()
            mass = pd.Series(weight).groupby(task).sum().to_numpy()
            draws = rng.multinomial(len(mass), np.full(len(mass), 1 / len(mass)), size=replicates)
            interval = np.quantile((draws @ cluster) / (draws @ mass)[:, None], [.025, .975], axis=0)
            for j, budget in enumerate([*BUDGETS, "ALC"]):
                rows.append(dict(shape=shape, role=role, model=name, budget=budget,
                                 brier=np.average(values[:, j], weights=weight),
                                 delta=np.average(delta[:, j], weights=weight),
                                 ci_low=interval[0,j], ci_high=interval[1,j],
                                 tasks=len(mass), trajectories=len(curve)))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path, nargs="+", required=True)
    p.add_argument("--reference", required=True, help="Exact reference model ID from scores")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    scores = pd.concat([pd.read_parquet(path) for path in a.scores], ignore_index=True)
    result = compare(scores, a.reference)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.output, index=False)


if __name__ == "__main__":
    main()
