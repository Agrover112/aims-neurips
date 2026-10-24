#!/usr/bin/env python3
"""Compute the information ceiling ALC_min described in docs/information-ceiling-note.md.

The question: with at most 31 binary responses, how precisely can a scalar
ability be known, and what Brier score does that imply even for a perfect
forecaster? If the Beta-Binomial baseline already sits at that floor, no
architecture can beat it and the ablation ladder is chasing a ghost.

Construction, following the note:

1. **Oracle item parameters.** Fit a 2PL ``(a_i, b_i)`` per item and ``theta_j``
   per task by alternating MLE on *all* labels of the split, evaluation pool
   included. This is deliberate cheating: it hands the forecaster item
   parameters it could never know, so whatever remains is purely the cost of
   estimating ability from a finite context.

   Fitting is per benchmark. As established, ability and difficulty are only
   separable up to one additive constant per benchmark (no subject spans two
   benchmarks), so a global fit would be unidentified. Each benchmark is
   centred independently.

2. **Laplace posterior for ability.** For each task, order, and budget, run the
   same penalized Newton MAP the production code uses, on the real acquired
   context and real labels, with the oracle item parameters. The curvature at
   the mode gives ``sigma_theta^2(n) = 1 / (I_n + 1/tau^2)``.

3. **Posterior-predictive Brier.** Integrate ``sigma(a_t (theta - b_t))`` over
   ``N(theta_MAP, sigma_theta^2)`` by Gauss-Hermite quadrature and score the
   result against the real evaluation labels.

Three numbers come out:

* ``alc_known_theta`` -- oracle item parameters *and* oracle ability. The
  absolute floor: nothing but Bernoulli noise and 2PL misspecification.
* ``alc_min`` -- oracle item parameters, ability estimated from the budget.
  **This is the information ceiling.** No forecaster can beat it without
  knowing item parameters better than a full-data fit.
* ``alc_beta`` -- the Beta-Binomial baseline on the identical rows, as the
  reference point every ablation table uses.

Usage:

    python scripts/information_ceiling.py --corpus /path/to/allbench-features-v1
    python scripts/information_ceiling.py --corpus DIR --split test --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
PRIOR_VARIANCE = 4.0
QUADRATURE_NODES = 41
FIT_ROUNDS = 60
RIDGE_A = 1.0
RIDGE_B = 1.0
A_FLOOR = 0.05
A_CEIL = 4.0
THETA_CLAMP = 4.0


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def fit_2pl(
    labels: np.ndarray,
    item_codes: np.ndarray,
    task_codes: np.ndarray,
    n_items: int,
    n_tasks: int,
    *,
    rounds: int = FIT_ROUNDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alternating MAP fit of a 2PL on one benchmark.

    Returns ``(a, b, theta)``. Ability is re-centred every round: within a
    benchmark the ability/difficulty split is identified only up to a shared
    additive constant, so the centring pins that constant rather than letting
    it drift.
    """

    a = np.ones(n_items)
    b = np.zeros(n_items)
    theta = np.zeros(n_tasks)

    for _ in range(rounds):
        # --- ability, given item parameters (Newton, one scalar per task) ---
        for _ in range(3):
            eta = a[item_codes] * (theta[task_codes] - b[item_codes])
            p = _sigmoid(eta)
            residual = labels - p
            gradient = np.bincount(
                task_codes, weights=a[item_codes] * residual, minlength=n_tasks
            ) - theta / PRIOR_VARIANCE
            curvature = np.bincount(
                task_codes,
                weights=a[item_codes] ** 2 * p * (1.0 - p),
                minlength=n_tasks,
            ) + 1.0 / PRIOR_VARIANCE
            theta = np.clip(theta + gradient / np.maximum(curvature, 1e-9), -THETA_CLAMP, THETA_CLAMP)
        theta -= theta.mean()

        # --- item parameters, given ability (Newton on the 2-vector per item) ---
        for _ in range(3):
            eta = a[item_codes] * (theta[task_codes] - b[item_codes])
            p = _sigmoid(eta)
            residual = labels - p
            weight = p * (1.0 - p)
            offset = theta[task_codes] - b[item_codes]

            grad_a = np.bincount(item_codes, weights=residual * offset, minlength=n_items)
            grad_a -= RIDGE_A * (a - 1.0)
            curv_a = np.bincount(
                item_codes, weights=weight * offset**2, minlength=n_items
            ) + RIDGE_A

            grad_b = np.bincount(item_codes, weights=-residual * a[item_codes], minlength=n_items)
            grad_b -= RIDGE_B * b
            curv_b = np.bincount(
                item_codes, weights=weight * a[item_codes] ** 2, minlength=n_items
            ) + RIDGE_B

            a = np.clip(a + grad_a / np.maximum(curv_a, 1e-9), A_FLOOR, A_CEIL)
            b = np.clip(b + grad_b / np.maximum(curv_b, 1e-9), -6.0, 6.0)

    return a, b, theta


def laplace_posterior(
    a_context: np.ndarray, b_context: np.ndarray, y_context: np.ndarray
) -> tuple[float, float]:
    """Penalized Newton MAP plus curvature, matching the production solver.

    Returns ``(theta_map, sigma_squared)`` where
    ``sigma_squared = 1 / (I_n(theta_map) + 1 / tau^2)``.
    """

    theta = 0.0
    if len(y_context):
        for _ in range(25):
            p = _sigmoid(a_context * (theta - b_context))
            gradient = float(np.sum(a_context * (y_context - p))) - theta / PRIOR_VARIANCE
            curvature = (
                float(np.sum(a_context**2 * p * (1.0 - p))) + 1.0 / PRIOR_VARIANCE
            )
            step = gradient / max(curvature, 1e-9)
            theta = float(np.clip(theta + step, -THETA_CLAMP, THETA_CLAMP))
            if abs(step) < 1e-10:
                break
        p = _sigmoid(a_context * (theta - b_context))
        information = float(np.sum(a_context**2 * p * (1.0 - p)))
    else:
        information = 0.0
    return theta, 1.0 / (information + 1.0 / PRIOR_VARIANCE)


def predictive_probability(
    a_target: np.ndarray, b_target: np.ndarray, theta_map: float, sigma_squared: float
) -> np.ndarray:
    """E_theta[sigma(a(theta - b))] under N(theta_map, sigma^2), by quadrature."""

    nodes, weights = np.polynomial.hermite_e.hermegauss(QUADRATURE_NODES)
    weights = weights / weights.sum()
    theta_grid = theta_map + np.sqrt(sigma_squared) * nodes
    surface = _sigmoid(
        a_target[None, :] * (theta_grid[:, None] - b_target[None, :])
    )
    return surface.T @ weights


def run(corpus: Path, split: str, orders: int, folds: int) -> dict[str, Any]:
    items = pd.read_parquet(corpus / "eval_items.parquet")
    order_frame = pd.read_parquet(corpus / "eval_orders.parquet")
    items = items[items["split"] == split].copy()
    items["task_id"] = items["task_id"].astype(str)
    order_frame["task_id"] = order_frame["task_id"].astype(str)
    print(f"{split}: {items['task_id'].nunique()} tasks, {len(items):,} rows", flush=True)

    weight_of = dict(zip(BUDGETS, WEIGHTS, strict=True))
    per_task: list[dict[str, Any]] = []
    coverage: list[float] = []

    for benchmark, block in items.groupby("benchmark_id", sort=True):
        all_tasks = sorted(block["task_id"].unique())
        n_folds = max(2, min(folds, len(all_tasks)))
        fold_of = {task: index % n_folds for index, task in enumerate(all_tasks)}
        base_rate = float(block["label"].mean())
        base_logit = float(np.log(max(base_rate, 1e-6) / max(1.0 - base_rate, 1e-6)))

        # Item parameters are fitted out-of-fold: a task's own labels never
        # inform the parameters used to score it. Without this the fit simply
        # memorizes -- items here carry only a handful of observations, and two
        # free parameters each would reproduce the labels they are scored on.
        a_of: dict[Any, float] = {}
        b_of: dict[Any, float] = {}
        theta_in_fold: dict[str, float] = {}
        for fold in range(n_folds):
            fit_block = block[block["task_id"].map(fold_of) != fold]
            if fit_block.empty:
                continue
            item_codes, item_ids = pd.factorize(fit_block["item_index"], sort=True)
            task_codes, task_ids = pd.factorize(fit_block["task_id"], sort=True)
            a, b, theta = fit_2pl(
                fit_block["label"].to_numpy(dtype=float),
                item_codes,
                task_codes,
                len(item_ids),
                len(task_ids),
            )
            held_out = {task for task, index in fold_of.items() if index == fold}
            for item_id, a_value, b_value in zip(item_ids, a, b, strict=True):
                # Only items the held-out tasks will actually need.
                a_of.setdefault((fold, item_id), float(a_value))
                b_of.setdefault((fold, item_id), float(b_value))
            for task_id, theta_value in zip(task_ids, theta, strict=True):
                theta_in_fold.setdefault(task_id, float(theta_value))
            del held_out

        # A full-data fit is kept only as the oracle-ability reference point.
        item_codes, item_ids = pd.factorize(block["item_index"], sort=True)
        task_codes, task_ids = pd.factorize(block["task_id"], sort=True)
        a_full, b_full, theta_full = fit_2pl(
            block["label"].to_numpy(dtype=float),
            item_codes,
            task_codes,
            len(item_ids),
            len(task_ids),
        )
        in_sample = _sigmoid(
            a_full[item_codes] * (theta_full[task_codes] - b_full[item_codes])
        )
        theta_of = dict(zip(task_ids, theta_full, strict=True))
        a_full_of = dict(zip(item_ids, a_full, strict=True))
        b_full_of = dict(zip(item_ids, b_full, strict=True))
        print(
            f"  {benchmark}: {len(task_ids)} tasks, {len(item_ids)} items, "
            f"{len(block) / max(len(item_ids), 1):.1f} obs/item, "
            f"in-sample brier={np.mean((in_sample - block['label'].to_numpy()) ** 2):.4f}",
            flush=True,
        )

        for task_id, task_rows in block.groupby("task_id", sort=True):
            fold = fold_of[task_id]

            def _out_of_fold(index_array, table, fallback, fold=fold):
                return np.array(
                    [table.get((fold, index), fallback) for index in index_array],
                    dtype=float,
                )

            targets = task_rows[task_rows["pool"] == "evaluation"]
            if targets.empty:
                continue
            target_items = targets["item_index"].to_numpy()
            a_target = _out_of_fold(target_items, a_of, 1.0)
            b_target = _out_of_fold(target_items, b_of, base_logit * -1.0)
            y_target = targets["label"].to_numpy(dtype=float)
            coverage.append(
                float(np.mean([(fold, i) in a_of for i in target_items]))
            )

            label_of = dict(
                zip(
                    task_rows["item_index"].to_numpy(),
                    task_rows["label"].to_numpy(dtype=float),
                    strict=True,
                )
            )

            # Oracle-ability reference uses the full-data fit on purpose: it is
            # the optimistic floor, not the ceiling being reported.
            a_known = np.array([a_full_of[i] for i in target_items], dtype=float)
            b_known = np.array([b_full_of[i] for i in target_items], dtype=float)
            known = _sigmoid(a_known * (theta_of[task_id] - b_known))
            brier_known = float(np.mean((known - y_target) ** 2))

            task_orders = order_frame[order_frame["task_id"] == task_id]
            seeds = sorted(task_orders["order_seed"].unique())[:orders]

            min_scores: list[float] = []
            beta_scores: list[float] = []
            for seed in seeds:
                sequence = (
                    task_orders[task_orders["order_seed"] == seed]
                    .sort_values("acquisition_rank")["item_index"]
                    .to_numpy()
                )
                alc_min = 0.0
                alc_beta = 0.0
                for budget in BUDGETS:
                    context = sequence[:budget]
                    y_context = np.array([label_of[i] for i in context], dtype=float)
                    a_context = _out_of_fold(context, a_of, 1.0)
                    b_context = _out_of_fold(context, b_of, base_logit * -1.0)

                    theta_map, sigma_squared = laplace_posterior(
                        a_context, b_context, y_context
                    )
                    predictive = predictive_probability(
                        a_target, b_target, theta_map, sigma_squared
                    )
                    alc_min += weight_of[budget] * float(
                        np.mean((predictive - y_target) ** 2)
                    )

                    beta = (2.0 + float(y_context.sum())) / (4.0 + budget)
                    alc_beta += weight_of[budget] * float(np.mean((beta - y_target) ** 2))
                min_scores.append(alc_min)
                beta_scores.append(alc_beta)

            per_task.append(
                {
                    "benchmark_id": benchmark,
                    "task_id": task_id,
                    "alc_known_theta": brier_known,
                    "alc_min": float(np.mean(min_scores)),
                    "alc_beta": float(np.mean(beta_scores)),
                }
            )

    frame = pd.DataFrame(per_task)
    gap = frame["alc_beta"] - frame["alc_min"]
    rng = np.random.default_rng(20261022)
    index = rng.integers(0, len(gap), size=(10_000, len(gap)))
    draws = gap.to_numpy()[index].mean(axis=1)

    return {
        "status": "information_ceiling",
        "split": split,
        "tasks": len(frame),
        "orders": int(orders),
        "alc_known_theta": float(frame["alc_known_theta"].mean()),
        "alc_min": float(frame["alc_min"].mean()),
        "alc_beta": float(frame["alc_beta"].mean()),
        "headroom_beta_minus_min": float(gap.mean()),
        "headroom_ci_low": float(np.quantile(draws, 0.025)),
        "headroom_ci_high": float(np.quantile(draws, 0.975)),
        "tasks_where_beta_beats_ceiling": int((gap < 0).sum()),
        "item_parameter_coverage": float(np.mean(coverage)) if coverage else 0.0,
        "folds": int(folds),
        "note": (
            "alc_min uses out-of-fold item parameters: a task's own labels never "
            "inform the parameters used to score it. alc_known_theta uses a "
            "full-data fit and is an optimistic floor, not a ceiling."
        ),
        "by_benchmark": frame.groupby("benchmark_id", as_index=False)
        .agg(
            tasks=("task_id", "size"),
            alc_known_theta=("alc_known_theta", "mean"),
            alc_min=("alc_min", "mean"),
            alc_beta=("alc_beta", "mean"),
        )
        .assign(headroom=lambda f: f["alc_beta"] - f["alc_min"])
        .to_dict(orient="records"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--orders", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args()

    result = run(arguments.corpus, arguments.split, arguments.orders, arguments.folds)

    print("\n" + "=" * 70)
    print(f"INFORMATION CEILING  ({result['split']}, {result['tasks']} tasks)")
    print("=" * 70)
    print(f"  oracle item params + oracle ability : {result['alc_known_theta']:.6f}")
    print(f"  oracle item params, ability from n  : {result['alc_min']:.6f}   <- ALC_min")
    print(f"  Beta-Binomial baseline              : {result['alc_beta']:.6f}")
    print(f"\n  out-of-fold item parameter coverage : {result['item_parameter_coverage']:.1%}")
    print(
        f"\n  headroom (Beta - ceiling): {result['headroom_beta_minus_min']:+.6f}"
        f"  95% CI [{result['headroom_ci_low']:+.6f}, {result['headroom_ci_high']:+.6f}]"
    )
    print(
        f"  tasks where Beta already beats the ceiling: "
        f"{result['tasks_where_beta_beats_ceiling']}/{result['tasks']}"
    )
    print("\n  per benchmark:")
    print(f"    {'benchmark':<16}{'tasks':>6}{'ceiling':>11}{'beta':>11}{'headroom':>11}")
    for row in result["by_benchmark"]:
        print(
            f"    {row['benchmark_id']:<16}{row['tasks']:>6}{row['alc_min']:>11.6f}"
            f"{row['alc_beta']:>11.6f}{row['headroom']:>+11.6f}"
        )

    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=float) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
