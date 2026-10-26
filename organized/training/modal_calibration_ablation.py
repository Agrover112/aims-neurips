"""Decompose and stress-test the per-budget probability calibrator.

The script rescores the selected CNP-AIRT checkpoint (and optionally
Online-AIRT) on validation and test, then fits several calibration variants on
validation only and evaluates them on test. It writes per-budget Brier/ECE and
weighted ALC, with standard errors across tasks.

Variants
--------
raw                 sigmoid(z)
platt_bce           sigmoid(s_b z + c_b)
anchor_bce          (1 - w_b) sigmoid(z) + w_b rho_C
full_bce            (1 - w_b) sigmoid(s_b z + c_b) + w_b rho_C
context             rho_C
platt_brier         Platt form fitted by weighted Brier instead of BCE
anchor_brier        anchor form fitted by weighted Brier
full_brier          full form fitted by weighted Brier
full_bce_task       full BCE with each task weighted equally
full_brier_task     full Brier with each task weighted equally
full_bce_shared     one shared full BCE calibrator across all budgets
full_brier_shared   one shared full Brier calibrator across all budgets
full_bce_zero_half  full BCE, but force budget 0 to 0.5
full_brier_zero_half full Brier, but force budget 0 to 0.5

All fits use validation rows only. Evaluation is on the disjoint test split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal
import numpy as np

APP_NAME = "aims-neurips-calibration-ablation"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "allbench-features-v1"
RESULT_ROOT = REMOTE_ROOT / "results" / "allbench-deterministic-2seed"
OUTPUT_ROOT = REMOTE_ROOT / "results" / "allbench-calibration-ablation"
UNIFORM_ROOT = REMOTE_ROOT / "results" / "allbench-uniform-v1" / "calibration"
UNIFORM_FOLDS = 5
MODES = ("masked", "visible")
SEEDS = (20261018, 20261019)
SUBJECT_SLICE = (384, 768)
ORDER_SEEDS = tuple(range(10))
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
ANCHORS = tuple(float(value) for value in np.linspace(0.0, 1.0, 21))
MIN_SAMPLES = 200

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2,<3",
        "pandas>=2.2,<3",
        "pyarrow>=17,<24",
        "scipy>=1.13,<2",
        "torch==2.5.1",
    )
    .add_local_dir(PROJECT_ROOT / "src", "/opt/aims/src", copy=True)
    .env(
        {
            "PYTHONPATH": "/opt/aims/src",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cnp_logits(model, x_T, context: dict[str, Any], budget: int):
    import torch

    device = next(model.parameters()).device
    x_C = torch.as_tensor(context["x_C_pool"][:budget], dtype=torch.float32, device=device)
    y_C = torch.as_tensor(context["y_C_pool"][:budget], dtype=torch.float32, device=device)
    with torch.no_grad():
        return model.logits(x_C, y_C, x_T).cpu().numpy()


def _online_logits(model, x_T, context: dict[str, Any], budget: int):
    import torch

    from aims_starter.faithful_np import map_theta_2pl

    device = next(model.parameters()).device
    a_T, b_T = model.item_parameters(x_T)
    x_C = torch.as_tensor(context["x_C_pool"][:budget], dtype=torch.float32, device=device)
    y_C = torch.as_tensor(context["y_C_pool"][:budget], dtype=torch.float32, device=device)
    theta = map_theta_2pl(model, x_C, y_C)
    with torch.no_grad():
        return (a_T * (theta - b_T)).cpu().numpy()


def _load_models(seed: int, mode: str, x_dim: int):
    import torch

    from aims_starter.faithful_np import CNPAIRT, Amortized2PL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_slice = SUBJECT_SLICE if mode == "masked" else None
    root = RESULT_ROOT / f"seed-{seed}-{mode}-checkpoints"
    cnp_state = torch.load(root / "cnp_airt.pt", map_location=device)
    cnp = CNPAIRT(x_dim, subject_feature_slice=subject_slice).to(device)
    cnp.load_state_dict(cnp_state)
    cnp.eval()
    online_state = torch.load(root / "online_airt_2pl.pt", map_location=device)
    online = Amortized2PL(
        x_dim,
        int(online_state["theta"].shape[0]),
        subject_feature_slice=subject_slice,
    ).to(device)
    online.load_state_dict(online_state)
    online.eval()
    return cnp, online


def _sigmoid(value):
    import numpy as np

    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def _context_rate(labels):
    import numpy as np

    values = np.asarray(labels, dtype=float)
    return (1.0 + float(values.sum())) / (2.0 + len(values))


def _flatten(samples: list[dict[str, Any]]):
    import numpy as np

    logits = np.concatenate([sample["logits"] for sample in samples])
    labels = np.concatenate([sample["labels"] for sample in samples])
    rates = np.concatenate(
        [
            np.full(len(sample["labels"]), _context_rate(sample["context_labels"]))
            for sample in samples
        ]
    )
    task_ids = np.concatenate(
        [np.full(len(sample["labels"]), sample["task_id"], dtype=object) for sample in samples]
    )
    budgets = np.concatenate(
        [np.full(len(sample["labels"]), sample["budget"], dtype=int) for sample in samples]
    )
    return logits, labels, rates, task_ids, budgets


def _task_weights(task_ids, balanced: bool):
    import numpy as np

    if not balanced:
        return np.ones(len(task_ids), dtype=float)
    names, counts = np.unique(task_ids, return_counts=True)
    mapping = {str(name): int(count) for name, count in zip(names, counts, strict=True)}
    return np.array([1.0 / mapping[str(task_id)] for task_id in task_ids], dtype=float)


def _fit_logistic(logits, labels, weights):
    import numpy as np

    features = np.column_stack([logits, np.ones_like(logits)])
    beta = np.zeros(2, dtype=float)
    for _ in range(30):
        probability = _sigmoid(features @ beta)
        gradient = features.T @ (weights * (labels - probability))
        curvature = (features * (weights * probability * (1.0 - probability))[:, None]).T @ features
        curvature += 1e-8 * np.eye(2)
        step = np.linalg.solve(curvature, gradient)
        beta += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def _weighted_bce(probability, labels, weights):
    import numpy as np

    clipped = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return float(
        np.average(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped), weights=weights)
        * -1.0
    )


def _weighted_brier(probability, labels, weights):
    import numpy as np

    return float(np.average((probability - labels) ** 2, weights=weights))


def _fit_platt(logits, labels, weights, loss: str):
    import numpy as np

    if loss == "bce":
        return _fit_logistic(logits, labels, weights)
    from scipy.optimize import minimize

    def objective(params):
        probability = _sigmoid(params[0] * logits + params[1])
        return _weighted_brier(probability, labels, weights)

    result = minimize(
        objective,
        np.array([1.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(-6.0, 6.0), (-6.0, 6.0)],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return float(result.x[0]), float(result.x[1])


def _fit_anchor(logits, labels, rates, weights, loss: str):

    base = _sigmoid(logits)
    best = None
    for anchor in ANCHORS:
        probability = (1.0 - anchor) * base + anchor * rates
        value = (
            _weighted_bce(probability, labels, weights)
            if loss == "bce"
            else _weighted_brier(probability, labels, weights)
        )
        if best is None or value < best[0]:
            best = (value, float(anchor))
    return (best[1],) if best is not None else (0.0,)


def _fit_full(logits, labels, rates, weights, loss: str):
    import numpy as np

    scale, bias = _fit_logistic(logits, labels, weights)
    best = None
    for anchor in ANCHORS:
        probability = (1.0 - anchor) * _sigmoid(scale * logits + bias) + anchor * rates
        value = (
            _weighted_bce(probability, labels, weights)
            if loss == "bce"
            else _weighted_brier(probability, labels, weights)
        )
        if best is None or value < best[0]:
            best = (value, scale, bias, float(anchor))
    assert best is not None
    initial = np.array([best[1], best[2], best[3]], dtype=float)
    if loss == "bce":
        return float(best[1]), float(best[2]), float(best[3])

    from scipy.optimize import minimize

    def objective(params):
        probability = (1.0 - params[2]) * _sigmoid(params[0] * logits + params[1])
        probability = probability + params[2] * rates
        return _weighted_brier(probability, labels, weights)

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-6.0, 6.0), (-6.0, 6.0), (0.0, 1.0)],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return float(result.x[0]), float(result.x[1]), float(result.x[2])


def _fit_variants(samples: list[dict[str, Any]], *, balanced: bool, shared: bool):
    import numpy as np

    variants: dict[str, Any] = {"raw": None, "context": None}
    groups: list[tuple[int | None, list[dict[str, Any]]]]
    if shared:
        groups = [(None, samples)]
    else:
        groups = [(budget, [sample for sample in samples if sample["budget"] == budget]) for budget in BUDGETS]

    fits: dict[int | None, dict[str, tuple[float, ...]]] = {}
    for budget, group in groups:
        if not group:
            continue
        logits, labels, rates, task_ids, _ = _flatten(group)
        weights = _task_weights(task_ids, balanced)
        if len(labels) < MIN_SAMPLES or len(np.unique(labels)) < 2:
            fits[budget] = {}
            continue
        fits[budget] = {
            "platt_bce": _fit_platt(logits, labels, weights, "bce"),
            "anchor_bce": _fit_anchor(logits, labels, rates, weights, "bce"),
            "full_bce": _fit_full(logits, labels, rates, weights, "bce"),
            "platt_brier": _fit_platt(logits, labels, weights, "brier"),
            "anchor_brier": _fit_anchor(logits, labels, rates, weights, "brier"),
            "full_brier": _fit_full(logits, labels, rates, weights, "brier"),
        }
    variants["fits"] = fits
    return variants


def _variant_probability(record: dict[str, Any], fits: dict[str, Any], variant: str):
    import numpy as np

    logits = np.asarray(record["logits"], dtype=float)
    rate = _context_rate(record["context_labels"])
    budget = int(record["budget"])
    if variant == "raw":
        return _sigmoid(logits)
    if variant == "context":
        return np.full(len(logits), rate)
    if variant in ("full_bce_zero_half", "full_brier_zero_half"):
        if budget == 0:
            return np.full(len(logits), 0.5)
        base = variant.replace("_zero_half", "")
        fit = fits["per_budget_row"]
        params = fit["fits"][budget][base]
    elif variant.endswith("_task"):
        base = variant[: -len("_task")]
        fit = fits["per_budget_task"]
        params = fit["fits"][budget][base]
    elif variant.endswith("_shared"):
        base = variant[: -len("_shared")]
        fit = fits["shared_row"]
        params = fit["fits"][None][base]
    else:
        base = variant
        fit = fits["per_budget_row"]
        params = fit["fits"][budget][base]
    if base.startswith("platt"):
        return _sigmoid(params[0] * logits + params[1])
    if base.startswith("anchor"):
        return (1.0 - params[0]) * _sigmoid(logits) + params[0] * rate
    return (1.0 - params[2]) * _sigmoid(params[0] * logits + params[1]) + params[2] * rate

def _fit_and_evaluate(seed: int, mode: str, models: tuple[str, ...], limit_tasks: int, orders: int):
    import numpy as np
    import pandas as pd
    import torch

    from aims_starter.metrics import calibration_ece

    item_embeddings = np.load(CORPUS_ROOT / "item_embeddings.npy")
    subject_embeddings = np.load(CORPUS_ROOT / "subject_embeddings.npy")
    x_dim = int(item_embeddings.shape[1] + subject_embeddings.shape[1])
    evaluation_items = pd.read_parquet(CORPUS_ROOT / "eval_items.parquet")
    order_frame = pd.read_parquet(CORPUS_ROOT / "eval_orders.parquet")
    from aims_starter.uniform_eval import validation_fold_map

    fold_map = validation_fold_map(evaluation_items, n_folds=UNIFORM_FOLDS)
    cnp, online = _load_models(seed, mode, x_dim)

    def features(rows):
        return np.concatenate(
            [
                item_embeddings[rows["item_index"].to_numpy()],
                subject_embeddings[rows["subject_index"].to_numpy()],
            ],
            axis=1,
        ).astype("float32")

    records: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        split_rows = evaluation_items[evaluation_items["split"] == split]
        task_ids = sorted(split_rows["task_id"].unique())
        if limit_tasks:
            task_ids = task_ids[:limit_tasks]
        for task_id in task_ids:
            task_rows = split_rows[split_rows["task_id"] == task_id]
            benchmark = str(task_rows["benchmark_id"].iloc[0])
            target_rows = task_rows[task_rows["pool"] == "evaluation"]
            x_T = torch.as_tensor(
                features(target_rows),
                dtype=torch.float32,
                device=next(cnp.parameters()).device,
            )
            y_T = target_rows["label"].to_numpy(dtype="float32")
            lookup = task_rows.set_index("item_index")
            for order_seed in range(orders):
                order_rows = (
                    order_frame[
                        (order_frame["task_id"] == task_id)
                        & (order_frame["order_seed"] == order_seed)
                    ]
                    .sort_values("acquisition_rank")
                    .reset_index(drop=True)
                )
                context_rows = lookup.loc[order_rows["item_index"].to_numpy()].reset_index()
                context = {
                    "x_C_pool": features(context_rows),
                    "y_C_pool": context_rows["label"].to_numpy(dtype="float32"),
                }
                for model_name, function in (
                    ("cnp_airt", _cnp_logits),
                    ("online_airt_2pl", _online_logits),
                ):
                    if model_name not in models:
                        continue
                    for budget in BUDGETS:
                        logits = function(
                            cnp if model_name == "cnp_airt" else online,
                            x_T,
                            context,
                            int(budget),
                        )
                        records.append(
                            {
                                "seed": seed,
                                "mode": mode,
                                "split": split,
                                "benchmark_id": benchmark,
                                "task_id": str(task_id),
                                "order_seed": int(order_seed),
                                "fold": int(fold_map.get(str(task_id), -1)),
                                "model": model_name,
                                "budget": int(budget),
                                "logits": np.asarray(logits, dtype=float),
                                "labels": np.asarray(y_T, dtype=float),
                                "context_labels": np.asarray(
                                    context["y_C_pool"][:budget], dtype=float
                                ),
                            }
                        )
            print(f"{split} {task_id} collected", flush=True)

    variants = {
        "raw",
        "platt_bce",
        "anchor_bce",
        "full_bce",
        "context",
        "platt_brier",
        "anchor_brier",
        "full_brier",
        "full_bce_task",
        "full_brier_task",
        "full_bce_shared",
        "full_brier_shared",
        "full_bce_zero_half",
        "full_brier_zero_half",
    }
    fits_by_model: dict[str, Any] = {}
    fit_rows: dict[str, Any] = {}
    for model_name in models:
        validation = [
            record
            for record in records
            if record["model"] == model_name and record["split"] == "validation"
        ]
        folds = sorted({int(record["fold"]) for record in validation})
        fits_by_model[model_name] = {
            "full": {
                "per_budget_row": _fit_variants(validation, balanced=False, shared=False),
                "per_budget_task": _fit_variants(validation, balanced=True, shared=False),
                "shared_row": _fit_variants(validation, balanced=False, shared=True),
            },
            "folds": {
                str(fold): {
                    "per_budget_row": _fit_variants(
                        [record for record in validation if int(record["fold"]) != fold],
                        balanced=False,
                        shared=False,
                    ),
                    "per_budget_task": _fit_variants(
                        [record for record in validation if int(record["fold"]) != fold],
                        balanced=True,
                        shared=False,
                    ),
                    "shared_row": _fit_variants(
                        [record for record in validation if int(record["fold"]) != fold],
                        balanced=False,
                        shared=True,
                    ),
                }
                for fold in folds
            },
        }
        fit_rows[model_name] = {
            "full": {
                "validation_rows": int(sum(len(record["labels"]) for record in validation)),
                "validation_tasks": len({record["task_id"] for record in validation}),
            },
            "folds": {
                str(fold): {
                    "fit_rows": int(
                        sum(
                            len(record["labels"])
                            for record in validation
                            if int(record["fold"]) != fold
                        )
                    ),
                    "fit_tasks": len(
                        {
                            record["task_id"]
                            for record in validation
                            if int(record["fold"]) != fold
                        }
                    ),
                    "held_out_tasks": len(
                        {
                            record["task_id"]
                            for record in validation
                            if int(record["fold"]) == fold
                        }
                    ),
                }
                for fold in folds
            },
        }

    evaluation_rows: list[dict[str, Any]] = []
    for record in records:
        if record["split"] == "validation":
            fits = fits_by_model[record["model"]]["folds"][str(record["fold"])]
        else:
            fits = fits_by_model[record["model"]]["full"]
        for variant in variants:
            probability = _variant_probability(record, fits, variant)
            labels = record["labels"]
            evaluation_rows.append(
                {
                    "model": record["model"],
                    "mode": mode,
                    "split": record["split"],
                    "fold": int(record["fold"]),
                    "task_id": record["task_id"],
                    "order_seed": record["order_seed"],
                    "budget": record["budget"],
                    "variant": variant,
                    "brier": float(np.mean((probability - labels) ** 2)),
                    "ece": float(calibration_ece(labels.tolist(), probability.tolist())),
                }
            )

    frame = pd.DataFrame(evaluation_rows)
    weights = dict(zip(BUDGETS, WEIGHTS, strict=True))
    frame["weighted_brier"] = frame["budget"].map(weights) * frame["brier"]
    frame["weighted_ece"] = frame["budget"].map(weights) * frame["ece"]
    UNIFORM_ROOT.mkdir(parents=True, exist_ok=True)
    stem = "-".join(models)
    frame.to_parquet(
        UNIFORM_ROOT / f"calibration-ablation-{seed}-{mode}-{stem}-rows.parquet",
        index=False,
    )
    by_budget = []
    for (model, split, variant, budget), group in frame.groupby(
        ["model", "split", "variant", "budget"], sort=True
    ):
        task_means = group.groupby("task_id", sort=False)[["brier", "ece"]].mean()
        by_budget.append(
            {
                "model": model,
                "split": split,
                "variant": variant,
                "budget": int(budget),
                "brier": float(task_means["brier"].mean()),
                "brier_se": float(task_means["brier"].std(ddof=1) / np.sqrt(len(task_means))),
                "ece": float(task_means["ece"].mean()),
                "ece_se": float(task_means["ece"].std(ddof=1) / np.sqrt(len(task_means))),
                "tasks": len(task_means),
            }
        )
    alc = []
    for (model, split, variant), group in frame.groupby(["model", "split", "variant"], sort=True):
        per_task = (
            group.groupby(["task_id", "order_seed"], sort=False)[["weighted_brier", "weighted_ece"]]
            .sum()
            .groupby("task_id", sort=False)
            .mean()
        )
        alc.append(
            {
                "model": model,
                "split": split,
                "variant": variant,
                "brier_alc": float(per_task["weighted_brier"].mean()),
                "brier_alc_se": float(
                    per_task["weighted_brier"].std(ddof=1) / np.sqrt(len(per_task))
                ),
                "ece_alc": float(per_task["weighted_ece"].mean()),
                "ece_alc_se": float(
                    per_task["weighted_ece"].std(ddof=1) / np.sqrt(len(per_task))
                ),
                "tasks": len(per_task),
            }
        )

    def _serialize_fit_block(block: dict[str, Any]) -> dict[str, Any]:
        return {
            key: {
                ("shared" if budget is None else str(budget)): value
                for budget, value in fit["fits"].items()
            }
            for key, fit in block.items()
        }

    artifact = {
        "status": "calibration_ablation_uniform",
        "seed": seed,
        "mode": mode,
        "models": list(models),
        "orders": orders,
        "limit_tasks": limit_tasks,
        "split_policy": "validation folds are out-of-fold; test uses the full validation fit",
        "folds": int(UNIFORM_FOLDS),
        "fit_rows": fit_rows,
        "by_budget": by_budget,
        "alc": alc,
        "fits": {
            model: {
                "full": _serialize_fit_block(model_fits["full"]),
                "folds": {
                    fold: _serialize_fit_block(block)
                    for fold, block in model_fits["folds"].items()
                },
            }
            for model, model_fits in fits_by_model.items()
        },
    }
    return artifact


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=4 * 60 * 60,
)
def run(
    seed: int = 20261018,
    mode: str = "masked",
    models: str = "cnp_airt",
    limit_tasks: int = 0,
    orders: int = 10,
) -> dict[str, Any]:
    names = tuple(name.strip() for name in models.split(",") if name.strip())
    if not names:
        raise ValueError("at least one model is required")
    artifact = _fit_and_evaluate(seed, mode, names, limit_tasks, orders)
    UNIFORM_ROOT.mkdir(parents=True, exist_ok=True)
    stem = "-".join(names)
    output = UNIFORM_ROOT / f"calibration-ablation-{seed}-{mode}-{stem}.json"
    _write_json(output, artifact)
    volume.commit()
    return artifact


@app.local_entrypoint()
def main(
    seed: int = 0,
    mode: str = "",
    models: str = "cnp_airt,online_airt_2pl",
    limit_tasks: int = 0,
    orders: int = 10,
) -> None:
    if seed and mode:
        print(
            json.dumps(
                run.remote(seed, mode, models, limit_tasks, orders),
                indent=2,
                sort_keys=True,
            )
        )
        return
    handles = [
        run.spawn(seed_value, mode_value, models, limit_tasks, orders)
        for seed_value in SEEDS
        for mode_value in MODES
    ]
    for handle in handles:
        handle.get()
