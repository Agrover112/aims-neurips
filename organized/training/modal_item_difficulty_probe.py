"""CPU-only benchmark-centered item-difficulty transfer probe.

The probe first fits a separate Rasch model inside every benchmark, then asks
whether frozen item-text embeddings predict the resulting centered item
difficulties in two settings:

1. within-benchmark item-fold cross-validation;
2. transfer from training benchmarks to entirely held-out benchmarks.

The target construction may use all responses in a benchmark because this is
a representation diagnostic, not a deployable predictor.  Regression splits
remain leakage-safe: held-out items or held-out benchmarks never fit the text
mapping used to predict them.

Run:
    modal run modal_item_difficulty_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal
import numpy as np

APP_NAME = "aims-item-difficulty-probe"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = "allbench-train-me5_instruct-cap128-v2"
RESULT_ROOT = REMOTE_ROOT / "results" / "item-difficulty-probe"
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2,<3",
        "pandas>=2.2,<3",
        "pyarrow>=17,<24",
        "scikit-learn>=1.5,<2",
    )
    .add_local_dir(PROJECT_ROOT / "src", "/opt/aims/src", copy=True)
    .env({"PYTHONPATH": "/opt/aims/src"})
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _add_benchmark_id(frame):
    """Recover the compact training schema's benchmark suffix when needed."""

    if "benchmark_id" in frame.columns:
        return frame
    output = frame.copy()
    output["benchmark_id"] = output["task_id"].astype(str).str.rsplit("::", n=1).str[-1]
    return output


def _item_targets(frame, split: str):
    """Fit one centered Rasch model per benchmark and return item targets."""

    import pandas as pd

    from aims_starter.item_difficulty_probe import fit_centered_rasch

    output = []
    for benchmark, block in frame.groupby("benchmark_id", sort=True):
        block = block.copy()
        item_codes, item_ids = pd.factorize(block["item_index"], sort=True)
        task_codes, _ = pd.factorize(block["task_id"].astype(str), sort=True)
        difficulty, _ = fit_centered_rasch(
            block["label"].to_numpy(dtype=float),
            item_codes,
            task_codes,
            n_items=len(item_ids),
            n_tasks=int(task_codes.max()) + 1,
        )
        count = np.bincount(item_codes, minlength=len(item_ids))
        successes = np.bincount(
            item_codes,
            weights=block["label"].to_numpy(dtype=float),
            minlength=len(item_ids),
        )
        output.append(
            pd.DataFrame(
                {
                    "split": split,
                    "benchmark_id": str(benchmark),
                    "item_index": np.asarray(item_ids, dtype=np.int64),
                    "difficulty": difficulty,
                    "response_count": count.astype(np.int64),
                    "success_rate": successes / np.maximum(count, 1),
                }
            )
        )
        print(
            f"targets split={split} benchmark={benchmark} "
            f"items={len(item_ids):,} tasks={int(task_codes.max()) + 1:,}",
            flush=True,
        )
    return pd.concat(output, ignore_index=True)


def _sample_per_benchmark(frame, maximum: int, seed: int):
    if maximum <= 0:
        return frame.copy()
    rng = np.random.default_rng(seed)
    pieces = []
    for _, group in frame.groupby("benchmark_id", sort=True):
        if len(group) <= maximum:
            pieces.append(group)
        else:
            pieces.append(group.iloc[np.sort(rng.choice(len(group), maximum, replace=False))])
    return frame.iloc[:0].copy() if not pieces else __import__("pandas").concat(pieces)


def _xy(frame, embeddings):
    x = np.asarray(embeddings[frame["item_index"].to_numpy(dtype=np.int64)], dtype=np.float32)
    y = frame["difficulty"].to_numpy(dtype=float)
    weight = np.sqrt(frame["response_count"].to_numpy(dtype=float))
    return x, y, weight


def _select_alpha(train, embeddings, maximum_per_benchmark: int) -> tuple[float, list[dict]]:
    from sklearn.linear_model import Ridge

    from aims_starter.item_difficulty_probe import center_by_group, regression_metrics

    tune = _sample_per_benchmark(train, min(250, maximum_per_benchmark), seed=20261025)
    groups = np.asarray(sorted(tune["benchmark_id"].unique()))
    fold_of = {group: index % min(5, len(groups)) for index, group in enumerate(groups)}
    folds = tune["benchmark_id"].map(fold_of).to_numpy(dtype=int)
    rows = []
    for alpha in ALPHAS:
        truth_parts = []
        prediction_parts = []
        weight_parts = []
        group_parts = []
        for fold in sorted(np.unique(folds)):
            fit = tune.iloc[np.flatnonzero(folds != fold)]
            held = tune.iloc[np.flatnonzero(folds == fold)]
            x_fit, y_fit, w_fit = _xy(fit, embeddings)
            x_held, y_held, w_held = _xy(held, embeddings)
            model = Ridge(alpha=float(alpha), solver="lsqr")
            model.fit(x_fit, y_fit, sample_weight=w_fit)
            truth_parts.append(y_held)
            prediction_parts.append(model.predict(x_held))
            weight_parts.append(w_held)
            group_parts.append(held["benchmark_id"].to_numpy())
        truth = np.concatenate(truth_parts)
        prediction = np.concatenate(prediction_parts)
        weights = np.concatenate(weight_parts)
        held_groups = np.concatenate(group_parts)
        prediction = center_by_group(prediction, held_groups, weights)
        metrics = regression_metrics(truth, prediction, weights)
        rows.append({"alpha": float(alpha), **metrics})
        print(f"alpha={alpha:g} grouped-r2={metrics['r2']:.4f}", flush=True)
    best = min(rows, key=lambda row: row["rmse"])
    return float(best["alpha"]), rows


def _summarize_predictions(frame, prediction: np.ndarray, protocol: str):
    from aims_starter.item_difficulty_probe import center_by_group, regression_metrics

    result = frame.copy()
    result["protocol"] = protocol
    weights = np.sqrt(result["response_count"].to_numpy(dtype=float))
    result["prediction"] = center_by_group(
        prediction, result["benchmark_id"].to_numpy(), weights
    )
    pooled = regression_metrics(
        result["difficulty"].to_numpy(dtype=float),
        result["prediction"].to_numpy(dtype=float),
        weights,
    )
    by_benchmark = []
    for benchmark, group in result.groupby("benchmark_id", sort=True):
        metrics = regression_metrics(
            group["difficulty"].to_numpy(dtype=float),
            group["prediction"].to_numpy(dtype=float),
            np.sqrt(group["response_count"].to_numpy(dtype=float)),
        )
        by_benchmark.append(
            {
                "protocol": protocol,
                "split": str(group["split"].iloc[0]),
                "benchmark_id": str(benchmark),
                "items": len(group),
                **metrics,
            }
        )
    summary = {
        "protocol": protocol,
        "split": str(result["split"].iloc[0]),
        "items": len(result),
        "benchmarks": result["benchmark_id"].nunique(),
        **pooled,
        "macro_r2": float(np.mean([row["r2"] for row in by_benchmark])),
        "macro_pearson": float(np.mean([row["pearson"] for row in by_benchmark])),
    }
    for minimum in (2, 5):
        subset = result[result["response_count"] >= minimum].copy()
        if subset.empty:
            continue
        subset_weights = np.sqrt(subset["response_count"].to_numpy(dtype=float))
        subset_truth = center_by_group(
            subset["difficulty"].to_numpy(dtype=float),
            subset["benchmark_id"].to_numpy(),
            subset_weights,
        )
        subset_prediction = center_by_group(
            subset["prediction"].to_numpy(dtype=float),
            subset["benchmark_id"].to_numpy(),
            subset_weights,
        )
        subset_metrics = regression_metrics(
            subset_truth, subset_prediction, subset_weights
        )
        summary[f"items_n{minimum}"] = len(subset)
        summary[f"r2_n{minimum}"] = subset_metrics["r2"]
        summary[f"pearson_n{minimum}"] = subset_metrics["pearson"]
    return result, summary, by_benchmark


def _heldout_transfer(train, held, embeddings, alpha: float, maximum: int):
    from sklearn.linear_model import Ridge

    fit = _sample_per_benchmark(train, maximum, seed=20261026)
    x_fit, y_fit, w_fit = _xy(fit, embeddings)
    x_held, _, _ = _xy(held, embeddings)
    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(x_fit, y_fit, sample_weight=w_fit)
    return _summarize_predictions(held, model.predict(x_held), "heldout_benchmark")


def _within_benchmark(frame, embeddings, maximum: int):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    sampled = _sample_per_benchmark(frame, maximum, seed=20261027)
    pieces = []
    predictions = []
    for benchmark, group in sampled.groupby("benchmark_id", sort=True):
        group = group.reset_index(drop=True)
        n_splits = min(5, len(group))
        if n_splits < 2:
            continue
        x, y, weight = _xy(group, embeddings)
        predicted = np.empty(len(group), dtype=float)
        chosen_alphas = []
        for fit_index, held_index in KFold(
            n_splits=n_splits, shuffle=True, random_state=20261025
        ).split(x):
            inner_splits = min(3, len(fit_index))
            alpha_loss = []
            for alpha in ALPHAS:
                squared_error = 0.0
                weight_sum = 0.0
                for inner_fit, inner_held in KFold(
                    n_splits=inner_splits, shuffle=True, random_state=20261026
                ).split(fit_index):
                    inner_model = Ridge(alpha=float(alpha), solver="lsqr")
                    inner_model.fit(
                        x[fit_index[inner_fit]],
                        y[fit_index[inner_fit]],
                        sample_weight=weight[fit_index[inner_fit]],
                    )
                    inner_prediction = inner_model.predict(x[fit_index[inner_held]])
                    inner_weight = weight[fit_index[inner_held]]
                    squared_error += float(
                        np.sum(inner_weight * (y[fit_index[inner_held]] - inner_prediction) ** 2)
                    )
                    weight_sum += float(inner_weight.sum())
                alpha_loss.append((squared_error / weight_sum, float(alpha)))
            selected_alpha = min(alpha_loss)[1]
            chosen_alphas.append(selected_alpha)
            model = Ridge(alpha=selected_alpha, solver="lsqr")
            model.fit(x[fit_index], y[fit_index], sample_weight=weight[fit_index])
            predicted[held_index] = model.predict(x[held_index])
        pieces.append(group)
        predictions.append(predicted)
        print(
            f"within benchmark={benchmark} items={len(group):,} "
            f"median_alpha={float(np.median(chosen_alphas)):g}",
            flush=True,
        )
    combined = __import__("pandas").concat(pieces, ignore_index=True)
    return _summarize_predictions(
        combined, np.concatenate(predictions), "within_benchmark_item_cv"
    )


@app.function(
    image=image,
    cpu=16,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=8 * 60 * 60,
)
def run_probe(
    corpus: str = DEFAULT_CORPUS,
    maximum_train_items_per_benchmark: int = 2000,
    maximum_within_items_per_benchmark: int = 5000,
) -> dict[str, Any]:
    import pandas as pd

    root = REMOTE_ROOT / "datasets" / "full" / corpus
    output = RESULT_ROOT / f"{corpus}.json"
    train_rows = _add_benchmark_id(pd.read_parquet(root / "train_tasks.parquet"))
    evaluation = _add_benchmark_id(pd.read_parquet(root / "eval_items.parquet"))
    embeddings = np.load(root / "item_embeddings.npy", mmap_mode="r")

    train_targets = _item_targets(train_rows, "train")
    held_targets = {
        split: _item_targets(evaluation[evaluation["split"] == split], split)
        for split in ("validation", "test")
        if bool((evaluation["split"] == split).any())
    }
    alpha, tuning = _select_alpha(
        train_targets, embeddings, maximum_train_items_per_benchmark
    )
    print(f"selected alpha={alpha:g}", flush=True)

    summaries = []
    by_benchmark = []
    prediction_frames = []
    for split, targets in held_targets.items():
        transferred, summary, detail = _heldout_transfer(
            train_targets,
            targets,
            embeddings,
            alpha,
            maximum_train_items_per_benchmark,
        )
        within, within_summary, within_detail = _within_benchmark(
            targets,
            embeddings,
            maximum_within_items_per_benchmark,
        )
        summaries.extend([summary, within_summary])
        by_benchmark.extend(detail + within_detail)
        prediction_frames.extend([transferred, within])
        print(
            f"{split}: transfer R2={summary['r2']:.4f}; "
            f"within R2={within_summary['r2']:.4f}",
            flush=True,
        )

    artifact = {
        "status": "complete",
        "corpus": corpus,
        "target": "benchmark-centered regularized Rasch item difficulty",
        "regressor": "linear ridge on frozen item embedding",
        "selected_alpha": alpha,
        "alpha_tuning": tuning,
        "maximum_train_items_per_benchmark": maximum_train_items_per_benchmark,
        "maximum_within_items_per_benchmark": maximum_within_items_per_benchmark,
        "summaries": summaries,
        "by_benchmark": by_benchmark,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(output, artifact)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        output.with_suffix(".predictions.parquet"), index=False
    )
    volume.commit()
    return artifact


@app.local_entrypoint()
def main(
    corpus: str = DEFAULT_CORPUS,
    maximum_train_items_per_benchmark: int = 2000,
    maximum_within_items_per_benchmark: int = 5000,
) -> None:
    result = run_probe.remote(
        corpus,
        maximum_train_items_per_benchmark,
        maximum_within_items_per_benchmark,
    )
    print(json.dumps(result["summaries"], indent=2, sort_keys=True))
