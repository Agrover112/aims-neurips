"""Benchmark-held-out evaluation on the full PAIEC protocol.

The protocol freezes 6828 eligible subject-benchmark tasks across 54 benchmarks:
43 train, 5 validation, 6 test. This app builds a Codabench-compatible feature
store (item text and subject metadata only) and runs the deterministic track
(CNP-AIRT and Online-AIRT) with the subject block masked or visible.

Protocol rules implemented here:

1. Meta-training uses train-benchmark tasks only. Validation and test benchmarks
   never enter training.
2. Each task's context is revealed in a frozen acquisition order; the evaluation
   pool is disjoint. Evaluation labels are used for scoring only.
3. Each task is capped to the platform's draw size: up to 500 evaluation items,
   so a source group stays inside 1000 items.
4. Metrics are averaged within a task first, then equally across tasks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "aims-neurips-allbench"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
PROTOCOL_ROOT = REMOTE_ROOT / "datasets" / "full" / "evaluation-protocol-v2"
PROCESSED_ROOT = REMOTE_ROOT / "datasets" / "full" / "processed"
EPISODES_ROOT = REMOTE_ROOT / "datasets" / "full" / "episodes"
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "allbench-features-v1"
RESULT_ROOT = REMOTE_ROOT / "results" / "allbench-deterministic-2seed"
MODES = ("masked", "visible")
SEEDS = (20261018, 20261019)
SUBJECT_SLICE = (384, 768)
TRAIN_ITEMS_PER_TASK = 128
EVAL_ITEMS_PER_TASK = 500
ORDER_SEEDS = tuple(range(10))
BUDGETS = (0, 1, 3, 7, 15, 31)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2,<3",
        "pandas>=2.2,<3",
        "pyarrow>=17,<24",
        "torch==2.5.1",
        "sentence-transformers==3.4.1",
    )
    .add_local_dir(PROJECT_ROOT / "src", "/opt/aims/src", copy=True)
    .env(
        {
            "PYTHONPATH": "/opt/aims/src",
            "HF_HOME": "/data/huggingface-cache",
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


def _rank(value: str) -> int:
    return int(hashlib.blake2b(value.encode(), digest_size=8).hexdigest(), 16)


def _encode(encoder, texts: list[str]):
    return encoder.encode(
        texts,
        batch_size=512,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")


SUBJECT_COLUMNS = (
    "normalized_name",
    "provider",
    "release_date",
    "access_date",
    "harness",
    "reasoning_effort",
    "harness_version",
    "subject_features_extra",
)


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=2 * 60 * 60,
)
def prepare_features() -> dict[str, Any]:
    """Build the Codabench-compatible feature store for the full protocol."""

    import numpy as np
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    existing = _read_json(CORPUS_ROOT / "manifest.json")
    if existing is not None and "feature_slices" in existing:
        return {"status": "reused", **existing}

    protocol = pd.read_parquet(PROTOCOL_ROOT / "eligible_tasks.parquet")
    protocol["task_id"] = protocol["task_id"].astype(str)
    split_of = dict(zip(protocol["task_id"], protocol["split_by_benchmark"], strict=True))
    benchmark_of = dict(zip(protocol["task_id"], protocol["benchmark_id"], strict=True))
    print(f"protocol: {len(protocol)} tasks", flush=True)

    episode_items = pd.read_parquet(
        EPISODES_ROOT / "episode_items.parquet",
        columns=[
            "task_id",
            "item_uid",
            "subject_uid",
            "pool",
            "label_observation_id",
            "acquisition_rank",
        ],
    )
    episode_items["task_id"] = episode_items["task_id"].astype(str)
    episode_items["item_uid"] = episode_items["item_uid"].astype(str)
    episode_items["subject_uid"] = episode_items["subject_uid"].astype(str)
    episode_items["split"] = episode_items["task_id"].map(split_of)
    if episode_items["split"].isna().any():
        raise RuntimeError("episode items reference tasks outside the protocol")
    print(f"episode items: {len(episode_items)} rows", flush=True)
    episode_items["rank"] = [
        _rank(f"{task}::{item}")
        for task, item in zip(episode_items["task_id"], episode_items["item_uid"], strict=True)
    ]

    train = episode_items[episode_items["split"] == "train"]
    train = (
        train.sort_values(["task_id", "rank", "item_uid"])
        .groupby("task_id", sort=True)
        .head(TRAIN_ITEMS_PER_TASK)
    )
    evaluation_side = episode_items[episode_items["split"].isin(["validation", "test"])]
    evaluation = (
        evaluation_side[evaluation_side["pool"] == "evaluation"]
        .sort_values(["task_id", "rank", "item_uid"])
        .groupby("task_id", sort=True)
        .head(EVAL_ITEMS_PER_TASK)
    )
    acquisition = evaluation_side[evaluation_side["pool"] == "acquisition"]
    selected = pd.concat([train, evaluation, acquisition], ignore_index=True)
    print(
        f"selected rows: train {len(train)}, evaluation {len(evaluation)}, "
        f"acquisition {len(acquisition)}",
        flush=True,
    )

    items = pd.read_parquet(
        PROCESSED_ROOT / "items.parquet", columns=["item_uid", "content", "item_features"]
    )
    items["item_uid"] = items["item_uid"].astype(str)
    items = items.drop_duplicates("item_uid").sort_values("item_uid").reset_index(drop=True)
    item_to_index = dict(zip(items["item_uid"], range(len(items)), strict=True))
    subjects = pd.read_parquet(PROCESSED_ROOT / "subjects.parquet")
    subjects["subject_uid"] = subjects["subject_uid"].astype(str)
    subjects = (
        subjects.drop_duplicates("subject_uid").sort_values("subject_uid").reset_index(drop=True)
    )
    subject_to_index = dict(zip(subjects["subject_uid"], range(len(subjects)), strict=True))
    missing_items = sorted(set(selected["item_uid"]) - set(item_to_index))
    missing_subjects = sorted(set(selected["subject_uid"]) - set(subject_to_index))
    if missing_items or missing_subjects:
        raise RuntimeError(
            f"feature join incomplete: {len(missing_items)} items, {len(missing_subjects)} subjects"
        )

    pairs = selected[["task_id", "item_uid"]].drop_duplicates()
    observations = pd.read_parquet(
        PROCESSED_ROOT / "observations.parquet",
        columns=["task_id", "item_uid", "observation_id", "label"],
    )
    observations["task_id"] = observations["task_id"].astype(str)
    observations["item_uid"] = observations["item_uid"].astype(str)
    observations = observations.merge(pairs, on=["task_id", "item_uid"], how="inner")
    observations = observations.sort_values("observation_id").drop_duplicates(
        ["task_id", "item_uid"], keep="first"
    )
    selected = selected.merge(
        observations[["task_id", "item_uid", "label"]], on=["task_id", "item_uid"], how="left"
    )
    if selected["label"].isna().any():
        raise RuntimeError("label join incomplete")
    selected["item_index"] = selected["item_uid"].map(item_to_index).astype("int32")
    selected["subject_index"] = selected["subject_uid"].map(subject_to_index).astype("int32")
    selected["benchmark_id"] = selected["task_id"].map(benchmark_of)

    train_output = selected[selected["split"] == "train"][
        ["task_id", "item_index", "subject_index", "label"]
    ].reset_index(drop=True)
    evaluation_output = selected[selected["split"].isin(["validation", "test"])][
        [
            "task_id",
            "split",
            "benchmark_id",
            "item_index",
            "subject_index",
            "label",
            "pool",
            "acquisition_rank",
        ]
    ].reset_index(drop=True)

    orders = pd.read_parquet(
        PROTOCOL_ROOT / "acquisition_orders.parquet",
        columns=["task_id", "item_uid", "order_seed", "acquisition_rank"],
        filters=[("acquisition_rank", "<=", 31)],
    )
    orders["task_id"] = orders["task_id"].astype(str)
    orders["item_uid"] = orders["item_uid"].astype(str)
    orders = orders[
        orders["task_id"].isin(set(evaluation_output["task_id"]))
        & orders["order_seed"].isin(ORDER_SEEDS)
    ].copy()
    orders["item_index"] = orders["item_uid"].map(item_to_index).astype("int32")
    orders = orders[["task_id", "order_seed", "acquisition_rank", "item_index"]].reset_index(
        drop=True
    )
    print(f"evaluation orders: {len(orders)} rows", flush=True)

    item_texts = [
        f"item: {content}\nmetadata: {features}"
        for content, features in zip(
            items["content"].fillna(""), items["item_features"].fillna(""), strict=True
        )
    ]
    subject_texts = [
        "\n".join(f"{column}: {getattr(row, column)}" for column in SUBJECT_COLUMNS)
        for row in subjects.fillna("").itertuples(index=False)
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    print("encoding items", flush=True)
    item_embeddings = _encode(encoder, item_texts)
    print("encoding subjects", flush=True)
    subject_embeddings = _encode(encoder, subject_texts)

    CORPUS_ROOT.mkdir(parents=True, exist_ok=False)
    items[["item_uid"]].to_parquet(CORPUS_ROOT / "items.parquet", index=False)
    subjects[["subject_uid"]].to_parquet(CORPUS_ROOT / "subjects.parquet", index=False)
    train_output.to_parquet(CORPUS_ROOT / "train_tasks.parquet", index=False)
    evaluation_output.to_parquet(CORPUS_ROOT / "eval_items.parquet", index=False)
    orders.to_parquet(CORPUS_ROOT / "eval_orders.parquet", index=False)
    np.save(CORPUS_ROOT / "item_embeddings.npy", item_embeddings)
    np.save(CORPUS_ROOT / "subject_embeddings.npy", subject_embeddings)
    manifest = {
        "format_version": 1,
        "input_contract": "codabench_item_and_subject_fields_only",
        "protocol": str(PROTOCOL_ROOT),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "feature_slices": {
            "item": [0, int(item_embeddings.shape[1])],
            "subject": [
                int(item_embeddings.shape[1]),
                int(item_embeddings.shape[1] + subject_embeddings.shape[1]),
            ],
        },
        "combined_x_dimension": int(item_embeddings.shape[1] + subject_embeddings.shape[1]),
        "unique_items": len(items),
        "unique_subjects": len(subjects),
        "train_tasks": int(train_output["task_id"].nunique()),
        "train_rows": len(train_output),
        "evaluation_tasks": int(evaluation_output["task_id"].nunique()),
        "evaluation_rows": len(evaluation_output),
        "order_rows": len(orders),
        "train_items_per_task": TRAIN_ITEMS_PER_TASK,
        "evaluation_items_per_task": EVAL_ITEMS_PER_TASK,
        "excluded_unavailable_fields": [
            "benchmark_description",
            "benchmark_name",
            "grading_criterion",
            "asset_manifest",
            "item_numeric",
        ],
    }
    _write_json(CORPUS_ROOT / "manifest.json", manifest)
    volume.commit()
    return manifest


def _task_batch(train, item_embeddings, subject_embeddings):
    import numpy as np

    tasks = []
    for task_id, rows in train.groupby("task_id", sort=True):
        tasks.append(
            {
                "task_id": str(task_id),
                "x": np.concatenate(
                    [
                        item_embeddings[rows["item_index"].to_numpy()],
                        subject_embeddings[rows["subject_index"].to_numpy()],
                    ],
                    axis=1,
                ).astype("float32"),
                "y": rows["label"].to_numpy(dtype="float32"),
            }
        )
    return tasks


def _run_mode(seed: int, mode: str) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch

    from aims_starter.faithful_np import (
        score_amortized_2pl,
        score_cnp,
        train_amortized_2pl,
        train_cnp,
        write_metrics,
    )
    from aims_starter.metrics import DEFAULT_BUDGETS, DEFAULT_WEIGHTS, calibration_ece

    output = RESULT_ROOT / f"seed-{seed}-{mode}.json"
    existing = _read_json(output)
    if existing is not None:
        return existing
    manifest = _read_json(CORPUS_ROOT / "manifest.json")
    if manifest is None or "feature_slices" not in manifest:
        raise FileNotFoundError("all-benchmark feature store is missing")
    subject_slice = tuple(manifest["feature_slices"]["subject"]) if mode == "masked" else None
    item_embeddings = np.load(CORPUS_ROOT / "item_embeddings.npy")
    subject_embeddings = np.load(CORPUS_ROOT / "subject_embeddings.npy")
    train = pd.read_parquet(CORPUS_ROOT / "train_tasks.parquet")
    evaluation_items = pd.read_parquet(CORPUS_ROOT / "eval_items.parquet")
    orders = pd.read_parquet(CORPUS_ROOT / "eval_orders.parquet")
    print(
        f"seed={seed} mode={mode}: {train['task_id'].nunique()} train tasks, "
        f"{evaluation_items['task_id'].nunique()} evaluation tasks",
        flush=True,
    )

    meta_tasks = _task_batch(train, item_embeddings, subject_embeddings)
    cnp, cnp_history = train_cnp(
        meta_tasks,
        seed=seed,
        steps=800,
        episodes_per_step=4,
        subject_feature_slice=subject_slice,
    )
    online, online_history = train_amortized_2pl(
        meta_tasks,
        seed=seed,
        steps=800,
        subject_feature_slice=subject_slice,
    )
    fold_root = RESULT_ROOT / f"seed-{seed}-{mode}-checkpoints"
    fold_root.mkdir(parents=True, exist_ok=True)
    torch.save(cnp.state_dict(), fold_root / "cnp_airt.pt")
    torch.save(online.state_dict(), fold_root / "online_airt_2pl.pt")
    write_metrics(fold_root / "cnp-history.json", {"history": cnp_history})
    write_metrics(fold_root / "online-history.json", {"history": online_history})

    def features(rows):
        return np.concatenate(
            [
                item_embeddings[rows["item_index"].to_numpy()],
                subject_embeddings[rows["subject_index"].to_numpy()],
            ],
            axis=1,
        ).astype("float32")

    rows = []
    for split in ("validation", "test"):
        split_rows = evaluation_items[evaluation_items["split"] == split]
        order_seeds = ORDER_SEEDS if split == "test" else (0,)
        for task_id, task_rows in split_rows.groupby("task_id", sort=True):
            benchmark = str(task_rows["benchmark_id"].iloc[0])
            target_rows = task_rows[task_rows["pool"] == "evaluation"]
            x_T = features(target_rows)
            y_T = target_rows["label"].to_numpy(dtype="float32")
            lookup = task_rows.set_index("item_index")
            for order_seed in order_seeds:
                order_rows = (
                    orders[(orders["task_id"] == task_id) & (orders["order_seed"] == order_seed)]
                    .sort_values("acquisition_rank")
                    .reset_index(drop=True)
                )
                context_rows = lookup.loc[order_rows["item_index"].to_numpy()].reset_index()
                task = {
                    "task_id": str(task_id),
                    "x_C_pool": features(context_rows),
                    "y_C_pool": context_rows["label"].to_numpy(dtype="float32"),
                    "x_T": x_T,
                    "y_T": y_T,
                }
                cnp_scores = score_cnp(cnp, task)
                online_scores = score_amortized_2pl(online, task)
                y_C = task["y_C_pool"]
                beta = {}
                for budget in DEFAULT_BUDGETS:
                    probability = (2.0 + float(y_C[:budget].sum())) / (4.0 + budget)
                    p_y_T = np.full(len(y_T), probability)
                    beta[str(budget)] = {
                        "brier": float(np.mean((p_y_T - y_T) ** 2)),
                        "ece": calibration_ece(y_T.tolist(), p_y_T.tolist()),
                    }
                beta["brier_alc"] = float(
                    sum(
                        weight * beta[str(budget)]["brier"]
                        for budget, weight in zip(DEFAULT_BUDGETS, DEFAULT_WEIGHTS, strict=True)
                    )
                )
                for model, scores in (
                    ("cnp_airt", cnp_scores),
                    ("online_airt_2pl", online_scores),
                    ("beta_binomial", beta),
                ):
                    for budget in DEFAULT_BUDGETS:
                        entry = {
                            "seed": seed,
                            "mode": mode,
                            "split": split,
                            "benchmark_id": benchmark,
                            "task_id": str(task_id),
                            "order_seed": int(order_seed),
                            "model": model,
                            "budget": int(budget),
                            "brier": float(scores[str(budget)]["brier"]),
                            "ece": float(scores[str(budget)]["ece"]),
                            "brier_alc": float(scores["brier_alc"]),
                        }
                        if "auc" in scores[str(budget)]:
                            entry["auc"] = float(scores[str(budget)]["auc"])
                        rows.append(entry)
        print(f"seed={seed} mode={mode} split={split} scored", flush=True)

    result = {
        "seed": seed,
        "mode": mode,
        "status": "allbench_deterministic",
        "rows": rows,
    }
    write_metrics(output, result)
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=3 * 60 * 60,
)
def run_masked(seed: int) -> dict[str, Any]:
    return _run_mode(seed, "masked")


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=3 * 60 * 60,
)
def run_visible(seed: int) -> dict[str, Any]:
    return _run_mode(seed, "visible")


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    volumes={str(REMOTE_ROOT): volume},
    timeout=30 * 60,
)
def aggregate_full() -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    results = [
        _read_json(RESULT_ROOT / f"seed-{seed}-{mode}.json") for seed in SEEDS for mode in MODES
    ]
    if any(result is None for result in results):
        raise FileNotFoundError("all-benchmark run incomplete")
    frame = pd.DataFrame([row for result in results for row in result["rows"]])
    frame = frame.astype({"budget": int, "order_seed": int, "seed": int})
    budgets = sorted(frame["budget"].unique().tolist())

    def summarise(subset: pd.DataFrame) -> list[dict[str, Any]]:
        task_means = subset.groupby(
            ["model", "mode", "split", "task_id", "budget"], as_index=False
        ).agg(brier=("brier", "mean"), ece=("ece", "mean"), auc=("auc", "mean"))
        output = []
        for (model, mode, split), group in task_means.groupby(
            ["model", "mode", "split"], sort=True
        ):
            for budget in budgets:
                rows = group[group["budget"] == budget]
                auc_value = rows["auc"].mean()
                output.append(
                    {
                        "model": model,
                        "mode": mode,
                        "split": split,
                        "budget": int(budget),
                        "brier": float(rows["brier"].mean()),
                        "brier_se": float(rows["brier"].std(ddof=1) / np.sqrt(len(rows))),
                        "ece": float(rows["ece"].mean()),
                        "auc": None if pd.isna(auc_value) else float(auc_value),
                        "tasks": len(rows),
                    }
                )
        return output

    alc = (
        frame.groupby(["model", "mode", "split", "seed", "task_id", "order_seed"])["brier_alc"]
        .mean()
        .reset_index()
    )
    alc_summary = (
        alc.groupby(["model", "mode", "split", "seed"], as_index=False)["brier_alc"]
        .mean()
        .groupby(["model", "mode", "split"], as_index=False)
        .agg(brier_alc=("brier_alc", "mean"), brier_alc_seed_std=("brier_alc", "std"))
    )
    per_benchmark = frame.groupby(
        ["mode", "model", "split", "benchmark_id", "budget"], as_index=False
    ).agg(brier=("brier", "mean"))
    artifact = {
        "status": "allbench_deterministic",
        "by_budget": summarise(frame),
        "brier_alc": alc_summary.to_dict(orient="records"),
        "by_benchmark": per_benchmark.to_dict(orient="records"),
        "modes": list(MODES),
        "seeds": list(SEEDS),
    }
    _write_json(RESULT_ROOT / "metrics-allbench.json", artifact)
    frame.to_parquet(RESULT_ROOT / "per_task_rows.parquet", index=False)
    volume.commit()
    return artifact


@app.local_entrypoint()
def main(prepare_only: bool = False, modes_only: bool = False) -> None:
    if prepare_only:
        print(json.dumps(prepare_features.remote(), indent=2, sort_keys=True))
        return
    if not modes_only:
        print(json.dumps(prepare_features.remote(), indent=2, sort_keys=True))
    handles = [run_masked.spawn(seed) for seed in SEEDS]
    handles += [run_visible.spawn(seed) for seed in SEEDS]
    for handle in handles:
        handle.get()
    print(json.dumps(aggregate_full.remote(), indent=2, sort_keys=True))
