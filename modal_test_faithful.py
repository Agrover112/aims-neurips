"""Test-faithful BeaverTails pilot: item text plus subject metadata only.

The Codabench input provides the item content, item features, interactors, and an
opaque benchmark alias, plus the subject metadata fields. It does not provide
benchmark descriptions or grading criteria. This app builds the pilot features
under that contract and compares the two subject-channel treatments:

- ``masked``: subject metadata is absent from the item path and the MAP anchor;
- ``visible``: subject metadata enters the item network as in the pre-fix run.

The models are CNP-AIRT and Online-AIRT, the two deterministic rows that were
competitive in the benchmark-aware pilot.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "aims-neurips-testfaithful"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "faithful-pilot-beavertails-v3-testfeatures"
RESULT_ROOT = REMOTE_ROOT / "results" / "testfaithful-pilot-cnp-online-3seed"
SEEDS = (20261018, 20261019, 20261020)
MODES = ("masked", "visible")
SUBJECT_SLICE = (384, 768)

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


def _hash(*values: object) -> str:
    return hashlib.sha256("::".join(map(str, values)).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _x_for_rows(rows, item_to_index, subject_to_index, item_embeddings, subject_embeddings):
    import numpy as np

    item_indices = np.asarray([item_to_index[str(value)] for value in rows["item_uid"]])
    subject_indices = np.asarray([subject_to_index[str(value)] for value in rows["subject_uid"]])
    return np.concatenate(
        [item_embeddings[item_indices], subject_embeddings[subject_indices]], axis=1
    ).astype("float32")


def _task_arrays(frame, feature_state):
    result = []
    for task_id, rows in frame.groupby("task_id", sort=True):
        result.append(
            {
                "task_id": str(task_id),
                "x": _x_for_rows(rows, **feature_state),
                "y": rows["label"].to_numpy(dtype="float32"),
            }
        )
    return result


def _evaluation_task(task_id, beaver, assignments, observation_lookup, feature_state):
    assignment = assignments[assignments["task_id"].astype(str) == task_id]
    acquisition_rows = assignment[assignment["pool"] == "acquisition"].sort_values(
        "acquisition_rank"
    )
    label_rows = observation_lookup.loc[
        acquisition_rows["label_observation_id"].astype(str).tolist()
    ]
    evaluation_items = set(
        assignment.loc[assignment["pool"] == "evaluation", "item_uid"].astype(str)
    )
    targets = beaver[
        (beaver["task_id"].astype(str) == task_id)
        & (beaver["item_uid"].astype(str).isin(evaluation_items))
    ]
    return {
        "task_id": task_id,
        "x_C_pool": _x_for_rows(label_rows, **feature_state),
        "y_C_pool": label_rows["label"].to_numpy(dtype="float32"),
        "x_T": _x_for_rows(targets, **feature_state),
        "y_T": targets["label"].to_numpy(dtype="float32"),
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32_768,
    volumes={str(REMOTE_ROOT): volume},
    timeout=60 * 60,
)
def prepare_corpus(meta_task_count: int = 128, items_per_task: int = 128) -> dict[str, Any]:
    """Build the test-faithful feature store for the BeaverTails pilot."""

    import numpy as np
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    existing = _read_json(CORPUS_ROOT / "manifest.json")
    if existing is not None and "feature_slices" in existing:
        return {"status": "reused", **existing}

    processed = REMOTE_ROOT / "datasets" / "full" / "processed"
    episodes = REMOTE_ROOT / "datasets" / "full" / "episodes"
    tasks = pd.read_parquet(processed / "tasks.parquet")
    candidates = tasks[
        (tasks["split_by_benchmark"] == "train")
        & (tasks["split_by_task"] == "train")
        & (tasks["n_items"] >= 80)
        & (tasks["benchmark_id"] != "beavertails")
    ].copy()
    candidates["selection_key"] = [
        _hash(benchmark, task)
        for benchmark, task in zip(candidates["benchmark_id"], candidates["task_id"], strict=True)
    ]
    candidates = candidates.sort_values(["benchmark_id", "selection_key"])
    candidates = candidates.groupby("benchmark_id", sort=True).head(3)
    selected = candidates.sort_values("selection_key").head(meta_task_count)
    selected_task_ids = selected["task_id"].astype(str).tolist()
    print(
        f"Selected {len(selected_task_ids)} tasks from {selected['benchmark_id'].nunique()} benchmarks",
        flush=True,
    )
    meta = pd.read_parquet(
        processed / "observations.parquet",
        filters=[("task_id", "in", selected_task_ids)],
    )
    meta = (
        meta.sort_values("observation_id")
        .groupby(["task_id", "item_uid"], as_index=False, sort=True)
        .first()
    )
    meta["sample_key"] = [
        _hash(task, item) for task, item in zip(meta["task_id"], meta["item_uid"], strict=True)
    ]
    meta = (
        meta.sort_values(["task_id", "sample_key"])
        .groupby("task_id", sort=True)
        .head(items_per_task)
        .drop(columns="sample_key")
        .reset_index(drop=True)
    )
    beaver = pd.read_parquet(
        processed / "observations.parquet", filters=[("benchmark_id", "==", "beavertails")]
    )
    beaver_episodes = pd.read_parquet(
        episodes / "episode_items.parquet", filters=[("benchmark_id", "==", "beavertails")]
    )
    item_ids = sorted(set(meta["item_uid"].astype(str)) | set(beaver["item_uid"].astype(str)))
    items = pd.read_parquet(processed / "items.parquet", filters=[("item_uid", "in", item_ids)])
    items = items.drop_duplicates("item_uid").sort_values("item_uid").reset_index(drop=True)
    if len(items) != len(item_ids):
        raise RuntimeError(f"item join incomplete: expected {len(item_ids)}, received {len(items)}")
    subject_ids = sorted(
        set(meta["subject_uid"].astype(str)) | set(beaver["subject_uid"].astype(str))
    )
    subjects = pd.read_parquet(
        processed / "subjects.parquet", filters=[("subject_uid", "in", subject_ids)]
    )
    subjects = (
        subjects.drop_duplicates("subject_uid").sort_values("subject_uid").reset_index(drop=True)
    )
    if len(subjects) != len(subject_ids):
        raise RuntimeError(
            f"subject join incomplete: expected {len(subject_ids)}, received {len(subjects)}"
        )

    # Only fields the Codabench item dictionary provides.
    item_texts = [
        f"item: {content}\nmetadata: {features}"
        for content, features in zip(
            items["content"].fillna(""), items["item_features"].fillna(""), strict=True
        )
    ]
    # Only fields the Codabench subject dictionary provides.
    subject_columns = (
        "normalized_name",
        "provider",
        "release_date",
        "access_date",
        "harness",
        "reasoning_effort",
        "harness_version",
        "subject_features_extra",
    )
    subject_texts = [
        "\n".join(f"{column}: {getattr(row, column)}" for column in subject_columns)
        for row in subjects.fillna("").itertuples(index=False)
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    item_embeddings = encoder.encode(
        item_texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    subject_embeddings = encoder.encode(
        subject_texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    CORPUS_ROOT.mkdir(parents=True, exist_ok=False)
    meta.to_parquet(CORPUS_ROOT / "meta_observations.parquet", index=False)
    beaver.to_parquet(CORPUS_ROOT / "beaver_observations.parquet", index=False)
    beaver_episodes.to_parquet(CORPUS_ROOT / "beaver_episode_items.parquet", index=False)
    items[["item_uid"]].to_parquet(CORPUS_ROOT / "items.parquet", index=False)
    subjects[["subject_uid"]].to_parquet(CORPUS_ROOT / "subjects.parquet", index=False)
    np.save(CORPUS_ROOT / "item_embeddings.npy", item_embeddings)
    np.save(CORPUS_ROOT / "subject_embeddings.npy", subject_embeddings)
    manifest = {
        "format_version": 3,
        "input_contract": "codabench_item_and_subject_fields_only",
        "meta_tasks": int(meta["task_id"].nunique()),
        "meta_benchmarks": int(meta["benchmark_id"].nunique()),
        "meta_rows": len(meta),
        "beavertails_tasks": int(beaver["task_id"].nunique()),
        "beavertails_rows": len(beaver),
        "unique_items": len(items),
        "unique_subjects": len(subjects),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "item_embedding_dimension": int(item_embeddings.shape[1]),
        "subject_embedding_dimension": int(subject_embeddings.shape[1]),
        "feature_slices": {
            "item": [0, int(item_embeddings.shape[1])],
            "subject": [
                int(item_embeddings.shape[1]),
                int(item_embeddings.shape[1] + subject_embeddings.shape[1]),
            ],
        },
        "excluded_unavailable_fields": [
            "benchmark_description",
            "benchmark_name",
            "grading_criterion",
            "asset_manifest",
            "item_numeric",
        ],
    }
    manifest["combined_x_dimension"] = int(manifest["feature_slices"]["subject"][1])
    (CORPUS_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    volume.commit()
    return manifest


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
        raise FileNotFoundError("test-faithful corpus is missing")
    subject_slice = tuple(manifest["feature_slices"]["subject"]) if mode == "masked" else None
    items = pd.read_parquet(CORPUS_ROOT / "items.parquet")
    subjects = pd.read_parquet(CORPUS_ROOT / "subjects.parquet")
    feature_state = {
        "item_to_index": dict(zip(items["item_uid"].astype(str), range(len(items)), strict=True)),
        "subject_to_index": dict(
            zip(subjects["subject_uid"].astype(str), range(len(subjects)), strict=True)
        ),
        "item_embeddings": np.load(CORPUS_ROOT / "item_embeddings.npy"),
        "subject_embeddings": np.load(CORPUS_ROOT / "subject_embeddings.npy"),
    }
    meta = pd.read_parquet(CORPUS_ROOT / "meta_observations.parquet")
    beaver = pd.read_parquet(CORPUS_ROOT / "beaver_observations.parquet")
    assignments = pd.read_parquet(CORPUS_ROOT / "beaver_episode_items.parquet")
    observation_lookup = beaver.set_index("observation_id", drop=False)
    meta_tasks = _task_arrays(meta, feature_state)
    beaver_tasks = {task["task_id"]: task for task in _task_arrays(beaver, feature_state)}
    task_ids = sorted(beaver_tasks)
    folds = []
    for fold, heldout in enumerate(task_ids):
        print(f"seed={seed} mode={mode} fold={fold + 1}/4 heldout={heldout}", flush=True)
        training_tasks = meta_tasks + [task for key, task in beaver_tasks.items() if key != heldout]
        evaluation = _evaluation_task(
            heldout, beaver, assignments, observation_lookup, feature_state
        )
        cnp, cnp_history = train_cnp(
            training_tasks,
            seed=seed + fold * 1000,
            steps=800,
            episodes_per_step=4,
            subject_feature_slice=subject_slice,
        )
        online, online_history = train_amortized_2pl(
            training_tasks,
            seed=seed + fold * 1000,
            steps=800,
            subject_feature_slice=subject_slice,
        )
        fold_root = RESULT_ROOT / f"seed-{seed}-{mode}-checkpoints" / f"fold-{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        torch.save(cnp.state_dict(), fold_root / "cnp_airt.pt")
        torch.save(online.state_dict(), fold_root / "online_airt_2pl.pt")
        if fold == 0:
            write_metrics(fold_root / "cnp-history.json", {"history": cnp_history})
            write_metrics(fold_root / "online-history.json", {"history": online_history})
        y_T = np.asarray(evaluation["y_T"], dtype=float)
        beta = {}
        for budget in DEFAULT_BUDGETS:
            y_C = evaluation["y_C_pool"][:budget]
            probability = (2.0 + float(y_C.sum())) / (4.0 + len(y_C))
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
        folds.append(
            {
                "fold": fold,
                "heldout_task": heldout,
                "models": {
                    "cnp_airt": score_cnp(cnp, evaluation),
                    "online_airt_2pl": score_amortized_2pl(online, evaluation),
                    "beta_binomial": beta,
                },
            }
        )
    result = {"seed": seed, "mode": mode, "folds": folds}
    write_metrics(output, result)
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32_768,
    volumes={str(REMOTE_ROOT): volume},
    timeout=2 * 60 * 60,
)
def run_masked(seed: int) -> dict[str, Any]:
    return _run_mode(seed, "masked")


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32_768,
    volumes={str(REMOTE_ROOT): volume},
    timeout=2 * 60 * 60,
)
def run_visible(seed: int) -> dict[str, Any]:
    return _run_mode(seed, "visible")


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    volumes={str(REMOTE_ROOT): volume},
    timeout=10 * 60,
)
def aggregate() -> dict[str, Any]:
    import numpy as np

    results = [
        _read_json(RESULT_ROOT / f"seed-{seed}-{mode}.json") for seed in SEEDS for mode in MODES
    ]
    if any(result is None for result in results):
        raise FileNotFoundError("test-faithful pilot incomplete")
    names = {
        "online_airt_2pl": "Online-AIRT",
        "cnp_airt": "CNP-AIRT",
        "beta_binomial": "Beta-Binomial",
    }
    aggregate: dict[str, dict[str, Any]] = {mode: {} for mode in MODES}
    for mode in MODES:
        mode_results = [result for result in results if result["mode"] == mode]
        for model in names:
            by_budget = {}
            for budget in (0, 1, 3, 7, 15, 31):
                rows = [
                    fold["models"][model][str(budget)]
                    for result in mode_results
                    for fold in result["folds"]
                ]
                by_budget[str(budget)] = {
                    metric: float(np.mean([row[metric] for row in rows]))
                    for metric in ("brier", "ece")
                }
                by_budget[str(budget)]["brier_se"] = float(
                    np.std([row["brier"] for row in rows], ddof=1) / np.sqrt(len(rows))
                )
            seed_alc = [
                float(np.mean([fold["models"][model]["brier_alc"] for fold in result["folds"]]))
                for result in mode_results
            ]
            aggregate[mode][model] = {
                "brier_alc": float(np.mean(seed_alc)),
                "brier_alc_seed_std": float(np.std(seed_alc, ddof=1)),
                "by_budget": by_budget,
            }
    result = {"status": "testfaithful_pilot", "aggregate": aggregate}
    (RESULT_ROOT / "metrics-testfeatures.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    volume.commit()
    return result


@app.local_entrypoint()
def main(prepare_only: bool = False, modes_only: bool = False) -> None:
    if prepare_only:
        print(json.dumps(prepare_corpus.remote(), indent=2, sort_keys=True))
        return
    if not modes_only:
        print(json.dumps(prepare_corpus.remote(), indent=2, sort_keys=True))
    handles = [run_masked.spawn(seed) for seed in SEEDS]
    handles += [run_visible.spawn(seed) for seed in SEEDS]
    for handle in handles:
        handle.get()
    print(json.dumps(aggregate.remote(), indent=2, sort_keys=True))
