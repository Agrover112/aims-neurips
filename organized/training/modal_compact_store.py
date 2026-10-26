"""Slice a full v2 embedding store into a training corpus at a chosen cap.

``modal_embed_v2.py`` encodes every item in the protocol universe once. This
step turns one of those stores into the compact corpus the training code
consumes, applying the per-task item cap here rather than baking it into the
feature store the way ``modal_allbench.TRAIN_ITEMS_PER_TASK`` does.

Separating the two matters because the cap is a training dial, not a property
of the corpus: embeddings are per-item and reusable, so one encode pass at
roughly $22 supports every cap you want to sweep. It is also the only honest
way to ask whether 128 was ever the right number -- nothing in ``docs/``
justifies it, and the competition's own rules (>= 80 distinct items per pair,
a 50/50 acquisition/evaluation split, and a 1,000 subject-item draw cap, which
is where ``EVAL_ITEMS_PER_TASK = 500`` legitimately comes from) say nothing
about the training side.

Two differences from the original builder, both deliberate:

* The cap samples **randomly** within a task. ``modal_allbench`` takes
  ``.head(N)`` after sorting by acquisition rank, which biases training toward
  early-ranked items.
* Evaluation and order tables are never capped. They define the scored
  protocol and must match across every training store.

Run::

    modal run modal_compact_store.py --arm me5_instruct --caps 128,256,1024
    modal run modal_compact_store.py --arm me5_instruct --caps 0        # no cap
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal
import numpy as np

APP_NAME = "aims-neurips-compact-store"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "allbench-features-v1"
EMBEDDING_ROOT = REMOTE_ROOT / "datasets" / "full"

DEFAULT_SEED = 20261024

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24"
)


def _source_root(arm: str) -> Path:
    return EMBEDDING_ROOT / f"allbench-embed-{arm}-v2"


def _target_root(arm: str, cap: int | None) -> Path:
    suffix = "uncapped" if cap is None else f"cap{cap}"
    return EMBEDDING_ROOT / f"allbench-train-{arm}-{suffix}-v2"


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


@app.function(
    image=image,
    cpu=8,
    memory=131_072,
    volumes={str(REMOTE_ROOT): volume},
    timeout=2 * 60 * 60,
)
def compact(arm: str, cap: int | None = 128, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    import pandas as pd

    source = _source_root(arm)
    source_manifest = _read_json(source / "manifest.json")
    if source_manifest is None:
        raise FileNotFoundError(f"no v2 embedding store at {source}; run modal_embed_v2.py first")
    target = _target_root(arm, cap)
    existing = _read_json(target / "manifest.json")
    if existing is not None:
        return {"status": "reused", **existing}

    train = pd.read_parquet(CORPUS_ROOT / "train_tasks.parquet")
    evaluation = pd.read_parquet(CORPUS_ROOT / "eval_items.parquet")
    orders = pd.read_parquet(CORPUS_ROOT / "eval_orders.parquet")
    print(f"{arm}: {len(train):,} train rows, {train['task_id'].nunique():,} train tasks", flush=True)

    # Cap the training side only. Sampling is random within a task, with the
    # kept rows left in their original order so acquisition rank still means
    # what it meant before.
    if cap is not None:
        rng = np.random.default_rng(seed)
        before = len(train)
        kept = []
        for _, group in train.groupby("task_id", sort=True):
            if len(group) <= cap:
                kept.append(group)
            else:
                positions = np.sort(rng.permutation(len(group))[:cap])
                kept.append(group.iloc[positions])
        train = pd.concat(kept).reset_index(drop=True)
        print(f"{arm}: capped train rows {before:,} -> {len(train):,}", flush=True)

    # The v2 store is ordered exactly like the protocol's item_index space:
    # processed items deduplicated by item_uid and sorted, which is how
    # modal_embedding_ablation.prepare built its indices. Verified below.
    item_embeddings_full = np.load(source / "item_embeddings.npy", mmap_mode="r")
    subject_embeddings_full = np.load(source / "subject_embeddings.npy", mmap_mode="r")
    item_uids_full = pd.read_parquet(source / "items.parquet")["item_uid"].to_numpy()
    subject_uids_full = pd.read_parquet(source / "subjects.parquet")["subject_uid"].to_numpy()

    item_indices = np.unique(
        np.concatenate(
            [
                train["item_index"].to_numpy(dtype=np.int64),
                evaluation["item_index"].to_numpy(dtype=np.int64),
            ]
        )
    )
    subject_indices = np.unique(
        np.concatenate(
            [
                train["subject_index"].to_numpy(dtype=np.int64),
                evaluation["subject_index"].to_numpy(dtype=np.int64),
            ]
        )
    )
    if int(item_indices.max()) >= len(item_uids_full):
        raise RuntimeError(
            f"protocol item_index {int(item_indices.max())} exceeds store rows {len(item_uids_full)}"
        )
    if int(subject_indices.max()) >= len(subject_uids_full):
        raise RuntimeError("protocol subject_index exceeds store rows")

    item_map = {int(old): int(new) for new, old in enumerate(item_indices)}
    subject_map = {int(old): int(new) for new, old in enumerate(subject_indices)}
    train = train.copy()
    evaluation = evaluation.copy()
    orders = orders.copy()
    train["item_index"] = train["item_index"].map(item_map).astype("int32")
    train["subject_index"] = train["subject_index"].map(subject_map).astype("int32")
    evaluation["item_index"] = evaluation["item_index"].map(item_map).astype("int32")
    evaluation["subject_index"] = evaluation["subject_index"].map(subject_map).astype("int32")
    orders["item_index"] = orders["item_index"].map(item_map).astype("int32")

    item_embeddings = np.asarray(item_embeddings_full[item_indices], dtype=np.float32)
    subject_embeddings = np.asarray(subject_embeddings_full[subject_indices], dtype=np.float32)
    item_uids = item_uids_full[item_indices]
    subject_uids = subject_uids_full[subject_indices]

    for name, array in (("items", item_embeddings), ("subjects", subject_embeddings)):
        if not np.isfinite(array).all():
            raise RuntimeError(f"{name} contain non-finite values")
        norms = np.linalg.norm(array, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise RuntimeError(f"{name} lost unit norm during the gather")
    if train["item_index"].isna().any() or evaluation["item_index"].isna().any():
        raise RuntimeError("reindexing left null item_index values")

    target.mkdir(parents=True, exist_ok=True)
    train.to_parquet(target / "train_tasks.parquet", index=False)
    evaluation.to_parquet(target / "eval_items.parquet", index=False)
    orders.to_parquet(target / "eval_orders.parquet", index=False)
    pd.DataFrame({"item_uid": item_uids}).to_parquet(target / "items.parquet", index=False)
    pd.DataFrame({"subject_uid": subject_uids}).to_parquet(target / "subjects.parquet", index=False)
    np.save(target / "item_embeddings.npy", item_embeddings)
    np.save(target / "subject_embeddings.npy", subject_embeddings)

    per_task = train.groupby("task_id").size()
    manifest = {
        "format_version": 1,
        "arm": arm,
        "source_store": str(source),
        "embedding_model": source_manifest["embedding_model"],
        "item_text": source_manifest["item_text"],
        "max_sequence_length": source_manifest["max_sequence_length"],
        "train_items_per_task": cap,
        "cap_sampling": "random within task, seeded" if cap is not None else "none",
        "cap_seed": int(seed),
        "item_embedding_dim": int(item_embeddings.shape[1]),
        "subject_embedding_dim": int(subject_embeddings.shape[1]),
        "combined_x_dimension": int(item_embeddings.shape[1] + subject_embeddings.shape[1]),
        "feature_slices": {
            "item": [0, int(item_embeddings.shape[1])],
            "subject": [
                int(item_embeddings.shape[1]),
                int(item_embeddings.shape[1] + subject_embeddings.shape[1]),
            ],
        },
        "unique_items": int(item_embeddings.shape[0]),
        "unique_subjects": int(subject_embeddings.shape[0]),
        "train_tasks": int(train["task_id"].nunique()),
        "train_rows": len(train),
        "train_items_per_task_min": int(per_task.min()),
        "train_items_per_task_median": float(per_task.median()),
        "train_items_per_task_max": int(per_task.max()),
        "evaluation_tasks": int(evaluation["task_id"].nunique()),
        "evaluation_rows": len(evaluation),
        "order_rows": len(orders),
    }
    _write_json(target / "manifest.json", manifest)
    volume.commit()
    print(
        f"{arm} cap={cap}: {manifest['unique_items']:,} items, "
        f"{manifest['train_rows']:,} train rows",
        flush=True,
    )
    return {"status": "built", **manifest}


@app.local_entrypoint()
def main(arm: str = "me5_instruct", caps: str = "128", seed: int = DEFAULT_SEED) -> None:
    # A cap of 0 means "no cap"; argument parsing has no clean None.
    values = [int(value.strip()) for value in caps.split(",") if value.strip()]
    requested: list[int | None] = [None if value == 0 else value for value in values]

    handles = [(cap, compact.spawn(arm, cap, seed)) for cap in requested]
    for cap, handle in handles:
        result = handle.get()
        print(f"\n===== {arm} cap={cap} =====")
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
