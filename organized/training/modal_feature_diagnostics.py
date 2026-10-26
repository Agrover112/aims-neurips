"""Pre-training diagnostics for the item/subject feature pipeline.

Three read-only measurements, all on the frozen ``allbench-features-v1`` store.
Nothing is trained and nothing is written back into the corpus.

1. **Token lengths.** How long are the item texts under the MiniLM tokenizer
   (the 256/512-token BERT family currently in use) and under the bge-m3
   tokenizer (8192-token XLM-R)? This decides whether truncation is a real
   problem or a rounding error before any re-encode is paid for.

2. **Item replication.** How many distinct tasks/subjects answer each item?
   Item difficulty ``b_i`` is only estimable from data when an item is answered
   by several subjects. With no replication there is no empirical ``b_i`` to
   regress an embedding against, and item features can only ever be supervised
   through the joint model.

3. **Variance components.** How is outcome variance split between subject,
   item, and benchmark? This bounds how much any item encoder can contribute:
   if between-item variance is small, no item representation helps.

Run:

    modal run modal_feature_diagnostics.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "aims-neurips-feature-diagnostics"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "allbench-features-v1"
PROCESSED_ROOT = REMOTE_ROOT / "datasets" / "full" / "processed"
OUTPUT_ROOT = REMOTE_ROOT / "results" / "feature-diagnostics"

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BGE_M3 = "BAAI/bge-m3"
LENGTH_CAPS = (128, 256, 512, 1024, 2048, 8192)
PERCENTILES = (50, 75, 90, 95, 99, 99.9, 100)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2,<3",
        "pandas>=2.2,<3",
        "pyarrow>=17,<24",
        "transformers>=4.44,<5",
        "tokenizers>=0.20",
        "sentencepiece>=0.2",
        "protobuf>=4",
        "huggingface-hub>=0.26",
    )
    .env({"HF_HOME": "/data/huggingface-cache", "TOKENIZERS_PARALLELISM": "true"})
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _item_texts(items) -> list[str]:
    """Rebuild the exact template used by ``modal_allbench.py``."""

    return [
        f"item: {content}\nmetadata: {features}"
        for content, features in zip(
            items["content"].fillna(""),
            items["item_features"].fillna(""),
            strict=True,
        )
    ]


def _length_summary(lengths, name: str, sample_size: int) -> dict[str, Any]:
    import numpy as np

    lengths = np.asarray(lengths, dtype=np.int32)
    return {
        "tokenizer": name,
        "items_measured": int(sample_size),
        "mean": float(lengths.mean()),
        "std": float(lengths.std(ddof=1)),
        "percentiles": {
            str(p): float(np.percentile(lengths, p)) for p in PERCENTILES
        },
        "truncation_rate": {
            str(cap): float((lengths > cap).mean()) for cap in LENGTH_CAPS
        },
        "tokens_lost_fraction": {
            str(cap): float(
                np.clip(lengths - cap, 0, None).sum() / max(int(lengths.sum()), 1)
            )
            for cap in LENGTH_CAPS
        },
    }


@app.function(
    image=image,
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=2 * 60 * 60,
)
def token_lengths(sample: int = 0) -> dict[str, Any]:
    """Measure item-text token lengths under both tokenizers."""

    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    # The feature store keeps only item_uid plus embeddings; the raw text lives
    # in the processed corpus. Restrict to the items the protocol actually uses
    # so the truncation rate describes the population being encoded.
    used = pd.read_parquet(CORPUS_ROOT / "items.parquet", columns=["item_uid"])
    source = pd.read_parquet(
        PROCESSED_ROOT / "items.parquet", columns=["item_uid", "content", "item_features"]
    )
    print(f"protocol items: {len(used):,}  processed items: {len(source):,}", flush=True)

    items = source.merge(used, on="item_uid", how="inner").reset_index(drop=True)
    missing = len(used) - len(items)
    if missing:
        print(f"warning: {missing:,} protocol items have no processed text row", flush=True)
    print(f"matched {len(items):,} items with text", flush=True)

    if sample and sample < len(items):
        items = items.sample(sample, random_state=20261022).reset_index(drop=True)
        print(f"sampled {len(items):,} items", flush=True)

    texts = _item_texts(items)
    characters = np.array([len(text) for text in texts], dtype=np.int64)

    summaries = []
    for name in (MINILM, BGE_M3):
        print(f"tokenizing with {name}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(name)
        lengths = np.empty(len(texts), dtype=np.int32)
        step = 10_000
        for start in range(0, len(texts), step):
            chunk = texts[start : start + step]
            encoded = tokenizer(
                chunk, add_special_tokens=True, truncation=False, padding=False
            )["input_ids"]
            lengths[start : start + len(chunk)] = [len(ids) for ids in encoded]
            if start % 200_000 == 0:
                print(f"  {start:,}/{len(texts):,}", flush=True)
        summaries.append(_length_summary(lengths, name, len(texts)))
        print(f"  {name}: mean={lengths.mean():.1f} p99={np.percentile(lengths, 99):.0f}", flush=True)

    result = {
        "status": "token_lengths",
        "corpus": str(CORPUS_ROOT),
        "text_source": str(PROCESSED_ROOT / "items.parquet"),
        "items_matched": len(items),
        "item_text_template": "item: {content}\\nmetadata: {item_features}",
        "characters": {
            "mean": float(characters.mean()),
            "percentiles": {
                str(p): float(np.percentile(characters, p)) for p in PERCENTILES
            },
        },
        "tokenizers": summaries,
    }
    _write_json(OUTPUT_ROOT / "token-lengths.json", result)
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=8,
    memory=131_072,
    volumes={str(REMOTE_ROOT): volume},
    timeout=2 * 60 * 60,
)
def replication_and_variance() -> dict[str, Any]:
    """Item/subject replication counts and an outcome variance decomposition."""

    import numpy as np
    import pandas as pd

    train = pd.read_parquet(CORPUS_ROOT / "train_tasks.parquet")
    print(f"train rows: {len(train):,}  columns: {list(train.columns)}", flush=True)

    # One row per (task, item). A task is one (subject, benchmark) pair, so the
    # number of distinct tasks answering an item is its replication.
    per_item = train.groupby("item_index", sort=False).agg(
        responses=("label", "size"),
        mean_label=("label", "mean"),
    )
    per_task = train.groupby("task_id", sort=False).agg(
        responses=("label", "size"),
        mean_label=("label", "mean"),
    )

    item_counts = per_item["responses"].to_numpy()
    task_counts = per_task["responses"].to_numpy()
    labels = train["label"].to_numpy(dtype=float)

    def _histogram(counts: np.ndarray) -> dict[str, int]:
        return {
            "1": int((counts == 1).sum()),
            "2-4": int(((counts >= 2) & (counts <= 4)).sum()),
            "5-9": int(((counts >= 5) & (counts <= 9)).sum()),
            "10-49": int(((counts >= 10) & (counts <= 49)).sum()),
            "50+": int((counts >= 50).sum()),
        }

    # Variance decomposition on the binary outcome. For each grouping factor,
    # the share of total variance explained by that factor's group means is the
    # classical eta-squared. The design is crossed, so these shares overlap and
    # need not sum to one; they are read as "how much does knowing only this
    # factor explain", not as an orthogonal partition.
    total_variance = float(labels.var())
    grand_mean = float(labels.mean())

    def _eta_squared(keys: str) -> dict[str, float]:
        group_mean = train.groupby(keys, sort=False)["label"].transform("mean").to_numpy()
        between = float(np.mean((group_mean - grand_mean) ** 2))
        return {
            "eta_squared": between / total_variance if total_variance else 0.0,
            "residual_variance": float(np.mean((labels - group_mean) ** 2)),
        }

    # Shrunken item means guard against single-response items looking perfectly
    # "explanatory" purely because one observation reproduces itself.
    prior_strength = 4.0
    shrunk = (
        per_item["mean_label"] * per_item["responses"] + grand_mean * prior_strength
    ) / (per_item["responses"] + prior_strength)
    shrunk_by_row = train["item_index"].map(shrunk).to_numpy()
    shrunk_item_eta = float(
        np.mean((shrunk_by_row - grand_mean) ** 2) / total_variance
    ) if total_variance else 0.0

    result = {
        "status": "replication_and_variance",
        "rows": len(train),
        "distinct_items": int(train["item_index"].nunique()),
        "distinct_tasks": int(train["task_id"].nunique()),
        "distinct_subjects": int(train["subject_index"].nunique()),
        "grand_mean_label": grand_mean,
        "total_variance": total_variance,
        "item_replication": {
            "mean_responses_per_item": float(item_counts.mean()),
            "median_responses_per_item": float(np.median(item_counts)),
            "max_responses_per_item": int(item_counts.max()),
            "items_answered_once": int((item_counts == 1).sum()),
            "fraction_answered_once": float((item_counts == 1).mean()),
            "histogram": _histogram(item_counts),
        },
        "task_size": {
            "mean_responses_per_task": float(task_counts.mean()),
            "median_responses_per_task": float(np.median(task_counts)),
            "min_responses_per_task": int(task_counts.min()),
            "histogram": _histogram(task_counts),
        },
        "variance_components": {
            "task": _eta_squared("task_id"),
            "subject": _eta_squared("subject_index"),
            "item_raw": _eta_squared("item_index"),
            "item_shrunk_eta_squared": shrunk_item_eta,
            "note": (
                "Crossed design: shares overlap and need not sum to one. "
                "item_raw is inflated whenever items have few responses; "
                "compare it against item_shrunk_eta_squared."
            ),
        },
    }
    _write_json(OUTPUT_ROOT / "replication-variance.json", result)
    volume.commit()
    return result


@app.local_entrypoint()
def main(sample: int = 0, skip_tokens: bool = False, skip_variance: bool = False) -> None:
    handles = []
    if not skip_tokens:
        handles.append(("token_lengths", token_lengths.spawn(sample)))
    if not skip_variance:
        handles.append(("replication_and_variance", replication_and_variance.spawn()))
    for name, handle in handles:
        print(f"\n===== {name} =====")
        print(json.dumps(handle.get(), indent=2, sort_keys=True))
