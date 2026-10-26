"""Full-corpus item/subject embedding stores, item_content only.

What changes against ``modal_embedding_ablation.py``:

1. **Item text is ``item_content`` alone.** The v1 arms encoded
   ``f"item: {content}\\nmetadata: {item_features}"``. ``item_features`` has no
   shared schema across benchmarks, carries per-item fingerprints
   (``appworld_task_id``, ``source_task_id``, ``datetime``) and at least one
   benchmark label in disguise (every ``devbench`` item repeats an identical
   180-character ``input_scope=`` sentence). That is the mechanism behind
   item-difficulty R^2 of 0.78-0.82 overall against roughly 0 within a
   benchmark. Subject text is left byte-identical to v1 so the item channel is
   the only thing that moves.

2. **Every item in the protocol universe is encoded, not the 218k compact
   subset.** The compact subset is a function of ``TRAIN_ITEMS_PER_TASK``,
   which is a training dial. Embeddings are per-item and reusable, so the full
   corpus is encoded once and any cap is applied later when a training store is
   built.

3. **Multilingual encoders.** The corpus is 9% non-English across 24 languages;
   Chinese alone spans 9 of 52 benchmarks. The v1 arms were English-only
   (MiniLM, bge-base) or covered none of those languages (Llama-3.1 supports
   en/fr/de/hi/it/pt/es/th). See ``docs/language-distribution.pdf``.

All arms emit 1024 dimensions. Qwen3 is natively 2560/4096 and is reduced by
Matryoshka truncation: slice the first 1024 components, then renormalize.

Run::

    modal run modal_embed_v2.py --arms me5_instruct,bge_m3_long,qwen3_4b,qwen3_8b
    modal run modal_embed_v2.py --arms me5_instruct          # one arm
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal
import numpy as np

APP_NAME = "aims-neurips-embed-v2"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROCESSED_ROOT = REMOTE_ROOT / "datasets" / "full" / "processed"
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "allbench-features-v1"
EMBEDDING_ROOT = REMOTE_ROOT / "datasets" / "full"
HF_HOME = REMOTE_ROOT / "huggingface-cache"

OUTPUT_DIM = 1024
VERIFY_SAMPLE = 512

# Subject text is unchanged from v1: these eight fields, newline joined as
# "{column}: {value}". Keeping it fixed isolates the item-channel change.
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

# e5 models are trained with an instruction prefix and lose several points
# without one. A single fixed instruction keeps it a constant, not a variable.
E5_INSTRUCTION = (
    "Instruct: Represent this evaluation item for predicting its difficulty\nQuery: "
)

ARMS: dict[str, dict[str, Any]] = {
    "me5_instruct": {
        "model": "intfloat/multilingual-e5-large-instruct",
        "native_dim": 1024,
        "max_length": 512,
        "pooling": "mean",
        "item_prefix": E5_INSTRUCTION,
        "subject_prefix": E5_INSTRUCTION,
        "gpu": "L40S",
        "token_budget": 131_072,
        "description": "XLM-R backbone, 1st on MMTEB, 3rd on AfriMTEB-Lite.",
    },
    "bge_m3_long": {
        "model": "BAAI/bge-m3",
        "native_dim": 1024,
        "max_length": 8192,
        "pooling": "cls",
        "item_prefix": "",
        "subject_prefix": "",
        "gpu": "L40S",
        "token_budget": 131_072,
        "description": "Same XLM-R backbone at 8192 tokens: the context-length control.",
    },
    "qwen3_4b": {
        "model": "Qwen/Qwen3-Embedding-4B",
        "native_dim": 2560,
        "max_length": 32768,
        "pooling": "last",
        "item_prefix": "",
        "subject_prefix": "",
        "gpu": "H100",
        "token_budget": 65_536,
        "description": "32K context, strongest Chinese, weakest African languages.",
    },
    "qwen3_8b": {
        "model": "Qwen/Qwen3-Embedding-8B",
        "native_dim": 4096,
        "max_length": 32768,
        "pooling": "last",
        "item_prefix": "",
        "subject_prefix": "",
        "gpu": "H100",
        "token_budget": 32_768,
        "description": "Larger Qwen3; contingent on qwen3_4b showing a gain.",
    },
}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2,<3",
        "pandas>=2.2,<3",
        "pyarrow>=17,<24",
        "torch==2.5.1",
        "transformers==4.51.3",
        "tokenizers>=0.21",
        "sentencepiece>=0.2",
        "protobuf>=4",
        "huggingface-hub>=0.26",
    )
    .env(
        {
            "HF_HOME": str(HF_HOME),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)


def _store_root(arm: str) -> Path:
    return EMBEDDING_ROOT / f"allbench-embed-{arm}-v2"


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _load_texts():
    """Canonical item and subject text, in the order the stores use.

    Items are deduplicated and sorted by ``item_uid`` and subjects by
    ``subject_uid``, matching how ``modal_embedding_ablation.prepare`` indexes
    them, so ``item_index`` from the protocol tables addresses these rows.
    """

    import pandas as pd

    items = pd.read_parquet(PROCESSED_ROOT / "items.parquet", columns=["item_uid", "content"])
    items["item_uid"] = items["item_uid"].astype(str)
    items = items.drop_duplicates("item_uid").sort_values("item_uid").reset_index(drop=True)

    subjects = pd.read_parquet(PROCESSED_ROOT / "subjects.parquet")
    subjects["subject_uid"] = subjects["subject_uid"].astype(str)
    subjects = (
        subjects.drop_duplicates("subject_uid").sort_values("subject_uid").reset_index(drop=True)
    )

    item_texts = items["content"].fillna("").astype(str).tolist()
    subject_texts = [
        "\n".join(f"{column}: {getattr(row, column)}" for column in SUBJECT_COLUMNS)
        for row in subjects.fillna("").itertuples(index=False)
    ]
    return items["item_uid"].to_numpy(), item_texts, subjects["subject_uid"].to_numpy(), subject_texts


def _encode(model, tokenizer, texts, spec, device, label: str):
    """Encode texts to L2-normalized float32 rows in the input order.

    Work is done length-sorted so padding does not dominate -- contents span
    three orders of magnitude -- and scattered back to the caller's order
    before returning. The scatter is the step most likely to go silently
    wrong, so ``build`` re-encodes a sample afterwards and compares.
    """

    import torch

    pooling = spec["pooling"]
    max_length = int(spec["max_length"])
    budget = int(spec["token_budget"])
    prefix = spec.get("prefix", "")
    if prefix:
        texts = [prefix + text for text in texts]

    # Cheap proxy for token count; only used to group similar lengths together.
    order = np.argsort([len(text) for text in texts], kind="stable")
    out = np.zeros((len(texts), int(spec["native_dim"])), dtype=np.float32)
    truncated = 0
    done = 0

    start = 0
    while start < len(order):
        # Grow the batch while the padded token count stays under the budget.
        longest = 0
        end = start
        while end < len(order):
            approx = min(max(len(texts[order[end]]) // 3, 1) + 2, max_length)
            candidate = max(longest, approx) * (end - start + 1)
            if end > start and candidate > budget:
                break
            longest = max(longest, approx)
            end += 1
        batch = [texts[i] for i in order[start:end]]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        truncated += int((encoded["attention_mask"].sum(dim=1) >= max_length).sum())

        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"]
            if pooling == "mean":
                weights = mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
            elif pooling == "cls":
                pooled = hidden[:, 0]
            elif pooling == "last":
                # Left padding puts the final real token last for every row.
                pooled = hidden[:, -1]
            else:
                raise ValueError(f"unknown pooling: {pooling}")
            pooled = pooled.float()
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-9)

        out[order[start:end]] = pooled.cpu().numpy()
        done += end - start
        if done % 100_000 < (end - start):
            print(f"  {label}: {done:,}/{len(texts):,}", flush=True)
        start = end

    return out, truncated


def _mrl(embeddings: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka reduction: slice, then renormalize. Order matters."""

    if embeddings.shape[1] == dim:
        return embeddings
    sliced = embeddings[:, :dim]
    norms = np.linalg.norm(sliced, axis=1, keepdims=True)
    return (sliced / np.clip(norms, 1e-9, None)).astype(np.float32)


@app.function(
    image=image,
    cpu=8,
    memory=131_072,
    volumes={str(REMOTE_ROOT): volume},
    secrets=[hf_secret],
    timeout=8 * 60 * 60,
)
def build(arm: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}; choose from {sorted(ARMS)}")
    root = _store_root(arm)
    existing = _read_json(root / "manifest.json")
    if existing is not None:
        print(f"{arm}: reusing existing store", flush=True)
        return {"status": "reused", **existing}

    spec = ARMS[arm]
    item_uids, item_texts, subject_uids, subject_texts = _load_texts()
    print(f"{arm}: {len(item_texts):,} items, {len(subject_texts):,} subjects", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(spec["model"])
    if spec["pooling"] == "last":
        # Last-token pooling is only correct with left padding, otherwise the
        # final position is a pad token and every vector is quietly wrong.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        spec["model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()

    item_spec = {**spec, "prefix": spec["item_prefix"]}
    subject_spec = {**spec, "prefix": spec["subject_prefix"]}
    print(f"{arm}: encoding items", flush=True)
    item_raw, item_truncated = _encode(model, tokenizer, item_texts, item_spec, device, "items")
    print(f"{arm}: encoding subjects", flush=True)
    subject_raw, subject_truncated = _encode(
        model, tokenizer, subject_texts, subject_spec, device, "subjects"
    )

    item_embeddings = _mrl(item_raw, OUTPUT_DIM)
    subject_embeddings = _mrl(subject_raw, OUTPUT_DIM)

    # --- verification -----------------------------------------------------
    for name, array in (("items", item_embeddings), ("subjects", subject_embeddings)):
        if array.shape[1] != OUTPUT_DIM:
            raise RuntimeError(f"{name} dim {array.shape[1]} != {OUTPUT_DIM}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"{name} contain non-finite values")
        norms = np.linalg.norm(array, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise RuntimeError(f"{name} are not unit norm (min {norms.min()}, max {norms.max()})")

    rng = np.random.default_rng(20261024)
    probe = rng.permutation(len(item_texts))[:VERIFY_SAMPLE]
    probe_raw, _ = _encode(
        model, tokenizer, [item_texts[i] for i in probe], item_spec, device, "verify"
    )
    delta = float(np.abs(_mrl(probe_raw, OUTPUT_DIM) - item_embeddings[probe]).max())
    if delta > 1e-2:
        raise RuntimeError(f"scatter check failed: max|delta| = {delta}")
    print(f"{arm}: scatter check ok, max|delta| = {delta:.2e}", flush=True)

    # --- write ------------------------------------------------------------
    import pandas as pd

    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "item_embeddings.npy", item_embeddings)
    np.save(root / "subject_embeddings.npy", subject_embeddings)
    pd.DataFrame({"item_uid": item_uids}).to_parquet(root / "items.parquet", index=False)
    pd.DataFrame({"subject_uid": subject_uids}).to_parquet(root / "subjects.parquet", index=False)

    manifest = {
        "format_version": 2,
        "arm": arm,
        "embedding_model": spec["model"],
        "description": spec["description"],
        "item_text": "content_only",
        "item_text_template": "{content}",
        "subject_text": "newline-joined eight subject fields (unchanged from v1)",
        "item_prefix": spec["item_prefix"],
        "pooling": spec["pooling"],
        "max_sequence_length": spec["max_length"],
        "native_dim": spec["native_dim"],
        "item_embedding_dim": OUTPUT_DIM,
        "subject_embedding_dim": OUTPUT_DIM,
        "mrl_truncated": spec["native_dim"] != OUTPUT_DIM,
        "normalized": True,
        "forward_dtype": "bfloat16",
        "storage_dtype": "float32",
        "unique_items": int(item_embeddings.shape[0]),
        "unique_subjects": int(subject_embeddings.shape[0]),
        "items_truncated": int(item_truncated),
        "subjects_truncated": int(subject_truncated),
        "scatter_check_max_abs_delta": delta,
        "gpu": spec["gpu"],
        "covers_full_corpus": True,
        "note": (
            "Full protocol item universe, not the TRAIN_ITEMS_PER_TASK-capped "
            "compact subset. Apply any per-task cap when building a training store."
        ),
    }
    _write_json(root / "manifest.json", manifest)
    volume.commit()
    print(f"{arm}: done, {item_truncated:,} items truncated at {spec['max_length']}", flush=True)
    return {"status": "built", **manifest}


@app.local_entrypoint()
def main(arms: str = "me5_instruct,bge_m3_long,qwen3_4b,qwen3_8b") -> None:
    names = [name.strip() for name in arms.split(",") if name.strip()]
    unknown = [name for name in names if name not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {sorted(ARMS)}")

    handles = []
    for name in names:
        gpu = ARMS[name]["gpu"]
        print(f"spawning {name} on {gpu}")
        handles.append((name, build.with_options(gpu=gpu).spawn(name)))
    for name, handle in handles:
        result = handle.get()
        print(f"\n===== {name} =====")
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
