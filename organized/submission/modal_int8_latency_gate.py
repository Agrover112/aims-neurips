"""Latency + drift gate for an int8 Qwen3-Embedding-8B CodaBench predictor.

The staged submission embeds every drawn item with Qwen3-Embedding-8B on CPU.
CodaBench caps a whole run at eight hours and draws up to 1,000 distinct source
items, so the per-item embedding budget is about 29 seconds. This job measures
the two unknowns that decide whether an int8 CPU predictor can hold that
budget, in one pass over the same items:

1. **Latency.** Wall time per item for the int8 model at the item's natural
   token length. No truncation: ``max_length`` stays at the store's 32,768 so
   long questions are encoded exactly as training encoded them.
2. **Drift.** Cosine similarity between each int8 vector and the stored
   bfloat16 vector for the same ``item_uid``. The store was built on an H100
   and is the reference the frozen checkpoint was trained against.

Latency is the gate. Drift only matters if latency passes, but it is free to
collect here, so it is collected here.

``cpu`` and ``memory`` are deliberately close to an ordinary CodaBench CPU
container. The absolute numbers still carry the caveat that CodaBench's host is
not Modal's, so treat the result as an estimate with a safety factor, not a
guarantee.

Run::

    modal run --profile agrover112 \
      organized/submission/modal_int8_latency_gate.py::main --n-items 20
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "aims-np-int8-latency-gate"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROCESSED_ROOT = REMOTE_ROOT / "datasets" / "full" / "processed"
STORE_ROOT = REMOTE_ROOT / "datasets" / "full" / "allbench-embed-qwen3_8b-v2"
HF_HOME = REMOTE_ROOT / "huggingface-cache"
RESULT_PATH = REMOTE_ROOT / "results" / "submission" / "int8-latency-gate.json"

MODEL_ID = "Qwen/Qwen3-Embedding-8B"
# Exactly the store's contract: last-token pooling, Matryoshka slice to the
# first 1,024 of 4,096 components, then renormalise. Order matters.
EMBEDDING_DIM = 1024
MAX_LENGTH = 32_768

# CodaBench's documented limits, used only to turn seconds/item into a verdict.
RUN_SECONDS = 8 * 60 * 60
DRAW_ITEMS = 1_000

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
    .env({"HF_HOME": str(HF_HOME), "TOKENIZERS_PARALLELISM": "false"})
)


def _quantize_in_place(model) -> int:
    """Swap every ``nn.Linear`` for a dynamic-int8 one, layer by layer.

    ``quantize_dynamic`` needs float32 input modules, and casting the whole
    bfloat16 8B model to float32 first would need about 30 GiB. Converting one
    layer at a time keeps the peak near the bfloat16 footprint plus a single
    layer, which is what lets this run in a CodaBench-sized container at all.
    """

    import torch
    from torch import nn
    from torch.ao.nn.quantized.dynamic import Linear as DynamicQuantizedLinear

    converted = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            as_float = nn.Linear(
                child.in_features, child.out_features, bias=child.bias is not None
            )
            with torch.no_grad():
                as_float.weight.copy_(child.weight.to(torch.float32))
                if child.bias is not None:
                    as_float.bias.copy_(child.bias.to(torch.float32))
            setattr(module, name, DynamicQuantizedLinear.from_float(as_float))
            del as_float, child
            converted += 1
    return converted


@app.function(
    image=image,
    cpu=8,
    memory=32_768,
    volumes={"/data": volume},
    secrets=[hf_secret],
    timeout=2 * 60 * 60,
)
def gate(n_items: int = 20, seed: int = 20261028) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(8)

    # ---- pick items spanning the length distribution ------------------------
    # Cost is linear in tokens, so a uniform sample of a long-tailed corpus
    # would understate the mean. Sample one item per content-length decile.
    items = pd.read_parquet(PROCESSED_ROOT / "items.parquet", columns=["item_uid", "content"])
    items["item_uid"] = items["item_uid"].astype(str)
    items = items.drop_duplicates("item_uid").sort_values("item_uid").reset_index(drop=True)
    items["content"] = items["content"].fillna("").astype(str)
    items["n_chars"] = items["content"].str.len()

    rng = np.random.default_rng(seed)
    deciles = np.quantile(items["n_chars"], np.linspace(0, 1, 11))
    picked: list[int] = []
    per_bin = max(1, n_items // 10)
    for low, high in zip(deciles[:-1], deciles[1:]):
        bin_rows = items.index[(items["n_chars"] >= low) & (items["n_chars"] <= high)]
        if len(bin_rows):
            take = min(per_bin, len(bin_rows))
            picked.extend(rng.choice(bin_rows, size=take, replace=False).tolist())
    picked = sorted(set(picked))[:n_items]
    sample = items.loc[picked].reset_index(drop=True)
    print(
        f"sampled {len(sample)} items, chars "
        f"{sample.n_chars.min():,}-{sample.n_chars.max():,} "
        f"(corpus median {int(items.n_chars.median()):,})",
        flush=True,
    )

    # ---- reference vectors from the store ----------------------------------
    store_ids = pd.read_parquet(STORE_ROOT / "items.parquet")["item_uid"].astype(str).to_numpy()
    store_pos = {uid: i for i, uid in enumerate(store_ids)}
    store_vectors = np.load(STORE_ROOT / "item_embeddings.npy", mmap_mode="r")

    # ---- load and quantize --------------------------------------------------
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.eval()
    load_seconds = time.perf_counter() - load_start
    print(f"loaded bfloat16 in {load_seconds:.1f}s", flush=True)

    quantize_start = time.perf_counter()
    converted = _quantize_in_place(model)
    quantize_seconds = time.perf_counter() - quantize_start
    print(f"quantized {converted} Linear layers in {quantize_seconds:.1f}s", flush=True)

    # ---- time one item at a time -------------------------------------------
    rows = []
    for row in sample.itertuples(index=False):
        encoded = tokenizer(
            row.content, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
        )
        n_tokens = int(encoded["input_ids"].shape[1])
        start = time.perf_counter()
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            vector = hidden[:, -1, :EMBEDDING_DIM].float()
            vector = torch.nn.functional.normalize(vector, p=2, dim=-1)
        elapsed = time.perf_counter() - start
        produced = vector[0].numpy().astype("float32")

        cosine = None
        position = store_pos.get(str(row.item_uid))
        if position is not None:
            reference = np.asarray(store_vectors[position], dtype=np.float32)
            cosine = float(produced @ reference)

        rows.append(
            {
                "item_uid": str(row.item_uid),
                "n_chars": int(row.n_chars),
                "n_tokens": n_tokens,
                "seconds": round(elapsed, 3),
                "cosine_vs_bf16_store": None if cosine is None else round(cosine, 5),
            }
        )
        print(
            f"  {n_tokens:>6} tok  {elapsed:>7.2f}s"
            + ("" if cosine is None else f"  cos={cosine:.4f}"),
            flush=True,
        )

    seconds = np.array([r["seconds"] for r in rows], dtype=float)
    tokens = np.array([r["n_tokens"] for r in rows], dtype=float)
    cosines = np.array(
        [r["cosine_vs_bf16_store"] for r in rows if r["cosine_vs_bf16_store"] is not None],
        dtype=float,
    )

    # Corpus-weighted projection: scale the measured per-token rate by the
    # corpus mean token count, approximated from the char/token ratio here.
    chars_per_token = float(np.sum([r["n_chars"] for r in rows]) / max(tokens.sum(), 1.0))
    corpus_mean_tokens = float(items["n_chars"].mean() / max(chars_per_token, 1e-9))
    seconds_per_token = float(seconds.sum() / max(tokens.sum(), 1.0))
    projected_per_item = seconds_per_token * corpus_mean_tokens
    budget_per_item = RUN_SECONDS / DRAW_ITEMS

    summary = {
        "model": MODEL_ID,
        "precision": "int8_dynamic_linear_over_bfloat16",
        "max_length": MAX_LENGTH,
        "cpu": 8,
        "n_items": len(rows),
        "linear_layers_quantized": converted,
        "load_seconds": round(load_seconds, 1),
        "quantize_seconds": round(quantize_seconds, 1),
        "sample_mean_seconds_per_item": round(float(seconds.mean()), 2),
        "sample_median_seconds_per_item": round(float(np.median(seconds)), 2),
        "seconds_per_token": round(seconds_per_token, 5),
        "sample_mean_tokens": round(float(tokens.mean()), 1),
        "corpus_mean_tokens_estimate": round(corpus_mean_tokens, 1),
        "projected_seconds_per_item_corpus": round(projected_per_item, 2),
        "budget_seconds_per_item": round(budget_per_item, 2),
        "projected_hours_for_1000_items": round(projected_per_item * DRAW_ITEMS / 3600, 2),
        "verdict": "PASS" if projected_per_item < budget_per_item else "FAIL",
        "cosine_vs_bf16_store": {
            "n": int(cosines.size),
            "mean": None if not cosines.size else round(float(cosines.mean()), 5),
            "min": None if not cosines.size else round(float(cosines.min()), 5),
            "p05": None if not cosines.size else round(float(np.quantile(cosines, 0.05)), 5),
        },
        "rows": rows,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    volume.commit()

    print("\n" + "=" * 68, flush=True)
    print(f"verdict:              {summary['verdict']}", flush=True)
    print(
        f"projected s/item:     {summary['projected_seconds_per_item_corpus']}"
        f"  (budget {summary['budget_seconds_per_item']})",
        flush=True,
    )
    print(f"projected 1000 items: {summary['projected_hours_for_1000_items']} h  (cap 8 h)", flush=True)
    print(f"cosine vs bf16 store: {summary['cosine_vs_bf16_store']}", flush=True)
    print("=" * 68, flush=True)
    return summary


@app.local_entrypoint()
def main(n_items: int = 20, seed: int = 20261028) -> None:
    gate.remote(n_items=n_items, seed=seed)
