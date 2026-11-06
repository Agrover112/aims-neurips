#!/usr/bin/env python3
"""Token-entropy features of item text from a small language model.

For each item, the language model reads the item's text and, at every token
position, reports how spread out its next-token distribution is (Shannon
entropy, in nats). The per-item summary is a set of entropy statistics: mean,
standard deviation, quantiles and the mean of the top-k values. These are the
only features this study uses; they need no labels, so they can be computed
at test time, before any response is revealed.

Column reads are used because some row groups of the item-text file fail when
read whole (corrupt snappy pages in the volume copy); reading one column at a
time avoids the bad pages where they do not overlap that column.

    python organized/testing/online/lm_uncertainty_features.py --pilot 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = Path.home() / "aims-replay-cache" / "lmfeat"
ITEMS = CACHE / "items.parquet"
RESULTS = ROOT / "organized" / "training" / "results" / "item-difficulty-b0.parquet"
OUT = ROOT / "organized" / "testing" / "online" / "results"
MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MAX_TOKENS = 512
TOP_K = 10
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def load_items(uids: set[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    f = pq.ParquetFile(ITEMS)
    parts = []
    for g in range(f.num_row_groups):
        uid = f.read_row_group(g, columns=["item_uid"]).column("item_uid").to_pylist()
        keep = [i for i, u in enumerate(uid) if u in uids]
        if not keep:
            continue
        content = f.read_row_group(g, columns=["content"]).column("content").to_pylist()
        parts.append(pd.DataFrame({"item_uid": [uid[i] for i in keep],
                                   "content": [content[i] for i in keep]}))
    return pd.concat(parts, ignore_index=True)


def sample_pilot(n: int, rng) -> pd.DataFrame:
    """Items sampled across all benchmarks, in proportion to their size but with a floor."""
    t = pd.read_parquet(RESULTS)[["item_uid", "benchmark_id", "benchmark_split", "observed"]]
    per = max(1, n // t["benchmark_id"].nunique())
    picked = t.groupby("benchmark_id", group_keys=False).apply(
        lambda g: g.sample(min(per, len(g)), random_state=int(rng.integers(1 << 31))))
    return picked.reset_index(drop=True)


def entropy_features(model, tok, texts: list[str], torch) -> np.ndarray:
    rows = []
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
        if enc["input_ids"].shape[1] < 2:
            rows.append([np.nan] * (5 + len(QUANTILES) + 1 + 1))
            continue
        with torch.inference_mode():
            logits = model(**enc).logits[0, :-1].float()       # position t predicts token t+1
        logp = torch.log_softmax(logits, dim=-1)
        h = -(logp.exp() * logp).sum(-1).cpu().numpy()          # entropy per position, nats
        q = np.quantile(h, QUANTILES)
        top = np.sort(h)[-TOP_K:].mean() if len(h) >= TOP_K else h.mean()
        rows.append([h.mean(), h.std(), *q, top, len(h)])
    names = ["ent_mean", "ent_std", *[f"ent_q{int(q * 100)}" for q in QUANTILES], "ent_topk", "n_tokens"]
    return pd.DataFrame(rows, columns=names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", type=int, default=200, help="items to sample across the 17 benchmarks")
    ap.add_argument("--threads", type=int, default=6, help="CPU threads; keep below the core count")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(20261102)
    sample = sample_pilot(a.pilot, rng)
    text = load_items(set(sample["item_uid"].astype(str)))
    sample = sample.merge(text, on="item_uid", how="left")
    missing = int(sample["content"].isna().sum())
    print(f"sampled {len(sample)} items over {sample.benchmark_id.nunique()} benchmarks; "
          f"text missing for {missing}", flush=True)
    sample = sample.dropna(subset=["content"]).reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    print("model loaded", flush=True)

    t0 = time.time()
    feats = entropy_features(model, tok, sample["content"].astype(str).tolist(), torch)
    secs = time.time() - t0
    out = pd.concat([sample.drop(columns=["content"]), feats], axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"lm-entropy-pilot-{len(out)}.parquet"
    out.to_parquet(path, index=False)
    per_item = secs / max(len(out), 1)
    print(f"done: {len(out)} items in {secs:.0f}s ({per_item:.2f}s/item); "
          f"extrapolated 22,000 items: {per_item * 22000 / 3600:.1f} h", flush=True)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
