"""Fit a 1024x1024 ridge map from a cheap embedding arm into qwen3_8b space.

The staged latent-CNP-128 checkpoint's weights are fit to the geometry of
``allbench-embed-qwen3_8b-v2``. Feeding it vectors from another arm loads
without error and silently produces nonsense, because the dimensions match and
the spaces do not. This job fits the linear map that makes a substitution
legitimate::

    W = (X^T X + lambda I)^-1 X^T Y

with ``X`` the source arm's item vectors and ``Y`` the qwen3_8b vectors for the
same items. Both stores cover the identical 1,356,312-item universe with the
same ``content_only`` text, so every row is an exact pair and no alignment
heuristic is needed.

The two 1024x1024 Gram matrices are accumulated in chunks, so neither store is
ever held in memory whole. The solve itself is a 1024x1024 system.

**Items are split, not rows.** Runtime needs this map to generalise to items the
fit never saw, so the held-out fold is a disjoint set of ``item_uid``s selected
by hash. Reporting in-sample error here would be meaningless.

The reported number to care about is held-out **cosine** between the mapped
source vector and the true qwen vector, after renormalising, since the
checkpoint consumes L2-normalised inputs. The raw cross-arm cosine is reported
alongside it as the do-nothing baseline.

Run::

    modal run --profile aims-test \
      organized/submission/modal_fit_ridge_adapter.py::main --source-arm me5_instruct
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "aims-np-ridge-adapter"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
EMBED_ROOT = REMOTE_ROOT / "datasets" / "full"
TARGET_ARM = "qwen3_8b"
OUT_ROOT = REMOTE_ROOT / "results" / "submission" / "ridge-adapter"

DIM = 1024
CHUNK = 65_536
HELD_OUT_FRAC = 0.10
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24"
)


def _store(arm: str) -> Path:
    return EMBED_ROOT / f"allbench-embed-{arm}-v2"


@app.function(
    image=image,
    cpu=4,
    memory=32_768,
    volumes={"/data": volume},
    timeout=60 * 60,
)
def fit(source_arm: str = "me5_instruct", seed: int = 20261028) -> dict:
    import hashlib

    import numpy as np
    import pandas as pd

    source_root, target_root = _store(source_arm), _store(TARGET_ARM)
    source_ids = pd.read_parquet(source_root / "items.parquet")["item_uid"].astype(str).to_numpy()
    target_ids = pd.read_parquet(target_root / "items.parquet")["item_uid"].astype(str).to_numpy()

    # Both stores are written from the same dedup-and-sort pass, so the rows
    # should already correspond. Assert it rather than assume it: a silent
    # off-by-one here would produce a plausible-looking but worthless map.
    aligned = source_ids.shape == target_ids.shape and bool((source_ids == target_ids).all())
    if not aligned:
        raise RuntimeError(
            f"row order differs: {source_arm}={source_ids.shape} {TARGET_ARM}={target_ids.shape}; "
            "an explicit join would be required"
        )
    print(f"aligned on {len(source_ids):,} items", flush=True)

    X = np.load(source_root / "item_embeddings.npy", mmap_mode="r")
    Y = np.load(target_root / "item_embeddings.npy", mmap_mode="r")
    if X.shape[1] != DIM or Y.shape[1] != DIM:
        raise RuntimeError(f"expected {DIM}-d stores, got {X.shape[1]} and {Y.shape[1]}")

    # Hash the uid so the split is reproducible and independent of row order.
    def _is_held_out(uid: str) -> bool:
        digest = hashlib.blake2b(f"{seed}:{uid}".encode("utf-8"), digest_size=8).digest()
        return (int.from_bytes(digest, "big") / float(1 << 64)) < HELD_OUT_FRAC

    held_out = np.fromiter((_is_held_out(u) for u in source_ids), dtype=bool, count=len(source_ids))
    n_train = int((~held_out).sum())
    print(f"train {n_train:,} items / held-out {int(held_out.sum()):,} items", flush=True)

    # ---- accumulate the Gram matrices over training items only -------------
    A = np.zeros((DIM, DIM), dtype=np.float64)
    B = np.zeros((DIM, DIM), dtype=np.float64)
    for start in range(0, len(source_ids), CHUNK):
        stop = min(start + CHUNK, len(source_ids))
        keep = ~held_out[start:stop]
        if not keep.any():
            continue
        x = np.asarray(X[start:stop], dtype=np.float64)[keep]
        y = np.asarray(Y[start:stop], dtype=np.float64)[keep]
        A += x.T @ x
        B += x.T @ y
        if (start // CHUNK) % 5 == 0:
            print(f"  gram {stop:,}/{len(source_ids):,}", flush=True)

    # ---- solve for each lambda, score on held-out items --------------------
    scale = float(np.trace(A) / DIM)  # keep lambda comparable across arms
    identity = np.eye(DIM, dtype=np.float64)
    results = []
    for lam in LAMBDAS:
        W = np.linalg.solve(A + lam * scale * identity, B)
        cos_sum, raw_sum, sq_err, sq_tot, count = 0.0, 0.0, 0.0, 0.0, 0
        for start in range(0, len(source_ids), CHUNK):
            stop = min(start + CHUNK, len(source_ids))
            keep = held_out[start:stop]
            if not keep.any():
                continue
            x = np.asarray(X[start:stop], dtype=np.float64)[keep]
            y = np.asarray(Y[start:stop], dtype=np.float64)[keep]
            predicted = x @ W
            norms = np.linalg.norm(predicted, axis=1, keepdims=True)
            unit = predicted / np.clip(norms, 1e-12, None)
            # Both y and unit are L2-normalised, so the dot product is cosine.
            cos_sum += float((unit * y).sum())
            raw_sum += float((x * y).sum())  # do-nothing baseline
            sq_err += float(((unit - y) ** 2).sum())
            sq_tot += float((y**2).sum())
            count += int(keep.sum())
        results.append(
            {
                "lambda": lam,
                "held_out_cosine": round(cos_sum / max(count, 1), 5),
                "raw_cross_arm_cosine": round(raw_sum / max(count, 1), 5),
                "held_out_r2_unit": round(1.0 - sq_err / max(sq_tot, 1e-12), 5),
                "n_held_out": count,
            }
        )
        print(f"  lambda={lam:<8g} cosine={results[-1]['held_out_cosine']:.4f}", flush=True)

    best = max(results, key=lambda r: r["held_out_cosine"])
    W = np.linalg.solve(A + best["lambda"] * scale * identity, B).astype(np.float32)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(OUT_ROOT / f"W-{source_arm}-to-{TARGET_ARM}.npy", W)
    summary = {
        "source_arm": source_arm,
        "target_arm": TARGET_ARM,
        "dim": DIM,
        "n_items": int(len(source_ids)),
        "n_train": n_train,
        "held_out_frac": HELD_OUT_FRAC,
        "seed": seed,
        "best": best,
        "sweep": results,
        "weight_file": f"W-{source_arm}-to-{TARGET_ARM}.npy",
    }
    (OUT_ROOT / f"metrics-{source_arm}.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()

    print("\n" + "=" * 64, flush=True)
    print(f"best lambda:          {best['lambda']}", flush=True)
    print(f"held-out cosine:      {best['held_out_cosine']}", flush=True)
    print(f"do-nothing baseline:  {best['raw_cross_arm_cosine']}", flush=True)
    print(f"held-out R2 (unit):   {best['held_out_r2_unit']}", flush=True)
    print("=" * 64, flush=True)
    return summary


@app.local_entrypoint()
def main(source_arm: str = "me5_instruct", seed: int = 20261028) -> None:
    fit.remote(source_arm=source_arm, seed=seed)
