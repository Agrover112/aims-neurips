"""BGE-M3 item embeddings under the replay's contract.

Item content only, CLS pooling, max length 8192, L2-normalised, the same
settings as the BGE-M3 store the kernel was tuned on. Vectors are cached in
memory by text, so each item is encoded once per worker process.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

MODEL_ID = "BAAI/bge-m3"
MAX_LENGTH = 8192
DIM = 1024
SMOKE_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
# The organizer validator sets this only while importing model.py; capture it
# now so the interface smoke path stays active for that whole validation.
LOCAL_SMOKE_TEST = bool(os.environ.get(SMOKE_ENV))

_state: dict = {}
_cache: dict[str, np.ndarray] = {}


def _smoke(text: str) -> np.ndarray:
    # Interface-only path for the local validator; never used on CodaBench.
    raw = hashlib.shake_256(text.encode("utf-8")).digest(4 * DIM)
    v = np.frombuffer(raw, dtype=np.uint32).astype(np.float32) / np.float32(2 ** 31) - 1.0
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _load() -> None:
    if _state:
        return
    import torch
    from transformers import AutoModel, AutoTokenizer

    on_cuda = torch.cuda.is_available()
    _state["device"] = torch.device("cuda" if on_cuda else "cpu")
    if not on_cuda:
        try:
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        except (RuntimeError, ValueError):
            pass
    _state["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True,
                                      torch_dtype=torch.bfloat16 if on_cuda else torch.float32)
    model.eval()
    _state["model"] = model.to(_state["device"])
    _state["batch"] = 16 if on_cuda else 4


def _encode(texts: list[str]) -> np.ndarray:
    import torch

    tok, model, device = _state["tokenizer"], _state["model"], _state["device"]
    out = []
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for start in range(0, len(order), _state["batch"]):
        chunk = [texts[i] for i in order[start:start + _state["batch"]]]
        enc = tok(chunk, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True).to(device)
        with torch.inference_mode():
            vec = model(**enc).last_hidden_state[:, 0].float()
            vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
        out.append(vec.cpu().numpy())
    flat = np.concatenate(out)
    result = np.empty_like(flat)
    result[order] = flat
    return result


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalised embeddings, one row per text, cached by text."""
    keys = [hashlib.blake2b(t.encode("utf-8"), digest_size=16).hexdigest() for t in texts]
    missing = [i for i, k in enumerate(keys) if k not in _cache]
    if missing:
        todo = [texts[i] for i in missing]
        if LOCAL_SMOKE_TEST:
            vecs = np.stack([_smoke(t) for t in todo])
        else:
            _load()
            vecs = _encode(todo)
        for i, v in zip(missing, vecs):
            _cache[keys[i]] = v.astype(np.float32)
    return np.stack([_cache[k] for k in keys])
