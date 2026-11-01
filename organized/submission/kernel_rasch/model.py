"""Kernel Rasch predictor for the AIMS online evaluation.

Pooled Rasch refit on every pair's revealed labels, with item difficulties
correlated by item-text similarity within a benchmark:

    logit p(s, i) = mu_b + theta_s - b_i
    mu_b ~ N(M0, 2^2),  theta_s ~ N(0, 1^2)
    b    ~ N(0, K),     K = tau^2 [(1 - rho) I + rho S],  S_ij = exp((cos(x_i, x_j) - 1) / T)

x_i is the BGE-M3 embedding of the item content (CLS pooling, L2-normalised,
the contract the replay used). Unlabelled items get the kriging estimate
b_i = K_iL K_LL^-1 b_L, so a label on a similar item moves them. Settings
were chosen on the benchmark-holdout validation replay; on simulated draws of
unseen test benchmarks this beat the beta-binomial by 0.0049 Brier-ALC and
pooled Rasch by 0.0013. See organized/testing/online/kernel_rasch_replay.py.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

M0 = 0.049      # logit of the training-benchmark label mean, 0.512
TAU_B = 2.0     # prior sd of a benchmark's base rate around M0
TAU_S = 1.0     # prior sd of subject ability
TAU_I = 1.4     # prior sd of item difficulty
RHO = 0.8       # share of difficulty variance shared between similar items
TEMP = 0.2      # similarity temperature
NEWTON_STEPS = 30

# --- BGE-M3 item embeddings (inlined: a separate module named "embedder" can be
# shadowed by a module of the same name in the evaluation environment) --------
EMB_MODEL_ID = "BAAI/bge-m3"
EMB_MAX_LENGTH = 8192
EMB_DIM = 1024
SMOKE_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
# The organizer validator sets this only while importing model.py; capture it
# now so the interface smoke path stays active for that whole validation.
LOCAL_SMOKE_TEST = bool(os.environ.get(SMOKE_ENV))

_state: dict = {}
_emb_cache: dict[str, np.ndarray] = {}


def _smoke(text: str) -> np.ndarray:
    # Interface-only path for the local validator; never used on CodaBench.
    raw = hashlib.shake_256(text.encode("utf-8")).digest(4 * EMB_DIM)
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
    _state["tokenizer"] = AutoTokenizer.from_pretrained(EMB_MODEL_ID, local_files_only=True)
    model = AutoModel.from_pretrained(EMB_MODEL_ID, local_files_only=True,
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
        enc = tok(chunk, return_tensors="pt", truncation=True, max_length=EMB_MAX_LENGTH, padding=True).to(device)
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
    missing = [i for i, k in enumerate(keys) if k not in _emb_cache]
    if missing:
        todo = [texts[i] for i in missing]
        if LOCAL_SMOKE_TEST:
            vecs = np.stack([_smoke(t) for t in todo])
        else:
            _load()
            vecs = _encode(todo)
        for i, v in zip(missing, vecs):
            _emb_cache[keys[i]] = v.astype(np.float32)
    return np.stack([_emb_cache[k] for k in keys])


_cache: dict[str, tuple] = {}


def _subject_key(subject) -> str:
    return json.dumps(subject, sort_keys=True, default=str)


def _text(item) -> str:
    return str((item or {}).get("item_content", "") or "")


def _item_key(item) -> str:
    item = item or {}
    payload = "\x1f".join(str(item.get(k, "")) for k in ("benchmark_id", "item_content", "item_features"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()


def _evidence(labeled):
    rows = []
    for entry in labeled or []:
        try:
            (subject, item), label = entry
        except (TypeError, ValueError):
            continue
        if label is None:
            continue
        rows.append(((item or {}).get("benchmark_id", ""), _subject_key(subject), _item_key(item),
                     _text(item), float(label)))
    return rows


def _kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.exp((a @ b.T - 1.0) / TEMP)


def _fit(rows):
    benches = sorted({r[0] for r in rows})
    subjects = sorted({r[1] for r in rows})
    items = sorted({(r[0], r[2]) for r in rows})
    text = {(r[0], r[2]): r[3] for r in rows}
    nb, ns, ni = len(benches), len(subjects), len(items)
    bi = {b: j for j, b in enumerate(benches)}
    si = {s: nb + j for j, s in enumerate(subjects)}
    ii = {it: nb + ns + j for j, it in enumerate(items)}
    n = nb + ns + ni
    X = np.zeros((len(rows), n))
    for r, (b, s, h, _, _) in enumerate(rows):
        X[r, bi[b]] = X[r, si[s]] = 1.0
        X[r, ii[(b, h)]] = -1.0
    y = np.array([r[4] for r in rows])
    prec = np.zeros((n, n))
    prec[np.arange(nb), np.arange(nb)] = TAU_B ** -2
    prec[np.arange(nb, nb + ns), np.arange(nb, nb + ns)] = TAU_S ** -2
    blocks = {}
    for b in benches:   # similarity only within a benchmark
        keys = [it for it in items if it[0] == b]
        idx = np.array([ii[it] for it in keys])
        vec = embed([text[it] for it in keys])
        K = TAU_I ** 2 * ((1 - RHO) * np.eye(len(keys)) + RHO * _kernel(vec, vec)) + 1e-6 * np.eye(len(keys))
        Kinv = np.linalg.inv(K)
        prec[np.ix_(idx, idx)] = Kinv
        blocks[b] = (keys, idx, vec, Kinv)
    mean = np.concatenate([np.full(nb, M0), np.zeros(ns + ni)])
    v = mean.copy()
    for _ in range(NEWTON_STEPS):
        p = 1.0 / (1.0 + np.exp(-(X @ v)))
        grad = X.T @ (p - y) + prec @ (v - mean)
        hess = (X * (p * (1 - p))[:, None]).T @ X + prec
        step = np.linalg.solve(hess, grad)
        v -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    mu = {b: v[j] for b, j in bi.items()}
    theta = {s: v[j] for s, j in si.items()}
    known = {it: v[j] for it, j in ii.items()}
    krig = {b: (vec, Kinv @ v[idx]) for b, (_, idx, vec, Kinv) in blocks.items()}
    return mu, theta, known, krig


def predict(input: list, labeled: list | None = None) -> float:
    """Probability that the subject in ``input`` answers its item correctly."""
    try:
        subject, item = input
        rows = _evidence(labeled)
        if not rows:
            return float(1.0 / (1.0 + np.exp(-M0)))
        key = hashlib.blake2b(repr(sorted((r[0], r[1], r[2], r[4]) for r in rows)).encode("utf-8"),
                              digest_size=16).hexdigest()
        if key not in _cache:
            _cache.clear()  # one checkpoint at a time; keep memory flat
            _cache[key] = _fit(rows)
        mu, theta, known, krig = _cache[key]
        b = (item or {}).get("benchmark_id", "")
        ik = (b, _item_key(item))
        if ik in known:
            diff = known[ik]
        elif b in krig:
            vec, alpha = krig[b]
            diff = float(TAU_I ** 2 * RHO * (_kernel(embed([_text(item)]), vec) @ alpha)[0])
        else:
            diff = 0.0
        z = mu.get(b, M0) + theta.get(_subject_key(subject), 0.0) - diff
        return float(min(max(1.0 / (1.0 + np.exp(-z)), 1e-4), 1 - 1e-4))
    except Exception:
        # Never fail a call: fall back to the training base rate.
        return float(1.0 / (1.0 + np.exp(-M0)))
