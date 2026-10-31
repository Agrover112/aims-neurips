"""Pooled hierarchical Rasch predictor for the AIMS online evaluation.

The platform passes every label purchased so far across all sampled
subject-benchmark pairs in ``labeled``. This predictor uses all of it:

    logit p(s, i) = mu_b + theta_s - b_h

    mu_b     benchmark base rate, pooled over every pair on that benchmark
    theta_s  subject ability, from that subject's labels on any benchmark
    b_h      item difficulty, from other subjects' labels on the same item
             (matched by content); 0 for an item nobody has labelled

The three are MAP estimates under Gaussian priors, refitted from ``labeled``
on each call (cached per distinct evidence set). With no evidence it predicts
sigmoid(M0), the training-benchmark base rate. Prior scales were chosen on the
benchmark-held-out validation replay: replayed on simulated draws of unseen
test benchmarks it beat the beta-binomial by 0.0035 Brier-ALC.
See organized/testing/online/modal_pooled_rasch.py.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

M0 = 0.049     # logit of the training-benchmark label mean, 0.512
TAU_B = 2.0    # prior sd of a benchmark's base rate around M0
TAU_S = 1.0    # prior sd of subject ability
TAU_I = 1.0    # prior sd of item difficulty
NEWTON_STEPS = 25

_cache: dict[str, tuple[dict, dict, dict]] = {}


def _subject_key(subject) -> str:
    return json.dumps(subject, sort_keys=True, default=str)


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
        rows.append(((item or {}).get("benchmark_id", ""), _subject_key(subject), _item_key(item), float(label)))
    return rows


def _fit(rows):
    """Newton's method on the convex MAP objective; returns three lookup dicts."""
    benches = sorted({r[0] for r in rows})
    subjects = sorted({r[1] for r in rows})
    items = sorted({r[2] for r in rows})
    nb, ns, ni = len(benches), len(subjects), len(items)
    bi = {b: j for j, b in enumerate(benches)}
    si = {s: nb + j for j, s in enumerate(subjects)}
    ii = {h: nb + ns + j for j, h in enumerate(items)}
    n = nb + ns + ni
    # Design matrix: +1 for benchmark and subject, -1 for item difficulty.
    X = np.zeros((len(rows), n))
    for r, (b, s, h, _) in enumerate(rows):
        X[r, bi[b]] = 1.0
        X[r, si[s]] = 1.0
        X[r, ii[h]] = -1.0
    y = np.array([r[3] for r in rows])
    prior_mean = np.concatenate([np.full(nb, M0), np.zeros(ns + ni)])
    prior_prec = np.concatenate([np.full(nb, TAU_B ** -2), np.full(ns, TAU_S ** -2), np.full(ni, TAU_I ** -2)])
    v = prior_mean.copy()
    for _ in range(NEWTON_STEPS):
        p = 1.0 / (1.0 + np.exp(-(X @ v)))
        grad = X.T @ (p - y) + prior_prec * (v - prior_mean)
        hess = (X * (p * (1 - p))[:, None]).T @ X + np.diag(prior_prec)
        step = np.linalg.solve(hess, grad)
        v -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return ({b: v[j] for b, j in bi.items()}, {s: v[j] for s, j in si.items()}, {h: v[j] for h, j in ii.items()})


def predict(input: list, labeled: list | None = None) -> float:
    """Probability that the subject in ``input`` answers its item correctly."""
    try:
        subject, item = input
        rows = _evidence(labeled)
        if not rows:
            return float(1.0 / (1.0 + np.exp(-M0)))
        key = hashlib.blake2b(repr(sorted(rows)).encode("utf-8"), digest_size=16).hexdigest()
        if key not in _cache:
            _cache.clear()  # one checkpoint at a time; keep memory flat
            _cache[key] = _fit(rows)
        mu, theta, diff = _cache[key]
        z = (mu.get((item or {}).get("benchmark_id", ""), M0) + theta.get(_subject_key(subject), 0.0)
             - diff.get(_item_key(item), 0.0))
        return float(min(max(1.0 / (1.0 + np.exp(-z)), 1e-4), 1 - 1e-4))
    except Exception:
        # Never fail a call: fall back to the training base rate.
        return float(1.0 / (1.0 + np.exp(-M0)))
