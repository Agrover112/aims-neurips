"""Closed-form Beta-mixture predictors (R3, E1) and the pilot switch rule (R4, E3).

All predict one probability per task from its own revealed labels (h correct out of n); nothing is trained at test time.

    mixture:  components (w_k, a_k, b_k); posterior weight of k is proportional to w_k * BetaBinomial(h; n, a_k, b_k);
              prediction = sum_k posterior_k * (a_k + h) / (a_k + b_k + n)
    switch:   after at least kmin labels, use Jeffreys when |h/n - m| > d, otherwise the fitted Beta(m, s) prior
"""
from __future__ import annotations
import numpy as np
from scipy.special import betaln, logsumexp


def mixture_predict(comps, h: float, n: float) -> float:
    c = np.asarray(comps, float)
    w, a, b = c[:, 0], c[:, 1], c[:, 2]
    logp = np.log(w) + betaln(h + a, n - h + b) - betaln(a, b)
    post = np.exp(logp - logsumexp(logp))
    return float((post * (a + h) / (a + b + n)).sum())


def switch_predict(m: float, s: float, d: float, kmin: int, h: float, n: float) -> float:
    if n >= kmin and abs(h / n - m) > d:
        return (h + 0.5) / (n + 1.0)                       # Jeffreys
    return (m * s + h) / (s + n)                            # fitted prior


def grid_components(means, strengths, weights=None):
    """Uniform (or given) mixture over a grid of Beta(m*s, (1-m)*s): Bayesian averaging over mean and strength."""
    comps = [(m * s, (1 - m) * s) for m in means for s in strengths]
    w = np.full(len(comps), 1.0 / len(comps)) if weights is None else np.asarray(weights, float)
    return [[float(wk), a, b] for wk, (a, b) in zip(w, comps)]
