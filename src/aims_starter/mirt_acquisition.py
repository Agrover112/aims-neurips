"""Pure acquisition helpers retained after the root Modal runner was removed.

These formulas are unchanged from ``modal_mirt_acquisition.py`` at commit
``c84f2fdda2c67de6dd12673e14a48cfcca69a41a``. Keeping them here lets numerical
tests run without importing or restoring a retired Modal experiment.
"""

from __future__ import annotations

import re

import numpy as np


def _rung_config(rung: str) -> tuple[int, float, int, float]:
    """Read ability and online difficulty priors from a MIRT rung name."""
    dimension = re.search(r"_d(\d+)(?:_|$)", rung)
    if dimension is None:
        raise ValueError(f"MIRT rung does not encode an ability dimension: {rung}")
    tau = re.search(r"_tau([0-9]+(?:\.[0-9]+)?)(?:_|$)", rung)
    adapter = re.search(r"_jointb(\d+)(?:_|$)", rung)
    adapter_tau = re.search(r"_btau([0-9]+(?:\.[0-9]+)?)(?:_|$)", rung)
    return (
        int(dimension.group(1)),
        float(tau.group(1)) if tau else 4.0,
        int(adapter.group(1)) if adapter else 0,
        float(adapter_tau.group(1)) if adapter_tau else 1.0,
    )


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def _posterior_covariance(design_selected, d_selected, state, prior_variance):
    """Laplace covariance of an ability or joint ability/difficulty state.

    The joint design has rows ``[a_i, -u_i]`` and state ``[theta, z]``.
    Curvature is ``X.T @ diag(p * (1-p)) @ X + diag(1 / prior_variance)``.
    """
    dim = state.shape[0]
    eye = np.eye(dim)
    prior_variance = np.broadcast_to(np.asarray(prior_variance, dtype=float), (dim,))
    if design_selected.shape[0] == 0:
        return np.diag(prior_variance)
    probability = _sigmoid(design_selected @ state - d_selected)
    weight = probability * (1.0 - probability)
    hessian = (design_selected.T * weight) @ design_selected
    hessian = hessian + np.diag(1.0 / prior_variance)
    return np.linalg.inv(hessian + 1e-6 * eye)
