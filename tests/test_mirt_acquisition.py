from __future__ import annotations

import numpy as np

from aims_starter.mirt_acquisition import _posterior_covariance, _rung_config


def test_joint_rung_config() -> None:
    assert _rung_config("mirt_scalar_d4_jointb4_tau1") == (4, 1.0, 4, 1.0)
    assert _rung_config("mirt_scalar_d4") == (4, 4.0, 0, 1.0)


def test_joint_posterior_covariance_uses_per_dimension_priors() -> None:
    design = np.empty((0, 3))
    covariance = _posterior_covariance(
        design,
        np.empty(0),
        np.zeros(3),
        np.array([1.0, 1.0, 0.25]),
    )

    np.testing.assert_allclose(covariance, np.diag([1.0, 1.0, 0.25]))


def test_observation_reduces_joint_posterior_volume() -> None:
    prior = np.ones(3)
    empty = _posterior_covariance(np.empty((0, 3)), np.empty(0), np.zeros(3), prior)
    posterior = _posterior_covariance(
        np.array([[1.0, 0.5, -1.0]]),
        np.zeros(1),
        np.zeros(3),
        prior,
    )

    assert np.linalg.det(posterior) < np.linalg.det(empty)
