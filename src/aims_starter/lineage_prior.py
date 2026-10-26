"""Subject-lineage prior for the ability vector.

A new subject arrives with no labels at budget 0 and only a handful after, so
the Newton MAP is dominated by its prior. The default prior is N(0, tau^2 I)
for every subject. Models from the same family (a base model and its
fine-tunes, or one model under two harnesses) have similar ability, so the
prior mean can be taken from training subjects that share a lineage instead
of from zero.

The estimate is a three-level partial pool

    global mean  <-  provider mean  <-  family (normalized_name) mean

where each level is shrunk toward its parent by ``strength`` pseudo-tasks:

    mean_level = (n * observed + strength * parent) / (n + strength).

A family seen once barely moves off its provider mean; a family seen forty
times is trusted. A subject with no name, or a name never seen in training,
falls back one level at a time, and a subject with neither returns the global
mean. Nothing here sees evaluation labels: it is fitted on training-task
abilities only.
"""

from __future__ import annotations

import numpy as np

UNKNOWN = ""


def _clean(value) -> str:
    return str(value or "").strip().lower()


class LineagePrior:
    """Partial-pooled prior mean over (provider, family) keys."""

    def __init__(self, strength: float = 4.0):
        if strength < 0.0:
            raise ValueError("strength must be non-negative")
        self.strength = float(strength)
        self.global_mean: np.ndarray | None = None
        self.provider_mean: dict[str, np.ndarray] = {}
        self.family_mean: dict[tuple[str, str], np.ndarray] = {}
        self.provider_count: dict[str, int] = {}
        self.family_count: dict[tuple[str, str], int] = {}

    def fit(self, families, providers, thetas) -> LineagePrior:
        """``thetas`` is (n_tasks, D); ``families``/``providers`` are per task."""

        thetas = np.asarray(thetas, dtype=float)
        if thetas.ndim != 2 or len(thetas) == 0:
            raise ValueError("thetas must be a non-empty (n, D) array")
        if not (len(families) == len(providers) == len(thetas)):
            raise ValueError("families, providers and thetas must have equal length")
        families = [_clean(f) for f in families]
        providers = [_clean(p) for p in providers]

        self.global_mean = thetas.mean(axis=0)
        k = self.strength

        by_provider: dict[str, list[int]] = {}
        for index, provider in enumerate(providers):
            if provider:
                by_provider.setdefault(provider, []).append(index)
        for provider, rows in by_provider.items():
            n = len(rows)
            observed = thetas[rows].mean(axis=0)
            self.provider_mean[provider] = (n * observed + k * self.global_mean) / (n + k)
            self.provider_count[provider] = n

        by_family: dict[tuple[str, str], list[int]] = {}
        for index, family in enumerate(families):
            if family:
                by_family.setdefault((providers[index], family), []).append(index)
        for key, rows in by_family.items():
            n = len(rows)
            observed = thetas[rows].mean(axis=0)
            parent = self.provider_mean.get(key[0], self.global_mean)
            self.family_mean[key] = (n * observed + k * parent) / (n + k)
            self.family_count[key] = n
        return self

    def level(self, family, provider) -> str:
        family, provider = _clean(family), _clean(provider)
        if (provider, family) in self.family_mean:
            return "family"
        if provider in self.provider_mean:
            return "provider"
        return "global"

    def predict(self, family, provider) -> np.ndarray:
        if self.global_mean is None:
            raise RuntimeError("fit the prior before predicting")
        family, provider = _clean(family), _clean(provider)
        if (provider, family) in self.family_mean:
            return self.family_mean[(provider, family)].copy()
        if family and not provider:
            # name without provider: pool every family of that name
            rows = [v for (p, f), v in self.family_mean.items() if f == family]
            if rows:
                return np.mean(rows, axis=0)
        if provider in self.provider_mean:
            return self.provider_mean[provider].copy()
        return self.global_mean.copy()
