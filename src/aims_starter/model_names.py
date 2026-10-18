"""Canonical display names for the measured model family.

Machine keys stay stable in stored metrics.json files; every plot, log table,
and paper draft renders the same human-facing names. Key namespaces differ
slightly between experiments: the faithful run stores "np", while the pilot
run stores "np_airt". ``display_name`` handles both.
"""

from __future__ import annotations

DISPLAY_NAMES = {
    "cnp": "CNP-AIRT",
    "np": "NP-AIRT",
    "anp": "ANP-AIRT",
    "stacked_mhsa_anp": "Stacked-MHSA-ANP-AIRT",
    "online_airt_2pl": "Online-AIRT",
    "tnpd_airt": "TfNP-D-AIRT",
    "beta_binomial": "Beta-Binomial",
    "tabpfn_v2": "TabPFN-v2",
}


def display_name(key: str) -> str:
    """Return the canonical display name for a stored model key."""

    if key.endswith("_airt") and key[:-5] in DISPLAY_NAMES:
        return DISPLAY_NAMES[key[:-5]]
    return DISPLAY_NAMES.get(key, key)
