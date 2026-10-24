"""Tests for canonical model display names."""

from __future__ import annotations

from aims_starter.model_names import DISPLAY_NAMES, display_name


def test_known_key_returns_display_name() -> None:
    assert display_name("cnp") == "CNP-AIRT"
    assert display_name("beta_binomial") == "Beta-Binomial"


def test_airt_suffix_is_stripped_and_mapped() -> None:
    """The faithful run stores "np"; the pilot run stores "np_airt"."""

    assert display_name("np_airt") == display_name("np")
    assert display_name("cnp_airt") == "CNP-AIRT"


def test_unknown_key_passes_through_unchanged() -> None:
    assert display_name("not_a_model") == "not_a_model"


def test_airt_suffix_with_unknown_prefix_falls_through() -> None:
    """The suffix branch requires the stripped prefix to be known.

    "unknown_airt" strips to "unknown", which is absent from DISPLAY_NAMES, so
    the lookup falls through and returns the original key verbatim rather than
    the stripped prefix.
    """

    assert display_name("unknown_airt") == "unknown_airt"


def test_key_that_is_itself_registered_with_airt_suffix_wins() -> None:
    """"tnpd_airt" is a dict key in its own right, not a suffixed alias."""

    assert "tnpd_airt" in DISPLAY_NAMES
    assert display_name("tnpd_airt") == "TfNP-D-AIRT"


def test_every_display_name_has_a_working_airt_variant() -> None:
    """Guards future DISPLAY_NAMES additions against suffix-handling drift."""

    for key, expected in DISPLAY_NAMES.items():
        assert display_name(key) == expected
        if not key.endswith("_airt"):
            assert display_name(f"{key}_airt") == expected
