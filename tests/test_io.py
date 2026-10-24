"""Tests for the dependency-free experiment-record helpers."""

from __future__ import annotations

import json

import pytest

from aims_starter.io import read_json, write_json


def test_round_trip_preserves_nested_value(tmp_path) -> None:
    value = {"alc": [0.1, 0.2], "meta": {"seed": 20261018, "mode": "masked"}, "ok": True}
    path = tmp_path / "record.json"

    write_json(path, value)
    assert read_json(path) == value


def test_write_creates_missing_parent_directories(tmp_path) -> None:
    path = tmp_path / "results" / "nested" / "record.json"
    write_json(path, {"a": 1})

    assert path.exists()
    assert read_json(path) == {"a": 1}


def test_write_leaves_no_temp_file_behind(tmp_path) -> None:
    """The atomic write renames its .tmp sibling rather than leaving it."""

    path = tmp_path / "record.json"
    write_json(path, {"a": 1})

    assert path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_is_byte_deterministic_regardless_of_key_order(tmp_path) -> None:
    """sort_keys=True means insertion order cannot change the stored bytes."""

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_json(first, {"b": 2, "a": 1, "c": 3})
    write_json(second, {"c": 3, "a": 1, "b": 2})

    assert first.read_bytes() == second.read_bytes()


def test_write_overwrites_an_existing_file(tmp_path) -> None:
    path = tmp_path / "record.json"
    write_json(path, {"version": 1})
    write_json(path, {"version": 2})

    assert read_json(path) == {"version": 2}


def test_written_file_ends_with_a_newline(tmp_path) -> None:
    path = tmp_path / "record.json"
    write_json(path, {"a": 1})

    assert path.read_text(encoding="utf-8").endswith("\n")


def test_read_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "absent.json")


def test_read_rejects_malformed_json(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_json(path)
