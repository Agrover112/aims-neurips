from pathlib import Path

from aims_starter.caching import content_signature


def test_signature_changes_with_source_and_config(tmp_path: Path) -> None:
    source = tmp_path / "train.py"
    source.write_text("steps = 1\n", encoding="utf-8")
    first = content_signature([source], {"steps": 1})
    source.write_text("steps = 2\n", encoding="utf-8")
    second = content_signature([source], {"steps": 1})
    third = content_signature([source], {"steps": 2})
    assert first != second
    assert second != third


def test_signature_is_config_order_independent(tmp_path: Path) -> None:
    source = tmp_path / "train.py"
    source.write_text("steps = 1\n", encoding="utf-8")
    first = content_signature([source], {"steps": 1, "seed": 7})
    second = content_signature([source], {"seed": 7, "steps": 1})
    assert first == second
