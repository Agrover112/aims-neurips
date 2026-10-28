#!/usr/bin/env python3
"""Build CodaBench archives for the completed BGE-M3 ANP runs."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "organized" / "submission"
OUTPUT = ROOT / "dist"

VARIANTS = {
    "anp-sa1-w64-bge-m3-long-random": {
        "checkpoint": "anp-sa1-w64-bgem3-20261028-bge_m3_long-seed20261018.pt",
        "run_label": "anp-sa1-w64-bgem3-20261028",
        "width": 64,
        "latent_dim": 32,
        "max_context": 31,
    },
    "anp-sa1-w128-bge-m3-long-random": {
        "checkpoint": "anp128-bgem3-l40s-100k-s20261018-bge_m3_long-seed20261018.pt",
        "run_label": "anp128-bgem3-l40s-100k-s20261018",
        "width": 128,
        "latent_dim": 128,
        "max_context": 64,
    },
}


def _metadata(variant: dict[str, object]) -> dict[str, object]:
    return {
        "checkpoint": variant["checkpoint"],
        "embedding": {
            "model": "BAAI/bge-m3",
            "item_text": "content_only",
            "native_dimension": 1024,
            "output_dimension": 1024,
            "max_length": 8192,
            "pooling": "cls",
            "l2_normalized": True,
        },
        "model": {
            "family": "anp_cross_attention",
            "width": variant["width"],
            "latent_dimension": variant["latent_dim"],
            "cross_attention_heads": 4,
            "context_self_attention_blocks": 1,
            "max_context": variant["max_context"],
        },
        "acquisition": "random_streaming_with_reserve_rule",
        "provenance": {
            "run_label": variant["run_label"],
            "seed": 20261018,
            "training_arm": "bge_m3_long",
            "checkpoint_selection": "fixed_validation_unseen_target_brier",
        },
    }


def _stage(staged: Path, variant: dict[str, object]) -> None:
    for name in ("model.py", "labeling.py", "requirements.txt", "prediction_config.json"):
        shutil.copy2(SOURCE / name, staged / name)
    (staged / "models.txt").write_text("BAAI/bge-m3\n")

    assets = staged / "assets"
    assets.mkdir()
    checkpoint = str(variant["checkpoint"])
    shutil.copy2(SOURCE / "assets" / checkpoint, assets / checkpoint)
    (assets / "metadata.json").write_text(json.dumps(_metadata(variant), indent=2) + "\n")

    package = staged / "aims_predictor"
    package.mkdir()
    for name in ("__init__.py", "predictor.py"):
        shutil.copy2(SOURCE / "aims_predictor" / name, package / name)
    shutil.copy2(ROOT / "organized" / "training" / "cnp_blocks.py", package / "cnp_blocks.py")
    shutil.copy2(
        ROOT / "organized" / "training" / "anp_crossattn_blocks.py",
        package / "anp_crossattn_blocks.py",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, variant in VARIANTS.items():
        checkpoint = SOURCE / "assets" / str(variant["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing completed checkpoint: {checkpoint}")
        with tempfile.TemporaryDirectory(prefix=f"{name}-") as temporary:
            staged = Path(temporary)
            _stage(staged, variant)
            archive = OUTPUT / f"{name}.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
                for path in sorted(staged.rglob("*")):
                    if path.is_file():
                        output.write(path, path.relative_to(staged))
            print(archive)


if __name__ == "__main__":
    main()
