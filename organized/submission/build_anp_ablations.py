#!/usr/bin/env python3
"""Build local-only CodaBench ZIPs for the saved ANP ablations.

Every archive uses the shared, deliberately random streaming acquisition
policy. This script does not contact CodaBench or create a submission.
"""

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
    "anp-ca-only-128-qwen3-8b-random": {
        "checkpoint": "anp-ca-only-128-qwen3-8b-seed20261018.pt",
        "description": "ANP, cross-attention only (zero context self-attention blocks).",
        "self_attention_blocks": 0,
    },
    "anp-sa1-128-qwen3-8b-random": {
        "checkpoint": "anp-sa1-128-qwen3-8b-seed20261018.pt",
        "description": "ANP, one context self-attention block before cross-attention.",
        "self_attention_blocks": 1,
    },
    "stacked-anp-sa2-128-qwen3-8b-random": {
        "checkpoint": "stacked-anp-sa2-128-qwen3-8b-seed20261018.pt",
        "description": "Stacked ANP, two context self-attention blocks before cross-attention.",
        "self_attention_blocks": 2,
    },
}


def _metadata(variant: dict[str, object]) -> dict[str, object]:
    return {
        "checkpoint": variant["checkpoint"],
        "embedding": {
            "model": "Qwen/Qwen3-Embedding-8B",
            "item_text": "content_only",
            "native_dimension": 4096,
            "mrl_dimension": 1024,
            "pooling": "last_token",
            "l2_normalized": True,
        },
        "model": {
            "family": "anp_cross_attention",
            "width": 128,
            "latent_dimension": 128,
            "cross_attention_heads": 4,
            "context_self_attention_blocks": variant["self_attention_blocks"],
            "max_context": 64,
        },
        "acquisition": "random_streaming_with_reserve_rule",
        "provenance_note": (
            "In-training best checkpoint from the filtered ANP ablation; "
            "not a fully completed/refit final model."
        ),
    }


def _stage_common(staged: Path, checkpoint: str, metadata: dict[str, object]) -> None:
    for name in ("model.py", "labeling.py", "models.txt", "requirements.txt", "prediction_config.json"):
        shutil.copy2(SOURCE / name, staged / name)
    assets = staged / "assets"
    assets.mkdir()
    shutil.copy2(SOURCE / "assets" / checkpoint, assets / checkpoint)
    (assets / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    package = staged / "aims_predictor"
    package.mkdir()
    for name in ("__init__.py", "predictor.py"):
        shutil.copy2(SOURCE / "aims_predictor" / name, package / name)
    shutil.copy2(ROOT / "organized" / "training" / "cnp_blocks.py", package / "cnp_blocks.py")
    shutil.copy2(ROOT / "organized" / "training" / "anp_crossattn_blocks.py", package / "anp_crossattn_blocks.py")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, variant in VARIANTS.items():
        checkpoint = str(variant["checkpoint"])
        if not (SOURCE / "assets" / checkpoint).is_file():
            raise FileNotFoundError(f"Missing checkpoint asset: {checkpoint}")
        with tempfile.TemporaryDirectory(prefix=f"{name}-") as temporary:
            staged = Path(temporary)
            _stage_common(staged, checkpoint, _metadata(variant))
            archive = OUTPUT / f"{name}.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
                for path in sorted(staged.rglob("*")):
                    if path.is_file():
                        zip_file.write(path, path.relative_to(staged))
            print(archive)


if __name__ == "__main__":
    main()
