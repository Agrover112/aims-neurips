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
    "benchmark-holdout-anp-w64-bhold-budgetladder": {
        "checkpoint": (
            "anp-crossattn-latent-w64-sa1-h4-z32-benchmark-heldout-es-fixed-"
            "anp-sa1-w64-bgem3-bhold-budgetladder-n15-20261029-"
            "bge_m3_long-seed20261018.pt"
        ),
        "run_label": "anp-sa1-w64-bgem3-bhold-budgetladder-n15-20261029",
        "width": 64,
        "latent_dim": 32,
        "max_context": 31,
        "benchmark_holdout": True,
    },
    "benchmark-holdout-anp-w64-sam005": {
        "checkpoint": (
            "anp-crossattn-latent-w64-sa1-h4-z32-benchmark-heldout-es-fixed-"
            "anp-sa1-w64-bgem3-bhold-sam005-unweighted-n15-cpu-20261029-"
            "bge_m3_long-seed20261018.pt"
        ),
        "run_label": "anp-sa1-w64-bgem3-bhold-sam005-unweighted-n15-cpu-20261029",
        "width": 64,
        "latent_dim": 32,
        "max_context": 31,
        "benchmark_holdout": True,
        "training": {"sam_rho": 0.05, "selected_step": 5450,
                     "context_size_sampling": "uniform_scored_budget_ladder"},
    },
    "benchmark-holdout-anp-w64-denseprefix": {
        # Frozen copy of the dense-prefix job's best checkpoint; the live file
        # is overwritten whenever that job improves.
        "checkpoint": (
            "anp-crossattn-latent-w64-sa1-h4-z32-denseprefix-unweighted-"
            "best-step4550-bge_m3_long-seed20261018.pt"
        ),
        "run_label": "anp-sa1-w64-bgem3-bhold-denseprefix-unweighted-n15-l4-20261030",
        "width": 64,
        "latent_dim": 32,
        "max_context": 31,
        "benchmark_holdout": True,
        "training": {"dense_prefix_training": True, "selected_step": 4550,
                     "context_size_sampling": "all_stream_prefixes_0_to_max_context"},
    },
    "benchmark-holdout-anp-w64-filtered": {
        "checkpoint": (
            "anp-crossattn-latent-w64-sa1-h4-z32-benchmark-heldout-es-fixed-"
            "anp-sa1-w64-bgem3-filtered-benchmarkholdout-n15-20261029-"
            "bge_m3_long-seed20261018.pt"
        ),
        "run_label": "anp-sa1-w64-bgem3-filtered-benchmarkholdout-n15-20261029",
        "width": 64,
        "latent_dim": 32,
        "max_context": 31,
        "benchmark_holdout": True,
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
            "benchmark_holdout": bool(variant.get("benchmark_holdout", False)),
            "filtering": (
                "per-benchmark iterative core: nonconstant items with >=15 responders; "
                "subjects with >=80 retained items; masked missing responses"
                if variant.get("benchmark_holdout", False) else None
            ),
            "training": variant.get("training"),
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="",
                        help="comma-separated variant names; default builds all")
    args = parser.parse_args()
    wanted = set(filter(None, args.only.split(",")))
    unknown = wanted - set(VARIANTS)
    if unknown:
        raise SystemExit(f"unknown variants: {sorted(unknown)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, variant in VARIANTS.items():
        if wanted and name not in wanted:
            continue
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
