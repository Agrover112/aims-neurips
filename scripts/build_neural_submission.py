#!/usr/bin/env python3
"""Build a neural submission ZIP from a trained CNP-AIRT or Online-AIRT checkpoint.

The model code is vendored from ``src/aims_starter`` at build time so the ZIP
always matches the checkpoint's training code. Run the reference kit checks on
the resulting ZIP before uploading.
"""

from __future__ import annotations

import argparse
import json
import py_compile
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "submission_neural"
MODULES = ("faithful_np.py", "metrics.py", "calibration.py")


def _calibration_entries(path: Path, model: str) -> list[dict]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    fitted = artifact["fitted"][model]
    return [
        {
            "budget": int(entry["budget"]),
            "scale": float(entry["scale"]),
            "bias": float(entry["bias"]),
            "anchor": float(entry["anchor"]),
            "samples": int(entry["samples"]),
            "fitted": bool(entry["fitted"]),
        }
        for entry in fitted
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("masked", "visible"), required=True)
    parser.add_argument("--x-dim", type=int, default=768)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--model", default="cnp_airt")
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--checkpoint-name")
    parser.add_argument("--labeling", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/submission-neural.zip")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    template = args.template.resolve()
    if not (template / "model.py").is_file():
        raise SystemExit(f"template not found: {template}")
    labeling = args.labeling.resolve() if args.labeling else None
    if labeling is not None and not labeling.is_file():
        raise SystemExit(f"labeling not found: {labeling}")
    checkpoint_name = args.checkpoint_name or checkpoint.name
    calibration = _calibration_entries(args.calibration, args.model) if args.calibration else None

    with tempfile.TemporaryDirectory(prefix="aims-neural-package-") as temporary:
        staged = Path(temporary) / "submission"
        shutil.copytree(template, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if labeling is not None:
            shutil.copyfile(labeling, staged / "labeling.py")
        package = staged / "aims_predictor"
        for module in MODULES:
            shutil.copyfile(ROOT / "src" / "aims_starter" / module, package / module)

        assets = staged / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint, assets / checkpoint_name)
        metadata = {
            "architecture": args.model,
            "checkpoint": checkpoint_name,
            "mode": args.mode,
            "x_dim": args.x_dim,
            "subject_slice": [args.x_dim // 2, args.x_dim],
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "calibration": calibration,
            "source_checkpoint": checkpoint.name,
        }
        (assets / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        for path in staged.rglob("*.py"):
            py_compile.compile(str(path), doraise=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staged.rglob("*")):
                if not path.is_file():
                    continue
                if "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                archive.write(path, path.relative_to(staged))
    print(args.output)


if __name__ == "__main__":
    main()
