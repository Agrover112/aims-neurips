#!/usr/bin/env python3
"""Build the staged latent-CNP-128 CodaBench ZIP without changing source files."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "organized" / "submission"
OUTPUT = ROOT / "dist" / "latent-cnp128-qwen3-8b-random.zip"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="latent-cnp128-submission-") as temporary:
        staged = Path(temporary)
        for name in ("model.py", "labeling.py", "models.txt", "requirements.txt", "prediction_config.json"):
            shutil.copy2(SOURCE / name, staged / name)
        metadata = json.loads((SOURCE / "assets" / "metadata.json").read_text())
        assets = staged / "assets"
        assets.mkdir()
        shutil.copy2(SOURCE / "assets" / "metadata.json", assets / "metadata.json")
        shutil.copy2(SOURCE / "assets" / metadata["checkpoint"], assets / metadata["checkpoint"])
        package = staged / "aims_predictor"
        package.mkdir()
        shutil.copy2(SOURCE / "aims_predictor" / "__init__.py", package / "__init__.py")
        shutil.copy2(SOURCE / "aims_predictor" / "predictor.py", package / "predictor.py")
        # Vendor both supported architectures. The predictor selects exactly
        # one from checkpoint metadata at runtime.
        shutil.copy2(ROOT / "organized" / "training" / "cnp_blocks.py", package / "cnp_blocks.py")
        shutil.copy2(ROOT / "organized" / "training" / "anp_crossattn_blocks.py", package / "anp_crossattn_blocks.py")
        with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staged.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staged))
    print(OUTPUT)


if __name__ == "__main__":
    main()
