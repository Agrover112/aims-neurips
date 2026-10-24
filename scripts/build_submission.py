#!/usr/bin/env python3
"""Build a competition ZIP and run checks that need no third-party packages."""

from __future__ import annotations

import argparse
import py_compile
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist/submission.zip")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "submission",
        help="submission source directory (default: submission)",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    required = source / "model.py"
    if not required.is_file():
        raise SystemExit("submission/model.py is missing")
    with tempfile.TemporaryDirectory(prefix="aims-package-") as temporary:
        staged = Path(temporary) / "submission"
        shutil.copytree(source, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
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
