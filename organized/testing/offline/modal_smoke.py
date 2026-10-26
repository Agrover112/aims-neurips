"""Run the organizer smoke test with the four-worker CPU pool.

The Codabench runtime pre-downloads declared HuggingFace models and then runs
offline. This app reproduces that: the image fetches MiniLM at build time, and
the smoke test runs against the packaged submission ZIP with concurrency 4.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "aims-neurips-smoke"
ROOT = Path(__file__).resolve().parent

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=2,<3", "transformers==4.48.3")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .env({"HF_HOME": "/opt/hf-cache"})
    .run_commands(
        "python -c \"from transformers import AutoModel, AutoTokenizer; "
        "AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); "
        "AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')\""
    )
    .env(
        {
            "HF_HOME": "/opt/hf-cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    .add_local_file(
        ROOT / "dist" / "submission-neural-final.zip", "/opt/smoke/submission.zip", copy=True
    )
    .add_local_dir(ROOT / "reference" / "starting_kit", "/opt/kit", copy=True)
)


@app.function(image=image, cpu=8, memory=32_768, timeout=30 * 60)
def run_smoke() -> str:
    import subprocess
    import sys
    import tempfile
    import zipfile

    with tempfile.TemporaryDirectory(prefix="paiec-smoke-") as directory:
        with zipfile.ZipFile("/opt/smoke/submission.zip") as archive:
            archive.extractall(directory)
        completed = subprocess.run(
            [
                sys.executable,
                "/opt/kit/tools/run_smoke_test.py",
                directory,
                "--concurrency",
                "4",
            ],
            capture_output=True,
            text=True,
            timeout=25 * 60,
        )
    output = completed.stdout + completed.stderr
    print(output, flush=True)
    completed.check_returncode()
    return output


@app.function(image=image, cpu=2, memory=8_192, timeout=15 * 60)
def run_check() -> str:
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "/opt/kit/tools/check_submission_zip.py",
            "/opt/smoke/submission.zip",
        ],
        capture_output=True,
        text=True,
        timeout=10 * 60,
    )
    output = completed.stdout + completed.stderr
    print(output, flush=True)
    completed.check_returncode()
    return output


@app.local_entrypoint()
def main() -> None:
    print(run_smoke.remote())
