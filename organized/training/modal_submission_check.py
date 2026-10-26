"""Validate the final neural submission in a competition-like Modal runtime."""

from __future__ import annotations

import modal

app = modal.App("aims-neurips-submission-check")
volume = modal.Volume.from_name("aims-neurips-data", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2,<3",
        "pandas>=2.2,<3",
        "torch==2.5.1",
        "transformers==4.48.3",
    )
    .add_local_file("dist/submission-neural-final.zip", "/opt/input/submission.zip", copy=True)
    .add_local_dir("reference/starting_kit", "/opt/kit", copy=True)
    .env(
        {
            "HF_HOME": "/data/huggingface-cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
)


@app.function(image=image, cpu=4, memory=16_384, timeout=15 * 60, volumes={"/data": volume})
def validate() -> dict[str, str]:
    import os
    import subprocess
    import zipfile
    from pathlib import Path

    root = Path("/tmp/submission")
    root.mkdir()
    with zipfile.ZipFile("/opt/input/submission.zip") as archive:
        archive.extractall(root)

    env = {**os.environ, "PYTHONPATH": "/opt/kit/tools"}
    commands = {
        "zip_check": ["python", "/opt/kit/tools/check_submission_zip.py", "/opt/input/submission.zip"],
        "smoke_1": ["python", "/opt/kit/tools/run_smoke_test.py", str(root)],
        "smoke_4": [
            "python",
            "/opt/kit/tools/run_smoke_test.py",
            str(root),
            "--concurrency",
            "4",
        ],
    }
    output = {}
    for name, command in commands.items():
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=600)
        output[name] = (result.stdout + result.stderr).strip()
        if result.returncode:
            raise RuntimeError(f"{name} failed ({result.returncode}): {output[name]}")
    return output


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(validate.remote(), indent=2))
