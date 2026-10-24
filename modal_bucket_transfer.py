"""Move the data Volume through a shared R2 bucket, between Modal workspaces.

Modal Volumes cannot be shared across workspaces, but a bucket can. Both
directions run inside Modal, so no data touches a laptop.

    export R2_BUCKET=aims-neurips-transfer R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com

    # In the workspace that owns the data:
    MODAL_PROFILE=agrover112   modal run modal_bucket_transfer.py --direction push
    # In the workspace that should receive it:
    MODAL_PROFILE=aims-neurips modal run modal_bucket_transfer.py --direction pull

Files whose size already matches are skipped, so an interrupted run resumes.
Create the bucket and the ``aims-r2`` Secret with ``scripts/setup_r2_bucket.py``.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

APP_NAME = "aims-neurips-bucket-transfer"
VOLUME_NAME = "aims-neurips-data"
SECRET_NAME = "aims-r2"
VOLUME_ROOT = Path("/data")
BUCKET_ROOT = Path("/bucket")
DEFAULT_PATHS = "datasets,results,tabpfn-cache,huggingface-cache"
WORKERS = 16

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
bucket = modal.CloudBucketMount(
    bucket_name=os.environ.get("R2_BUCKET", "aims-neurips-transfer"),
    bucket_endpoint_url=os.environ.get("R2_ENDPOINT"),
    secret=modal.Secret.from_name(
        SECRET_NAME, required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    ),
)
image = modal.Image.debian_slim(python_version="3.11")


def _copy_tree(source: Path, target: Path) -> tuple[int, int, int]:
    files = [p for p in source.rglob("*") if p.is_file()]
    copied = skipped = 0
    size = 0

    def copy(path: Path) -> bool:
        destination = target / path.relative_to(source)
        if destination.exists() and destination.stat().st_size == path.stat().st_size:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        return True

    with ThreadPoolExecutor(WORKERS) as pool:
        for path, did_copy in zip(files, pool.map(copy, files)):
            if did_copy:
                copied += 1
                size += path.stat().st_size
            else:
                skipped += 1
    return copied, skipped, size


def _transfer(source_root: Path, target_root: Path, paths: str) -> None:
    for name in paths.split(","):
        source = source_root / name
        if not source.exists():
            print(f"{name}: not found in source, skipped")
            continue
        copied, skipped, size = _copy_tree(source, target_root / name)
        print(f"{name}: copied {copied} files ({size / 1e9:.2f} GB), skipped {skipped}", flush=True)


@app.function(image=image, volumes={str(VOLUME_ROOT): volume, str(BUCKET_ROOT): bucket}, cpu=4, timeout=24 * 3600)
def push(paths: str = DEFAULT_PATHS) -> None:
    """Volume -> bucket."""
    volume.reload()
    _transfer(VOLUME_ROOT, BUCKET_ROOT, paths)


@app.function(image=image, volumes={str(VOLUME_ROOT): volume, str(BUCKET_ROOT): bucket}, cpu=4, timeout=24 * 3600)
def pull(paths: str = DEFAULT_PATHS) -> None:
    """Bucket -> volume."""
    _transfer(BUCKET_ROOT, VOLUME_ROOT, paths)
    volume.commit()


@app.local_entrypoint()
def main(direction: str, paths: str = DEFAULT_PATHS) -> None:
    if direction not in {"push", "pull"}:
        raise SystemExit("--direction must be 'push' or 'pull'")
    {"push": push, "pull": pull}[direction].remote(paths)
