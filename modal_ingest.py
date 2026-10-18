"""Credential-isolated Modal ingestion for the gated Measurement Data Bank.

Create a Modal Secret named ``huggingface-secret`` with an ``HF_TOKEN`` key,
then run either:

    modal run modal_ingest.py --dataset-name pilot --limit 2
    modal run modal_ingest.py --dataset-name full --limit 0
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import modal

APP_NAME = "aims-neurips-ingest"
VOLUME_NAME = "aims-neurips-data"
HF_SECRET_NAME = "huggingface-secret"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface-hub>=1.0,<2", "pandas>=2.2,<3", "pyarrow>=17,<24")
    .add_local_dir(PROJECT_ROOT / "src", "/opt/aims/src", copy=True)
    .add_local_dir(PROJECT_ROOT / "scripts", "/opt/aims/scripts", copy=True)
    .env(
        {
            "PYTHONPATH": "/opt/aims/src:/opt/aims/scripts",
            "HF_HOME": "/tmp/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)


def _safe_name(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", value):
        raise ValueError("dataset names must use lowercase letters, digits, '.', '_' or '-'")
    return value


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _dataset_report(name: str) -> dict[str, Any]:
    root = REMOTE_ROOT / "datasets" / name
    raw = root / "raw"
    processed = root / "processed"
    return {
        "name": name,
        "raw_ready": (raw / "download_manifest.json").exists(),
        "processed_ready": (processed / "manifest.json").exists(),
        "raw_bytes": _directory_bytes(raw),
        "processed_bytes": _directory_bytes(processed),
        "download_manifest": _read_json(raw / "download_manifest.json"),
        "processed_manifest": _read_json(processed / "manifest.json"),
    }


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={str(REMOTE_ROOT): volume},
    cpu=4,
    memory=16_384,
    timeout=12 * 60 * 60,
)
def prepare_dataset(
    dataset_name: str = "pilot",
    limit: int = 2,
    include_traces: bool = False,
    minimum_items: int = 0,
) -> dict[str, Any]:
    """Download, preprocess, and atomically publish a revision-pinned snapshot."""

    from download_measurement_tables import download_measurement_tables

    from aims_starter.data import prepare_measurement_data

    dataset_name = _safe_name(dataset_name)
    if limit < 0 or minimum_items < 0:
        raise ValueError("limit and minimum_items must be non-negative")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            f"Modal Secret {HF_SECRET_NAME!r} must provide the HF_TOKEN environment variable"
        )

    destination = REMOTE_ROOT / "datasets" / dataset_name
    raw_destination = destination / "raw"
    processed_destination = destination / "processed"
    report = _dataset_report(dataset_name)
    if report["raw_ready"] and report["processed_ready"]:
        report["status"] = "reused"
        return report
    if destination.exists():
        raise RuntimeError(
            f"Incomplete dataset directory exists at {destination}; inspect it before replacing it"
        )

    with tempfile.TemporaryDirectory(prefix=f"aims-{dataset_name}-") as temporary:
        stage = Path(temporary)
        raw_stage = stage / "raw"
        processed_stage = stage / "processed"
        download_measurement_tables(
            raw_stage,
            limit=limit,
            include_traces=include_traces,
            minimum_items=minimum_items,
        )
        prepare_measurement_data(raw_stage, processed_stage)

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir()
        shutil.copytree(raw_stage, raw_destination)
        shutil.copytree(processed_stage, processed_destination)
        volume.commit()

    report = _dataset_report(dataset_name)
    report["status"] = "created"
    return report


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={str(REMOTE_ROOT): volume},
    cpu=2,
    memory=4_096,
    timeout=12 * 60 * 60,
)
def sync_traces(dataset_name: str = "full") -> dict[str, Any]:
    """Persist available trace tables separately, with resumable commits."""

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    dataset_name = _safe_name(dataset_name)
    dataset_root = REMOTE_ROOT / "datasets" / dataset_name
    download_manifest_path = dataset_root / "raw" / "download_manifest.json"
    if not download_manifest_path.exists():
        raise FileNotFoundError(f"Dataset download manifest not found: {download_manifest_path}")
    download_manifest = _read_json(download_manifest_path)
    assert download_manifest is not None
    repo = str(download_manifest["repo"])
    revision = str(download_manifest["revision"])
    benchmarks = [str(name) for name in download_manifest["benchmarks"]]

    api = HfApi()
    info = api.repo_info(repo, repo_type="dataset", revision=revision)
    repo_files = {entry.rfilename for entry in info.siblings}
    requested = {
        benchmark: f"{benchmark}/formatted_tables/traces.parquet"
        for benchmark in benchmarks
    }
    available = {name: path for name, path in requested.items() if path in repo_files}
    missing = sorted(set(requested) - set(available))
    path_info = {}
    remote_paths = list(available.values())
    for start in range(0, len(remote_paths), 50):
        for entry in api.get_paths_info(
            repo,
            remote_paths[start : start + 50],
            repo_type="dataset",
            revision=revision,
            expand=True,
        ):
            path_info[entry.path] = int(entry.size)

    traces_root = dataset_root / "traces"
    traces_root.mkdir(parents=True, exist_ok=True)
    completed = []
    for number, (benchmark, remote_path) in enumerate(sorted(available.items()), 1):
        expected_bytes = path_info[remote_path]
        destination = traces_root / f"{benchmark}.parquet"
        if destination.exists() and destination.stat().st_size == expected_bytes:
            parquet = pq.ParquetFile(destination)
            completed.append(
                {
                    "benchmark": benchmark,
                    "file": destination.name,
                    "bytes": expected_bytes,
                    "rows": parquet.metadata.num_rows,
                }
            )
            print(f"Reused trace {number}/{len(available)}: {benchmark}")
            continue
        if destination.exists():
            raise RuntimeError(
                f"Existing trace has the wrong size and will not be overwritten: {destination}"
            )

        partial = traces_root / f".{benchmark}.partial"
        if partial.exists():
            partial.unlink()
        with tempfile.TemporaryDirectory(prefix=f"aims-trace-{benchmark}-") as cache:
            source = hf_hub_download(
                repo,
                remote_path,
                repo_type="dataset",
                revision=revision,
                cache_dir=cache,
            )
            shutil.copyfile(source, partial)
        if partial.stat().st_size != expected_bytes:
            raise RuntimeError(f"Trace size mismatch after download: {benchmark}")
        parquet = pq.ParquetFile(partial)
        required = {"response_id", "subject_id", "item_id", "benchmark_id", "trace"}
        missing_columns = sorted(required - set(parquet.schema_arrow.names))
        if missing_columns:
            raise ValueError(f"{benchmark}: trace table missing columns {missing_columns}")
        partial.rename(destination)
        completed.append(
            {
                "benchmark": benchmark,
                "file": destination.name,
                "bytes": expected_bytes,
                "rows": parquet.metadata.num_rows,
            }
        )
        volume.commit()
        print(
            f"Downloaded trace {number}/{len(available)}: {benchmark} "
            f"({expected_bytes / 1024**3:.3f} GiB)"
        )

    manifest = {
        "repo": repo,
        "revision": revision,
        "dataset_name": dataset_name,
        "files": completed,
        "missing_benchmarks": missing,
        "total_bytes": sum(int(row["bytes"]) for row in completed),
    }
    manifest_path = traces_root / "trace_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    action: str = "dataset",
    dataset_name: str = "pilot",
    limit: int = 2,
    include_traces: bool = False,
    minimum_items: int = 80,
) -> None:
    if action == "dataset":
        result = prepare_dataset.remote(
            dataset_name=dataset_name,
            limit=limit,
            include_traces=include_traces,
            minimum_items=minimum_items,
        )
    elif action == "traces":
        result = sync_traces.remote(dataset_name=dataset_name)
    else:
        raise ValueError("action must be dataset or traces")
    print(json.dumps(result, indent=2, sort_keys=True))
