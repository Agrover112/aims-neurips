#!/usr/bin/env python3
"""Download only model-training tables from the Measurement Data Bank."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "aims-foundations/measurement-db"


def _download(repo_path: str, revision: str, destination: Path) -> None:
    cached = hf_hub_download(REPO_ID, repo_path, repo_type="dataset", revision=revision)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, destination)


def download_measurement_tables(
    output: str | Path,
    *,
    limit: int = 0,
    benchmarks: Sequence[str] = (),
    include_traces: bool = False,
    minimum_items: int = 0,
) -> dict[str, Any]:
    """Download an immutable, revision-pinned table snapshot.

    The callable form is shared by the local CLI and Modal ingestion job. The
    Hugging Face client reads authentication from its standard configuration,
    including the ``HF_TOKEN`` environment variable used by Modal Secrets.
    """

    if limit < 0 or minimum_items < 0:
        raise ValueError("limit and minimum_items must be non-negative")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")

    info = HfApi().repo_info(REPO_ID, repo_type="dataset", revision="main")
    revision = info.sha
    files = {entry.rfilename for entry in info.siblings}
    benchmark_paths = sorted(
        path for path in files if path.endswith("/formatted_tables/benchmarks.parquet")
    )
    requested = set(benchmarks)

    candidates = []
    for index, benchmark_path in enumerate(benchmark_paths, 1):
        cached = hf_hub_download(REPO_ID, benchmark_path, repo_type="dataset", revision=revision)
        row = pd.read_parquet(cached).iloc[0]
        directory = benchmark_path.removesuffix("/benchmarks.parquet")
        benchmark_name = directory.split("/", 1)[0]
        if requested and benchmark_name not in requested:
            continue
        if str(row.get("response_type", "")).lower() != "binary":
            continue
        if str(row.get("granularity", "")).lower() != "item":
            continue
        n_items = int(row.get("n_items", 0) or 0)
        if n_items < minimum_items:
            continue
        candidates.append(
            {
                "directory": directory,
                "benchmark": benchmark_name,
                "benchmark_path": benchmark_path,
                "n_responses": int(row.get("n_responses", 0) or 0),
                "n_items": n_items,
                "cached_benchmark": cached,
            }
        )
        if index % 100 == 0:
            print(f"Scanned {index}/{len(benchmark_paths)} benchmark metadata tables")

    if benchmarks:
        found = {candidate["benchmark"] for candidate in candidates}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"Requested benchmarks not found or not eligible: {missing}")
    candidates.sort(key=lambda row: (row["n_responses"], row["benchmark"]))
    if limit:
        candidates = candidates[:limit]
    if not candidates:
        raise ValueError("No eligible benchmark tables selected")

    table_names = ["benchmarks", "subjects", "items", "responses"]
    if include_traces:
        table_names.append("traces")
    copied = []
    for number, candidate in enumerate(candidates, 1):
        directory = candidate["directory"]
        destination = output / directory
        destination.mkdir(parents=True, exist_ok=True)
        for table in table_names:
            repo_path = f"{directory}/{table}.parquet"
            if table == "responses" and repo_path not in files:
                repo_path = f"{directory}/response.parquet"
            if repo_path not in files:
                if table == "traces":
                    continue
                raise FileNotFoundError(f"Missing required Hub table: {repo_path}")
            target = destination / f"{table}.parquet"
            if table == "benchmarks":
                shutil.copyfile(candidate["cached_benchmark"], target)
            else:
                _download(repo_path, revision, target)
            copied.append(str(target.relative_to(output)))
        print(
            f"Downloaded {number}/{len(candidates)}: {candidate['benchmark']} "
            f"({candidate['n_responses']} responses)"
        )

    manifest = {
        "repo": REPO_ID,
        "revision": revision,
        "benchmarks": [candidate["benchmark"] for candidate in candidates],
        "files": copied,
        "includes_traces": include_traces,
    }
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Downloaded table snapshot: {output}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New or empty output directory")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Download the N smallest eligible benchmarks; 0 downloads all",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=[],
        help="Exact benchmark directory names to download instead of automatic selection",
    )
    parser.add_argument("--include-traces", action="store_true")
    parser.add_argument(
        "--minimum-items",
        type=int,
        default=0,
        help="Skip benchmarks with fewer than this many source items",
    )
    args = parser.parse_args()
    download_measurement_tables(
        args.out,
        limit=args.limit,
        benchmarks=args.benchmarks,
        include_traces=args.include_traces,
        minimum_items=args.minimum_items,
    )


if __name__ == "__main__":
    main()
