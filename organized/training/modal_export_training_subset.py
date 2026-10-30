"""Export the minimum data a benchmark-holdout ANP run needs, for transfer.

The full ``allbench-embed-bge_m3_long-v2`` store is 1,356,312 items and about
5.6 GB, but the benchmark-holdout filter keeps only ~123,035 of them, so a run
never touches the other 91%. Moving the whole store between Modal workspaces
would cost eleven times the bytes for no extra information.

This writes two artifacts under ``/data/export/<name>/``:

``datasets/full/allbench-embed-<arm>-v2/``
    ``items.parquet`` and ``item_embeddings.npy`` holding only the retained
    items, in the same row order, at the *same relative path the trainer
    already expects*. Nothing in the training code needs to know it is a
    subset.
``filtered-responses.parquet``
    The filter's own output, so the destination workspace does not need
    ``datasets/full/processed/`` (9.5M raw rows) at all. The filter is
    deterministic given its parameters, and they are recorded in
    ``manifest.json`` alongside a hash of each file, so the destination can
    assert it loaded the same data rather than trusting the file name.

Run::

    modal run --profile aims-test \
      organized/training/modal_export_training_subset.py::main
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "aims-export-training-subset"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
DATASETS = Path("/data/datasets/full")
EXPORT_ROOT = Path("/data/export")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24"
)
_FILTERS = (Path(__file__).resolve().parent / "benchmark_holdout_filters.py")
if _FILTERS.exists():
    image = image.add_local_file(_FILTERS, "/root/benchmark_holdout_filters.py")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(image=image, cpu=4, memory=32_768,
              volumes={"/data": volume}, timeout=60 * 60)
def export(arm: str = "bge_m3_long", split_seed: int = 20261018,
           min_item_responders: int = 15, min_items_per_subject: int = 80,
           name: str = "") -> dict:
    import sys

    import numpy as np
    import pandas as pd

    sys.path.insert(0, "/root")
    import benchmark_holdout_filters as filters

    if not name:
        name = (f"bhold-{arm}-n{min_item_responders}-m{min_items_per_subject}"
                f"-s{split_seed}")
    destination = EXPORT_ROOT / name
    store_out = destination / "datasets" / "full" / f"allbench-embed-{arm}-v2"
    store_out.mkdir(parents=True, exist_ok=True)

    responses, audit = filters.load_benchmark_holdout_filtered(
        PROCESSED, min_item_responders=min_item_responders,
        min_items_per_subject=min_items_per_subject, split_seed=split_seed,
    )
    print(f"filtered: {len(responses):,} rows, "
          f"{responses['benchmark_id'].nunique()} benchmarks", flush=True)

    store = DATASETS / f"allbench-embed-{arm}-v2"
    store_ids = pd.read_parquet(store / "items.parquet")["item_uid"].astype(str).to_numpy()
    embeddings = np.load(store / "item_embeddings.npy", mmap_mode="r")
    position = {uid: index for index, uid in enumerate(store_ids)}

    # Keep exactly the trainer's own intersection, so the subset cannot be
    # narrower than what a run would have used.
    responses = responses[responses["item_uid"].astype(str).isin(position)].copy()
    kept = sorted(responses["item_uid"].astype(str).unique())
    rows = [position[uid] for uid in kept]
    print(f"items: {len(store_ids):,} in store -> {len(kept):,} retained "
          f"({100 * len(kept) / len(store_ids):.1f}%)", flush=True)

    # Copy in chunks; the source is memory-mapped and must not be realised whole.
    subset = np.empty((len(rows), embeddings.shape[1]), dtype=np.float32)
    for start in range(0, len(rows), 8192):
        block = rows[start:start + 8192]
        subset[start:start + len(block)] = np.asarray(embeddings[block], dtype=np.float32)
    np.save(store_out / "item_embeddings.npy", subset)
    pd.DataFrame({"item_uid": kept}).to_parquet(store_out / "items.parquet", index=False)

    responses_path = destination / "filtered-responses.parquet"
    responses.to_parquet(responses_path, index=False)

    files = {
        "datasets/full/allbench-embed-%s-v2/item_embeddings.npy" % arm:
            store_out / "item_embeddings.npy",
        "datasets/full/allbench-embed-%s-v2/items.parquet" % arm:
            store_out / "items.parquet",
        "filtered-responses.parquet": responses_path,
    }
    manifest = {
        "name": name,
        "arm": arm,
        "source_volume": VOLUME_NAME,
        "filter_parameters": {
            "split_seed": split_seed,
            "min_item_responders": min_item_responders,
            "min_items_per_subject": min_items_per_subject,
        },
        "filter_audit": audit,
        "rows": int(len(responses)),
        "items_in_store": int(len(store_ids)),
        "items_retained": int(len(kept)),
        "embedding_dimension": int(embeddings.shape[1]),
        "split_counts": {
            str(role): int(count)
            for role, count in responses["benchmark_split"].value_counts().items()
        },
        "files": {
            relative: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for relative, path in files.items()
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    volume.commit()

    total = sum(entry["bytes"] for entry in manifest["files"].values())
    print("\n" + "=" * 60, flush=True)
    for relative, entry in manifest["files"].items():
        print(f"  {entry['bytes'] / 1e6:9.1f} MB  {relative}", flush=True)
    print(f"  {total / 1e6:9.1f} MB  TOTAL", flush=True)
    print(f"export path: {destination}", flush=True)
    print("=" * 60, flush=True)
    return manifest


@app.local_entrypoint()
def main(arm: str = "bge_m3_long", split_seed: int = 20261018,
         min_item_responders: int = 15, min_items_per_subject: int = 80,
         name: str = "") -> None:
    export.remote(arm=arm, split_seed=split_seed,
                  min_item_responders=min_item_responders,
                  min_items_per_subject=min_items_per_subject, name=name)
