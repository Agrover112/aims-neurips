"""CPU audit for the benchmark-transfer filter before launching a trainer.

Example:
  .venv/bin/modal run --profile aims-test \
    organized/training/modal_benchmark_holdout_filter_audit.py::main
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "aims-neurips-benchmark-holdout-filter-audit"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
AUDIT_PATH = Path("/data/results/data-quality/benchmark-holdout-filter-audit.json")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24"
)
_HERE = Path(__file__).resolve().parent
_FILTERS = _HERE / "benchmark_holdout_filters.py"
if str(_HERE) != "/root" and _FILTERS.exists():
    image = image.add_local_file(_FILTERS, "/root/benchmark_holdout_filters.py")


@app.function(image=image, cpu=8, memory=32_768, volumes={"/data": volume},
              timeout=60 * 60)
def audit(min_item_responders: int = 15, min_items_per_subject: int = 80,
          split_seed: int = 20261018) -> dict:
    import benchmark_holdout_filters as filters

    _, result = filters.load_benchmark_holdout_filtered(
        PROCESSED,
        min_item_responders=min_item_responders,
        min_items_per_subject=min_items_per_subject,
        split_seed=split_seed,
    )
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    volume.commit()
    return result


@app.local_entrypoint()
def main(min_item_responders: int = 15, min_items_per_subject: int = 80,
         split_seed: int = 20261018) -> None:
    result = audit.remote(min_item_responders, min_items_per_subject, split_seed)
    print(json.dumps(result, indent=2, sort_keys=True))
