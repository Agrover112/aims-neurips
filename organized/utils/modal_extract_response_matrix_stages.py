"""Extract a matched raw-versus-filtered response-matrix visualisation sample.

This is a CPU-only read of one benchmark. It does not train a model or modify
the Modal volume. The local entrypoint writes a compact ``.npz`` so plotting
can happen locally.

The filtered panel applies the response-support part of the data audit:
deduplicate subject--item responses, remove all-correct/all-incorrect items,
require at least 30 respondents per item, then retain subjects with at least
80 remaining item responses.

Example:
    .venv/bin/modal run --profile agrover112 \
      organized/utils/modal_extract_response_matrix_stages.py::main \
      --benchmark afrieval --output docs/response-matrix-before-after.npz
"""

from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "aims-neurips-response-matrix-stages"
VOLUME_NAME = "aims-neurips-data"
OBS = Path("/data/datasets/full/processed/observations.parquet")
MAX_SUBJECTS = 128
MAX_ITEMS = 384

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24"
)


def _matrix(frame, pd, np):
    """Densest readable submatrix, ordered weak→strong and hard→easy."""
    top_subjects = frame["subject_uid"].value_counts().head(MAX_SUBJECTS).index
    panel = frame[frame["subject_uid"].isin(top_subjects)]
    top_items = panel["item_uid"].value_counts().head(MAX_ITEMS).index
    panel = panel[panel["item_uid"].isin(top_items)]
    matrix = panel.pivot(index="subject_uid", columns="item_uid", values="label")
    subject_order = matrix.mean(axis=1).sort_values().index
    item_order = matrix.mean(axis=0).sort_values().index
    matrix = matrix.loc[subject_order, item_order]
    dense = np.where(np.isnan(matrix.to_numpy(dtype="float32")), -1,
                     matrix.to_numpy(dtype="int8")).astype("int8")
    return dense


@app.function(image=image, cpu=8, memory=32_768, volumes={"/data": volume},
              timeout=60 * 60)
def extract(benchmark: str) -> dict:
    import numpy as np
    import pandas as pd

    obs = pd.read_parquet(
        OBS,
        columns=["subject_uid", "item_uid", "label"],
        filters=[("benchmark_id", "=", benchmark)],
    )
    raw_rows = len(obs)
    raw = obs.groupby(["subject_uid", "item_uid"], as_index=False)["label"].mean()
    raw["label"] = (raw["label"] >= 0.5).astype("int8")
    raw_matrix = _matrix(raw, pd, np)

    item_mean = raw.groupby("item_uid")["label"].mean()
    variable_items = item_mean[(item_mean > 0) & (item_mean < 1)].index
    filtered = raw[raw["item_uid"].isin(variable_items)]
    respondent_count = filtered.groupby("item_uid")["subject_uid"].nunique()
    supported_items = respondent_count[respondent_count >= 30].index
    filtered = filtered[filtered["item_uid"].isin(supported_items)]
    subject_count = filtered.groupby("subject_uid")["item_uid"].nunique()
    supported_subjects = subject_count[subject_count >= 80].index
    filtered = filtered[filtered["subject_uid"].isin(supported_subjects)].copy()
    filtered_matrix = _matrix(filtered, pd, np)

    return {
        "benchmark": benchmark,
        "raw_matrix": raw_matrix,
        "filtered_matrix": filtered_matrix,
        "raw_rows": int(raw_rows),
        "deduplicated_rows": int(len(raw)),
        "filtered_rows": int(len(filtered)),
        "raw_subjects": int(raw["subject_uid"].nunique()),
        "raw_items": int(raw["item_uid"].nunique()),
        "filtered_subjects": int(filtered["subject_uid"].nunique()),
        "filtered_items": int(filtered["item_uid"].nunique()),
    }


@app.local_entrypoint()
def main(benchmark: str = "afrieval",
         output: str = "docs/response-matrix-before-after.npz") -> None:
    import json
    import numpy as np

    result = extract.remote(benchmark)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {key: value for key, value in result.items() if not key.endswith("_matrix")}
    np.savez_compressed(destination, raw=result["raw_matrix"], filtered=result["filtered_matrix"],
                        metadata=json.dumps(metadata, sort_keys=True))
    print(json.dumps({**metadata, "output": str(destination)}, indent=2, sort_keys=True))
