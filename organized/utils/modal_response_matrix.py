"""Raw subject x item response matrices for a few benchmarks, split by role.

Figure 8 of the amortized-evaluation paper shows the response matrix itself
(blue correct, red incorrect, white missing). This extracts the same view for
two benchmarks from each of our train / validation / test roles, straight from
``processed/observations.parquet``. To keep a panel readable it keeps the
subjects with the most responses and the items answered by the most subjects,
then orders rows by subject accuracy and columns by item pass rate.

Run:  modal run --detach modal_response_matrix.py
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "aims-neurips-response-matrix"
VOLUME_NAME = "aims-neurips-data"
OBS = Path("/data/datasets/full/processed/observations.parquet")
OUT = Path("/data/results/data-analysis/response-matrices.npz")
ROLES = {"train": ("ai2d_test", "hallusionbench"),
         "validation": ("cmmlu", "afrimedqa"),
         "test": ("bertaqa", "dave")}
MAX_SUBJECTS, MAX_ITEMS = 90, 400

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24")


@app.function(image=image, cpu=8, memory=131_072, volumes={"/data": volume},
              timeout=2 * 60 * 60)
def run() -> dict:
    import numpy as np
    import pandas as pd

    wanted = [b for pair in ROLES.values() for b in pair]
    obs = pd.read_parquet(OBS, columns=["subject_uid", "item_uid", "benchmark_id", "label"],
                          filters=[("benchmark_id", "in", wanted)])
    print(f"observations kept: {len(obs):,}", flush=True)
    arrays, meta = {}, {}
    for role, benches in ROLES.items():
        for bench in benches:
            d = obs[obs["benchmark_id"] == bench]
            d = d.groupby(["subject_uid", "item_uid"], as_index=False)["label"].mean()
            d["label"] = (d["label"] >= 0.5).astype("int8")
            n_sub_all, n_item_all = d["subject_uid"].nunique(), d["item_uid"].nunique()
            top_sub = d["subject_uid"].value_counts().head(MAX_SUBJECTS).index
            d = d[d["subject_uid"].isin(top_sub)]
            top_item = d["item_uid"].value_counts().head(MAX_ITEMS).index
            d = d[d["item_uid"].isin(top_item)]
            m = d.pivot(index="subject_uid", columns="item_uid", values="label")
            sub_acc = m.mean(axis=1).sort_values()
            item_rate = m.mean(axis=0).sort_values()
            m = m.loc[sub_acc.index, item_rate.index]
            grid = m.to_numpy(dtype="float32")
            grid = np.where(np.isnan(grid), -1, grid).astype("int8")
            arrays[bench] = grid
            meta[bench] = (role, int(n_sub_all), int(n_item_all), float(np.mean(grid[grid >= 0])),
                           float((grid < 0).mean()))
            print(f"{role:<11}{bench:<16}shown {grid.shape}  of {n_sub_all} x {n_item_all}  "
                  f"missing {(grid < 0).mean():.0%}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    volume.commit()
    return {k: list(v) for k, v in meta.items()}


@app.local_entrypoint()
def main() -> None:
    print(run.remote())
