"""Does the existing embedding store cover the n>=30 filtered item set?

Before training an amortized model (difficulty predicted from question text) we
need an embedding for every item that survives filtering. The embedding stores
under ``allbench-features-v1`` were built over the full item universe, with the
per-task cap applied only when a downstream training store is assembled -- so
coverage is expected to be complete. This job checks that rather than assuming
it, and reports exactly which items would still need encoding.

CPU only, a couple of minutes.

Run:  modal run organized/utils/modal_embedding_coverage.py
      modal run organized/utils/modal_embedding_coverage.py --min-subjects 15
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "aims-embedding-coverage"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
DATASETS = Path("/data/datasets/full")
OUT = Path("/data/results/data-analysis/embedding-coverage.parquet")

# Same filter pipeline as modal_classic_irt_filter_test.py; keep in step.
MIN_SUBJECTS = 30
MIN_ITEMS_PER_TASK = 80

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24")


def _filtered_items(min_subjects: int):
    """Item uids surviving the standard filter, plus per-benchmark counts."""
    import pandas as pd

    obs = pd.read_parquet(PROCESSED / "observations.parquet",
                          columns=["subject_uid", "item_uid", "benchmark_id", "label"])
    items_meta = pd.read_parquet(PROCESSED / "items.parquet",
                                 columns=["item_uid", "content_hash"])
    obs = obs.groupby(["subject_uid", "item_uid", "benchmark_id"], as_index=False)["label"].mean()
    obs = obs.merge(items_meta, on="item_uid", how="left")
    print(f"observations (deduped subject x item): {len(obs):,}", flush=True)

    kept, per_bench = [], []
    for bench, dd in obs.groupby("benchmark_id"):
        raw_items = dd["item_uid"].nunique()
        dd = dd.drop_duplicates(["content_hash", "subject_uid"])
        # Zero-variance items carry no discriminative signal.
        var = dd.groupby("item_uid")["label"].nunique()
        dd = dd[dd["item_uid"].isin(var[var > 1].index)]
        n_sub_per_item = dd.groupby("item_uid")["subject_uid"].nunique()
        dd = dd[dd["item_uid"].isin(n_sub_per_item[n_sub_per_item >= min_subjects].index)]
        n_item_per_task = dd.groupby("subject_uid")["item_uid"].nunique()
        dd = dd[dd["subject_uid"].isin(
            n_item_per_task[n_item_per_task >= MIN_ITEMS_PER_TASK].index)]
        if dd.empty:
            per_bench.append({"benchmark_id": bench, "items_raw": int(raw_items),
                              "items_kept": 0})
            continue
        uids = dd["item_uid"].unique()
        kept.append(pd.Series(uids, name="item_uid"))
        per_bench.append({"benchmark_id": bench, "items_raw": int(raw_items),
                          "items_kept": int(len(uids))})

    all_kept = pd.concat(kept).drop_duplicates() if kept else pd.Series([], name="item_uid")
    return all_kept.astype(str), pd.DataFrame(per_bench)


@app.function(image=image, volumes={"/data": volume}, timeout=60 * 30)
def run(min_subjects: int = MIN_SUBJECTS) -> dict:
    import numpy as np
    import pandas as pd

    kept, per_bench = _filtered_items(min_subjects)
    print(f"\nfiltered item universe (n>={min_subjects}): {len(kept):,} items", flush=True)

    # Each store is its own flat directory: allbench-features-v1 and the
    # allbench-embed-*-v* arms. Training stores (allbench-train-*) are the
    # capped subsets and are deliberately excluded.
    stores = sorted(p for p in DATASETS.iterdir()
                    if p.is_dir() and not p.name.startswith("allbench-train-")
                    and (p / "items.parquet").exists()
                    and (p / "item_embeddings.npy").exists())
    if not stores:
        print(f"no embedding stores found under {DATASETS}", flush=True)
        return {"filtered_items": int(len(kept)), "arms": []}

    kept_set = set(kept)
    rows, missing_by_arm = [], {}
    for store_dir in stores:
        arm = store_dir.name
        store = pd.read_parquet(store_dir / "items.parquet")
        have = set(store["item_uid"].astype(str))
        missing = kept_set - have
        emb_path = store_dir / "item_embeddings.npy"
        dim = int(np.load(emb_path, mmap_mode="r").shape[1]) if emb_path.exists() else -1
        rows.append({"arm": arm, "min_subjects": min_subjects,
                     "store_items": len(have), "filtered_items": len(kept_set),
                     "covered": len(kept_set) - len(missing), "missing": len(missing),
                     "coverage_pct": 100.0 * (len(kept_set) - len(missing)) / max(len(kept_set), 1),
                     "embedding_dim": dim})
        missing_by_arm[arm] = missing
        print(f"  {arm:<34} store={len(have):>8,}  covered={len(kept_set) - len(missing):>7,}"
              f"  missing={len(missing):>6,}  ({rows[-1]['coverage_pct']:.2f}%)  dim={dim}",
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT.with_name(f"{OUT.stem}-n{min_subjects}{OUT.suffix}"), index=False)
    per_bench.to_parquet(
        OUT.with_name(f"{OUT.stem}-n{min_subjects}-per-benchmark{OUT.suffix}"), index=False)

    # Only worth writing the uid list if something actually needs encoding.
    for arm, missing in missing_by_arm.items():
        if missing:
            pd.DataFrame({"item_uid": sorted(missing)}).to_parquet(
                OUT.with_name(f"{OUT.stem}-n{min_subjects}-missing-{arm}{OUT.suffix}"),
                index=False)
    volume.commit()

    print(f"\nbenchmarks retained: {(per_bench['items_kept'] > 0).sum()} / {len(per_bench)}",
          flush=True)
    return {"filtered_items": int(len(kept)),
            "arms": {r["arm"]: {"missing": r["missing"], "coverage_pct": r["coverage_pct"]}
                     for r in rows}}


@app.local_entrypoint()
def main(min_subjects: int = MIN_SUBJECTS) -> None:
    print(run.remote(min_subjects))
