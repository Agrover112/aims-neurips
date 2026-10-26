"""How much of an item's pass rate is just "who answered it"?

For every item, compare its pass rate with the average strength of the
subjects that answered it. A subject's strength is its accuracy on its *other*
items (leave-one-item-out), so an item cannot predict itself. If the two are
tightly related, raw pass rate mostly reflects the answerer mix, not the item.

Uses ``processed/pairs.parquet`` (one row per subject-item pair with trial
counts). ``subject_uid`` never spans benchmarks, so a subject's strength is
within its own benchmark.

Run:  modal run --detach modal_item_answerer.py
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "aims-neurips-item-answerer"
VOLUME_NAME = "aims-neurips-data"
PAIRS = Path("/data/datasets/full/processed/pairs.parquet")
OUT = Path("/data/results/data-analysis/item-answerer.parquet")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24")


@app.function(image=image, cpu=8, memory=131_072, volumes={"/data": volume},
              timeout=2 * 60 * 60)
def run() -> dict:
    import pandas as pd

    p = pd.read_parquet(PAIRS, columns=["subject_uid", "item_uid", "benchmark_id",
                                        "n_trials", "successes"])
    print(f"pairs: {len(p):,}", flush=True)
    subj = p.groupby("subject_uid")[["successes", "n_trials"]].sum()
    p = p.join(subj.rename(columns={"successes": "S", "n_trials": "N"}), on="subject_uid")
    other_n = p["N"] - p["n_trials"]
    # accuracy on the subject's other items; undefined when it answered nothing else
    p["loo_acc"] = ((p["S"] - p["successes"]) / other_n.where(other_n > 0)).astype("float32")
    p = p.dropna(subset=["loo_acc"])
    g = p.groupby("item_uid", sort=False)
    table = pd.DataFrame({
        "benchmark_id": g["benchmark_id"].first(),
        "n_responses": g["n_trials"].sum(),
        "n_subjects": g["subject_uid"].nunique(),
        "successes": g["successes"].sum(),
        "answerer_acc": g["loo_acc"].mean(),
    }).reset_index()
    table["pass_rate"] = (table["successes"] / table["n_responses"]).astype("float32")
    table = table.drop(columns="successes")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT, index=False)
    volume.commit()
    return {"items": len(table)}


@app.local_entrypoint()
def main() -> None:
    print(run.remote())
