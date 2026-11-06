"""Fit the training-only Beta prior before frozen validation draws arrive."""
import json
from pathlib import Path
from time import monotonic

import modal

app = modal.App("eandd-bb-fit")
volume = modal.Volume.from_name("aims-neurips-data", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "scipy", "pandas", "pyarrow")
         .add_local_file(Path(__file__).with_name("bb_grid.py"), "/root/bb_grid.py"))


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={"/data": volume})
def fit():
    import pandas as pd
    from bb_grid import fit_empirical_bayes

    start = monotonic()
    volume.reload()
    source = Path("/data/datasets/full/tabpfn-lora-benchmark-pca64-s20261018-v1/filtered.parquet")
    root = Path("/data/results/eandd-analysis/id-v1")
    audit = json.loads((root / "subjects/manifest.json").read_text())
    train = pd.read_parquet(source, columns=["benchmark_id", "subject_uid", "label", "benchmark_split"],
                            filters=[("benchmark_split", "==", "train")])
    mean, strength = fit_empirical_bayes(train)
    result = dict(mean=mean, strength=strength, alpha=mean*strength, beta=(1-mean)*strength,
                  source=str(source), source_sha256=audit["response_sha256"], source_bytes=source.stat().st_size,
                  train_rows=len(train), training_benchmarks=train.benchmark_id.nunique(),
                  task_weighting="equal benchmarks; equal tasks within benchmark",
                  scope="OOD replay only; includes subjects held out of the separate ID control",
                  test_read=False, runtime_seconds=monotonic()-start)
    (root / "empirical-beta-prior.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    print(json.dumps(fit.remote(), indent=2))
