"""Run Person B's matched item-difficulty probes on existing Modal data."""
from pathlib import Path
import modal

HERE = Path(__file__).parent
app = modal.App("eandd-wp7")
volume = modal.Volume.from_name("aims-neurips-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "pandas", "pyarrow", "scipy", "scikit-learn").env(
    {"OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
if modal.is_local():
    image = image.add_local_file(HERE / "probes.py", "/root/probes.py").add_local_file(
        HERE.parent / "online/difficulty_rank_metrics.py", "/opt/aims/organized/testing/online/difficulty_rank_metrics.py")
    image = image.add_local_file(HERE.parents[1] / "training/benchmark_holdout_filters.py", "/root/benchmark_holdout_filters.py")


@app.function(image=image, cpu=4, memory=32768, timeout=2400, volumes={"/data": volume})
def execute():
    import sys
    import json
    from benchmark_holdout_filters import load_benchmark_holdout_filtered
    from probes import run
    sys.path.insert(0, "/opt/aims/organized/testing/online")
    from difficulty_rank_metrics import score
    volume.reload()
    output = Path("/data/results/eandd-analysis/wp7-v1")
    source = output.parent / "inputs" / "filtered-responses.parquet"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        frame, audit = load_benchmark_holdout_filtered(Path("/data/datasets/full/processed"), split_seed=20261018)
        if not 6_335_000 <= len(frame) < 6_345_000:
            raise ValueError("Shared source does not match the paper's canonical response count")
        frame.to_parquet(source, index=False)
        (source.parent / "split-audit.json").write_text(json.dumps(audit, indent=2))
        del frame
        volume.commit()
    try:
        return run(source,
                   Path("/data/datasets/full/allbench-embed-bge_m3_long-v2"),
                   Path("/data/datasets/full/processed/items.parquet"), output, score)
    finally:
        if output.exists():
            volume.commit()


@app.local_entrypoint()
def main():
    print(execute.spawn().object_id)
