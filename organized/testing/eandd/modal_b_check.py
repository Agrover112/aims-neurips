"""Rebuild the canonical OOD panel, then gate validation-only WP2 on the documented replay check."""
import json
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] if modal.is_local() else Path("/opt/aims")
REMOTE = "/opt/aims/organized/testing"
app = modal.App("eandd-b-check")
volume = modal.Volume.from_name("aims-neurips-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "scipy", "pandas", "pyarrow", "scikit-learn")
if modal.is_local():
    for name in ("common.py", "run_replay.py", "bb_grid.py", "analysis.py"):
        image = image.add_local_file(HERE / name, REMOTE + "/eandd/" + name)
    for name in ("benchmark_holdout_replay.py", "kernel_rasch_replay.py"):
        image = image.add_local_file(HERE.parent / "online" / name, REMOTE + "/online/" + name)
    image = image.add_local_file(ROOT / "paper/eandd/data/draws.parquet", "/opt/frozen-draws.parquet")


@app.function(image=image, cpu=6, memory=16384, timeout=1200, volumes={"/data": volume})
def check(run_name: str, revision: str):
    import os
    import shutil
    import subprocess
    import sys
    import pandas as pd

    volume.reload()
    src = Path("/data/datasets/full/tabpfn-lora-benchmark-pca64-s20261018-v1/filtered.parquet")
    emb = Path("/data/datasets/full/allbench-embed-bge_m3_long-v2")
    cache = Path("/data/results/eandd-analysis/ood-cache-v1")
    panel = cache / "panel"
    data = cache / "paper-eandd"
    data.mkdir(parents=True, exist_ok=True)
    frozen = Path("/opt/frozen-draws.parquet")
    draws = data / "draws.parquet"
    if draws.exists() and draws.read_bytes() != frozen.read_bytes():
        raise ValueError("Cached draws differ from Git's frozen draws")
    if not draws.exists():
        shutil.copyfile(frozen, draws)
    env = dict(os.environ, AIMS_RESPONSES=str(src), AIMS_EMBEDDINGS=str(emb),
               AIMS_REPLAY_CACHE=str(cache), AIMS_EANDD_DATA=str(data),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if not (panel / "candidate_orders.parquet").exists():
        sys.path.insert(0, REMOTE + "/online")
        from benchmark_holdout_replay import make_panel
        rows = pd.read_parquet(src, filters=[("benchmark_split", "in", ["validation", "test"])])
        tasks, items, orders, _ = make_panel(rows, split_seed=20261018, min_source_items=80,
                                             max_source_items=100000, order_seeds=range(10))
        frozen_tasks = pd.read_parquet(draws, columns=["role", "task_id"]).drop_duplicates()
        expected = set(zip(frozen_tasks.role, frozen_tasks.task_id))
        if set(zip(tasks.benchmark_split, tasks.task_id)) != expected:
            raise ValueError("Reconstructed panel task IDs do not match the frozen draws; stop")
        panel.mkdir(parents=True, exist_ok=True)
        for name, frame in (("tasks", tasks), ("items", items), ("candidate_orders", orders)):
            frame.to_parquet(panel / (name + ".parquet"), index=False)
        volume.commit()
    subprocess.run([sys.executable, REMOTE + "/eandd/run_replay.py", "--run", run_name, "--shape", "competition",
                    "--role", "test", "--models", "jeffreys,rasch-old,rasch-retuned,kernel-old",
                    "--no-preds"], env=env, check=True)
    scores_path = data / "scores" / (run_name + ".parquet")
    scores = pd.read_parquet(scores_path)
    tb = scores.groupby(["model", "task_id", "budget"]).brier.mean().unstack()
    alc = tb.mul(pd.Series({0:.1, 1:.2, 3:.2, 7:.2, 15:.2, 31:.1})).sum(axis=1).groupby("model").mean()
    checks = {name: round(float(alc[name] - alc["jeffreys"]), 4) for name in ("rasch-retuned", "kernel-old", "rasch-old")}
    checks["jeffreys"] = round(float(alc["jeffreys"]), 4)
    record = data / "runs" / (run_name + ".json")
    metadata = json.loads(record.read_text())
    metadata["git_commit"] = revision
    metadata["reconstructed_panel"] = True
    metadata["reproduction_check"] = checks
    record.write_text(json.dumps(metadata, indent=2))
    volume.commit()
    if checks != {"rasch-retuned": -.0032, "kernel-old": -.0024, "rasch-old": -.0011, "jeffreys": .2499}:
        raise ValueError(f"Reproduction gate failed; no grid run: {checks}")
    output = data / (run_name + "-wp2")
    subprocess.run([sys.executable, REMOTE + "/eandd/bb_grid.py", "--responses", str(src), "--panel", str(panel),
                    "--draws", str(draws), "--output", str(output)], env=env, check=True)
    subprocess.run([sys.executable, REMOTE + "/eandd/analysis.py", "--scores", str(scores_path), "--reference",
                    "jeffreys", "--output", str(output / "paired-check.csv")], env=env, check=True)
    volume.commit()
    return {"check": checks, "wp2": json.loads((output / "priors.json").read_text())}


@app.local_entrypoint()
def main(run_name: str = "b-check-20261106-v1"):
    import subprocess
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    print(check.spawn(run_name, revision).object_id)
