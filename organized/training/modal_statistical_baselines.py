"""Five CPU baselines with identical prepared data, episode banks and Forge keys."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pickle
import modal

app = modal.App("aims-statistical-benchmark-baselines")
volume = modal.Volume.from_name("aims-neurips-data", create_if_missing=False)
PREPARED = "/data/datasets/full/tabpfn-lora-benchmark-pca64-s20261018-v1"
RESULTS = Path("/data/results/statistical-baselines")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu"
).pip_install("numpy>=2,<3", "scipy>=1.11,<2", "scikit-learn>=1.5,<2", "wandb>=0.25,<1").env(
    {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
HERE = Path(__file__).resolve().parent
if str(HERE) != "/root":
    for name in ("statistical_baselines.py", "tabpfn_lora_core.py"):
        image = image.add_local_file(HERE / name, "/root/" + name)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(data)
    return h.hexdigest()

def save(obj, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(obj, stream, protocol=5)
    temporary.replace(path)

@app.function(image=image, cpu=2, memory=8192, timeout=2*3600, volumes={"/data": volume},
              secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])])
def run_model(kind: str, run_label: str, smoke=False, seed=20261018):
    import time
    import numpy as np
    import wandb
    import statistical_baselines as core
    from tabpfn_lora_core import make_bank
    if kind not in core.MODELS:
        raise ValueError(kind)
    prepared = Path(PREPARED)
    info = json.loads((prepared / "prepared.json").read_text())
    if info["benchmark_counts"] != {"train": 11, "validation": 3, "test": 3}:
        raise ValueError("Unexpected benchmark split")
    for name in ("tasks.pkl", "features.npy", "projection.npz", "split-audit.json"):
        if sha(prepared / name) != info["artifact_hashes"][name]:
            raise ValueError(f"Prepared artifact changed: {name}")
    with (prepared / "tasks.pkl").open("rb") as stream:
        tasks = pickle.load(stream)
    features = np.load(prepared / "features.npy", mmap_mode="r")
    train_items = np.unique(np.concatenate([t["items"] for t in tasks if t["role"] == "train"]))
    phi, transform = (None, None) if kind in ("beta_binomial", "rasch") else core.feature_map(features, train_items, kind, seed)
    # These calls are the very same make_bank used by the TabPFN run.
    validation_bank = make_bank(tasks, "validation", seed + 2000, limit=3 if smoke else 64)
    training_bank = make_bank(tasks, "train", seed + 5000, limit=11 if smoke else 64)
    fractions = (.01,) if smoke else core.FRACTIONS
    config = {"model": kind, "seed": seed, "fractions": fractions, "smoke": smoke,
              "prepared": info, "core_sha256": sha(core.__file__), "bank_code_sha256": sha("/root/tabpfn_lora_core.py"),
              "fit_iterations": 3 if smoke else 30, "curve_axis": "fraction of training response labels",
              "diagnostic_population": "heldout_subjects means validation benchmarks, not test benchmarks",
              "diagnostic_context": "full 50% or 80% support; diagnostics exceed competition label budgets",
              "training_diagnostics": "in-sample training-role tasks; globally fitted parameters may have seen target labels"}
    identity = hashlib.sha256(pickle.dumps(config, protocol=5)).hexdigest()
    output = RESULTS / run_label / kind
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "completed.json"
    if completed.exists():
        result = json.loads(completed.read_text())
        if result["identity"] != identity:
            raise ValueError("Completed run configuration differs")
        return result
    wb_id = "stat-" + hashlib.sha256(f"{run_label}/{kind}".encode()).hexdigest()[:18]
    wb = wandb.init(project="aims-predictive-evaluation", group=run_label,
                    name=f"{kind}--benchmark11-3-3--{run_label}", id=wb_id, resume="allow", config=config,
                    tags=["statistical", "benchmark-heldout", kind])
    wb.define_metric("training_fraction")
    wb.define_metric("diagnostic/*", step_metric="training_fraction")
    wb.define_metric("validation/*", step_metric="training_fraction")
    wb.define_metric("training/*", step_metric="training_fraction")
    print(f"{kind}: W&B {wb.url}", flush=True)
    fixed_beta = {"kind": "beta_binomial", "alpha": 2., "beta": 2.}
    reference = core.evaluate(fixed_beta, tasks, validation_bank, None)
    wb.summary["reference/fixed_beta_2_2_validation_alc"] = reference["brier_alc"]
    start, best, history, next_stage = time.monotonic(), float("inf"), [], 0
    last = output / "last.pkl"
    if last.exists():
        with last.open("rb") as stream:
            previous = pickle.load(stream)
        if previous["identity"] != identity:
            raise ValueError("Resume identity differs")
        best, history, next_stage = previous["best"], previous["history"], previous["next_stage"]
    for stage in range(next_stage, len(fractions)):
        fraction = fractions[stage]
        cells = core.training_cells(tasks, fraction, seed)
        state = core.fit(kind, cells, phi, len(tasks), len(features), iterations=config["fit_iterations"])
        validation = core.evaluate(state, tasks, validation_bank, phi)
        record = {"training_fraction": fraction, "training/observed_cells": len(cells[0]),
                  "validation/brier_alc": validation["brier_alc"],
                  **{f"validation/brier_budget_{b}": v for b,v in validation["by_budget"].items()},
                  **core.diagnostics(state, tasks, training_bank, phi, "training_subjects"),
                  **core.diagnostics(state, tasks, validation_bank, phi, "heldout_subjects")}
        wb.log(record, step=stage)
        history.append(record)
        snapshot = {"identity": identity, "config": config, "state": state, "transform": transform,
                    "fraction": fraction, "next_stage": stage + 1, "history": history,
                    "best": min(best, validation["brier_alc"]), "wandb_id": wb.id,
                    "validation_bank": validation_bank, "training_bank": training_bank,
                    "resume_method": "deterministic complete-stage restart; no hidden optimizer state"}
        if validation["brier_alc"] < best:
            best = validation["brier_alc"]
            save(snapshot, output / "best.pkl")
            (output / "best-validation.json").write_text(json.dumps(validation, indent=2))
        save(snapshot, last)
        (output / "curves.json").write_text(json.dumps(history, indent=2))
        volume.commit()
        print(f"{kind}: fraction={fraction:.2f} cells={len(cells[0]):,} validation_ALC={validation['brier_alc']:.6f}", flush=True)
    if not smoke:
        with (output / "best.pkl").open("rb") as stream:
            selected = pickle.load(stream)
        for role in ("validation", "test"):
            bank = make_bank(tasks, role, seed + 3000)
            result = core.evaluate(selected["state"], tasks, bank, phi)
            result["fixed_beta_2_2_brier_alc"] = core.evaluate(fixed_beta, tasks, bank, None)["brier_alc"]
            (output / f"final-{role}.json").write_text(json.dumps(result, indent=2))
            wb.summary[f"final/{role}_brier_alc"] = result["brier_alc"]
            wb.summary[f"final/{role}_benchmark_macro_alc"] = result["benchmark_macro_alc"]
    result = {"model": kind, "identity": identity, "best_validation_alc": best, "output": str(output),
              "elapsed_seconds": time.monotonic() - start, "wandb_url": wb.url, "smoke": smoke}
    completed.write_text(json.dumps(result, indent=2))
    volume.commit()
    wb.finish()
    return result

@app.function(cpu=.125, memory=256, timeout=3*3600)
def workflow(run_label, smoke):
    models = ("beta_binomial", "rasch", "feature_rasch", "linear_regression", "bayesian_rbf")
    calls = [(name, run_model.spawn(name, run_label, smoke)) for name in models]
    results = []
    for name, call in calls:
        print(f"{name}: {call.object_id}", flush=True)
    errors = []
    for name, call in calls:
        try:
            result = call.get()
            print(json.dumps(result), flush=True)
            results.append(result)
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return results

@app.local_entrypoint()
def main(smoke: bool = False, run_label: str = "statistical-benchmark-20261029-v1"):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_label):
        raise ValueError("Invalid run label")
    if smoke:
        run_label += "-smoke"
    call = workflow.spawn(run_label, smoke)
    print(f"Spawned workflow {call.object_id}; use --detach", flush=True)
