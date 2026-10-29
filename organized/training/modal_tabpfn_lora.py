"""Episodic TabPFN-v2 LoRA using the canonical benchmark-holdout filters.

Run from repo root with --profile arshiaswe. Preparation is CPU-only; training
uses one L4. Checkpoints and frozen preprocessing live on aims-neurips-data.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import sys
import modal

app = modal.App("aims-tabpfn-lora-benchmark")
volume = modal.Volume.from_name("aims-neurips-data", create_if_missing=False)
ROOT = Path("/data/datasets/full")
RESULTS = Path("/data/results/tabpfn-lora")
BASE_REPO = "Prior-Labs/TabPFN-v2-clf"
BASE_FILE = "tabpfn-v2-classifier-v2_default.ckpt"
HERE = Path(__file__).resolve().parent
cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scikit-learn>=1.5,<2", "huggingface-hub>=0.23,<2"
).env({"HF_HOME": "/data/huggingface-cache", "TABPFN_NO_BROWSER": "1"})
gpu_image = cpu_image.pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu124").pip_install(
    "tabpfn==9.2.0", "wandb>=0.25,<1"
).env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TABPFN_MODEL_CACHE_DIR": "/data/tabpfn-cache"})
if str(HERE) != "/root":
    cpu_image = cpu_image.add_local_file(HERE / "benchmark_holdout_filters.py", "/root/benchmark_holdout_filters.py")
    gpu_image = gpu_image.add_local_file(HERE / "tabpfn_lora_core.py", "/root/tabpfn_lora_core.py")
    gpu_image = gpu_image.add_local_file(HERE / "benchmark_holdout_filters.py", "/root/benchmark_holdout_filters.py")

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

@app.function(image=cpu_image, cpu=8, memory=32_768, volumes={"/data": volume}, timeout=3600)
def prepare(split_seed=20261018, feature_dim=64):
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from huggingface_hub import HfApi, hf_hub_download
    import benchmark_holdout_filters as filters
    directory = ROOT / f"tabpfn-lora-benchmark-pca{feature_dim}-s{split_seed}-v1"
    identity = {"split_seed": split_seed, "feature_dim": feature_dim,
                "filter_sha256": digest(filters.__file__), "base_repo": BASE_REPO, "base_file": BASE_FILE}
    ready = directory / "prepared.json"
    if ready.exists():
        result = json.loads(ready.read_text())
        if result["identity"] != identity:
            raise ValueError("Prepared data identity changed; use a fresh cache version")
        for name, checksum in result["artifact_hashes"].items():
            if digest(directory / name) != checksum:
                raise ValueError(f"Prepared artifact checksum mismatch: {name}")
        return str(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identity_path = directory / "preparation-identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Partial preparation belongs to a different configuration")
    identity_path.write_text(json.dumps(identity, sort_keys=True))
    filtered_path = directory / "filtered.parquet"
    if filtered_path.exists() and (directory / "split-audit.json").exists():
        frame = pd.read_parquet(filtered_path)
        audit = json.loads((directory / "split-audit.json").read_text())
    else:
        print("Running canonical benchmark filters", flush=True)
        frame, audit = filters.load_benchmark_holdout_filtered(ROOT / "processed", split_seed=split_seed)
        temporary = filtered_path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(filtered_path)
        (directory / "split-audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
        volume.commit()
    split_map = audit["benchmark_split_map"]
    counts = {r: sum(v == r for v in split_map.values()) for r in ("train", "validation", "test")}
    if counts != {"train": 11, "validation": 3, "test": 3}:
        raise ValueError(f"Canonical data differs from requested 11/3/3 split: {counts}")
    (directory / "split-audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(f"Canonical benchmark split: {counts}", flush=True)
    store = ROOT / "allbench-embed-bge_m3_long-v2"
    ids = pd.read_parquet(store / "items.parquet", columns=["item_uid"])["item_uid"].astype(str)
    lookup = {uid: i for i, uid in enumerate(ids)}
    selected_ids = sorted(frame["item_uid"].astype(str).unique())
    missing = set(selected_ids) - lookup.keys()
    if missing:
        raise ValueError(f"{len(missing)} canonical items have no embeddings; refusing to change the split")
    raw = np.load(store / "item_embeddings.npy", mmap_mode="r")
    positions = np.array([lookup[i] for i in selected_ids])
    item_index = {uid: i for i, uid in enumerate(selected_ids)}
    train_ids = sorted(frame.loc[frame["benchmark_split"] == "train", "item_uid"].astype(str).unique())
    rng = np.random.default_rng(split_seed + 1000)
    fit_ids = rng.choice(train_ids, min(50_000, len(train_ids)), replace=False)
    fit_rows = np.array([lookup[i] for i in fit_ids])
    projection = PCA(n_components=feature_dim, svd_solver="randomized", random_state=split_seed)
    projection.fit(np.asarray(raw[fit_rows], dtype=np.float32))
    np.savez(directory / "projection.npz", mean=projection.mean_, components=projection.components_,
             fit_item_ids=fit_ids, explained_variance_ratio=projection.explained_variance_ratio_)
    features = np.lib.format.open_memmap(directory / "features.npy", mode="w+", dtype=np.float32,
                                          shape=(len(selected_ids), feature_dim))
    for start in range(0, len(positions), 4096):
        features[start:start + 4096] = projection.transform(raw[positions[start:start + 4096]])
    features.flush()
    tasks = []
    frame["_item_index"] = frame["item_uid"].astype(str).map(item_index).astype(np.int64)
    for (subject, benchmark), group in frame.groupby(["subject_uid", "benchmark_id"], sort=True):
        tasks.append({"subject": str(subject), "benchmark": str(benchmark),
                      "role": str(group["benchmark_split"].iloc[0]),
                      "items": group["_item_index"].to_numpy(dtype=np.int64),
                      "labels": group["label"].to_numpy(dtype=np.int64)})
    with (directory / "tasks.pkl").open("wb") as stream:
        pickle.dump(tasks, stream, protocol=5)
    print(f"Prepared {len(tasks)} tasks, {len(selected_ids)} items, PCA fit on {len(fit_ids)} training items", flush=True)
    base_revision = HfApi().model_info(BASE_REPO).sha
    base_path = hf_hub_download(BASE_REPO, BASE_FILE, revision=base_revision)
    result = {"identity": identity, "base_revision": base_revision, "base_path": base_path,
              "base_sha256": digest(base_path), "benchmark_counts": counts,
              "task_counts": {r: sum(t["role"] == r for t in tasks) for r in counts},
              "artifact_hashes": {name: digest(directory / name) for name in
                                  ("features.npy", "tasks.pkl", "projection.npz", "split-audit.json")}}
    ready.write_text(json.dumps(result, indent=2, sort_keys=True))
    volume.commit()
    return str(directory)

@app.function(image=gpu_image, gpu=["L4", "A10G"], cpu=4, memory=32_768, volumes={"/data": volume},
              secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])], timeout=6 * 3600)
def train(prepared_dir: str, run_label: str, steps=3000, seed=20261018, rank=8, alpha=16.,
          learning_rate=1e-4, validate_every=100, save_every=100, validation_tasks=64,
          episodes_per_step=4, query_size=32, smoke=False, interrupt_after=0):
    import copy
    import pickle
    import random
    import time
    import numpy as np
    import torch
    import wandb
    from tabpfn.model_loading import load_model_criterion_config
    import tabpfn_lora_core as core
    directory = Path(prepared_dir)
    info = json.loads((directory / "prepared.json").read_text())
    output = RESULTS / run_label
    output.mkdir(parents=True, exist_ok=True)
    config = {"format_version": 1, "seed": seed, "rank": rank, "alpha": alpha, "learning_rate": learning_rate,
              "validate_every": validate_every, "save_every": save_every, "validation_tasks": validation_tasks,
              "episodes_per_step": episodes_per_step, "query_size": query_size, "smoke": smoke,
              "prepared_dir": prepared_dir, "prepared_identity": info,
              "core_sha256": digest(core.__file__), "tabpfn_version": "9.2.0", "base_version": "v2",
              "zero_and_one_class_policy": "Beta(2,2) fallback", "precision": "float32", "evaluation_query_chunk": 256}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    completed = output / "completed.json"
    if completed.exists():
        previous = json.loads(completed.read_text())
        if previous["config_hash"] != config_hash or previous["steps"] != steps:
            raise ValueError("Completed run is immutable; use a new label")
        return previous
    features = np.load(directory / "features.npy", mmap_mode="r")
    with (directory / "tasks.pkl").open("rb") as stream:
        tasks = pickle.load(stream)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    if digest(info["base_path"]) != info["base_sha256"]:
        raise ValueError("Frozen base checkpoint checksum mismatch")
    models, _, _, _ = load_model_criterion_config(
        Path(info["base_path"]), check_bar_distribution_criterion=False,
        estimator_type="classifier", version="v2", download_if_not_exists=False)
    # The upstream loader can cache model instances. Never parametrize/mutate
    # that cached base, particularly when the same container handles resume.
    model = copy.deepcopy(models[0]).to("cuda")
    targets = core.attach_lora(model, rank, alpha)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1., (s + 1) / 50))
    bank = core.make_bank(tasks, "validation", info["identity"]["split_seed"] + 2000,
                          limit=3 if smoke else validation_tasks)
    last_path, best_path = output / "last.pt", output / "best.pt"
    order, history, next_step, best_score, best_step = [], [], 0, float("inf"), 0
    skipped = 0
    wb = wandb.init(project="aims-tabpfn-lora", name=run_label, id="lora-" + hashlib.sha256(run_label.encode()).hexdigest()[:18],
                    resume="allow", config={**config, "steps": steps, "trainable_parameters": sum(p.numel() for p in trainable),
                                            "adapter_targets": targets, "run_label": run_label})
    print(f"W&B: {wb.url}", flush=True)
    print(f"LoRA targets={len(targets)}, trainable={sum(p.numel() for p in trainable):,}, GPU={torch.cuda.get_device_name()}", flush=True)
    def checkpoint(step, best=False):
        state = {"format_version": 1, "config_hash": config_hash, "config": config,
                 "adapter_state": core.adapter_state(model), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "grad_scaler": None, "next_step": step,
                 "best_score": best_score, "best_step": best_step, "rng": core.capture_rng(rng),
                 "epoch_order": order, "history": history, "skipped_one_class_episodes": skipped,
                 "validation_bank": bank, "wandb_run_id": wb.id,
                 "projection": dict(np.load(directory / "projection.npz")),
                 "split_audit": json.loads((directory / "split-audit.json").read_text())}
        core.atomic_save(state, best_path if best else last_path)
        volume.commit()
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["config_hash"] != config_hash:
            raise ValueError("Resume configuration/data/base/code differs")
        core.load_adapters(model, state["adapter_state"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        next_step, best_score, best_step = state["next_step"], state["best_score"], state["best_step"]
        order, history, bank = state["epoch_order"], state["history"], state["validation_bank"]
        skipped = state["skipped_one_class_episodes"]
        core.restore_rng(state["rng"], rng)
        print(f"Resumed exact state at optimizer step {next_step}", flush=True)
    else:
        baseline = core.evaluate(model, tasks, bank, features, "cuda")
        (output / "baseline-validation.json").write_text(json.dumps(baseline, indent=2))
        best_score = baseline["brier_alc"]
        checkpoint(0, best=True)
        checkpoint(0)
        print(f"Base validation ALC={best_score:.6f}", flush=True)
    training_ids = [i for i, t in enumerate(tasks) if t["role"] == "train"]
    start_time = time.monotonic()
    for step in range(next_step, steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for _ in range(episodes_per_step):
            for attempt in range(1000):
                if not order:
                    order = rng.permutation(training_ids).tolist()
                task = tasks[order.pop()]
                k = int(rng.choice(core.BUDGETS[2:]))
                perm = rng.permutation(len(task["items"]))
                ci, ti = perm[:k], perm[k:k + query_size]
                yc = torch.tensor(task["labels"][ci], device="cuda")
                if len(torch.unique(yc)) == 2:
                    break
                skipped += 1
            else:
                raise RuntimeError("Could not sample a binary training context")
            xc = torch.tensor(features[task["items"][ci]], device="cuda")
            xt = torch.tensor(features[task["items"][ti]], device="cuda")
            yt = torch.tensor(task["labels"][ti], device="cuda", dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(core.logits(model, xc, yc, xt), yt) / episodes_per_step
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            total_loss += float(loss.detach())
        if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable):
            raise RuntimeError("LoRA gradients are zero; refusing no-op training")
        torch.nn.utils.clip_grad_norm_(trainable, 1.)
        optimizer.step()
        scheduler.step()
        record = {"step": step + 1, "train/loss": total_loss, "train/lr": scheduler.get_last_lr()[0]}
        if (step + 1) % validate_every == 0 or step + 1 == steps:
            validation = core.evaluate(model, tasks, bank, features, "cuda")
            record["validation/brier_alc"] = validation["brier_alc"]
            record.update({f"validation/brier_budget_{b}": v for b, v in validation["by_budget"].items()})
            if validation["brier_alc"] < best_score - 1e-6:
                best_score, best_step = validation["brier_alc"], step + 1
                (output / "best-validation.json").write_text(json.dumps(validation, indent=2))
                checkpoint(step + 1, best=True)
            print(f"step={step + 1} loss={total_loss:.6f} validation_ALC={validation['brier_alc']:.6f} best={best_score:.6f}", flush=True)
        history.append(record)
        wb.log(record, step=step + 1)
        if (step + 1) % save_every == 0 or step + 1 == steps:
            checkpoint(step + 1)
        if step == next_step:
            print(f"First optimizer update complete: loss={total_loss:.6f}", flush=True)
        if interrupt_after and step + 1 >= interrupt_after:
            checkpoint(step + 1)
            wb.finish()
            return {"interrupted_for_resume_test": True, "next_step": step + 1}
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    core.load_adapters(model, best["adapter_state"])
    if not smoke:
        for role in ("validation", "test"):
            result = core.evaluate(model, tasks, core.make_bank(tasks, role, info["identity"]["split_seed"] + 3000), features, "cuda")
            (output / f"final-{role}.json").write_text(json.dumps(result, indent=2))
            wb.summary[f"final/{role}_brier_alc"] = result["brier_alc"]
    summary = {"config_hash": config_hash, "steps": steps, "best_step": best_step, "best_validation_alc": best_score,
               "elapsed_seconds": time.monotonic() - start_time, "output": str(output), "wandb_url": wb.url, "smoke": smoke}
    completed.write_text(json.dumps(summary, indent=2))
    volume.commit()
    wb.finish()
    return summary

@app.local_entrypoint()
def main(run_label: str = "tabpfn-v2-lora-bge64-benchmark-20261029-v1", steps: int = 3000,
         seed: int = 20261018, split_seed: int = 20261018, feature_dim: int = 64,
         rank: int = 8, alpha: float = 16., smoke: bool = False, prepare_only: bool = False):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_label):
        raise ValueError("Invalid run label")
    call = workflow.spawn(run_label, steps, seed, split_seed, feature_dim, rank, alpha, smoke, prepare_only)
    print(f"Spawned detached workflow {call.object_id}; use modal run --detach", flush=True)

@app.function(cpu=.125, memory=256, timeout=8 * 3600)
def workflow(run_label, steps, seed, split_seed, feature_dim, rank, alpha, smoke, prepare_only):
    prepared = prepare.remote(split_seed, feature_dim)
    print(f"Prepared data: {prepared}", flush=True)
    if prepare_only:
        return
    options = dict(prepared_dir=prepared, run_label=run_label + ("-smoke" if smoke else ""),
                   steps=2 if smoke else steps, seed=seed, rank=rank, alpha=alpha, smoke=smoke)
    if smoke:
        options.update(validate_every=1, save_every=1, episodes_per_step=1)
        call = train.spawn(**options, interrupt_after=1)
        print(f"Smoke first-update call: {call.object_id}", flush=True)
        print(json.dumps(call.get(), indent=2), flush=True)
        call = train.spawn(**options)
        print(f"Smoke resume call: {call.object_id}", flush=True)
        print(json.dumps(call.get(), indent=2), flush=True)
    else:
        call = train.spawn(**options)
        print(f"Training call: {call.object_id}", flush=True)
        print(json.dumps(call.get(), indent=2), flush=True)
