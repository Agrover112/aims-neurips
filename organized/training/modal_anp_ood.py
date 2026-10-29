"""Four matched CPU ANP OOD experiments with benchmark-balanced episodes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pickle
import modal

app = modal.App("aims-anp-ood-four-way")
volume = modal.Volume.from_name("aims-neurips-data", create_if_missing=False)
PREPARED = Path("/data/datasets/full/tabpfn-lora-benchmark-pca64-s20261018-v1")
RAW = Path("/data/datasets/full/anp-raw-bgem3-benchmark-s20261018-v1")
RESULTS = Path("/data/results/anp-ood")
HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu"
).pip_install("numpy>=2,<3", "scipy>=1.11,<2", "scikit-learn>=1.5,<2", "wandb>=0.25,<1").env(
    {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
prepare_image = image.pip_install("pandas>=2.2,<3", "pyarrow>=17,<24")
if str(HERE) != "/root":
    for name in ("anp_ood_core.py", "cnp_blocks.py", "anp_crossattn_blocks.py", "tabpfn_lora_core.py"):
        image = image.add_local_file(HERE / name, "/root/" + name)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

@app.function(image=prepare_image, cpu=2, memory=8192, timeout=1800, volumes={"/data": volume})
def prepare_raw():
    import numpy as np
    import pandas as pd
    info = json.loads((PREPARED / "prepared.json").read_text())
    if info["benchmark_counts"] != {"train": 11, "validation": 3, "test": 3}:
        raise ValueError("Expected canonical 11/3/3 split")
    identity = {"prepared": info, "filtered_sha256": sha(PREPARED / "filtered.parquet")}
    RAW.mkdir(parents=True, exist_ok=True)
    ready = RAW / "ready.json"
    if ready.exists():
        result = json.loads(ready.read_text())
        if result["identity"] != identity or sha(RAW / "features.npy") != result["features_sha256"]:
            raise ValueError("Raw feature cache identity/checksum mismatch")
        return result
    ids = sorted(pd.read_parquet(PREPARED / "filtered.parquet", columns=["item_uid"])["item_uid"].astype(str).unique())
    if len(ids) != np.load(PREPARED / "features.npy", mmap_mode="r").shape[0]:
        raise ValueError("Item universe differs")
    store = Path("/data/datasets/full/allbench-embed-bge_m3_long-v2")
    source_ids = pd.read_parquet(store / "items.parquet", columns=["item_uid"])["item_uid"].astype(str)
    position = {uid: i for i,uid in enumerate(source_ids)}
    selected = np.array([position[uid] for uid in ids])
    source = np.load(store / "item_embeddings.npy", mmap_mode="r")
    temporary = RAW / "features.npy.tmp"
    out = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(ids), source.shape[1]))
    for start in range(0, len(ids), 2048):
        out[start:start+2048] = source[selected[start:start+2048]]
    out.flush()
    del out
    temporary.replace(RAW / "features.npy")
    result = {"identity": identity, "features_sha256": sha(RAW / "features.npy"), "shape": [len(ids), int(source.shape[1])],
              "source_manifest": json.loads((store / "manifest.json").read_text())}
    ready.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(f"Prepared frozen raw BGE-M3: {result['shape']}", flush=True)
    return result

@app.function(image=image, cpu=2, memory=8192, timeout=6*3600, volumes={"/data": volume},
              secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])])
def train(variant: str, group: str, raw_info: dict, steps=3000, smoke=False, interrupt_after=0):
    import copy
    import random
    import time
    import numpy as np
    import torch
    import wandb
    import anp_ood_core as core
    import anp_crossattn_blocks as architecture
    import tabpfn_lora_core as protocol
    if variant not in core.VARIANTS:
        raise ValueError(variant)
    torch.set_num_threads(2)
    seed = 20261018
    info = json.loads((PREPARED / "prepared.json").read_text())
    for name in ("tasks.pkl", "split-audit.json"):
        if sha(PREPARED / name) != info["artifact_hashes"][name]:
            raise ValueError("Prepared task/split checksum mismatch")
    if sha(RAW / "features.npy") != raw_info["features_sha256"]:
        raise ValueError("Raw features changed")
    with (PREPARED / "tasks.pkl").open("rb") as stream:
        tasks = pickle.load(stream)
    features = torch.from_numpy(np.load(RAW / "features.npy"))
    groups = core.training_groups(tasks)
    if len(groups) != 11:
        raise ValueError("Wrong training benchmark count")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = architecture.AttentiveNeuralProcess(features.shape[1], width=64, self_attention_blocks=1,
                    cross_attention_heads=4, use_latent=True, latent_dim=32, dropout=0.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    validate_every = 1 if smoke else 100
    swa_start, swa_every = (1, 1) if smoke else (600, 10)
    config = {"variant": variant, "seed": seed, "steps": steps, "smoke": smoke,
              "architecture": {"embedding_dim": int(features.shape[1]), "width": 64, "sa": 1, "heads": 4, "latent_dim": 32},
              "raw": raw_info, "optimizer": "AdamW", "lr": 3e-4, "weight_decay": .01, "max_context": 64,
              "max_targets": 32 if smoke else 128, "episodes_per_step": 11, "sampling": "one uniformly sampled task per training benchmark",
              "swa_start": swa_start, "swa_every": swa_every, "sam_rho": .05, "groupdro_eta": .01,
              "groupdro_loss_ema": .9, "validation_tasks": 3 if smoke else 64,
              "diagnostic_tasks": 11 if smoke else 32, "diagnostic_context_cap": 64, "diagnostic_target_cap": 512,
              "test_evaluation": False, "core_sha256": sha(core.__file__), "architecture_sha256": sha(architecture.__file__),
              "projection_code_sha256": sha("/root/cnp_blocks.py"), "protocol_sha256": sha(protocol.__file__)}
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output = RESULTS / group / variant
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "completed.json"
    if completed.exists():
        result = json.loads(completed.read_text())
        if result["identity"] != identity:
            raise ValueError("Completed configuration differs")
        return result
    bank = protocol.make_bank(tasks, "validation", seed+2000, limit=3 if smoke else 64)
    diagnostic_bank = protocol.make_bank(tasks, "train", seed+5000, limit=11 if smoke else 32)
    wb = wandb.init(project="aims-predictive-evaluation", group=group, name=f"anp-{variant}--w64-sa1-h4-z32--{group}",
                    id="ood-" + hashlib.sha256(f"{group}/{variant}".encode()).hexdigest()[:18], resume="allow", config=config,
                    tags=["anp", "benchmark-heldout", "ood", variant])
    print(f"{variant}: W&B {wb.url}", flush=True)
    history, next_step, best, best_step = [], 0, float("inf"), None
    dro, swa, n_averaged = None, None, 0
    draws = np.zeros(len(tasks), dtype=np.int64)
    target_counts = np.zeros(len(features), dtype=np.int64)
    last, best_path = output / "last.pt", output / "best.pt"
    def save_checkpoint(step, selected_model=None, selected=False):
        state = {"format_version": 1, "identity": identity, "config": config, "next_step": step,
                 "online_state_dict": model.state_dict(), "state_dict": (selected_model or model).state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": None, "rng": protocol.capture_rng(rng),
                 "groupdro": dro, "swa_state": swa, "n_averaged": n_averaged,
                 "history": history, "best_validation_alc": best, "best_step": best_step,
                 "validation_bank": bank, "diagnostic_bank": diagnostic_bank,
                 "task_draw_counts": draws, "target_counts": target_counts, "wandb_id": wb.id,
                 "weight_source": "swa" if selected and variant == "swa" and n_averaged else "online"}
        protocol.atomic_save(state, best_path if selected else last)
        volume.commit()
    if last.exists():
        state = torch.load(last, weights_only=False, map_location="cpu")
        if state["identity"] != identity:
            raise ValueError("Resume configuration/data/code mismatch")
        model.load_state_dict(state["online_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        protocol.restore_rng(state["rng"], rng)
        next_step, best, best_step = state["next_step"], state["best_validation_alc"], state["best_step"]
        dro, swa, n_averaged = state["groupdro"], state["swa_state"], state["n_averaged"]
        history, draws, target_counts = state["history"], state["task_draw_counts"], state["target_counts"]
        bank, diagnostic_bank = state["validation_bank"], state["diagnostic_bank"]
        print(f"{variant}: resumed optimizer step {next_step}", flush=True)
    else:
        save_checkpoint(0)
    started = time.monotonic()
    for step in range(next_step, steps):
        model.train()
        batch, metadata = core.sample_batch(tasks, groups, features, rng, max_targets=config["max_targets"])
        for index, benchmark, ci, ti in metadata:
            draws[index] += 1
            np.add.at(target_counts, ti, 1)
        if variant == "sam":
            objective, perturbed = core.sam_step(model, optimizer, lambda: core.losses(model, batch).mean(), rho=.05)
        else:
            optimizer.zero_grad(set_to_none=True)
            group_losses = core.losses(model, batch)
            if variant == "groupdro":
                dro, q = core.dro_weights(dro, group_losses)
                loss = (q * group_losses).sum()
            else:
                loss = group_losses.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            objective = float(loss.detach())
        if variant == "swa" and step+1 >= swa_start and (step+1-swa_start) % swa_every == 0:
            swa, n_averaged = core.average_state(swa, n_averaged, model.state_dict())
        record = {"optimizer_step": step+1, "training/objective": objective, "training/episodes": (step+1)*len(groups)}
        if variant == "sam":
            record["training/sam_perturbed_objective"] = perturbed
        if (step+1) % validate_every == 0 or step+1 == steps:
            evaluated = model
            if variant == "swa" and n_averaged:
                evaluated = copy.deepcopy(model)
                evaluated.load_state_dict(swa)
            validation = core.evaluate(evaluated, tasks, bank, features)
            record.update({"validation/brier_alc": validation["brier_alc"],
                           **{f"validation/brier_budget_{b}": v for b,v in validation["by_budget"].items()},
                           **core.diagnostics(evaluated, tasks, diagnostic_bank, features, "training_subjects"),
                           **core.diagnostics(evaluated, tasks, bank, features, "heldout_subjects"),
                           "swa/n_averaged": n_averaged})
            q = dro["logq"].exp().numpy() if dro is not None else np.full(len(groups), 1/len(groups))
            for i,name in enumerate(groups):
                record[f"groups/{name}/weight"] = float(q[i])
                record[f"groups/{name}/episodes"] = int(draws[groups[name]].sum())
            eligible = variant != "swa" or n_averaged >= (1 if smoke else 5)
            history.append(record.copy())
            if eligible and validation["brier_alc"] < best:
                best, best_step = validation["brier_alc"], step+1
                save_checkpoint(step+1, evaluated, selected=True)
                (output / "best-validation.json").write_text(json.dumps(validation, indent=2))
            (output / "curves.json").write_text(json.dumps(history, indent=2))
            save_checkpoint(step+1)
            print(f"{variant}: step={step+1} validation_ALC={validation['brier_alc']:.6f} best={best:.6f} swa_count={n_averaged}", flush=True)
        wb.log(record, step=step+1)
        if step == next_step:
            print(f"{variant}: first update complete, loss={objective:.6f}", flush=True)
        if interrupt_after and step+1 >= interrupt_after:
            save_checkpoint(step+1)
            wb.finish()
            return {"interrupted_for_resume_test": True, "next_step": step+1}
    selected = torch.load(best_path, weights_only=False, map_location="cpu")
    model.load_state_dict(selected["state_dict"])
    if not smoke:
        final = core.evaluate(model, tasks, protocol.make_bank(tasks, "validation", seed+3000), features)
        (output / "final-validation.json").write_text(json.dumps(final, indent=2))
        wb.summary["final/full_validation_alc"] = final["brier_alc"]
    result = {"variant": variant, "identity": identity, "steps": steps, "best_validation_alc": best,
              "best_step": best_step, "weight_source": selected["weight_source"], "output": str(output),
              "wandb_url": wb.url, "elapsed_seconds": time.monotonic()-started, "test_evaluated": False, "smoke": smoke}
    completed.write_text(json.dumps(result, indent=2))
    volume.commit()
    wb.finish()
    return result

@app.function(cpu=.125, memory=256, timeout=8*3600)
def experiment(variant, group, raw_info, smoke):
    if smoke:
        print(json.dumps(train.remote(variant, group, raw_info, steps=2, smoke=True, interrupt_after=1)), flush=True)
    result = train.remote(variant, group, raw_info, steps=2 if smoke else 3000, smoke=smoke)
    print(json.dumps(result), flush=True)
    return result

@app.function(cpu=.125, memory=256, timeout=8*3600)
def workflow(group, smoke):
    raw_info = prepare_raw.remote()
    if not smoke:
        checks = [(v, experiment.spawn(v, group + "-smoke", raw_info, True))
                  for v in ("balanced_erm", "groupdro", "swa", "sam")]
        for name, call in checks:
            print(f"Checking {name}: {call.object_id}", flush=True)
        # A failure prevents every full run from being launched.
        for name, call in checks:
            call.get()
            print(f"Smoke/resume passed: {name}", flush=True)
    calls = [(v, experiment.spawn(v, group, raw_info, smoke)) for v in ("balanced_erm", "groupdro", "swa", "sam")]
    for name, call in calls:
        print(f"{name}: {call.object_id}", flush=True)
    results, errors = [], []
    for name, call in calls:
        try:
            results.append(call.get())
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return results

@app.local_entrypoint()
def main(smoke: bool = False, group: str = "anp-ood-bge1024-20261029-v1"):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", group):
        raise ValueError("Invalid group")
    call = workflow.spawn(group + ("-smoke" if smoke else ""), smoke)
    print(f"Spawned workflow {call.object_id}; use --detach", flush=True)
