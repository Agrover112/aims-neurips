"""Does the frozen latent-CNP-128 survive ridge-projected me5 vectors?

The ridge map reaches 0.899 held-out cosine against the qwen3_8b store. Cosine
does not answer the question that matters, because the checkpoint's first layer
is linear over these vectors and a 20% variance miss may or may not move its
predictions. This job answers it directly, by scoring the *frozen* checkpoint on
its own fixed validation bank twice:

* **true** -- the qwen3_8b vectors it was trained on. This must reproduce the
  checkpoint's recorded 0.1272059134, which is what validates the harness. If it
  does not, the comparison below means nothing and the number to fix is this one.
* **projected** -- ``normalize(me5 @ W)``. This is the submission-time vector.

Only the embedding matrix differs between the two passes: identical bank,
identical episodes, identical weights. The delta is therefore attributable to
the projection alone.

The split, the warm-item carve and the bank are rebuilt by replaying the
trainer's random draws in its exact order, since a fixed bank is only fixed
relative to that sequence. ``_load_filtered``, ``_episode_indices`` and
``_fixed_validation_metrics`` below are verbatim copies from
``codex_es_fixed_modal_cnp_train.py`` rather than imports, because importing
that module would construct its Modal app as a side effect. They must be kept
in step with it.

Run::

    modal run --profile aims-test \
      organized/submission/modal_ridge_brier_check.py::main
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "aims-np-ridge-brier-check"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
DATASETS = Path("/data/datasets/full")
ADAPTER_ROOT = Path("/data/results/submission/ridge-adapter")
RESULT_PATH = Path("/data/results/submission/ridge-brier-check.json")

TARGET_ARM = "qwen3_8b"
# Trainer constants. Changing any of these invalidates the bank.
MIN_SUBJECTS = 30
MIN_ITEMS_PER_TASK = 80
HELD_OUT_FRAC = 0.20
VAL_FRAC = 0.15
MIN_HALF = 4
MAX_CONTEXT = 64
FIXED_VAL_EPISODES = 64
FIXED_VAL_MAX_TARGETS = 256

# The checkpoint's own recorded selection score, for the harness check.
EXPECTED_TRUE_BRIER = 0.12720591342804255

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "torch==2.5.1"
)

_HERE = Path(__file__).resolve().parent
_BLOCKS = (_HERE.parent / "training" / "cnp_blocks.py").resolve()
_CHECKPOINT = (_HERE / "assets" / "latent-cnp128-qwen3-8b-seed20261018.pt").resolve()
if _BLOCKS.exists():
    image = image.add_local_file(_BLOCKS, "/root/cnp_blocks.py")
if _CHECKPOINT.exists():
    image = image.add_local_file(_CHECKPOINT, "/root/checkpoint.pt")


def _load_filtered(min_subjects):
    """Verbatim from codex_es_fixed_modal_cnp_train.py."""
    import pandas as pd

    obs = pd.read_parquet(PROCESSED / "observations.parquet",
                          columns=["subject_uid", "item_uid", "benchmark_id", "label"])
    meta = pd.read_parquet(PROCESSED / "items.parquet",
                           columns=["item_uid", "content_hash"])
    obs = obs.groupby(["subject_uid", "item_uid", "benchmark_id"], as_index=False)["label"].mean()
    obs["label"] = (obs["label"] >= 0.5).astype("int8")
    d = obs.merge(meta, on="item_uid", how="left")
    d = d.drop_duplicates(["content_hash", "subject_uid"])
    var = d.groupby("item_uid")["label"].agg(["mean"])
    d = d[d["item_uid"].isin(var[(var["mean"] > 0) & (var["mean"] < 1)].index)]
    n_sub = d.groupby("item_uid")["subject_uid"].nunique()
    d = d[d["item_uid"].isin(n_sub[n_sub >= min_subjects].index)]
    n_item = d.groupby("subject_uid")["item_uid"].nunique()
    d = d[d["subject_uid"].isin(n_item[n_item >= MIN_ITEMS_PER_TASK].index)]
    return d


def _episode_indices(task, rng, max_context=MAX_CONTEXT,
                     max_targets=FIXED_VAL_MAX_TARGETS):
    """Verbatim from codex_es_fixed_modal_cnp_train.py."""
    n = len(task["items"])
    k = int(rng.integers(0, min(max_context, n - 1) + 1))
    permutation = rng.permutation(n)
    target = permutation[k:k + max_targets]
    if len(target) == 0:
        target = permutation[-1:]
    return permutation[:k], target


def _fixed_validation_metrics(model, tasks, episode_bank, emb, device):
    """Verbatim from codex_es_fixed_modal_cnp_train.py, plus per-episode Brier."""
    import torch
    import torch.nn.functional as F

    context_nll, target_brier, target_nll = [], [], []
    for task_index, context_index, target_index in episode_bank:
        task = tasks[task_index]
        item_index = task["items"]
        labels = task["labels"]
        x_c = emb[torch.as_tensor(item_index[context_index], dtype=torch.long, device=device)]
        y_c = torch.as_tensor(labels[context_index], dtype=torch.float32, device=device)
        x_t = emb[torch.as_tensor(item_index[target_index], dtype=torch.long, device=device)]
        y_t = torch.as_tensor(labels[target_index], dtype=torch.float32, device=device)
        target_logits = model.logits(x_c, y_c, x_t)
        target_probability = torch.sigmoid(target_logits)
        target_brier.append(float(((target_probability - y_t) ** 2).mean()))
        target_nll.append(float(F.binary_cross_entropy_with_logits(target_logits, y_t)))
        if len(context_index):
            context_logits = model.logits(x_c, y_c, x_c)
            context_nll.append(float(F.binary_cross_entropy_with_logits(context_logits, y_c)))
    return {
        "context_reconstruction_nll": float(sum(context_nll) / len(context_nll)),
        "unseen_target_nll": float(sum(target_nll) / len(target_nll)),
        "unseen_target_brier": float(sum(target_brier) / len(target_brier)),
        "per_episode_brier": target_brier,
    }


@app.function(
    image=image,
    cpu=4,
    memory=32_768,
    volumes={"/data": volume},
    timeout=60 * 60,
)
def check(source_arm: str = "me5_instruct") -> dict:
    import numpy as np
    import pandas as pd
    import torch

    import cnp_blocks as cb

    device = torch.device("cpu")
    checkpoint = torch.load("/root/checkpoint.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed = int(config["seed"])
    print(f"checkpoint seed={seed} arm={config.get('arm')} width={config.get('width')}", flush=True)
    if str(config.get("arm")) != TARGET_ARM:
        raise RuntimeError(f"checkpoint arm is {config.get('arm')!r}, expected {TARGET_ARM!r}")

    # ---- embedding stores, aligned ----------------------------------------
    target_root = DATASETS / f"allbench-embed-{TARGET_ARM}-v2"
    source_root = DATASETS / f"allbench-embed-{source_arm}-v2"
    target_ids = pd.read_parquet(target_root / "items.parquet")["item_uid"].astype(str).to_numpy()
    source_ids = pd.read_parquet(source_root / "items.parquet")["item_uid"].astype(str).to_numpy()
    if target_ids.shape != source_ids.shape or not bool((target_ids == source_ids).all()):
        raise RuntimeError("store row order differs; an explicit join would be required")
    pos = {uid: i for i, uid in enumerate(target_ids)}
    target_all = np.load(target_root / "item_embeddings.npy", mmap_mode="r")
    source_all = np.load(source_root / "item_embeddings.npy", mmap_mode="r")
    W = np.load(ADAPTER_ROOT / f"W-{source_arm}-to-{TARGET_ARM}.npy").astype(np.float32)
    print(f"adapter W {W.shape}", flush=True)

    # ---- replay the trainer's draws, in its order -------------------------
    d = _load_filtered(MIN_SUBJECTS)
    print(f"filtered (n>={MIN_SUBJECTS}): {len(d):,} rows", flush=True)
    rng = np.random.default_rng(seed)

    d = d[d["item_uid"].isin(pos.keys())]
    items_all = sorted(d["item_uid"].unique())
    i_map = {u: i for i, u in enumerate(items_all)}
    rows = [pos[u] for u in items_all]

    tasks = []
    for (sub, bench), g in d.groupby(["subject_uid", "benchmark_id"]):
        tasks.append({"subject": sub, "benchmark": bench,
                      "items": g["item_uid"].map(i_map).to_numpy(),
                      "labels": g["label"].to_numpy().astype(np.float32)})
    print(f"pooled tasks: {len(tasks):,}  items: {len(items_all):,}", flush=True)

    n_s = len(tasks)
    perm = rng.permutation(n_s)
    n_test = max(1, int(n_s * HELD_OUT_FRAC))
    n_val = max(1, int(n_s * VAL_FRAC))
    test_ix = set(perm[:n_test].tolist())
    val_ix = set(perm[n_test:n_test + n_val].tolist())
    held_i = set(rng.choice(len(items_all), max(1, int(len(items_all) * HELD_OUT_FRAC)),
                            replace=False).tolist())
    held_i_arr = np.fromiter(held_i, dtype=np.int64)

    def _warm(t):
        keep = ~np.isin(t["items"], held_i_arr)
        return {**t, "items": t["items"][keep], "labels": t["labels"][keep]}

    val_tasks = [_warm(t) for i, t in enumerate(tasks) if i in val_ix]
    val_tasks = [t for t in val_tasks if len(t["items"]) > 2 * MIN_HALF]
    print(f"val tasks: {len(val_tasks):,}  held-out items: {len(held_i):,}", flush=True)

    fixed_rng = np.random.default_rng(seed + 10_000)
    bank = []
    fixed_task_indices = fixed_rng.choice(
        len(val_tasks), size=min(FIXED_VAL_EPISODES, len(val_tasks)), replace=False
    )
    for task_index in fixed_task_indices:
        task_index = int(task_index)
        bank.append((task_index, *_episode_indices(val_tasks[task_index], fixed_rng)))
    print(f"fixed validation bank: {len(bank)} episodes", flush=True)

    # ---- the two embedding matrices ---------------------------------------
    true_emb = torch.as_tensor(np.asarray(target_all[rows], dtype=np.float32)).to(device)
    source = np.asarray(source_all[rows], dtype=np.float32)
    projected = source @ W
    projected /= np.clip(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12, None)
    projected_emb = torch.as_tensor(projected.astype(np.float32)).to(device)
    cosine = float((projected * np.asarray(target_all[rows], dtype=np.float32)).sum(axis=1).mean())
    print(f"item matrix {tuple(true_emb.shape)}  cosine on these items: {cosine:.4f}", flush=True)

    # ---- the frozen model --------------------------------------------------
    model = cb.NeuralProcess(
        true_emb.shape[1],
        width=int(config["width"]), n_blocks=int(config["n_blocks"]),
        encoder=str(config["encoder"]), pooling=str(config["pooling"]),
        use_latent=bool(config["use_latent"]), latent_dim=int(config["latent_dim"]),
        residual_proj=bool(config["residual_proj"]), dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    true_metrics = _fixed_validation_metrics(model, val_tasks, bank, true_emb, device)
    projected_metrics = _fixed_validation_metrics(model, val_tasks, bank, projected_emb, device)

    true_brier = true_metrics["unseen_target_brier"]
    projected_brier = projected_metrics["unseen_target_brier"]
    harness_delta = abs(true_brier - EXPECTED_TRUE_BRIER)

    per_true = np.array(true_metrics["per_episode_brier"])
    per_projected = np.array(projected_metrics["per_episode_brier"])

    summary = {
        "source_arm": source_arm,
        "seed": seed,
        "n_episodes": len(bank),
        "mean_item_cosine": round(cosine, 5),
        "true": {
            "unseen_target_brier": round(true_brier, 7),
            "unseen_target_nll": round(true_metrics["unseen_target_nll"], 6),
        },
        "projected": {
            "unseen_target_brier": round(projected_brier, 7),
            "unseen_target_nll": round(projected_metrics["unseen_target_nll"], 6),
        },
        "brier_delta": round(projected_brier - true_brier, 7),
        "brier_delta_pct": round(100.0 * (projected_brier - true_brier) / true_brier, 2),
        "episodes_worse": int((per_projected > per_true).sum()),
        "harness_check": {
            "expected_true_brier": EXPECTED_TRUE_BRIER,
            "observed_true_brier": round(true_brier, 7),
            "abs_delta": round(harness_delta, 7),
            "reproduced": bool(harness_delta < 1e-3),
        },
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()

    print("\n" + "=" * 66, flush=True)
    print(f"harness reproduced:  {summary['harness_check']['reproduced']}"
          f"  (expected {EXPECTED_TRUE_BRIER:.7f}, got {true_brier:.7f})", flush=True)
    print(f"true  Brier:         {true_brier:.7f}", flush=True)
    print(f"proj  Brier:         {projected_brier:.7f}", flush=True)
    print(f"delta:               {summary['brier_delta']:+.7f}"
          f"  ({summary['brier_delta_pct']:+.2f}%)", flush=True)
    print(f"episodes worse:      {summary['episodes_worse']}/{len(bank)}", flush=True)
    print("=" * 66, flush=True)
    return summary


@app.local_entrypoint()
def main(source_arm: str = "me5_instruct") -> None:
    check.remote(source_arm=source_arm)
