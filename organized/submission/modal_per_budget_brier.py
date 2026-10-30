"""Per-budget Brier for a benchmark-holdout NP against the shipped baseline.

The aggregate fixed-bank number hides the thing that decides the score. The
ALC puts 70% of its weight on budgets of at most seven labels, and a single
averaged Brier cannot say whether a model is weak there, weak everywhere, or
ignoring its context labels altogether. A flat curve in ``k`` means the model
is not reading ``y_c`` at all, which is a defect rather than a modelling gap,
and the two call for opposite responses.

Every reference below is scored on the *same* episodes as the model, so the
comparison is a difference in estimator and nothing else:

``baseline``
    The shipped 0.1486 submission's estimator, copied from
    ``submission/aims_predictor/predictor.py:91`` rather than reimplemented:
    nested beta-binomial shrinkage over the subject's acquired labels. It uses
    no item information whatsoever, which is the point -- it is the bar that a
    model holding 1,024-dimensional item embeddings has to clear.
``const_half``
    The baseline's own budget-0 output, since both shrinkage stages collapse
    to ``prior_mean`` with no evidence. Scores 0.25 by construction.
``const_train_mean``
    The best possible constant, fit on the training benchmarks. Separates
    "knows the base rate" from "knows anything else".
``subject_mean_oracle``
    The task's true target mean, which no submission can see. This is the
    ceiling for any estimator that predicts one number per subject, so the gap
    between ``baseline`` and this is all that subject-ability modelling can
    ever buy, and anything beyond it must come from item difficulty.

Budgets are nested as in scoring, and the target set is held fixed across
budgets at ``perm[max(BUDGETS):][:MAX_TARGETS]`` so that a change in Brier is
attributable to context size alone. Scoring the shrinking remainder instead
would confound budget with target composition.

Run::

    modal run --profile aims-test \
      organized/submission/modal_per_budget_brier.py::main \
      --run-label anp-sa1-w64-bgem3-filtered-benchmarkholdout-n15-20261029
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "aims-np-per-budget-brier"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
DATASETS = Path("/data/datasets/full")
CHECKPOINTS = Path("/data/results/cnp/checkpoints")
OUT_ROOT = Path("/data/results/submission/per-budget-brier")

BUDGETS = (0, 1, 3, 7, 15, 31)
BUDGET_WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
MAX_TARGETS = 256
MIN_TARGETS = 8
N_TASKS = 256          # validation subjects to score; 64 is too noisy to rank on

# Copied from submission/aims_predictor/predictor.py, not re-derived.
PRIOR_MEAN = 0.5
PRIOR_STRENGTH = 4.0
SUBJECT_STRENGTH = 8.0

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "torch==2.5.1"
)

_HERE = Path(__file__).resolve().parent
_TRAINING = (_HERE.parent / "training").resolve()
for _name in ("cnp_blocks.py", "anp_crossattn_blocks.py", "benchmark_holdout_filters.py"):
    _path = _TRAINING / _name
    if _path.exists():
        image = image.add_local_file(_path, f"/root/{_name}")


def _baseline_probability(successes: int, count: int) -> float:
    """The shipped submission's estimate for a context drawn from one task.

    Context items share the subject *and* the benchmark, so the predictor's
    subject-level and group-level tallies coincide; both stages are kept
    anyway so this stays a copy of the shipped math rather than a
    simplification of it.
    """
    subject_mean = (PRIOR_STRENGTH * PRIOR_MEAN + successes) / (PRIOR_STRENGTH + count)
    estimate = (SUBJECT_STRENGTH * subject_mean + successes) / (SUBJECT_STRENGTH + count)
    return float(min(1.0, max(0.0, estimate)))


@app.function(
    image=image,
    cpu=4,
    memory=32_768,
    volumes={"/data": volume},
    timeout=60 * 60,
)
def check(run_label: str, arm: str = "bge_m3_long", seed: int = 20261018,
          min_item_responders: int = 15, min_items_per_subject: int = 80,
          n_tasks: int = N_TASKS) -> dict:
    import sys

    import numpy as np
    import pandas as pd
    import torch

    sys.path.insert(0, "/root")
    import anp_crossattn_blocks as anp_cb
    import benchmark_holdout_filters as benchmark_filters

    device = torch.device("cpu")
    torch.set_num_threads(4)

    # ---- the checkpoint names its own architecture; nothing is assumed -----
    matches = sorted(
        p for p in CHECKPOINTS.glob("*.pt")
        if run_label in p.name and not p.name.endswith(".resume.pt")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"run_label {run_label!r} matched {len(matches)} checkpoints: "
            f"{[p.name for p in matches]}"
        )
    checkpoint = torch.load(matches[0], map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    selection = checkpoint.get("selection", {})
    print(f"checkpoint {matches[0].name}", flush=True)
    print(f"  recorded selection: {selection}", flush=True)
    if cfg["arm"] != arm:
        raise RuntimeError(
            f"checkpoint was trained on arm {cfg['arm']!r}; refusing to score it on "
            f"{arm!r}. Same width, different space -- it would load and lie."
        )

    # ---- data, exactly as the trainer carves it ----------------------------
    d, audit = benchmark_filters.load_benchmark_holdout_filtered(
        PROCESSED,
        min_item_responders=min_item_responders,
        min_items_per_subject=min_items_per_subject,
        split_seed=seed,
    )
    store = DATASETS / f"allbench-embed-{arm}-v2"
    emb_ids = pd.read_parquet(store / "items.parquet")["item_uid"].astype(str).to_numpy()
    emb_all = np.load(store / "item_embeddings.npy", mmap_mode="r")
    pos = {u: i for i, u in enumerate(emb_ids)}

    d = d[d["item_uid"].isin(pos.keys())]
    items_all = sorted(d["item_uid"].unique())
    i_map = {u: i for i, u in enumerate(items_all)}
    emb = torch.as_tensor(
        np.asarray(emb_all[[pos[u] for u in items_all]], dtype=np.float32)
    ).to(device)
    print(f"pooled item matrix: {tuple(emb.shape)}; splits {audit['final']['by_split']}",
          flush=True)

    tasks, train_label_sum, train_label_n = [], 0.0, 0
    for (subject, benchmark), g in d.groupby(["subject_uid", "benchmark_id"]):
        role = str(g["benchmark_split"].iloc[0])
        labels = g["label"].to_numpy().astype(np.float32)
        if role == "train":
            train_label_sum += float(labels.sum())
            train_label_n += int(labels.size)
            continue
        if role != "validation":
            continue
        tasks.append({"subject": subject, "benchmark": benchmark,
                      "items": g["item_uid"].map(i_map).to_numpy(), "labels": labels})
    train_mean = train_label_sum / max(train_label_n, 1)
    print(f"validation tasks: {len(tasks):,}; training label mean {train_mean:.4f}",
          flush=True)

    model = anp_cb.AttentiveNeuralProcess(
        emb.shape[1],
        width=int(cfg["width"]),
        self_attention_blocks=int(cfg["n_blocks"]),
        cross_attention_heads=int(cfg["cross_attention_heads"]),
        use_latent=bool(cfg["use_latent"]),
        latent_dim=int(cfg["latent_dim"]),
        residual_proj=bool(cfg["residual_proj"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    # ---- score every estimator on identical episodes ----------------------
    top = max(BUDGETS)
    rng = np.random.default_rng(seed + 777)
    order = rng.permutation(len(tasks))
    squares = {name: {k: [] for k in BUDGETS}
               for name in ("model", "baseline", "const_half",
                            "const_train_mean", "subject_mean_oracle")}
    scored = 0

    with torch.no_grad():
        for task_index in order:
            task = tasks[int(task_index)]
            items, labels = task["items"], task["labels"]
            if len(items) < top + MIN_TARGETS:
                continue
            perm = rng.permutation(len(items))
            target = perm[top:top + MAX_TARGETS]
            y = labels[target]
            x_t = emb[torch.as_tensor(items[target], dtype=torch.long)]
            oracle = float(y.mean())

            for k in BUDGETS:
                context = perm[:k]
                y_c = labels[context]
                x_c = emb[torch.as_tensor(items[context], dtype=torch.long)]
                p = torch.sigmoid(
                    model.logits(x_c, torch.as_tensor(y_c, dtype=torch.float32), x_t)
                ).numpy()
                squares["model"][k].append(float(((p - y) ** 2).mean()))
                flat = _baseline_probability(int(y_c.sum()), int(k))
                squares["baseline"][k].append(float(((flat - y) ** 2).mean()))
                squares["const_half"][k].append(float(((0.5 - y) ** 2).mean()))
                squares["const_train_mean"][k].append(float(((train_mean - y) ** 2).mean()))
                squares["subject_mean_oracle"][k].append(float(((oracle - y) ** 2).mean()))

            scored += 1
            if scored >= n_tasks:
                break

    if not scored:
        raise RuntimeError(
            f"no validation task had >= {top + MIN_TARGETS} items; nothing was scored"
        )

    per_budget = {name: {k: float(np.mean(v)) for k, v in budgets.items()}
                  for name, budgets in squares.items()}
    alc = {name: float(sum(w * per_budget[name][k] for k, w in zip(BUDGETS, BUDGET_WEIGHTS)))
           for name in per_budget}

    print("", flush=True)
    header = f"{'budget':>7} " + "".join(f"{n:>20}" for n in per_budget)
    print(header, flush=True)
    for k in BUDGETS:
        print(f"{k:>7} " + "".join(f"{per_budget[n][k]:>20.5f}" for n in per_budget),
              flush=True)
    print(f"{'ALC':>7} " + "".join(f"{alc[n]:>20.5f}" for n in per_budget), flush=True)
    print("", flush=True)
    spread = per_budget["model"][min(BUDGETS)] - per_budget["model"][max(BUDGETS)]
    print(f"model Brier gain from 0 -> {max(BUDGETS)} labels: {spread:+.5f}", flush=True)
    if abs(spread) < 0.005:
        print("  FLAT in k: the model is barely reading its context labels.", flush=True)
    print(f"model ALC minus baseline ALC: {alc['model'] - alc['baseline']:+.5f}", flush=True)

    summary = {
        "run_label": run_label,
        "checkpoint": matches[0].name,
        "recorded_selection": selection,
        "arm": arm,
        "n_tasks_scored": scored,
        "train_label_mean": train_mean,
        "budgets": list(BUDGETS),
        "budget_weights": list(BUDGET_WEIGHTS),
        "per_budget_brier": {n: {str(k): v for k, v in b.items()}
                             for n, b in per_budget.items()},
        "alc_brier": alc,
        "model_gain_0_to_max": spread,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"{run_label}.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    return summary


@app.local_entrypoint()
def main(run_label: str, arm: str = "bge_m3_long", seed: int = 20261018,
         min_item_responders: int = 15, min_items_per_subject: int = 80,
         n_tasks: int = N_TASKS) -> None:
    check.remote(run_label=run_label, arm=arm, seed=seed,
                 min_item_responders=min_item_responders,
                 min_items_per_subject=min_items_per_subject, n_tasks=n_tasks)
