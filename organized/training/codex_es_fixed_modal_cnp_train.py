"""Train the CNP/NP family with fixed-bank validation checkpoint selection.

Architecture lives in ``cnp_blocks.py`` so it is unit-testable without Modal;
this file is data, training loop, validation and logging.

    --encoder   linear | mhsa | set_transformer | sgformer
    --pooling   mean | mhsa
    --use-latent            latent NP (ELBO) instead of deterministic CNP
    --width 64              working width (see cnp_blocks docstring: 64, not 256)

Validation is a held-out **subject** split, not a held-out slice of the same
subjects' responses: the quantity we care about is generalising to a new
test-taker, so the validation loss has to be measured that way or it will look
healthy while the thing we need degrades.

This is a copy of ``modal_cnp_train.py`` with one training-control change:
training always completes its scheduled steps, and the restored checkpoint is
the lowest Brier score on one fixed bank of held-out-subject episodes. The
original script chose the lowest loss from newly sampled validation episodes,
which made checkpoint selection dominated by episode noise.

W&B is optional. With a ``wandb`` Modal secret present the run streams live
curves; without one it still trains and writes the curve to the volume, so a
missing secret degrades to "no live plot" rather than "no run".

Run:  modal run --detach codex_es_fixed_modal_cnp_train.py::main --encoder sgformer
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "aims-neurips-cnp-train-codex-es-fixed"
VOLUME_NAME = "aims-neurips-data"
PROCESSED = Path("/data/datasets/full/processed")
DATASETS = Path("/data/datasets/full")
OUT = Path("/data/results/cnp/cnp-codex-es-fixed.parquet")
CHECKPOINTS = Path("/data/results/cnp/checkpoints")

ARM = "qwen3_8b"
MIN_SUBJECTS = 30
MIN_ITEMS_PER_TASK = 80
HELD_OUT_FRAC = 0.20
VAL_FRAC = 0.15
MIN_HALF = 4
MIN_MATRIX = 20
MAX_CONTEXT = 64
STEPS = 3000
EPISODES_PER_STEP = 8
LR = 3e-4
WEIGHT_DECAY = 1e-2   # AdamW default; decoupled, so it does not scale with LR
VAL_EVERY = 50
VAL_EPISODES = 32      # fresh episodic loss, logged only as a diagnostic
FIXED_VAL_EPISODES = 64
FIXED_VAL_MAX_TARGETS = 256
COVERAGE_HIST_EVERY = 250  # W&B histogram; scalar coverage logs every validation
PATIENCE = 10_000      # deliberately exceeds the 3000-step budget
MAX_EVAL_TARGETS = 512
CURVE_SUBJECTS = 24    # subsample for the periodic per-axis curves
MAX_EPISODE_TARGETS = 512  # targets include context; cap for cost only
                       # per budget is the same estimator as scoring a sample
SEEDS = (20261018, 20261019, 20261020)
# Budget ladder lives in organized/testing/online/, not here.

WANDB_ENTITY = "agrgnn-kth-royal-institute-of-technology"
WANDB_PROJECT = "aims-predictive-evaluation"

# Names used in W&B and the saved curve parquet.  The short internal names
# (``cell`` etc.) were convenient code handles, but not adequate chart labels.
PROTOCOL_LABELS = {
    "cell": "same_subject_response_holdout",
    "subject_splithalf": "same_subject_split_half",
    "item_coldstart": "held_out_item_cold_start",
}
SCORE_LABELS = {"auc": "roc_auc", "brier": "brier_score"}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "scikit-learn>=1.5,<2",
    "tqdm>=4.66,<5", "wandb>=0.17,<1")
image = image.pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")

# Ship the architecture module. The container copy of this file lives at /root,
# so guard on BOTH existence and not-already-in-the-container: a bare
# .exists() check passes inside the container after shipping and re-adds the
# layer during Modal's re-import, and a bare parents[] index raises IndexError
# there. Both of those crash-looped jobs earlier in this project.
_HERE = Path(__file__).resolve().parent
_BLOCKS = _HERE / "cnp_blocks.py"
if str(_HERE) != "/root" and _BLOCKS.exists():
    image = image.add_local_file(_BLOCKS, "/root/cnp_blocks.py")

# Optional: live W&B curves need a secret holding WANDB_API_KEY.
try:
    _SECRETS = [modal.Secret.from_name("wandb")]
except Exception:
    _SECRETS = []


def _load_filtered(min_subjects):
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


class _TrainingCoverage:
    """Exact coverage of the *training* episode stream, not an estimate.

    Response cells are indexed within their subject-functions. Item counts
    aggregate those target draws across training subjects, so they answer both
    ``did every response contribute?`` and ``which items were repeatedly
    trained on?`` without retaining predictions or labels.
    """

    def __init__(self, tasks, n_items):
        import numpy as np

        self.tasks = tasks
        self.response_target_count = [
            np.zeros(len(task["items"]), dtype=np.uint16) for task in tasks
        ]
        self.item_target_count = np.zeros(n_items, dtype=np.uint32)
        self.eligible_item = np.zeros(n_items, dtype=bool)
        for task in tasks:
            self.eligible_item[task["items"]] = True
        self.task_draw_count = np.zeros(len(tasks), dtype=np.uint32)
        self.response_target_seen = 0
        self.item_target_seen = 0
        self.task_seen = 0
        self.target_draws_total = 0

    def record(self, task_index, target_index):
        """Record one target set. Target indices are unique within an episode."""
        import numpy as np

        response_counts = self.response_target_count[task_index]
        unseen_response = response_counts[target_index] == 0
        self.response_target_seen += int(unseen_response.sum())
        response_counts[target_index] += 1

        item_index = self.tasks[task_index]["items"][target_index]
        unseen_item = self.item_target_count[item_index] == 0
        self.item_target_seen += int(unseen_item.sum())
        self.item_target_count[item_index] += 1

        if self.task_draw_count[task_index] == 0:
            self.task_seen += 1
        self.task_draw_count[task_index] += 1
        self.target_draws_total += len(target_index)

    def metrics(self):
        import numpy as np

        item_counts = self.item_target_count[self.eligible_item]
        n_responses = sum(len(counts) for counts in self.response_target_count)
        n_items = len(item_counts)
        n_tasks = len(self.tasks)
        return {
            "coverage/training_response_target_unique_fraction": self.response_target_seen / n_responses,
            "coverage/training_response_target_unseen_count": n_responses - self.response_target_seen,
            "coverage/training_item_target_unique_fraction": self.item_target_seen / n_items,
            "coverage/training_item_target_unseen_count": n_items - self.item_target_seen,
            "coverage/training_item_target_exposure_mean": float(item_counts.mean()),
            "coverage/training_item_target_exposure_min": int(item_counts.min()),
            "coverage/training_item_target_exposure_p10": float(np.quantile(item_counts, 0.10)),
            "coverage/training_item_target_exposure_p50": float(np.quantile(item_counts, 0.50)),
            "coverage/training_item_target_exposure_p90": float(np.quantile(item_counts, 0.90)),
            "coverage/training_task_unique_fraction": self.task_seen / n_tasks,
            "coverage/training_task_exposure_min": int(self.task_draw_count.min()),
            "coverage/training_task_exposure_p50": float(np.quantile(self.task_draw_count, 0.50)),
            "coverage/training_task_exposure_max": int(self.task_draw_count.max()),
            "coverage/training_target_draws_total": self.target_draws_total,
        }

    def item_frame(self, item_uids):
        import pandas as pd

        keep = self.eligible_item
        return pd.DataFrame({
            "item_uid": item_uids[keep],
            "target_exposures": self.item_target_count[keep],
            "was_target": self.item_target_count[keep] > 0,
        })

    def task_frame(self):
        import numpy as np
        import pandas as pd

        return pd.DataFrame({
            "subject_uid": [task["subject"] for task in self.tasks],
            "benchmark_id": [task["benchmark"] for task in self.tasks],
            "eligible_responses": [len(counts) for counts in self.response_target_count],
            "responses_targeted": [int((counts > 0).sum()) for counts in self.response_target_count],
            "response_target_coverage": [float((counts > 0).mean()) for counts in self.response_target_count],
            "target_exposure_mean": [float(counts.mean()) for counts in self.response_target_count],
            "target_exposure_min": [int(counts.min()) for counts in self.response_target_count],
            "task_draws": self.task_draw_count,
        })


def _episode(task, rng, device, emb, max_context=MAX_CONTEXT,
             coverage=None, task_index=None):
    """One task, random context size, targets INCLUDING the context (CNP §2.3).

    The paper: "N ~ uniform[0,...,n-1]. We condition on the subset O_N ... Thus,
    the targets it scores Q_theta on include both the observed and unobserved
    values." So targets are the task's full observation set, not the complement.

    Disjoint context/target is the NP/ANP convention and is what this used to
    do; it is not what the original CNP specifies. Scoring the context points
    too is also what makes the model honest at budget 0 -- it has to explain
    what it was just told, not only what it was not.

    This is the "cell" training style: random subset of ONE task's responses as
    context, score that task's responses. Subject- and item-holdout are test
    protocols, not training recipes -- the paper draws a fresh f ~ P each step
    and has no notion of holding out x locations at all.
    """
    import torch

    items, labels = task["items"], task["labels"]
    n = len(items)
    k = int(rng.integers(0, min(max_context, n - 1) + 1))
    perm = rng.permutation(n)
    c = perm[:k]
    # Disjoint targets, following "Active Evaluation Acquisition" sec B.2:
    # "we randomly sample two non-overlapping subsets from each model as
    # Y^(o) and Y^(u)". That paper solves this exact task. Garnelo's CNP
    # instead scores targets that include the context; we take the
    # Active-Evaluation convention here.
    t = perm[k:k + MAX_EPISODE_TARGETS]
    if len(t) == 0:
        t = perm[-1:]
    if coverage is not None:
        coverage.record(task_index, t)
    gi = lambda idx: emb[torch.as_tensor(items[idx], dtype=torch.long, device=device)]
    gl = lambda idx: torch.as_tensor(labels[idx], dtype=torch.float32, device=device)
    return gi(c), gl(c), gi(t), gl(t)


def _episode_indices(task, rng, max_context=MAX_CONTEXT,
                     max_targets=FIXED_VAL_MAX_TARGETS):
    """Freeze one disjoint context/target carve without storing embeddings."""
    n = len(task["items"])
    k = int(rng.integers(0, min(max_context, n - 1) + 1))
    permutation = rng.permutation(n)
    target = permutation[k:k + max_targets]
    if len(target) == 0:
        target = permutation[-1:]
    return permutation[:k], target


def _fixed_validation_metrics(model, tasks, episode_bank, emb, device):
    """Deterministic held-out validation for checkpoint selection.

    The context/target indices are created once before optimisation.  For a
    latent NP, ``model.logits`` uses its prior mean, matching deterministic
    evaluation instead of adding posterior-sampling noise to model selection.
    """
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
    }


def _score(y, p):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    return auc, float(np.mean((p - y) ** 2))



def _axis_scores(model, tasks, *, held_i_arr, emb, device, rng, n_subjects,
                 max_targets, min_half, holdout_frac, max_context):
    """cell / subject_splithalf / item_coldstart on a subsample of tasks.

    Used both for the periodic curves during training (on VALIDATION subjects,
    never test -- early stopping reads these, so test here would leak into
    model selection) and for the final report (on test subjects).

    All three are the same CNP operation: encode a context set, decode targets.
    Only the carve differs.
    """
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    acc = {}
    # no_grad: this is called mid-training, so without it the tensors still
    # require grad and .numpy() raises -- and we would build a useless graph.
    torch.set_grad_enabled(False)
    picks = rng.choice(len(tasks), min(n_subjects, len(tasks)), replace=False)
    for ix in picks:
        t = tasks[ix]
        keep = ~np.isin(t["items"], held_i_arr)
        w_it, w_lb = t["items"][keep], t["labels"][keep]
        c_it, c_lb = t["items"][~keep], t["labels"][~keep]
        if len(w_it) < 2 * min_half:
            continue
        order = rng.permutation(len(w_it))

        def ev(axis, ctx, tgt_items, tgt_labels):
            if len(tgt_items) < 2 or len(np.unique(tgt_labels)) < 2:
                return
            x_c = emb[torch.as_tensor(w_it[ctx], dtype=torch.long, device=device)]
            y_c = torch.as_tensor(w_lb[ctx], dtype=torch.float32, device=device)
            x_t = emb[torch.as_tensor(tgt_items, dtype=torch.long, device=device)]
            p = torch.sigmoid(model.logits(x_c, y_c, x_t)).cpu().numpy()
            try:
                auc = float(roc_auc_score(tgt_labels, p))
            except ValueError:
                return
            acc.setdefault(axis, []).append(
                (auc, float(np.mean((p - tgt_labels) ** 2))))

        cut = int(len(order) * (1.0 - holdout_frac))
        ti = order[cut:cut + max_targets]
        ev("cell", order[:cut], w_it[ti], w_lb[ti])

        half = len(order) // 2
        ti = order[half:half + max_targets]
        ev("subject_splithalf", order[:half], w_it[ti], w_lb[ti])

        if len(c_it) >= 2:
            ev("item_coldstart", order[:min(len(order), max_context)],
               c_it[:max_targets], c_lb[:max_targets])

    torch.set_grad_enabled(True)
    out = {}
    for axis, vals in acc.items():
        a = np.array(vals)
        out[f"{axis}/auc"] = float(a[:, 0].mean())
        out[f"{axis}/brier"] = float(a[:, 1].mean())
    return out


def _described_protocol_metrics(population, scores):
    """Turn internal protocol keys into unambiguous W&B metric paths."""
    return {
        f"diagnostic/{population}/{PROTOCOL_LABELS[axis]}/{SCORE_LABELS[score]}": value
        for key, value in scores.items()
        for axis, score in [key.rsplit("/", 1)]
    }


# CPU: measured 0.036 s/step on 4 threads (3000 steps ~ 2 min). 171k
# params, <=64 context and <=512 target items per episode -- the matmuls
# are far too small for a GPU to help, and L4 capacity was queueing.
@app.function(image=image, cpu=8, memory=32_768, volumes={"/data": volume},
              secrets=_SECRETS, timeout=6 * 60 * 60)
def run(seed: int = SEEDS[0], arm: str = ARM, encoder: str = "linear",
        pooling: str = "mean", use_latent: bool = False, width: int = 64,
        n_blocks: int = 2, steps: int = STEPS, residual_proj: bool = False,
        wandb_mode: str = "online", run_label: str = "") -> dict:
    import os
    import sys
    import time
    import uuid

    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/root")
    import cnp_blocks as cb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training device: {device}", flush=True)
    torch.manual_seed(seed)

    # The display name and saved files must be unique: W&B permits duplicate
    # names, which made the previous 64-wide runs indistinguishable.
    if not run_label:
        run_label = f"run-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    run_label = "".join(c if c.isalnum() or c in "-_" else "-" for c in run_label)
    checkpoint_tag = (
        f"{encoder}-{pooling}{'-latent' if use_latent else ''}-w{width}"
        f"-es-fixed-{run_label}-{arm}-seed{seed}"
    )
    checkpoint_path = CHECKPOINTS / f"{checkpoint_tag}.pt"

    cfg = {"seed": seed, "arm": arm, "encoder": encoder, "pooling": pooling,
           "use_latent": use_latent, "width": width, "n_blocks": n_blocks,
           "steps": steps, "lr": LR, "weight_decay": WEIGHT_DECAY,
           "run_label": run_label,
           "optimizer": "AdamW", "residual_proj": residual_proj,
           "episodes_per_step": EPISODES_PER_STEP, "max_context": MAX_CONTEXT,
           "min_subjects": MIN_SUBJECTS, "val_frac": VAL_FRAC,
           "training_objective": (
               "negative_elbo_per_target" if use_latent else "target_nll"
           ),
           "checkpoint_metric": "validation/fixed_bank/unseen_target_brier_score",
           "fixed_val_episodes": FIXED_VAL_EPISODES,
           "fixed_val_max_targets": FIXED_VAL_MAX_TARGETS,
           "coverage_tracking": "exact_training_response_and_item_target_counts",
           "coverage_hist_every_steps": COVERAGE_HIST_EVERY}

    wb = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            tag = f"{encoder}-{pooling}{'-latent' if use_latent else ''}-w{width}-es-fixed-{run_label}"
            wb = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
                            name=f"cnp-{tag}-{arm}-seed{seed}", job_type="cnp-train",
                            config=cfg, tags=["cnp", "es-fixed", encoder, pooling, arm],
                            mode=wandb_mode, reinit=True)
            print("wandb: live", flush=True)
        except Exception as exc:                      # never let logging kill a run
            print(f"wandb disabled ({exc})", flush=True)
            wb = None
    else:
        print("wandb: no WANDB_API_KEY secret, logging to volume only", flush=True)

    store = DATASETS / f"allbench-embed-{arm}-v2"
    emb_ids = pd.read_parquet(store / "items.parquet")["item_uid"].astype(str).to_numpy()
    emb_all = np.load(store / "item_embeddings.npy", mmap_mode="r")
    pos = {u: i for i, u in enumerate(emb_ids)}
    print(f"embedding store {arm}: {emb_all.shape}", flush=True)

    d = _load_filtered(MIN_SUBJECTS)
    print(f"filtered (n>={MIN_SUBJECTS}): {len(d):,} rows, "
          f"{d.benchmark_id.nunique()} benchmarks", flush=True)

    rows, curves, rng = [], [], np.random.default_rng(seed)

    # ---- POOL every benchmark into one task set -------------------------------
    # Subjects never span benchmarks (checked: 6,264 subjects, all in exactly
    # one), so pooling cannot merge two different abilities into one task.
    # Training per benchmark was wrong twice over: `editbench` has 28 training
    # subjects and would see each ~857 times, and a per-benchmark model cannot
    # be evaluated under the grouped-benchmark-fold protocol at all, since it
    # has no parameters for a benchmark it never saw. The model never receives
    # a benchmark id -- an item is only its embedding -- so nothing about the
    # architecture needs to change.
    d = d[d["item_uid"].isin(pos.keys())]
    items_all = sorted(d["item_uid"].unique())
    i_map = {u: i for i, u in enumerate(items_all)}
    emb = torch.as_tensor(
        np.asarray(emb_all[[pos[u] for u in items_all]], dtype=np.float32)).to(device)
    print(f"pooled item matrix: {tuple(emb.shape)}", flush=True)

    tasks = []
    for (sub, bench), g in d.groupby(["subject_uid", "benchmark_id"]):
        tasks.append({"subject": sub, "benchmark": bench,
                      "items": g["item_uid"].map(i_map).to_numpy(),
                      "labels": g["label"].to_numpy().astype(np.float32)})
    print(f"pooled tasks (subject x benchmark): {len(tasks):,}", flush=True)

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

    train_tasks = [_warm(t) for i, t in enumerate(tasks)
                   if i not in test_ix and i not in val_ix]
    val_tasks = [_warm(t) for i, t in enumerate(tasks) if i in val_ix]
    train_tasks = [t for t in train_tasks if len(t["items"]) > 2 * MIN_HALF]
    val_tasks = [t for t in val_tasks if len(t["items"]) > 2 * MIN_HALF]
    print(f"train/val/test subjects: {len(train_tasks):,}/{len(val_tasks):,}/{len(test_ix):,}"
          f"   held-out items: {len(held_i):,}", flush=True)
    if not train_tasks or not val_tasks:
        raise RuntimeError("pooled split left no training or validation tasks")
    coverage = _TrainingCoverage(train_tasks, len(items_all))
    print(f"coverage tracker: {sum(len(t['items']) for t in train_tasks):,} training responses, "
          f"{int(coverage.eligible_item.sum()):,} eligible training items", flush=True)
    fixed_rng = np.random.default_rng(seed + 10_000)
    fixed_validation_bank = []
    fixed_task_indices = fixed_rng.choice(
        len(val_tasks), size=min(FIXED_VAL_EPISODES, len(val_tasks)), replace=False
    )
    for task_index in fixed_task_indices:
        task_index = int(task_index)
        fixed_validation_bank.append(
            (task_index, *_episode_indices(val_tasks[task_index], fixed_rng))
        )
    print(f"fixed validation bank: {len(fixed_validation_bank)} held-out-subject episodes",
          flush=True)

    model = cb.NeuralProcess(emb.shape[1], width=width, n_blocks=n_blocks,
                             encoder=encoder, pooling=pooling,
                             use_latent=use_latent,
                             residual_proj=residual_proj).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"params: {sum(q.numel() for q in model.parameters()):,}", flush=True)

    epoch_order: list[int] = []
    best_brier, best_state, best_step = float("inf"), None, None
    # These create live W&B panels with paired curves, rather than forcing a
    # reader to compare unrelated auto-generated panels by eye.
    panel_history = {
        "step": [], "train_objective": [], "resampled_val_objective": [],
        "fixed_context_nll": [], "fixed_target_nll": [],
        "fixed_target_brier": [], "best_target_brier": [],
    }
    if use_latent:
        panel_history.update({
            "train_posterior_nll": [], "val_posterior_nll": [],
            "train_kl": [], "val_kl": [],
        })
    diagnostic_panel_history = {
        axis: {"step": [], "train_auc": [], "val_auc": [],
               "train_brier": [], "val_brier": []}
        for axis in PROTOCOL_LABELS
    }
    t0 = time.time()
    for step in tqdm(range(steps), desc=f"seed={seed}", unit="step", mininterval=20.0):
        model.train()
        opt.zero_grad()
        # Shuffled epochs, not sampling with replacement: every subject is
        # seen once per pass. With replacement and early stopping at ~step 400
        # only ~54% of the 4,071 train subjects would ever be visited.
        batch_indices = []
        for _ in range(EPISODES_PER_STEP):
            if not epoch_order:
                epoch_order.extend(rng.permutation(len(train_tasks)).tolist())
            batch_indices.append(epoch_order.pop())
        batch_losses = [
            model.loss(*_episode(train_tasks[task_index], rng, device, emb,
                                 coverage=coverage, task_index=task_index))
            for task_index in batch_indices
        ]
        losses = [loss for loss, _ in batch_losses]
        tr = torch.stack(losses).mean()
        tr.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % VAL_EVERY == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                sampled_val = [
                    model.loss(*_episode(val_tasks[int(rng.integers(len(val_tasks)))],
                                         rng, device, emb))
                    for _ in range(VAL_EPISODES)
                ]
                sampled_val_loss = torch.stack([loss for loss, _ in sampled_val]).mean()
                fixed_val = _fixed_validation_metrics(
                    model, val_tasks, fixed_validation_bank, emb, device
                )
            curves.append({"seed": seed, "step": step,
                           "training_objective": float(tr),
                           "validation_resampled_objective": float(sampled_val_loss),
                           **{f"validation_fixed_bank_{name}": value
                              for name, value in fixed_val.items()},
                           **{key.replace("/", "_"): value
                              for key, value in coverage.metrics().items()}})
            if use_latent:
                curves[-1]["training_posterior_target_nll"] = float(
                    np.mean([aux["nll"] for _, aux in batch_losses])
                )
                curves[-1]["training_latent_kl_total"] = float(
                    np.mean([aux["kl"] for _, aux in batch_losses])
                )
                curves[-1]["validation_resampled_posterior_target_nll"] = float(
                    np.mean([aux["nll"] for _, aux in sampled_val])
                )
                curves[-1]["validation_resampled_latent_kl_total"] = float(
                    np.mean([aux["kl"] for _, aux in sampled_val])
                )
            # Same three carves on BOTH pools: the train/val gap per axis is
            # the generalisation gap for that axis. One model, one objective --
            # these are evaluation carves, not separate training heads.
            kw = dict(held_i_arr=held_i_arr, emb=emb, device=device,
                      n_subjects=CURVE_SUBJECTS, max_targets=MAX_EVAL_TARGETS,
                      min_half=MIN_HALF, holdout_frac=HELD_OUT_FRAC,
                      max_context=MAX_CONTEXT)
            val_axis = _axis_scores(model, val_tasks,
                                    rng=np.random.default_rng(seed), **kw)
            tr_axis = _axis_scores(model, train_tasks,
                                   rng=np.random.default_rng(seed + 1), **kw)
            curves[-1].update({
                key.replace("/", "_"): value
                for key, value in _described_protocol_metrics(
                    "heldout_subjects", val_axis
                ).items()
            })
            curves[-1].update({
                key.replace("/", "_"): value
                for key, value in _described_protocol_metrics(
                    "training_subjects", tr_axis
                ).items()
            })
            if wb is not None:
                objective_name = (
                    "negative_elbo_per_target" if use_latent else "target_nll"
                )
                wandb_metrics = {
                    f"loss/{objective_name}/training": float(tr),
                    f"loss/{objective_name}/validation_resampled_episodes": float(sampled_val_loss),
                    "loss/context_reconstruction_nll/validation_fixed_bank": fixed_val["context_reconstruction_nll"],
                    "loss/unseen_target_nll/validation_fixed_bank": fixed_val["unseen_target_nll"],
                    "brier_score/unseen_target/validation_fixed_bank": fixed_val["unseen_target_brier"],
                    "brier_score/unseen_target/best_fixed_bank_so_far": min(
                        best_brier, fixed_val["unseen_target_brier"]
                    ),
                    **_described_protocol_metrics("heldout_subjects", val_axis),
                    **_described_protocol_metrics("training_subjects", tr_axis),
                    **coverage.metrics(),
                }
                if step % COVERAGE_HIST_EVERY == 0 or step == steps - 1:
                    wandb_metrics["coverage/training_item_target_exposure_histogram"] = (
                        wandb.Histogram(
                            coverage.item_target_count[coverage.eligible_item]
                        )
                    )
                if use_latent:
                    wandb_metrics.update({
                        "loss/posterior_target_nll/training": curves[-1]["training_posterior_target_nll"],
                        "loss/posterior_target_nll/validation_resampled_episodes": curves[-1]["validation_resampled_posterior_target_nll"],
                        "loss/latent_kl_total/training": curves[-1]["training_latent_kl_total"],
                        "loss/latent_kl_total/validation_resampled_episodes": curves[-1]["validation_resampled_latent_kl_total"],
                    })
                panel_history["step"].append(step)
                panel_history["train_objective"].append(float(tr))
                panel_history["resampled_val_objective"].append(float(sampled_val_loss))
                panel_history["fixed_context_nll"].append(fixed_val["context_reconstruction_nll"])
                panel_history["fixed_target_nll"].append(fixed_val["unseen_target_nll"])
                panel_history["fixed_target_brier"].append(fixed_val["unseen_target_brier"])
                panel_history["best_target_brier"].append(min(best_brier, fixed_val["unseen_target_brier"]))
                wandb_metrics.update({
                    "panels/loss/objective_training_vs_resampled_validation": wandb.plot.line_series(
                        xs=[panel_history["step"], panel_history["step"]],
                        ys=[panel_history["train_objective"], panel_history["resampled_val_objective"]],
                        keys=["training", "validation (resampled episodes)"],
                        title=f"{objective_name}: training vs validation",
                        xname="optimizer step",
                    ),
                    "panels/loss/fixed_validation_context_vs_unseen_target": wandb.plot.line_series(
                        xs=[panel_history["step"], panel_history["step"]],
                        ys=[panel_history["fixed_context_nll"], panel_history["fixed_target_nll"]],
                        keys=["context reconstruction", "unseen target"],
                        title="Fixed-bank validation NLL by prediction role",
                        xname="optimizer step",
                    ),
                    "panels/brier_score/fixed_validation_vs_best_checkpoint": wandb.plot.line_series(
                        xs=[panel_history["step"], panel_history["step"]],
                        ys=[panel_history["fixed_target_brier"], panel_history["best_target_brier"]],
                        keys=["fixed validation", "best checkpoint so far"],
                        title="Fixed-bank unseen-target Brier score",
                        xname="optimizer step",
                    ),
                })
                if use_latent:
                    panel_history["train_posterior_nll"].append(curves[-1]["training_posterior_target_nll"])
                    panel_history["val_posterior_nll"].append(curves[-1]["validation_resampled_posterior_target_nll"])
                    panel_history["train_kl"].append(curves[-1]["training_latent_kl_total"])
                    panel_history["val_kl"].append(curves[-1]["validation_resampled_latent_kl_total"])
                    wandb_metrics.update({
                        "panels/loss/posterior_target_nll_training_vs_validation": wandb.plot.line_series(
                            xs=[panel_history["step"], panel_history["step"]],
                            ys=[panel_history["train_posterior_nll"], panel_history["val_posterior_nll"]],
                            keys=["training", "validation (resampled episodes)"],
                            title="Posterior target NLL: training vs validation",
                            xname="optimizer step",
                        ),
                        "panels/loss/latent_kl_training_vs_validation": wandb.plot.line_series(
                            xs=[panel_history["step"], panel_history["step"]],
                            ys=[panel_history["train_kl"], panel_history["val_kl"]],
                            keys=["training", "validation (resampled episodes)"],
                            title="Latent KL: training vs validation",
                            xname="optimizer step",
                        ),
                    })
                # These are periodic diagnostics, never the test set. Pair
                # the same protocol across train/held-out subjects so a gap
                # is visible without comparing separate W&B panels.
                for axis, protocol in PROTOCOL_LABELS.items():
                    auc_key, brier_key = f"{axis}/auc", f"{axis}/brier"
                    if not all(key in tr_axis and key in val_axis
                               for key in (auc_key, brier_key)):
                        continue
                    hist = diagnostic_panel_history[axis]
                    hist["step"].append(step)
                    hist["train_auc"].append(tr_axis[auc_key])
                    hist["val_auc"].append(val_axis[auc_key])
                    hist["train_brier"].append(tr_axis[brier_key])
                    hist["val_brier"].append(val_axis[brier_key])
                    wandb_metrics.update({
                        f"panels/diagnostics/{protocol}/roc_auc_training_vs_heldout": wandb.plot.line_series(
                            xs=[hist["step"], hist["step"]],
                            ys=[hist["train_auc"], hist["val_auc"]],
                            keys=["training subjects", "held-out subjects (validation)"],
                            title=f"{protocol}: ROC-AUC",
                            xname="optimizer step",
                        ),
                        f"panels/diagnostics/{protocol}/brier_training_vs_heldout": wandb.plot.line_series(
                            xs=[hist["step"], hist["step"]],
                            ys=[hist["train_brier"], hist["val_brier"]],
                            keys=["training subjects", "held-out subjects (validation)"],
                            title=f"{protocol}: Brier score",
                            xname="optimizer step",
                        ),
                    })
                wb.log(wandb_metrics, step=step)
            # Train the full fixed budget. Select the best checkpoint by one
            # stable, held-out-subject Brier bank rather than a fresh BCE draw.
            if fixed_val["unseen_target_brier"] < best_brier - 1e-6:
                best_brier = fixed_val["unseen_target_brier"]
                best_step = step
                # Keep the selected checkpoint on the Modal volume while the
                # job runs. This is atomic: an interruption leaves the prior
                # complete best checkpoint, never a partially written file.
                best_state = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()
                }
                CHECKPOINTS.mkdir(parents=True, exist_ok=True)
                checkpoint_tmp = checkpoint_path.with_suffix(".pt.tmp")
                torch.save({
                    "format_version": 1,
                    "state_dict": best_state,
                    "config": cfg,
                    "selection": {
                        "metric": "validation/fixed_bank/unseen_target_brier_score",
                        "value": float(best_brier),
                        "step": int(best_step),
                    },
                }, checkpoint_tmp)
                checkpoint_tmp.replace(checkpoint_path)
                volume.commit()
                print(f"saved best checkpoint step={best_step} Brier={best_brier:.6f} "
                      f"path={checkpoint_path}", flush=True)
                if wb is not None:
                    wb.summary["checkpoint/volume_path"] = str(checkpoint_path)
                    wb.summary["checkpoint/best_fixed_bank_brier_score"] = float(best_brier)
                    wb.summary["checkpoint/best_step"] = int(best_step)
        if step == 49:
            rate = (time.time() - t0) / 50
            print(f"{rate:.3f} s/step -> {rate * steps / 60:.1f} min for {steps} steps",
                  flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"restored best fixed-validation Brier {best_brier:.6f} at step={best_step}",
          flush=True)

    # ---- OFFLINE evaluation: the same three protocols as the IRT jobs -------
    # The budget ladder is deliberately NOT here. A ladder without an
    # acquisition policy is just the random-acquisition arm of the online
    # evaluation, and scoring it inside the training job would quietly pass it
    # off as an offline result. It belongs in organized/testing/online/.
    #
    # For a CNP every protocol below is the same operation -- encode a context
    # set, decode targets -- differing only in how context and target are
    # carved. Nothing is fitted at test time.
    model.eval()
    with torch.no_grad():
        for i in tqdm(sorted(test_ix), desc="eval", unit="subj", mininterval=20.0):
            t = tasks[i]
            keep = ~np.isin(t["items"], held_i_arr)
            w_it, w_lb = t["items"][keep], t["labels"][keep]
            c_it, c_lb = t["items"][~keep], t["labels"][~keep]
            if len(w_it) < 2 * MIN_HALF:
                continue
            base = {"benchmark_id": t["benchmark"], "seed": seed, "arm": arm,
                    "encoder": encoder, "pooling": pooling,
                    "use_latent": use_latent, "width": width}
            order = rng.permutation(len(w_it))

            def _emit(axis, ctx_idx, tgt_idx):
                if len(tgt_idx) < 2 or len(np.unique(w_lb[tgt_idx])) < 2:
                    return
                x_c = emb[torch.as_tensor(w_it[ctx_idx], dtype=torch.long, device=device)]
                y_c = torch.as_tensor(w_lb[ctx_idx], dtype=torch.float32, device=device)
                x_t = emb[torch.as_tensor(w_it[tgt_idx], dtype=torch.long, device=device)]
                p = torch.sigmoid(model.logits(x_c, y_c, x_t)).cpu().numpy()
                auc, brier = _score(w_lb[tgt_idx], p)
                rows.append({**base, "axis": axis, "n_score": int(len(tgt_idx)),
                             "auc": auc, "brier": brier})

            # cell: 80% of this subject's responses as context, 20% as target.
            # The IRT counterpart gets to fit this subject's own theta; the CNP
            # fits nothing, so matching it here is the stronger result.
            cut = int(len(order) * (1.0 - HELD_OUT_FRAC))
            _emit("cell", order[:cut], order[cut:cut + MAX_EVAL_TARGETS])

            # subject split-half: matches the IRT generalization protocol.
            half = len(order) // 2
            _emit("subject_splithalf", order[:half], order[half:half + MAX_EVAL_TARGETS])

            # item cold start: target items the model never trained on, with a
            # normal context from this subject's warm items.
            if len(c_it) >= 2 and len(np.unique(c_lb)) > 1:
                ctx = order[:min(len(order), MAX_CONTEXT)]
                x_c = emb[torch.as_tensor(w_it[ctx], dtype=torch.long, device=device)]
                y_c = torch.as_tensor(w_lb[ctx], dtype=torch.float32, device=device)
                cc = c_it[:MAX_EVAL_TARGETS]
                x_ci = emb[torch.as_tensor(cc, dtype=torch.long, device=device)]
                pc = torch.sigmoid(model.logits(x_c, y_c, x_ci)).cpu().numpy()
                a2, b2 = _score(c_lb[:MAX_EVAL_TARGETS], pc)
                rows.append({**base, "axis": "item_coldstart", "n_score": int(len(cc)),
                             "auc": a2, "brier": b2})

    frame = pd.DataFrame(rows)
    curve = pd.DataFrame(curves)
    final_coverage = coverage.metrics()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tag = f"{encoder}-{pooling}{'-latent' if use_latent else ''}-w{width}-es-fixed-{run_label}"
    frame.to_parquet(OUT.with_name(f"{OUT.stem}-{tag}-{arm}-seed{seed}{OUT.suffix}"), index=False)
    curve.to_parquet(OUT.with_name(f"{OUT.stem}-curve-{tag}-{arm}-seed{seed}{OUT.suffix}"), index=False)
    coverage.item_frame(np.asarray(items_all, dtype=str)).to_parquet(
        OUT.with_name(f"{OUT.stem}-item-coverage-{tag}-{arm}-seed{seed}{OUT.suffix}"),
        index=False,
    )
    coverage.task_frame().to_parquet(
        OUT.with_name(f"{OUT.stem}-task-coverage-{tag}-{arm}-seed{seed}{OUT.suffix}"),
        index=False,
    )
    volume.commit()
    if wb is not None:
        for name, value in final_coverage.items():
            wb.summary[name] = value
        if len(frame):
            final_test = (frame.groupby("axis", as_index=False)[["auc", "brier"]].mean()
                           .rename(columns={"axis": "protocol", "auc": "roc_auc",
                                            "brier": "brier_score"}))
            final_test["protocol"] = final_test["protocol"].map(PROTOCOL_LABELS)
            test_table = wandb.Table(dataframe=final_test)
            wb.log({
                "tables/test/final_scores": test_table,
                "panels/test/final_roc_auc_by_protocol": wandb.plot.bar(
                    test_table, "protocol", "roc_auc", title="Final test ROC-AUC by protocol"
                ),
                "panels/test/final_brier_score_by_protocol": wandb.plot.bar(
                    test_table, "protocol", "brier_score", title="Final test Brier score by protocol"
                ),
            })
            for axis, g in frame.groupby("axis"):
                protocol = PROTOCOL_LABELS[axis]
                wb.summary[f"final/test_subjects/{protocol}/roc_auc"] = float(g.auc.mean())
                wb.summary[f"final/test_subjects/{protocol}/brier_score"] = float(g.brier.mean())
        wb.finish()
    return {"rows": len(frame), "curve_points": len(curve), "seed": seed,
            "encoder": encoder, "pooling": pooling, "use_latent": use_latent,
            "run_label": run_label, "checkpoint_path": str(checkpoint_path),
            "best_step": best_step, "best_fixed_bank_brier_score": float(best_brier),
            "benchmarks": int(frame["benchmark_id"].nunique()) if len(frame) else 0}


@app.local_entrypoint()
def main(arm: str = ARM, encoder: str = "linear", pooling: str = "mean",
         use_latent: bool = False, width: int = 64, n_blocks: int = 2,
         steps: int = STEPS, residual_proj: bool = False,
         run_label: str = "") -> None:
    if not run_label:
        from datetime import datetime, timezone
        run_label = datetime.now(timezone.utc).strftime("sweep-%Y%m%dT%H%M%SZ")
    handles = [run.spawn(s, arm, encoder, pooling, use_latent, width, n_blocks,
                         steps, residual_proj, "online", run_label) for s in SEEDS]
    for h in handles:
        print(h.get())
