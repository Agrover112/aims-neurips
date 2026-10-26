"""Corrected-protocol ablation ladder for the ANP/GP-IRT direction.

This runner is separate from ``modal_r_c_ablation.py``.  It trains only on a
fixed acquisition half of each training task and scores only the untouched
evaluation half.  The evaluation grid, folds, and calibration are the same as
the uniform validation protocol.

v4.1 uses per-visit rotating disjoint splits, padded 16-task batches, and a
global task-budget pair epoch.  The deterministic rungs use the batched trainer;
the latent controls still use the legacy sequential path.

Phase 1 (architecture ladder):
    modal run modal_faithful_ladder.py --rungs 2pl,gp_2pl,2pl_anp,gp_anp

Phase 2 (objective ladder on the winner):
    modal run modal_faithful_ladder.py --rungs gp_anp --objective bce,bce_brier,brier_alc

Phase 3 (latent controls, full v4.1 pair epoch):
    modal run modal_faithful_ladder.py --rungs latent_np,latent_anp_noirt
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import modal
import numpy as np

APP_NAME = "aims-neurips-faithful-ladder"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "allbench-embed-minilm512-v1"
CORPUS_PARENT = REMOTE_ROOT / "datasets" / "full"


def _corpus_root(name: str = "") -> Path:
    """Resolve the feature store this run reads.

    The default keeps every published number reproducible. Passing a store
    name selects another one, which is how the v2 item_content-only arms
    (``allbench-train-{arm}-cap{N}-v2``, built by ``modal_compact_store.py``)
    are evaluated without editing a module constant.
    """

    if not name:
        return CORPUS_ROOT
    root = CORPUS_PARENT / name
    if not root.exists():
        raise FileNotFoundError(f"no feature store at {root}")
    return root
PROTOCOL_ROOT = REMOTE_ROOT / "datasets" / "full" / "evaluation-protocol-v2"
RESULT_ROOT = REMOTE_ROOT / "results" / "faithful-ladder"
VERSION = "v4.1-faithful-ladder"
SPLIT_SEED = 20261017
DEFAULT_BATCH_SIZE = 16
DEFAULT_PROBE_COUNT = 53
DEFAULT_PROBE_BENCHMARKS = 4
SEEDS = (20261018, 20261019, 20261020)
MODES = ("masked",)
OBJECTIVES = ("bce", "bce_brier", "brier_alc")
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
UNIFORM_FOLDS = 5
FIT_SAMPLE_LIMIT = 1_000_000
RUNGS = {
    "2pl": {
        "family": "anchor",
        "use_gp": False,
        "use_anp": False,
        "use_subject_prior": False,
    },
    "2pl_anp": {
        "family": "anchor",
        "use_gp": False,
        "use_anp": True,
        "use_subject_prior": False,
    },
    "gp_2pl": {
        "family": "anchor",
        "use_gp": True,
        "use_anp": False,
        "use_subject_prior": False,
    },
    "gp_anp": {
        "family": "anchor",
        "use_gp": True,
        "use_anp": True,
        "use_subject_prior": False,
    },
    "gp_anp_prior": {
        "family": "anchor",
        "use_gp": True,
        "use_anp": True,
        "use_subject_prior": True,
    },
    "latent_np": {
        "family": "latent",
        "variant": "np",
        "residual_scale": 0.0,
    },
    "latent_anp": {
        "family": "latent",
        "variant": "anp",
        "residual_scale": 0.1,
        "direct_decoder": False,
    },
    "latent_anp_noirt": {
        "family": "latent",
        "variant": "anp",
        "residual_scale": 0.0,
        "direct_decoder": True,
    },
}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2,<3",
        "pandas>=2.2,<3",
        "pyarrow>=17,<24",
        "torch==2.5.1",
    )
    .add_local_dir(PROJECT_ROOT / "src", "/opt/aims/src", copy=True)
    .env({"PYTHONPATH": "/opt/aims/src", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
)


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_path(name: str) -> Path:
    mounted = Path("/opt/aims/src") / name
    if mounted.exists():
        return mounted
    return PROJECT_ROOT / "src" / name


def _code_signature() -> str:
    from aims_starter.caching import content_signature

    return content_signature(
        [
            _source_path("aims_starter/faithful_ladder.py"),
            _source_path("aims_starter/faithful_np.py"),
            _source_path("aims_starter/r_c_attention.py"),
            _source_path("aims_starter/uniform_eval.py"),
            _source_path("aims_starter/calibration.py"),
        ],
        {
            "version": VERSION,
            "queue": "v4.1-pair-epoch",
            "batch_size": DEFAULT_BATCH_SIZE,
            "split_seed": SPLIT_SEED,
            "seeds": list(SEEDS),
            "modes": list(MODES),
            "objectives": list(OBJECTIVES),
            "budgets": list(BUDGETS),
            "weights": list(WEIGHTS),
            "uniform_folds": UNIFORM_FOLDS,
            "fit_sample_limit": FIT_SAMPLE_LIMIT,
        },
    )


def _features(rows, item_embeddings, subject_embeddings):
    return np.concatenate(
        [
            item_embeddings[rows["item_index"].to_numpy()],
            subject_embeddings[rows["subject_index"].to_numpy()],
        ],
        axis=1,
    )


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _task_list(train, item_embeddings, subject_embeddings, benchmark_of):
    from aims_starter.faithful_ladder import make_task

    tasks = []
    for task_id, rows in train.groupby("task_id", sort=True):
        task = make_task(
            str(task_id),
            _features(rows, item_embeddings, subject_embeddings),
            rows["label"].to_numpy(dtype="float32"),
            split_seed=SPLIT_SEED,
        )
        task["benchmark_id"] = benchmark_of.get(str(task_id), "unknown")
        tasks.append(task)
    return tasks


def _select_probe_benchmarks(tasks, count: int, minimum_tasks: int = 0):
    from aims_starter.faithful_ladder import stable_hash

    counts: dict[str, int] = {}
    for task in tasks:
        benchmark = str(task["benchmark_id"])
        if benchmark == "unknown":
            continue
        counts[benchmark] = counts.get(benchmark, 0) + 1
    ordered = sorted(counts, key=stable_hash)
    selected: list[str] = []
    covered = 0
    for benchmark in ordered:
        selected.append(benchmark)
        covered += counts[benchmark]
        if len(selected) >= max(0, int(count)) and covered >= int(minimum_tasks):
            break
    return tuple(selected)


def _split_fit_probe(tasks, probe_count: int, probe_benchmarks: tuple[str, ...]):
    from aims_starter.faithful_ladder import stable_hash

    if probe_count <= 0:
        return tasks, []
    selected = set(probe_benchmarks)
    candidates = [
        task for task in tasks if str(task["benchmark_id"]) in selected
    ]
    if not candidates:
        candidates = list(tasks)
    by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for task in candidates:
        by_benchmark.setdefault(str(task["benchmark_id"]), []).append(task)
    for values in by_benchmark.values():
        values.sort(key=lambda task: stable_hash(task["task_id"]))
    probe: list[dict[str, Any]] = []
    cursor = 0
    ordered_benchmarks = sorted(by_benchmark, key=stable_hash)
    while len(probe) < int(probe_count) and ordered_benchmarks:
        benchmark = ordered_benchmarks[cursor % len(ordered_benchmarks)]
        values = by_benchmark[benchmark]
        if values:
            probe.append(values.pop(0))
        else:
            ordered_benchmarks.remove(benchmark)
            continue
        cursor += 1
    probe_ids = {task["task_id"] for task in probe}
    fit = [task for task in tasks if task["task_id"] not in probe_ids]
    return fit, probe


def _make_model(
    rung: str,
    x_dim: int,
    subject_slice: tuple[int, int] | None,
    subject_prior_slice: tuple[int, int],
):
    from aims_starter.faithful_ladder import AnchorCorrectionIRT
    from aims_starter.faithful_np import FaithfulNPAIRT

    config = RUNGS[rung]
    if config["family"] == "latent":
        return FaithfulNPAIRT(
            x_dim,
            variant=str(config["variant"]),
            d_model=128,
            d_z=32,
            n_heads=8,
            residual_scale=float(config["residual_scale"]),
            anchor_scale=0.0,
            subject_feature_slice=subject_slice,
            direct_decoder=bool(config.get("direct_decoder", False)),
        )
    return AnchorCorrectionIRT(
        x_dim,
        use_gp=bool(config["use_gp"]),
        use_anp=bool(config["use_anp"]),
        use_subject_prior=bool(config["use_subject_prior"]),
        d_model=128,
        n_heads=8,
        subject_feature_slice=subject_slice,
        subject_prior_slice=subject_prior_slice,
    )


def _train(
    model,
    fit_tasks,
    probe_tasks,
    *,
    seed: int,
    latent: bool,
    steps: int,
    episodes_per_step: int,
    objective: str,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup: int = 100,
    eta_min_ratio: float = 0.05,
    latent_beta: float = 1.0,
    probe_every: int = 250,
    probe_count: int = 64,
    log_every: int = 25,
):
    """Train with a coverage sampler and select on a held-out benchmark probe."""

    import torch

    torch.manual_seed(seed + 202)

    from aims_starter.faithful_ladder import (
        brier_alc_on_tasks,
        deterministic_episode_loss,
        latent_episode_loss,
        sample_episode,
    )

    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_steps = max(1, min(int(warmup), max(1, steps // 10)))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    rng = np.random.default_rng(seed + 101)
    order = rng.permutation(len(fit_tasks))
    cursor = 0
    history: list[dict[str, Any]] = []
    ema_loss: float | None = None
    best_probe = float("inf")
    best_step: int | None = None
    best_state: dict[str, Any] | None = None
    min_probe_step = max(0, steps // 10)

    for step in range(steps):
        losses = []
        diagnostics = []
        for _ in range(episodes_per_step):
            if cursor >= len(order):
                order = rng.permutation(len(fit_tasks))
                cursor = 0
            task = fit_tasks[int(order[cursor])]
            cursor += 1
            x_c, y_c, x_t, y_t, _ = sample_episode(task, rng, device)
            if latent:
                loss, diagnostic = latent_episode_loss(
                    model,
                    x_c,
                    y_c,
                    x_t,
                    y_t,
                    objective=objective,
                    beta=latent_beta,
                    n_z_samples=1,
                )
            else:
                loss, diagnostic = deterministic_episode_loss(
                    model,
                    x_c,
                    y_c,
                    x_t,
                    y_t,
                    objective=objective,
                )
            losses.append(loss)
            diagnostics.append(diagnostic)
        batch_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        value = float(batch_loss.detach())
        ema_loss = value if ema_loss is None else 0.98 * ema_loss + 0.02 * value
        record: dict[str, Any] = {
            "step": step,
            "loss": value,
            "ema_loss": ema_loss,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        for key in ("bce", "brier", "kl", "penalty", "auxiliary"):
            values = [diagnostic[key] for diagnostic in diagnostics if key in diagnostic]
            if values:
                record[key] = float(np.mean(values))
        if probe_tasks and (step % probe_every == 0 or step == steps - 1):
            probe = brier_alc_on_tasks(
                model,
                probe_tasks,
                device=device,
                max_tasks=probe_count,
                latent_samples=64,
            )
            record["probe_brier_alc"] = float(probe["brier_alc"])
            model.train()
            if step >= min_probe_step and probe["brier_alc"] < best_probe:
                best_probe = float(probe["brier_alc"])
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        if step % log_every == 0 or step == steps - 1:
            history.append(record)
            print(
                f"seed={seed} step={step} loss={record['loss']:.6f} "
                f"ema={record['ema_loss']:.6f} "
                f"probe={record.get('probe_brier_alc', float('nan')):.6f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(
            {key: value.to(device) for key, value in best_state.items()}
        )
        selected_step = int(best_step) if best_step is not None else steps - 1
    else:
        selected_step = steps - 1
    model.eval()
    return model, history, {
        "selected_step": selected_step,
        "best_probe_brier_alc": best_probe if best_state is not None else float("nan"),
    }


def _train_batched(
    model,
    fit_tasks,
    probe_tasks,
    *,
    seed: int,
    objective: str,
    pair_epochs: int = 1,
    batch_size: int = 16,
    steps_limit: int = 0,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup: int = 100,
    eta_min_ratio: float = 0.05,
    probe_every: int = 250,
    probe_count: int = 53,
    log_every: int = 25,
):
    """Train the deterministic ladder with padded multi-task batches."""

    import torch

    torch.manual_seed(seed + 202)

    from aims_starter.faithful_ladder import (
        brier_alc_on_tasks,
        build_pair_epoch_batches,
        collate_episodes,
        deterministic_batch_loss,
        sample_pair_episode,
    )

    device = next(model.parameters()).device
    model.train()
    scheduled: list[tuple[int, list[Any]]] = []
    pairs_per_epoch = 0
    batches_per_epoch = 0
    for epoch in range(int(pair_epochs)):
        epoch_batches = build_pair_epoch_batches(
            fit_tasks,
            seed=seed + 1009 * epoch,
            batch_size=int(batch_size),
        )
        if epoch == 0:
            batches_per_epoch = len(epoch_batches)
            pairs_per_epoch = sum(len(batch) for batch in epoch_batches)
        scheduled.extend((epoch, batch) for batch in epoch_batches)
    if steps_limit > 0:
        scheduled = scheduled[: int(steps_limit)]
    if not scheduled:
        raise ValueError("no training batches were scheduled")

    total_steps = len(scheduled)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    warmup_steps = max(1, min(int(warmup), max(1, total_steps // 10)))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    rng = np.random.default_rng(seed + 101)
    history: list[dict[str, Any]] = []
    ema_loss: float | None = None
    best_probe = float("inf")
    best_step: int | None = None
    best_state: dict[str, Any] | None = None
    min_probe_step = max(0, total_steps // 10)
    pairs_seen = 0

    for step, (epoch, batch) in enumerate(scheduled):
        episodes = []
        for visit in batch:
            episode = sample_pair_episode(
                fit_tasks[visit.task_index],
                budget=int(visit.budget),
                rng=rng,
                split_seed=SPLIT_SEED,
                visit_key=f"e{epoch}:b{visit.budget}",
            )
            episode["task_index"] = int(visit.task_index)
            episodes.append(episode)
        padded = collate_episodes(episodes, device)
        loss, diagnostic = deterministic_batch_loss(
            model,
            padded,
            objective=objective,
            batch_size=int(batch_size),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        pairs_seen += len(batch)

        value = float(loss.detach())
        ema_loss = value if ema_loss is None else 0.98 * ema_loss + 0.02 * value
        record: dict[str, Any] = {
            "step": step,
            "epoch": int(epoch),
            "pairs_seen": int(pairs_seen),
            "batch_tasks": len(batch),
            "loss": value,
            "ema_loss": ema_loss,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        for key in ("bce", "brier", "prediction", "penalty", "auxiliary"):
            if key in diagnostic:
                record[key] = float(diagnostic[key])
        if probe_tasks and (step % probe_every == 0 or step == total_steps - 1):
            probe = brier_alc_on_tasks(
                model,
                probe_tasks,
                device=device,
                max_tasks=probe_count,
                latent_samples=64,
            )
            record["probe_brier_alc"] = float(probe["brier_alc"])
            model.train()
            if step >= min_probe_step and probe["brier_alc"] < best_probe:
                best_probe = float(probe["brier_alc"])
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        if step % log_every == 0 or step == total_steps - 1:
            history.append(record)
            print(
                f"seed={seed} step={step}/{total_steps} "
                f"loss={record['loss']:.6f} ema={record['ema_loss']:.6f} "
                f"bce={record.get('bce', float('nan')):.6f} "
                f"probe={record.get('probe_brier_alc', float('nan')):.6f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(
            {key: value.to(device) for key, value in best_state.items()}
        )
        selected_step = int(best_step) if best_step is not None else total_steps - 1
    else:
        selected_step = total_steps - 1
    model.eval()
    return model, history, {
        "selected_step": selected_step,
        "best_probe_brier_alc": (
            best_probe if best_state is not None else float("nan")
        ),
        "steps_run": int(total_steps),
        "pair_epochs": int(pair_epochs),
        "batches_per_epoch": int(batches_per_epoch),
        "pairs_per_epoch": int(pairs_per_epoch),
        "steps_per_epoch": int(batches_per_epoch),
        "batch_size": int(batch_size),
        "scheduled_pairs": int(sum(len(batch) for _, batch in scheduled)),
    }


def _train_latent_v41(
    model,
    fit_tasks,
    probe_tasks,
    *,
    seed: int,
    objective: str,
    pair_epochs: int = 1,
    batch_size: int = 16,
    steps_limit: int = 0,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup: int = 100,
    eta_min_ratio: float = 0.05,
    latent_beta: float = 1.0,
    probe_every: int = 250,
    probe_count: int = 53,
    log_every: int = 25,
):
    """Train a latent NP/ANP on the complete v4.1 task-budget queue.

    Latent distributions are currently single-episode objects, so the forward
    passes are sequential within each optimizer batch. The loss is averaged
    across the same distinct-task batches used by the padded deterministic
    trainer. This preserves the important v4.1 contract: every task sees every
    budget exactly once per epoch with rotating, disjoint context/target sets.
    """

    import torch

    torch.manual_seed(seed + 202)
    from aims_starter.faithful_ladder import (
        brier_alc_on_tasks,
        build_pair_epoch_batches,
        latent_episode_loss,
        sample_pair_episode,
    )

    device = next(model.parameters()).device
    scheduled: list[tuple[int, list[Any]]] = []
    pairs_per_epoch = 0
    batches_per_epoch = 0
    for epoch in range(int(pair_epochs)):
        epoch_batches = build_pair_epoch_batches(
            fit_tasks,
            seed=seed + 1009 * epoch,
            batch_size=int(batch_size),
        )
        if epoch == 0:
            batches_per_epoch = len(epoch_batches)
            pairs_per_epoch = sum(len(batch) for batch in epoch_batches)
        scheduled.extend((epoch, batch) for batch in epoch_batches)
    if steps_limit > 0:
        scheduled = scheduled[: int(steps_limit)]
    if not scheduled:
        raise ValueError("no latent training batches were scheduled")

    total_steps = len(scheduled)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    warmup_steps = max(1, min(int(warmup), max(1, total_steps // 10)))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    rng = np.random.default_rng(seed + 101)
    history: list[dict[str, Any]] = []
    ema_loss: float | None = None
    best_probe = float("inf")
    best_step: int | None = None
    best_state: dict[str, Any] | None = None
    min_probe_step = max(0, total_steps // 10)
    pairs_seen = 0
    model.train()

    for step, (epoch, batch) in enumerate(scheduled):
        losses = []
        diagnostics = []
        for visit in batch:
            episode = sample_pair_episode(
                fit_tasks[visit.task_index],
                budget=int(visit.budget),
                rng=rng,
                split_seed=SPLIT_SEED,
                visit_key=f"e{epoch}:b{visit.budget}",
            )
            x_c = torch.as_tensor(episode["x_c"], dtype=torch.float32, device=device)
            y_c = torch.as_tensor(episode["y_c"], dtype=torch.float32, device=device)
            x_t = torch.as_tensor(episode["x_t"], dtype=torch.float32, device=device)
            y_t = torch.as_tensor(episode["y_t"], dtype=torch.float32, device=device)
            loss, diagnostic = latent_episode_loss(
                model,
                x_c,
                y_c,
                x_t,
                y_t,
                objective=objective,
                beta=latent_beta,
                n_z_samples=1,
            )
            losses.append(loss)
            diagnostics.append(diagnostic)

        batch_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        pairs_seen += len(batch)

        value = float(batch_loss.detach())
        ema_loss = value if ema_loss is None else 0.98 * ema_loss + 0.02 * value
        record: dict[str, Any] = {
            "step": step,
            "epoch": int(epoch),
            "pairs_seen": int(pairs_seen),
            "batch_tasks": len(batch),
            "loss": value,
            "ema_loss": ema_loss,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        for key in ("bce", "brier", "kl"):
            values = [diagnostic[key] for diagnostic in diagnostics if key in diagnostic]
            if values:
                record[key] = float(np.mean(values))
        if probe_tasks and (step % probe_every == 0 or step == total_steps - 1):
            probe = brier_alc_on_tasks(
                model,
                probe_tasks,
                device=device,
                max_tasks=probe_count,
                latent_samples=64,
            )
            record["probe_brier_alc"] = float(probe["brier_alc"])
            model.train()
            if step >= min_probe_step and probe["brier_alc"] < best_probe:
                best_probe = float(probe["brier_alc"])
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        if step % log_every == 0 or step == total_steps - 1:
            history.append(record)
            print(
                f"seed={seed} step={step}/{total_steps} "
                f"pairs={pairs_seen} loss={record['loss']:.6f} "
                f"ema={record['ema_loss']:.6f} kl={record.get('kl', float('nan')):.6f} "
                f"probe={record.get('probe_brier_alc', float('nan')):.6f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(
            {key: value.to(device) for key, value in best_state.items()}
        )
        selected_step = int(best_step) if best_step is not None else total_steps - 1
    else:
        selected_step = total_steps - 1
    model.eval()
    return model, history, {
        "selected_step": selected_step,
        "best_probe_brier_alc": (
            best_probe if best_state is not None else float("nan")
        ),
        "steps_run": int(total_steps),
        "pair_epochs": int(pair_epochs),
        "batches_per_epoch": int(batches_per_epoch),
        "pairs_per_epoch": int(pairs_per_epoch),
        "steps_per_epoch": int(batches_per_epoch),
        "batch_size": int(batch_size),
        "scheduled_pairs": int(sum(len(batch) for _, batch in scheduled)),
    }


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def _logit_array(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-8, 1.0 - 1e-8)
    return np.log(clipped / (1.0 - clipped))


def _context_rate(labels: np.ndarray) -> float:
    values = np.asarray(labels, dtype=float)
    return (1.0 + float(values.sum())) / (2.0 + len(values))


def _score_validation(
    model,
    seed: int,
    mode: str,
    orders: int = 10,
    latent_samples: int = 128,
    max_tasks: int = 0,
    corpus_root: Path | None = None,
):
    import pandas as pd
    import torch

    from aims_starter.faithful_ladder import predict_probability
    from aims_starter.uniform_eval import validation_fold_map

    root = corpus_root or CORPUS_ROOT
    item_embeddings = np.load(root / "item_embeddings.npy")
    subject_embeddings = np.load(root / "subject_embeddings.npy")
    evaluation = pd.read_parquet(root / "eval_items.parquet")
    order_frame = pd.read_parquet(root / "eval_orders.parquet")
    fold_map = validation_fold_map(evaluation, n_folds=UNIFORM_FOLDS)
    device = next(model.parameters()).device
    torch.manual_seed(seed + 9999)
    model.eval()

    records = []
    beta_records = []
    validation = evaluation[evaluation["split"] == "validation"]
    task_ids = sorted(validation["task_id"].astype(str).unique())
    if max_tasks > 0:
        task_ids = task_ids[: int(max_tasks)]
    for task_id in task_ids:
        task_rows = validation[validation["task_id"].astype(str) == task_id]
        benchmark = str(task_rows["benchmark_id"].iloc[0])
        target_rows = task_rows[task_rows["pool"] == "evaluation"].reset_index(drop=True)
        x_t_np = _features(target_rows, item_embeddings, subject_embeddings)
        y_t = target_rows["label"].to_numpy(dtype="float32")
        lookup = task_rows.set_index("item_index")
        with torch.no_grad():
            x_t = torch.as_tensor(x_t_np, dtype=torch.float32, device=device)
        for order_seed in range(orders):
            order_rows = (
                order_frame[
                    (order_frame["task_id"].astype(str) == task_id)
                    & (order_frame["order_seed"] == order_seed)
                ]
                .sort_values("acquisition_rank")
                .reset_index(drop=True)
            )
            context_rows = lookup.loc[order_rows["item_index"].to_numpy()].reset_index()
            x_c_np = _features(context_rows, item_embeddings, subject_embeddings)
            y_c = context_rows["label"].to_numpy(dtype="float32")
            with torch.no_grad():
                x_c = torch.as_tensor(x_c_np, dtype=torch.float32, device=device)
                y_c_t = torch.as_tensor(y_c, dtype=torch.float32, device=device)
            for budget in BUDGETS:
                base = {
                    "seed": seed,
                    "mode": mode,
                    "split": "validation",
                    "benchmark_id": benchmark,
                    "task_id": task_id,
                    "order_seed": int(order_seed),
                    "fold": int(fold_map.get(task_id, -1)),
                    "budget": int(budget),
                    "labels": np.asarray(y_t, dtype=float),
                    "context_labels": np.asarray(y_c[:budget], dtype=float),
                }
                with torch.no_grad():
                    probability = predict_probability(
                        model,
                        x_c[:budget],
                        y_c_t[:budget],
                        x_t,
                        n_z_samples=latent_samples,
                    )
                logits = _logit_array(probability.detach().cpu().numpy())
                records.append({**base, "logits": logits})
                beta_probability = (2.0 + float(y_c[:budget].sum())) / (4.0 + budget)
                beta_records.append(
                    {
                        **base,
                        "logits": np.full(len(y_t), _logit_array(
                            np.asarray([beta_probability])
                        )[0], dtype=float),
                    }
                )
        print(f"seed={seed} scored {task_id}", flush=True)
    return records, beta_records


def _fit_calibrator(samples, rng):
    from aims_starter.calibration import AnchorCalibrator

    logits = np.concatenate([sample["logits"] for sample in samples])
    labels = np.concatenate([sample["labels"] for sample in samples])
    rates = np.concatenate(
        [
            np.full(len(sample["labels"]), _context_rate(sample["context_labels"]))
            for sample in samples
        ]
    )
    budgets = np.concatenate(
        [np.full(len(sample["labels"]), sample["budget"], dtype=int) for sample in samples]
    )
    if len(labels) > FIT_SAMPLE_LIMIT:
        index = rng.choice(len(labels), size=FIT_SAMPLE_LIMIT, replace=False)
        logits, labels, rates, budgets = (
            logits[index],
            labels[index],
            rates[index],
            budgets[index],
        )
    return AnchorCalibrator.fit(logits, labels, rates, budgets)


def _calibrators(records):
    rng = np.random.default_rng(20261020)
    folds = sorted({int(record["fold"]) for record in records})
    return {
        "full": _fit_calibrator(records, rng),
        "folds": {
            fold: _fit_calibrator(
                [record for record in records if int(record["fold"]) != fold], rng
            )
            for fold in folds
        },
    }


def _metric_rows(records, calibrators, arm: str, calibration: str):
    rows = []
    for record in records:
        labels = np.asarray(record["labels"], dtype=float)
        logits = np.asarray(record["logits"], dtype=float)
        if calibration == "raw" or arm == "beta":
            probability = _sigmoid(logits)
        else:
            calibrator = calibrators["folds"][int(record["fold"])]
            probability = calibrator.transform(
                logits,
                int(record["budget"]),
                np.asarray(record["context_labels"], dtype=float),
            )
        rows.append(
            {
                "arm": arm,
                "calibration": calibration,
                "task_id": record["task_id"],
                "order_seed": int(record["order_seed"]),
                "budget": int(record["budget"]),
                "brier": float(np.mean((probability - labels) ** 2)),
            }
        )
    return rows


def _summarize(rows):
    import pandas as pd

    frame = pd.DataFrame(rows)
    weight = dict(zip(BUDGETS, WEIGHTS, strict=True))
    frame["weighted"] = frame["budget"].map(weight) * frame["brier"]
    task_alc = (
        frame.groupby(["arm", "calibration", "task_id", "order_seed"], sort=False)["weighted"]
        .sum()
        .reset_index()
        .groupby(["arm", "calibration", "task_id"], sort=False)["weighted"]
        .mean()
        .reset_index()
    )
    alc = []
    for (arm, calibration), group in task_alc.groupby(["arm", "calibration"], sort=True):
        alc.append(
            {
                "arm": arm,
                "calibration": calibration,
                "split": "validation",
                "brier_alc": float(group["weighted"].mean()),
                "se": float(group["weighted"].std(ddof=1) / np.sqrt(len(group))),
                "tasks": len(group),
            }
        )
    task_records = task_alc.to_dict(orient="records")
    for row in task_records:
        row["weighted"] = float(row["weighted"])
    return alc, task_records


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=8 * 60 * 60,
)
def run(
    rung: str = "2pl",
    seed: int = 20261018,
    objective: str = "bce",
    mode: str = "masked",
    pair_epochs: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    steps: int = 0,
    episodes_per_step: int = 16,
    task_limit: int = 0,
    validation_limit: int = 0,
    probe_count: int = DEFAULT_PROBE_COUNT,
    probe_every: int = 250,
    probe_benchmark_count: int = DEFAULT_PROBE_BENCHMARKS,
    corpus: str = "",
) -> dict[str, Any]:
    if rung not in RUNGS:
        raise ValueError(f"rung must be one of {tuple(RUNGS)}")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if pair_epochs < 1:
        raise ValueError("pair_epochs must be positive")
    if batch_size != DEFAULT_BATCH_SIZE:
        raise ValueError(
            f"v4.1 batch contract requires batch_size={DEFAULT_BATCH_SIZE}"
        )
    if steps < 0 or episodes_per_step < 1:
        raise ValueError("steps must be non-negative and episodes_per_step positive")
    if task_limit < 0 or validation_limit < 0:
        raise ValueError("task_limit and validation_limit must be non-negative")
    if probe_count < 0 or probe_every < 1:
        raise ValueError("probe_count must be non-negative and probe_every must be positive")
    if probe_benchmark_count < 0:
        raise ValueError("probe_benchmark_count must be non-negative")

    limited = bool(task_limit or validation_limit or steps)
    run_name = f"ladder-v41-{rung}-{objective}-{seed}-{mode}"
    if corpus:
        run_name += f"-{corpus}"
    if int(pair_epochs) != 1:
        run_name += f"-pe{int(pair_epochs)}"
    if limited:
        run_name += "-smoke"
    output = RESULT_ROOT / f"{run_name}.json"
    code_signature = _code_signature()
    existing = _read_json(output)
    if existing is not None:
        protocol = existing.get("protocol", {})
        expected = {
            "split_seed": SPLIT_SEED,
            "pair_epochs": int(pair_epochs),
            "batch_size": int(batch_size),
            "steps_requested": int(steps),
            "task_limit": int(task_limit),
            "validation_limit": int(validation_limit),
            "episodes_per_step": None,
            "probe_count": int(probe_count),
            "probe_every": int(probe_every),
            "probe_benchmark_count": int(probe_benchmark_count),
        }
        mismatch = {
            key: (protocol.get(key), value)
            for key, value in expected.items()
            if protocol.get(key) != value
        }
        if (
            mismatch
            or existing.get("version") != VERSION
            or existing.get("rung") != rung
            or existing.get("objective") != objective
            or existing.get("mode") != mode
        ):
            raise RuntimeError(
                f"cached result has a different protocol: {mismatch or 'metadata mismatch'}; "
                f"remove {output} before rerunning"
            )
        if existing.get("code_signature") != code_signature:
            raise RuntimeError(
                "cached result was produced by different training or calibration code; "
                f"remove {output} before rerunning"
            )
        return existing

    import pandas as pd
    import torch

    corpus_root = _corpus_root(corpus)
    manifest = _read_json(corpus_root / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"compact corpus missing: {corpus_root}")
    item_embeddings = np.load(corpus_root / "item_embeddings.npy")
    subject_embeddings = np.load(corpus_root / "subject_embeddings.npy")
    item_dim = int(item_embeddings.shape[1])
    subject_prior_slice = (item_dim, item_dim + int(subject_embeddings.shape[1]))
    subject_slice = subject_prior_slice if mode == "masked" else None

    protocol = pd.read_parquet(
        PROTOCOL_ROOT / "eligible_tasks.parquet",
        columns=["task_id", "benchmark_id"],
    )
    benchmark_of = dict(
        zip(
            protocol["task_id"].astype(str),
            protocol["benchmark_id"].astype(str),
            strict=True,
        )
    )
    train = pd.read_parquet(corpus_root / "train_tasks.parquet")
    tasks = _task_list(train, item_embeddings, subject_embeddings, benchmark_of)
    probe_benchmarks = _select_probe_benchmarks(
        tasks,
        probe_benchmark_count,
        minimum_tasks=probe_count,
    )
    fit_tasks, probe_tasks = _split_fit_probe(
        tasks, probe_count, probe_benchmarks
    )
    if task_limit:
        from aims_starter.faithful_ladder import stable_hash

        fit_tasks = sorted(
            fit_tasks, key=lambda task: stable_hash(task["task_id"])
        )[: int(task_limit)]
    if not fit_tasks:
        raise ValueError("no fit tasks available")
    config = RUNGS[rung]
    latent = config["family"] == "latent"

    # Seed before model construction.  This is the bug in the old runner.
    _seed_everything(seed)
    model = _make_model(
        rung,
        int(tasks[0]["x"].shape[1]),
        subject_slice,
        subject_prior_slice,
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    if latent:
        model, history, selection = _train_latent_v41(
            model,
            fit_tasks,
            probe_tasks,
            seed=seed,
            objective=objective,
            pair_epochs=int(pair_epochs),
            batch_size=int(batch_size),
            steps_limit=int(steps),
            probe_every=int(probe_every),
            probe_count=int(probe_count),
        )
        selection["queue_mode"] = "v4.1 pair epoch; sequential latent forwards"
    else:
        model, history, selection = _train_batched(
            model,
            fit_tasks,
            probe_tasks,
            seed=seed,
            objective=objective,
            pair_epochs=int(pair_epochs),
            batch_size=int(batch_size),
            steps_limit=int(steps),
            probe_every=int(probe_every),
            probe_count=int(probe_count),
        )
        selection["queue_mode"] = "v4.1 pair epoch"

    checkpoint_root = RESULT_ROOT / f"{run_name}-checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_root / "model.pt")

    records, beta_records = _score_validation(
        model,
        seed,
        mode,
        orders=10,
        max_tasks=int(validation_limit),
        corpus_root=corpus_root,
    )
    calibrators = _calibrators(records)
    rows = []
    for calibration in ("raw", "oof"):
        rows.extend(_metric_rows(records, calibrators, rung, calibration))
        rows.extend(_metric_rows(beta_records, calibrators, "beta", calibration))
    alc, task_alc = _summarize(rows)
    steps_run = int(selection.get("steps_run", steps))
    scheduled_pairs = int(
        selection.get("scheduled_pairs", steps_run * episodes_per_step)
    )
    result = {
        "status": "complete",
        "version": VERSION,
        "code_signature": code_signature,
        "rung": rung,
        "arm": rung,
        "objective": objective,
        "seed": seed,
        "mode": mode,
        "split": "validation",
        "orders": 10,
        "manifest": manifest,
        "protocol": {
            "task_split": "per-visit rotating disjoint acquisition/evaluation split",
            "split_seed": SPLIT_SEED,
            "context": "uniform sample from acquisition half",
            "targets": "evaluation half only",
            "queue_mode": selection.get("queue_mode"),
            "objectives": list(OBJECTIVES),
            "budgets": list(BUDGETS),
            "weights": list(WEIGHTS),
            "budget_weighting": (
                "brier_alc applies 6*w_b to prediction loss only"
                if not latent
                else None
            ),
            "pair_epochs": int(pair_epochs),
            "batch_size": int(batch_size),
            "steps_requested": int(steps),
            "steps": steps_run,
            "pairs_epoch": selection.get("pairs_per_epoch"),
            "batches_epoch": selection.get("batches_per_epoch"),
            "scheduled_pairs": scheduled_pairs,
            "coverage_passes": float(scheduled_pairs / max(1, len(fit_tasks))),
            "episodes_per_step": None,
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "warmup": max(1, min(100, max(1, steps_run // 10))),
            "schedule": "warmup + cosine to 5%",
            "fit_tasks": len(fit_tasks),
            "task_limit": int(task_limit),
            "validation_limit": int(validation_limit),
            "probe_tasks": len(probe_tasks),
            "probe_count": int(probe_count),
            "probe_every": int(probe_every),
            "probe_benchmark_count": int(probe_benchmark_count),
            "probe_benchmarks": list(probe_benchmarks),
            "probe_selection": (
                "task-level holdout across 4 hash-selected train benchmarks; "
                "round-robin by benchmark"
            ),
            "selection": "best held-out train-probe Brier ALC",
            "latent_kl": "KL(q(z|C union Q) || q(z|C)) / |Q|" if latent else None,
            "anchor": "detached 2PL MAP ability" if not latent else "none",
            "decoder": (
                "direct Bernoulli ANP"
                if config.get("direct_decoder")
                else "latent 2PL-shaped response"
                if latent
                else "anchored response"
            ),
            "anp_shrinkage": 1e-3 if config.get("use_anp") else None,
        },
        "model": config,
        "selection": selection,
        "training": history,
        "alc": alc,
        "task_alc": task_alc,
    }
    _write_json(output, result)
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    rungs: str = "2pl,gp_2pl,2pl_anp,gp_anp",
    seeds: str = "20261018,20261019,20261020",
    objective: str = "bce",
    mode: str = "masked",
    pair_epochs: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    steps: int = 0,
    episodes_per_step: int = 16,
    task_limit: int = 0,
    validation_limit: int = 0,
    probe_count: int = DEFAULT_PROBE_COUNT,
    probe_every: int = 250,
    probe_benchmark_count: int = DEFAULT_PROBE_BENCHMARKS,
    parallel: bool = False,
    corpus: str = "",
):
    rung_values = [value for value in rungs.split(",") if value]
    seed_values = [int(value) for value in seeds.split(",") if value]
    objective_values = [value for value in objective.split(",") if value]
    for rung in rung_values:
        if rung not in RUNGS:
            raise ValueError(f"rung must be one of {tuple(RUNGS)}")
    for objective_value in objective_values:
        if objective_value not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    if not parallel:
        for rung in rung_values:
            for seed in seed_values:
                for objective_value in objective_values:
                    print(
                        f"=== {rung} seed={seed} objective={objective_value} ===",
                        flush=True,
                    )
                    result = run.remote(
                        rung,
                        seed,
                        objective_value,
                        mode,
                        int(pair_epochs),
                        int(batch_size),
                        int(steps),
                        int(episodes_per_step),
                        int(task_limit),
                        int(validation_limit),
                        int(probe_count),
                        int(probe_every),
                        int(probe_benchmark_count),
                        corpus,
                    )
                    print(
                        json.dumps(
                            {"status": result.get("status"), "alc": result.get("alc")},
                            indent=2,
                        )
                    )
        return

    calls = [
        (
            rung,
            seed,
            objective_value,
            run.spawn(
                rung,
                seed,
                objective_value,
                mode,
                int(pair_epochs),
                int(batch_size),
                int(steps),
                int(episodes_per_step),
                int(task_limit),
                int(validation_limit),
                int(probe_count),
                int(probe_every),
                int(probe_benchmark_count),
                corpus,
            ),
        )
        for rung in rung_values
        for seed in seed_values
        for objective_value in objective_values
    ]
    for rung, seed, objective_value, call in calls:
        print(f"=== {rung} seed={seed} objective={objective_value} ===", flush=True)
        result = call.get()
        print(json.dumps({"status": result.get("status"), "alc": result.get("alc")}, indent=2))
