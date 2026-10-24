"""Competition-faithful ablation ladder for the ANP/GP-IRT question.

This module is intentionally isolated from the production CNP/ANP classes in
``faithful_np.py``.  It implements the corrected protocol:

* a fixed 50/50 acquisition/evaluation split per training task;
* nested contexts drawn only from the acquisition half;
* the evaluation half is never a training target;
* a target-specific ANP correction on top of an item-only 2PL anchor;
* an optional finite-basis GP residual over ability;
* a corrected latent-NP/ANP loss with ``KL / |T|``.

The GP residual is a finite-basis MAP correction, not a full GPIRT posterior.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import kl_divergence

from .r_c_attention import ContextSelfAttention, TargetContextAttention

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)


def stable_hash(value: Any) -> int:
    """Return a deterministic 64-bit hash for task-level splits."""

    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def split_task_indices(
    n: int,
    task_id: str,
    *,
    split_seed: int = 20261017,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one training task into fixed acquisition and evaluation halves.

    The split depends only on ``task_id`` and ``split_seed``, so every model,
    objective, and model seed sees exactly the same task split.
    """

    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        raise ValueError("a task must contain at least one item")
    if n == 1:
        return np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64)
    rng = np.random.default_rng(stable_hash(f"{split_seed}:{task_id}"))
    order = rng.permutation(n)
    n_acquisition = max(1, min(n - 1, n // 2))
    acquisition = np.sort(order[:n_acquisition]).astype(np.int64)
    evaluation = np.sort(order[n_acquisition:]).astype(np.int64)
    return acquisition, evaluation


def make_task(
    task_id: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    split_seed: int = 20261017,
) -> dict[str, Any]:
    """Package one training task with a fixed acquisition/evaluation split."""

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    acquisition, evaluation = split_task_indices(
        len(y), str(task_id), split_seed=split_seed
    )
    return {
        "task_id": str(task_id),
        "x": x,
        "y": y,
        "acquisition": acquisition,
        "evaluation": evaluation,
    }


def sample_budget(rng: np.random.Generator) -> int:
    """Sample a budget with the competition ALC weights.

    The expected Brier loss under this sampler is the Brier ALC, so the
    stochastic objective is unbiased for the competition metric.
    """

    index = int(rng.choice(len(BUDGETS), p=np.asarray(WEIGHTS, dtype=float)))
    return int(BUDGETS[index])


def sample_episode(
    task: dict[str, Any],
    rng: np.random.Generator,
    device: torch.device,
    *,
    budget: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Draw a nested context from the acquisition half and score evaluation items."""

    if budget is None:
        budget = sample_budget(rng)
    acquisition = task["acquisition"]
    n_context = min(int(budget), len(acquisition))
    if len(acquisition):
        order = rng.permutation(acquisition)
        context_indices = order[:n_context]
    else:
        context_indices = acquisition[:0]
    target_indices = task["evaluation"]
    x_c = torch.as_tensor(task["x"][context_indices], dtype=torch.float32, device=device)
    y_c = torch.as_tensor(task["y"][context_indices], dtype=torch.float32, device=device)
    x_t = torch.as_tensor(task["x"][target_indices], dtype=torch.float32, device=device)
    y_t = torch.as_tensor(task["y"][target_indices], dtype=torch.float32, device=device)
    return x_c, y_c, x_t, y_t, n_context


ALC_PAIR_WEIGHTS = {
    int(budget): float(6.0 * weight) for budget, weight in zip(BUDGETS, WEIGHTS)
}


@dataclass(frozen=True)
class PairVisit:
    """One ``(task, budget, pass)`` visit in a pair epoch."""

    task_index: int
    budget: int
    pass_index: int


@dataclass
class PaddedEpisodeBatch:
    """Padded, masked batch of episodes."""

    x_c: torch.Tensor
    y_c: torch.Tensor
    context_mask: torch.Tensor
    x_t: torch.Tensor
    y_t: torch.Tensor
    target_mask: torch.Tensor
    budgets: torch.Tensor
    task_indices: torch.Tensor


def sample_pair_episode(
    task: dict[str, Any],
    *,
    budget: int,
    rng: np.random.Generator,
    split_seed: int = 20261017,
    visit_key: str,
) -> dict[str, Any]:
    """Sample one visit with a rotating disjoint C/Q split.

    The split key excludes the model seed, so every arm sees the same C/Q split
    for the same task, budget, and pass.  The context order still follows the
    model-seed RNG.
    """

    acquisition, evaluation = split_task_indices(
        len(task["y"]),
        f"{task['task_id']}:{visit_key}",
        split_seed=split_seed,
    )
    n_context = min(int(budget), len(acquisition))
    if len(acquisition):
        order = rng.permutation(acquisition)
        context_indices = order[:n_context]
    else:
        context_indices = acquisition[:0]
    return {
        "task_index": int(task.get("task_index", -1)),
        "budget": int(budget),
        "x_c": task["x"][context_indices],
        "y_c": task["y"][context_indices],
        "x_t": task["x"][evaluation],
        "y_t": task["y"][evaluation],
    }


def collate_episodes(
    episodes: list[dict[str, Any]],
    device: torch.device,
) -> PaddedEpisodeBatch:
    """Pad variable-length C/Q episodes into the v4.1 tensor contract."""

    if not episodes:
        raise ValueError("cannot collate an empty episode list")
    feature_dim = int(episodes[0]["x_t"].shape[-1])
    max_context = max(int(episode["x_c"].shape[0]) for episode in episodes)
    max_target = max(int(episode["x_t"].shape[0]) for episode in episodes)
    count = len(episodes)
    x_c = torch.zeros((count, max_context, feature_dim), dtype=torch.float32, device=device)
    y_c = torch.zeros((count, max_context), dtype=torch.float32, device=device)
    x_t = torch.zeros((count, max_target, feature_dim), dtype=torch.float32, device=device)
    y_t = torch.zeros((count, max_target), dtype=torch.float32, device=device)
    context_mask = torch.zeros((count, max_context), dtype=torch.bool, device=device)
    target_mask = torch.zeros((count, max_target), dtype=torch.bool, device=device)
    budgets = torch.zeros(count, dtype=torch.long, device=device)
    task_indices = torch.full((count,), -1, dtype=torch.long, device=device)
    for row, episode in enumerate(episodes):
        n_context = int(episode["x_c"].shape[0])
        n_target = int(episode["x_t"].shape[0])
        if n_context:
            x_c[row, :n_context] = torch.as_tensor(
                episode["x_c"], dtype=torch.float32, device=device
            )
            y_c[row, :n_context] = torch.as_tensor(
                episode["y_c"], dtype=torch.float32, device=device
            )
            context_mask[row, :n_context] = True
        if n_target:
            x_t[row, :n_target] = torch.as_tensor(
                episode["x_t"], dtype=torch.float32, device=device
            )
            y_t[row, :n_target] = torch.as_tensor(
                episode["y_t"], dtype=torch.float32, device=device
            )
            target_mask[row, :n_target] = True
        budgets[row] = int(episode["budget"])
        task_indices[row] = int(episode.get("task_index", -1))
    return PaddedEpisodeBatch(
        x_c=x_c,
        y_c=y_c,
        context_mask=context_mask,
        x_t=x_t,
        y_t=y_t,
        target_mask=target_mask,
        budgets=budgets,
        task_indices=task_indices,
    )


def build_pair_epoch_batches(
    tasks: list[dict[str, Any]],
    *,
    seed: int,
    batch_size: int = 16,
) -> list[list[PairVisit]]:
    """Build one v4.1 pair epoch.

    Every task gets six visits, one at each budget.  Full batches contain
    distinct tasks.  Budgets are assigned greedily from each task's remaining
    budget set, which keeps every batch near ``16 / 6`` per budget while
    guaranteeing that each task covers all six budgets exactly once.  The task
    count remainder is a separate tail pool so the last batch is small and can
    be size-weighted by the trainer.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    n_tasks = len(tasks)
    if n_tasks == 0:
        return []
    order = sorted(
        range(n_tasks), key=lambda index: stable_hash(str(tasks[index]["task_id"]))
    )
    offsets = np.empty(n_tasks, dtype=np.int64)
    for rank, index in enumerate(order):
        offsets[index] = rank % len(BUDGETS)

    tail_size = n_tasks % batch_size
    tails: list[list[int]] = [[] for _ in BUDGETS]
    if tail_size and len(BUDGETS) * tail_size <= n_tasks:
        tail_pool = order[: len(BUDGETS) * tail_size]
        for pass_index in range(len(BUDGETS)):
            tail = [
                int(index)
                for index in tail_pool[
                    pass_index * tail_size : (pass_index + 1) * tail_size
                ]
            ]
            # Spread the 6 * tail_size tail visits evenly over the six budgets.
            for local_index, index in enumerate(tail):
                desired_index = (pass_index * tail_size + local_index) % len(
                    BUDGETS
                )
                offsets[index] = (desired_index - pass_index) % len(BUDGETS)
            tails[pass_index] = tail

    batches: list[list[PairVisit]] = []
    for pass_index in range(len(BUDGETS)):
        tail_set = set(tails[pass_index])
        groups: dict[int, list[int]] = {offset: [] for offset in range(len(BUDGETS))}
        for index in range(n_tasks):
            if index not in tail_set:
                groups[int(offsets[index])].append(index)
        for offset, group in groups.items():
            rng = np.random.default_rng(
                stable_hash(f"{seed}:pair-pass:{pass_index}:offset:{offset}")
            )
            rng.shuffle(group)
        remaining = sum(len(group) for group in groups.values())
        rotation = 0
        while remaining:
            batch: list[PairVisit] = []
            for slot in range(batch_size):
                chosen = None
                for step in range(len(BUDGETS)):
                    offset = (rotation + slot + step) % len(BUDGETS)
                    if groups[offset]:
                        chosen = groups[offset].pop()
                        break
                if chosen is None:
                    break
                budget_index = int((pass_index + offsets[chosen]) % len(BUDGETS))
                batch.append(
                    PairVisit(
                        task_index=int(chosen),
                        budget=int(BUDGETS[budget_index]),
                        pass_index=int(pass_index),
                    )
                )
            if not batch:
                break
            batches.append(batch)
            remaining -= len(batch)
            rotation = (rotation + 1) % len(BUDGETS)

    by_budget: dict[int, list[PairVisit]] = {budget: [] for budget in BUDGETS}
    for pass_index in range(len(BUDGETS)):
        for index in tails[pass_index]:
            budget_index = int((pass_index + offsets[index]) % len(BUDGETS))
            by_budget[int(BUDGETS[budget_index])].append(
                PairVisit(
                    task_index=int(index),
                    budget=int(BUDGETS[budget_index]),
                    pass_index=int(pass_index),
                )
            )
    total_tail = sum(len(values) for values in by_budget.values())
    if (
        batch_size == 16
        and total_tail == 54
        and all(len(by_budget[budget]) == 9 for budget in BUDGETS)
    ):
        patterns = (
            {BUDGETS[0]: 2, BUDGETS[1]: 2, BUDGETS[2]: 3, BUDGETS[3]: 3, BUDGETS[4]: 3, BUDGETS[5]: 3},
            {BUDGETS[0]: 3, BUDGETS[1]: 3, BUDGETS[2]: 2, BUDGETS[3]: 2, BUDGETS[4]: 3, BUDGETS[5]: 3},
            {BUDGETS[0]: 3, BUDGETS[1]: 3, BUDGETS[2]: 3, BUDGETS[3]: 3, BUDGETS[4]: 2, BUDGETS[5]: 2},
            {budget: 1 for budget in BUDGETS},
        )
        for pattern in patterns:
            chunk = []
            for budget, count in pattern.items():
                for _ in range(count):
                    chunk.append(by_budget[budget].pop())
            batches.append(chunk)
        return batches

    tail_chunk = 0
    while any(by_budget.values()):
        chunk: list[PairVisit] = []
        while len(chunk) < batch_size and any(by_budget.values()):
            for step in range(len(BUDGETS)):
                budget = BUDGETS[(tail_chunk + step) % len(BUDGETS)]
                if by_budget[budget] and len(chunk) < batch_size:
                    chunk.append(by_budget[budget].pop())
        if chunk:
            batches.append(chunk)
            tail_chunk += 1
    return batches


def _masked_task_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    counts = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / counts


def deterministic_batch_loss(
    model: nn.Module,
    batch: PaddedEpisodeBatch,
    *,
    objective: str = "bce",
    batch_size: int = 16,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Masked loss for a padded deterministic batch.

    The prediction term is summed over tasks and divided by the full batch size,
    not by the number of tasks in this batch.  This is the size-weighting needed
    for the small final batch.  For ``brier_alc``, each task-budget pair gets the
    ``6 * w_b`` ALC multiplier; the latent KL is not touched here.
    """

    logits, _target_mask = model.forward_batch(
        batch.x_c,
        batch.y_c,
        batch.x_t,
        batch.context_mask,
        batch.target_mask,
    )
    labels = batch.y_t
    bce_items = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    brier_items = (torch.sigmoid(logits) - labels).square()
    bce_task = _masked_task_mean(bce_items, batch.target_mask)
    brier_task = _masked_task_mean(brier_items, batch.target_mask)
    if objective == "bce":
        prediction = bce_task
        pair_weights = torch.ones_like(prediction)
    elif objective == "brier" or objective == "brier_alc":
        prediction = brier_task
        if objective == "brier_alc":
            pair_weights = torch.as_tensor(
                [ALC_PAIR_WEIGHTS[int(budget)] for budget in batch.budgets.tolist()],
                dtype=prediction.dtype,
                device=prediction.device,
            )
        else:
            pair_weights = torch.ones_like(prediction)
    elif objective == "bce_brier":
        prediction = bce_task + brier_task
        pair_weights = torch.ones_like(prediction)
    else:
        raise ValueError(f"unknown objective: {objective}")
    denominator = float(max(1, int(batch_size)))
    value = (pair_weights * prediction).sum() / denominator
    penalty = model.gp_penalty() if hasattr(model, "gp_penalty") else logits.new_zeros(())
    auxiliary = model.aux_loss() if hasattr(model, "aux_loss") else logits.new_zeros(())
    loss = value + penalty + auxiliary
    return loss, {
        "bce": float(bce_task.mean().detach()),
        "brier": float(brier_task.mean().detach()),
        "prediction": float(value.detach()),
        "penalty": float(penalty.detach()),
        "auxiliary": float(auxiliary.detach()),
        "tasks": float(len(batch.budgets)),
        "valid_targets": float(batch.target_mask.sum().detach()),
    }


def _brier(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) - labels).square().mean()


def _objective_value(
    logits: torch.Tensor,
    labels: torch.Tensor,
    objective: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    brier = _brier(logits, labels)
    if objective == "bce":
        value = bce
    elif objective in {"brier", "brier_alc"}:
        value = brier
    elif objective == "bce_brier":
        value = bce + brier
    else:
        raise ValueError(f"unknown objective: {objective}")
    return value, bce, brier


def deterministic_episode_loss(
    model: nn.Module,
    x_c: torch.Tensor,
    y_c: torch.Tensor,
    x_t: torch.Tensor,
    y_t: torch.Tensor,
    *,
    objective: str = "bce",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss for an anchored deterministic model."""

    logits = model(x_c, y_c, x_t)
    value, bce, brier = _objective_value(logits, y_t, objective)
    penalty = model.gp_penalty() if hasattr(model, "gp_penalty") else logits.new_zeros(())
    auxiliary = model.aux_loss() if hasattr(model, "aux_loss") else logits.new_zeros(())
    loss = value + penalty + auxiliary
    return loss, {
        "bce": float(bce.detach()),
        "brier": float(brier.detach()),
        "penalty": float(penalty.detach()),
        "auxiliary": float(auxiliary.detach()),
    }


def latent_episode_loss(
    model: nn.Module,
    x_c: torch.Tensor,
    y_c: torch.Tensor,
    x_t: torch.Tensor,
    y_t: torch.Tensor,
    *,
    objective: str = "bce",
    beta: float = 1.0,
    n_z_samples: int = 1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Corrected NP/ANP loss: target posterior plus ``KL / |Q|``.

    The old production loss averaged target- and context-posterior NLL and
    added the full KL.  The official (A)NP code divides the KL by the number of
    target points because the target log-likelihood is averaged.  This function
    follows that convention and does not use a context-posterior NLL term.
    """

    if float(model.anchor_scale) != 0.0:
        raise ValueError("the latent ladder expects anchor_scale=0.0")
    x_c_encoded = model.encode_x(x_c)
    x_t_encoded = model.encode_x(x_t)
    x_t_item = model.encode_item_x(x_t)
    q_z_c = model.q_z(x_c_encoded, y_c)
    q_z_t = model.q_z(
        torch.cat([x_c_encoded, x_t_encoded], dim=0),
        torch.cat([y_c, y_t], dim=0),
    )
    z = q_z_t.rsample((int(n_z_samples),))
    r_star = model.r_star(x_c_encoded, y_c, x_t_encoded)
    logits = model.decode_logits(
        z,
        x_t_item,
        r_star,
        x_t.new_zeros(()),
    )
    labels = y_t[None, :].expand_as(logits)
    value, bce, brier = _objective_value(logits, labels, objective)
    kl = kl_divergence(q_z_t, q_z_c) / max(1, y_t.numel())
    loss = value + beta * kl
    return loss, {
        "bce": float(bce.detach()),
        "brier": float(brier.detach()),
        "kl": float(kl.detach()),
        "penalty": 0.0,
        "auxiliary": 0.0,
    }


@torch.no_grad()
def latent_predict_probability(
    model: nn.Module,
    x_c: torch.Tensor,
    y_c: torch.Tensor,
    x_t: torch.Tensor,
    *,
    n_z_samples: int = 128,
) -> torch.Tensor:
    """Average target probabilities under the context posterior."""

    if float(model.anchor_scale) != 0.0:
        raise ValueError("the latent ladder expects anchor_scale=0.0")
    x_c_encoded = model.encode_x(x_c)
    x_t_encoded = model.encode_x(x_t)
    q_z_c = model.q_z(x_c_encoded, y_c)
    z = q_z_c.rsample((int(n_z_samples),))
    r_star = model.r_star(x_c_encoded, y_c, x_t_encoded)
    logits = model.decode_logits(
        z,
        model.encode_item_x(x_t),
        r_star,
        x_t.new_zeros(()),
    )
    return torch.sigmoid(logits).mean(dim=0)


@torch.no_grad()
def predict_probability(
    model: nn.Module,
    x_c: torch.Tensor,
    y_c: torch.Tensor,
    x_t: torch.Tensor,
    *,
    n_z_samples: int = 128,
) -> torch.Tensor:
    """Dispatch to the deterministic or latent prediction path."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_c, y_c, x_t)
    return latent_predict_probability(
        model,
        x_c,
        y_c,
        x_t,
        n_z_samples=n_z_samples,
    )


@torch.no_grad()
def brier_alc_on_tasks(
    model: nn.Module,
    tasks: list[dict[str, Any]],
    *,
    device: torch.device,
    max_tasks: int | None = None,
    latent_samples: int = 64,
) -> dict[str, Any]:
    """Raw Brier ALC on held-out probe tasks at every budget."""

    selected = tasks if max_tasks is None else tasks[: int(max_tasks)]
    if not selected:
        return {"brier_alc": float("nan"), "by_budget": {}}
    model.eval()
    by_budget = {budget: [] for budget in BUDGETS}
    with torch.random.fork_rng():
        torch.manual_seed(12345)
        for task in selected:
            x_t = torch.as_tensor(
                task["x"][task["evaluation"]], dtype=torch.float32, device=device
            )
            y_t = torch.as_tensor(
                task["y"][task["evaluation"]], dtype=torch.float32, device=device
            )
            for budget in BUDGETS:
                indices = task["acquisition"][:budget]
                x_c = torch.as_tensor(task["x"][indices], dtype=torch.float32, device=device)
                y_c = torch.as_tensor(task["y"][indices], dtype=torch.float32, device=device)
                probability = predict_probability(
                    model,
                    x_c,
                    y_c,
                    x_t,
                    n_z_samples=latent_samples,
                )
                by_budget[budget].append(
                    float((probability - y_t).square().mean().detach().cpu())
                )
    means = {budget: float(np.mean(values)) for budget, values in by_budget.items()}
    alc = float(sum(weight * means[budget] for budget, weight in zip(BUDGETS, WEIGHTS)))
    return {
        "brier_alc": alc,
        "by_budget": {str(budget): value for budget, value in means.items()},
    }


class AnchorCorrectionIRT(nn.Module):
    """Item-only 2PL anchor with optional GP and target-specific ANP corrections.

    ``use_gp`` adds a finite-basis GP residual over ability:

    ``z_t(theta) = a_t (theta - b_t) + c_t^T phi(theta)``

    ``use_anp`` adds a target-specific context correction:

    ``z_t += gamma * delta_t(C)``

    ``delta_t(C)`` is produced by self-attention over label-conditioned context
    tokens followed by target-to-context cross-attention.  It is mean-centered
    across the target set, zero-initialized, and L2-shrunk, so the model starts
    at the anchor.  The ability MAP is detached; the correction path is trained
    directly.  This is deliberate: the anchor is an estimator, not a claim that
    the context encoder was trained through the MAP solve.
    """

    def __init__(
        self,
        x_dim: int,
        *,
        use_gp: bool = False,
        use_anp: bool = False,
        d_model: int = 128,
        basis_size: int = 8,
        lengthscale: float = 1.0,
        gp_lambda: float = 1e-3,
        anp_lambda: float = 1e-3,
        n_heads: int = 8,
        n_context_layers: int = 2,
        subject_feature_slice: tuple[int, int] | None = None,
        subject_prior_slice: tuple[int, int] | None = None,
        use_subject_prior: bool = False,
    ):
        super().__init__()
        if use_subject_prior and subject_prior_slice is None:
            raise ValueError("subject_prior_slice is required for a subject prior")
        self.use_gp = bool(use_gp)
        self.use_anp = bool(use_anp)
        self.use_subject_prior = bool(use_subject_prior)
        self.subject_feature_slice = subject_feature_slice
        self.subject_prior_slice = subject_prior_slice
        self.basis_size = int(basis_size)
        self.lengthscale = float(lengthscale)
        self.gp_lambda = float(gp_lambda)
        self.anp_lambda = float(anp_lambda)

        self.item_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.item_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )

        if self.use_gp:
            self.coefficient_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, basis_size),
            )
            nn.init.zeros_(self.coefficient_head[-1].weight)
            nn.init.zeros_(self.coefficient_head[-1].bias)
            centers = torch.linspace(-3.0, 3.0, basis_size)
            self.register_buffer("centers", centers)
            delta = centers[:, None] - centers[None, :]
            kernel = torch.exp(-delta.square() / (2.0 * self.lengthscale**2))
            kernel = kernel + 1e-4 * torch.eye(basis_size)
            self.register_buffer("kernel_inv", torch.linalg.inv(kernel))

        if self.use_anp:
            self.context_encoder = nn.Sequential(
                nn.Linear(d_model + 1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            self.context_layers = nn.ModuleList(
                [ContextSelfAttention(d_model, n_heads) for _ in range(n_context_layers)]
            )
            self.cross_attention = TargetContextAttention(d_model, n_heads)
            self.delta_head = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)
            self.gamma = nn.Parameter(torch.tensor(1.0))

        if self.use_subject_prior:
            start, stop = subject_prior_slice
            self.subject_prior_head = nn.Sequential(
                nn.Linear(stop - start, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
            )
            nn.init.zeros_(self.subject_prior_head[-1].weight)
            nn.init.zeros_(self.subject_prior_head[-1].bias)
        self.register_buffer("_gp_penalty", torch.zeros(()))
        self.register_buffer("_anp_penalty", torch.zeros(()))

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        if self.subject_feature_slice is not None:
            start, stop = self.subject_feature_slice
            x = torch.cat(
                [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]],
                dim=-1,
            )
        return self.item_encoder(x)

    @staticmethod
    def _split_parameters(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.nn.functional.softplus(raw[..., 0]) + 0.05, raw[..., 1]

    def _item_parameters(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split_parameters(self.item_head(h))

    def _coefficients(self, h: torch.Tensor) -> torch.Tensor:
        if not self.use_gp:
            raise RuntimeError("GP coefficients requested from a non-GP model")
        return self.coefficient_head(h)

    def _subject_prior(
        self, x_c: torch.Tensor, x_t: torch.Tensor
    ) -> torch.Tensor | None:
        if not self.use_subject_prior:
            return None
        reference = x_t if x_t.shape[0] else x_c
        if reference.shape[0] == 0:
            return None
        start, stop = self.subject_prior_slice
        subject = reference[:, start:stop].mean(dim=0)
        return torch.tanh(self.subject_prior_head(subject)).reshape(())

    def _subject_prior_batch(
        self,
        x_c: torch.Tensor,
        x_t: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.use_subject_prior:
            return None
        start, stop = self.subject_prior_slice
        target_count = target_mask.sum(dim=1, keepdim=True).clamp_min(1).to(x_t.dtype)
        context_count = context_mask.sum(dim=1, keepdim=True).clamp_min(1).to(
            x_c.dtype
        )
        target_sum = (x_t[..., start:stop] * target_mask[..., None]).sum(dim=1)
        context_sum = (x_c[..., start:stop] * context_mask[..., None]).sum(dim=1)
        subject = torch.where(
            target_mask.any(dim=1, keepdim=True),
            target_sum / target_count,
            context_sum / context_count,
        )
        prior = torch.tanh(self.subject_prior_head(subject)).squeeze(-1)
        has_reference = target_mask.any(dim=1) | context_mask.any(dim=1)
        return torch.where(has_reference, prior, torch.zeros_like(prior))

    @staticmethod
    def _map_theta(
        a_c: torch.Tensor,
        b_c: torch.Tensor,
        c_c: torch.Tensor | None,
        y_c: torch.Tensor,
        centers: torch.Tensor | None,
        lengthscale: float,
        *,
        prior_mean: float | torch.Tensor = 0.0,
        steps: int = 12,
    ) -> torch.Tensor:
        if isinstance(prior_mean, torch.Tensor):
            prior = prior_mean.to(device=a_c.device, dtype=a_c.dtype).reshape(())
        else:
            prior = a_c.new_tensor(float(prior_mean))
        theta = prior
        if a_c.shape[0] == 0:
            return theta
        a_c = a_c.detach()
        b_c = b_c.detach()
        y_c = y_c.to(dtype=a_c.dtype)
        if c_c is not None:
            c_c = c_c.detach()
            if centers is None:
                raise ValueError("centers are required for a GP residual")
        for _ in range(steps):
            if c_c is None:
                value = a_c * (theta - b_c)
                slope = a_c
                curvature = a_c.new_zeros(())
            else:
                delta = theta - centers
                phi = torch.exp(-delta.square() / (2.0 * lengthscale**2))
                dphi = phi * (-delta / lengthscale**2)
                ddphi = phi * (
                    delta.square() / lengthscale**4 - 1.0 / lengthscale**2
                )
                value = a_c * (theta - b_c) + c_c @ phi
                slope = a_c + c_c @ dphi
                curvature = c_c @ ddphi
            probability = torch.sigmoid(value)
            gradient = torch.sum((probability - y_c) * slope) + (theta - prior) / 4.0
            hessian = torch.sum(
                probability * (1.0 - probability) * slope.square()
                + (probability - y_c) * curvature
            ) + 0.25
            theta = (
                theta - (gradient / hessian.clamp_min(1e-6)).clamp(-1.0, 1.0)
            ).clamp(-4.0, 4.0)
        return theta

    def _set_penalty(
        self,
        c_c: torch.Tensor | None,
        c_t: torch.Tensor | None,
    ) -> None:
        if not self.use_gp or c_c is None or c_t is None:
            self._gp_penalty = self._gp_penalty.new_zeros(())
            return
        coefficients = torch.cat([c_c, c_t], dim=0)
        if coefficients.shape[0] == 0:
            self._gp_penalty = coefficients.sum() * 0.0
            return
        quadratic = torch.einsum(
            "nm,mk,nk->n", coefficients, self.kernel_inv, coefficients
        )
        self._gp_penalty = self.gp_lambda * quadratic.mean()

    @staticmethod
    def _map_theta_batch(
        a_c: torch.Tensor,
        b_c: torch.Tensor,
        c_c: torch.Tensor | None,
        y_c: torch.Tensor,
        context_mask: torch.Tensor,
        centers: torch.Tensor | None,
        lengthscale: float,
        *,
        prior_mean: torch.Tensor | float = 0.0,
        steps: int = 12,
    ) -> torch.Tensor:
        """Batched MAP ability solve with padded context masks."""

        batch = a_c.shape[0]
        if isinstance(prior_mean, torch.Tensor):
            prior = prior_mean.to(device=a_c.device, dtype=a_c.dtype).reshape(batch)
        else:
            prior = a_c.new_full((batch,), float(prior_mean))
        theta = prior
        if a_c.shape[1] == 0:
            return theta
        a_c = a_c.detach()
        b_c = b_c.detach()
        y_c = y_c.to(dtype=a_c.dtype)
        mask = context_mask.to(dtype=a_c.dtype)
        if c_c is not None:
            c_c = c_c.detach()
            if centers is None:
                raise ValueError("centers are required for a GP residual")
        for _ in range(steps):
            centered = theta[:, None]
            if c_c is None:
                value = a_c * (centered - b_c)
                slope = a_c
                curvature = torch.zeros_like(a_c)
            else:
                delta = centered[:, :, None] - centers[None, None, :]
                phi = torch.exp(-delta.square() / (2.0 * lengthscale**2))
                dphi = phi * (-delta / lengthscale**2)
                ddphi = phi * (
                    delta.square() / lengthscale**4 - 1.0 / lengthscale**2
                )
                value = a_c * (centered - b_c) + torch.einsum(
                    "blk,bk->bl", c_c, phi[:, 0, :]
                )
                slope = a_c + torch.einsum("blk,bk->bl", c_c, dphi[:, 0, :])
                curvature = torch.einsum("blk,bk->bl", c_c, ddphi[:, 0, :])
            probability = torch.sigmoid(value)
            gradient = ((probability - y_c) * slope * mask).sum(dim=1) + (
                theta - prior
            ) / 4.0
            hessian = (
                (
                    probability * (1.0 - probability) * slope.square()
                    + (probability - y_c) * curvature
                )
                * mask
            ).sum(dim=1) + 0.25
            theta = (
                theta - (gradient / hessian.clamp_min(1e-6)).clamp(-1.0, 1.0)
            ).clamp(-4.0, 4.0)
        return theta

    def _set_penalty_batch(
        self,
        c_c: torch.Tensor | None,
        c_t: torch.Tensor | None,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> None:
        if not self.use_gp or c_c is None or c_t is None:
            self._gp_penalty = self._gp_penalty.new_zeros(())
            return
        coefficients = torch.cat([c_c, c_t], dim=1)
        mask = torch.cat([context_mask, target_mask], dim=1)
        if coefficients.shape[1] == 0:
            self._gp_penalty = coefficients.sum() * 0.0
            return
        projected = coefficients @ self.kernel_inv
        quadratic = (projected * coefficients).sum(dim=-1)
        denominator = mask.sum().clamp_min(1).to(quadratic.dtype)
        self._gp_penalty = self.gp_lambda * (
            (quadratic * mask.to(quadratic.dtype)).sum() / denominator
        )

    def _target_correction_batch(
        self,
        h_c: torch.Tensor,
        h_t: torch.Tensor,
        y_c: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_anp or h_c.shape[1] == 0 or h_t.shape[1] == 0:
            self._anp_penalty = h_t.new_zeros(())
            return h_t.new_zeros(h_t.shape[:2])
        context = self.context_encoder(torch.cat([h_c, y_c[..., None]], dim=-1))
        for layer in self.context_layers:
            context = layer(context, context_mask)
        target_context = self.cross_attention(
            h_t, h_c, context, context_mask=context_mask
        )
        delta = self.delta_head(torch.cat([h_t, target_context], dim=-1)).squeeze(-1)
        has_context = context_mask.any(dim=1, keepdim=True)
        delta = delta * has_context.to(dtype=delta.dtype)
        valid = target_mask.to(dtype=delta.dtype)
        count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (delta * valid).sum(dim=1, keepdim=True) / count
        delta = (delta - mean) * valid
        self._anp_penalty = self.anp_lambda * (
            delta.square().sum() / valid.sum().clamp_min(1.0)
        )
        return delta

    def forward_batch(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Padded multi-task forward pass.

        Shapes are ``[B, Lc, D]`` and ``[B, Lt, D]`` with boolean masks.  Padded
        positions are ignored by the MAP, GP penalty, ANP mean, and loss.
        """

        h_c = self.encode_item(x_c)
        h_t = self.encode_item(x_t)
        a_c, b_c = self._item_parameters(h_c)
        c_c = self._coefficients(h_c) if self.use_gp else None
        prior_mean = self._subject_prior_batch(
            x_c, x_t, context_mask, target_mask
        )
        theta = self._map_theta_batch(
            a_c,
            b_c,
            c_c,
            y_c,
            context_mask,
            self.centers if self.use_gp else None,
            self.lengthscale,
            prior_mean=0.0 if prior_mean is None else prior_mean,
        )
        if h_t.shape[1] == 0:
            if self.use_gp:
                empty = h_t.new_zeros((h_t.shape[0], 0, self.basis_size))
                self._set_penalty_batch(c_c, empty, context_mask, target_mask)
            else:
                self._set_penalty_batch(None, None, context_mask, target_mask)
            self._anp_penalty = h_t.new_zeros(())
            return h_t.new_zeros(h_t.shape[:2]), target_mask
        a_t, b_t = self._item_parameters(h_t)
        logits = a_t * (theta[:, None] - b_t)
        if self.use_gp:
            c_t = self._coefficients(h_t)
            phi = torch.exp(
                -(theta[:, None] - self.centers[None, :]).square()
                / (2.0 * self.lengthscale**2)
            )
            logits = logits + torch.einsum("blk,bk->bl", c_t, phi)
            self._set_penalty_batch(c_c, c_t, context_mask, target_mask)
        else:
            self._set_penalty_batch(None, None, context_mask, target_mask)
        if self.use_anp:
            logits = logits + self.gamma * self._target_correction_batch(
                h_c, h_t, y_c, context_mask, target_mask
            )
        return logits, target_mask

    def gp_penalty(self) -> torch.Tensor:
        return self._gp_penalty

    def aux_loss(self) -> torch.Tensor:
        return self._anp_penalty

    def _target_correction(
        self,
        h_c: torch.Tensor,
        h_t: torch.Tensor,
        y_c: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_anp or h_c.shape[0] == 0 or h_t.shape[0] == 0:
            self._anp_penalty = h_t.new_zeros(())
            return h_t.new_zeros(h_t.shape[0])
        context = self.context_encoder(torch.cat([h_c, y_c[:, None]], dim=-1))
        for layer in self.context_layers:
            context = layer(context)
        target_context = self.cross_attention(h_t, h_c, context)
        delta = self.delta_head(torch.cat([h_t, target_context], dim=-1)).squeeze(-1)
        delta = delta - delta.mean()
        self._anp_penalty = self.anp_lambda * delta.square().mean()
        return delta

    def forward(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        h_c = self.encode_item(x_c)
        h_t = self.encode_item(x_t)
        a_c, b_c = self._item_parameters(h_c)
        c_c = self._coefficients(h_c) if self.use_gp else None
        prior_mean = self._subject_prior(x_c, x_t)
        theta = self._map_theta(
            a_c,
            b_c,
            c_c,
            y_c,
            self.centers if self.use_gp else None,
            self.lengthscale,
            prior_mean=0.0 if prior_mean is None else prior_mean,
        )
        if h_t.shape[0] == 0:
            if self.use_gp:
                empty = h_t.new_zeros((0, self.basis_size))
                self._set_penalty(c_c, empty)
            else:
                self._set_penalty(None, None)
            self._anp_penalty = h_t.new_zeros(())
            return h_t.new_zeros((0,))
        a_t, b_t = self._item_parameters(h_t)
        logits = a_t * (theta - b_t)
        if self.use_gp:
            c_t = self._coefficients(h_t)
            phi = torch.exp(
                -(theta - self.centers).square() / (2.0 * self.lengthscale**2)
            )
            logits = logits + c_t @ phi
            self._set_penalty(c_c, c_t)
        else:
            self._set_penalty(None, None)
        if self.use_anp:
            logits = logits + self.gamma * self._target_correction(h_c, h_t, y_c)
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_c, y_c, x_t))
