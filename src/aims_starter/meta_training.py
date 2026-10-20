"""Episodic meta-training protocol for IRT-anchored neural predictors.

This module is the consolidation target for the training loops currently
spread across ``faithful_np.py`` (``train_model``, ``train_cnp``,
``train_tnpd``, ``train_amortized_1pl``, ``train_amortized_2pl``). Model
definitions stay in ``faithful_np.py``; only the protocol lives here.

Settled protocol rules:

1. One task is one subject-benchmark pair. Every row in a task comes from the
   same subject, so subject identity is constant within an episode.
2. Context and target sets are disjoint. This follows the regression-NP split,
   not the image-inpainting split. No target label may reach the posterior.
3. Context size is drawn from the official budget schedule, so training
   matches the deployment budgets.
4. Item features must not contain subject identity. Subject metadata may enter
   a context-side encoder and inform the prior over ability; it must not enter
   the item parameter net. ``encode_item_x`` and ``subject_feature_slice`` in
   ``faithful_np.py`` implement runtime masking; explicit feature separation
   is the intended replacement.
5. Ability is an analytic MAP estimate, never a network output. Only item-side
   and correction parameters are learned.

Two model tracks share this protocol:

- Deterministic: item-only ``a(x)``, ``b(x)`` plus a MAP ability anchor.
- Latent: per-item latent route, no MAP anchor. Currently a negative control.

Model contract (duck typed, torch tensors on the model's device)::

    training_loss(x_context, y_context, x_target, y_target) -> scalar tensor
    predict(x_context, y_context, x_target) -> (probabilities, epistemic_variance)

TODO(open):
- Move the subject slice out of ``modal_faithful.py`` into the corpus manifest.
  It is currently hard-coded as ``(389, 773)`` in ten places.
- Add ROC AUC to ``metrics.py``; the score functions do not report it.
- Make label scaling consistent between the latent path and the deterministic
  path (``s_encoder`` applies ``label_scale``; ``r_encoder`` does not).
- Keep the decoder residual path disabled while its input ``r_star`` carries
  subject metadata, or mask it the same way as the item path.
- Add an acquisition policy hook once the predictor is frozen.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import DEFAULT_BUDGETS, brier_alc, brier_score, calibration_ece


@dataclass(frozen=True)
class Episode:
    """One training episode drawn from a single task."""

    x_context: np.ndarray
    y_context: np.ndarray
    x_target: np.ndarray
    y_target: np.ndarray
    budget: int


def sample_episode(
    task: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    disjoint: bool = True,
    min_targets: int = 32,
    max_targets: int | None = None,
) -> Episode:
    """Draw one context/target split from ``task``.

    ``task`` must provide ``x`` (n, d) and ``y`` (n,). With ``disjoint=True``
    the target set excludes every context row. The legacy overlapping split is
    kept only for reproducing old runs and is not paper-faithful.
    """

    x = np.asarray(task["x"])
    y = np.asarray(task["y"], dtype=np.float64)
    n = len(y)
    if n < 2:
        raise ValueError("a task needs at least two rows")
    order = rng.permutation(n)
    budget = min(int(rng.choice(np.asarray(budgets))), n - 1)
    if disjoint and min_targets:
        budget = min(budget, max(0, n - min_targets))
    context_index = order[:budget]
    if disjoint:
        target_index = order[budget:]
    else:
        target_size = min(n, max(min_targets, budget + 32))
        target_index = order[:target_size]
    if max_targets is not None:
        target_index = target_index[:max_targets]
    return Episode(
        x_context=x[context_index],
        y_context=y[context_index],
        x_target=x[target_index],
        y_target=y[target_index],
        budget=budget,
    )


def budget_contexts(
    task: Mapping[str, np.ndarray],
    order: np.ndarray,
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return nested context prefixes from a frozen acquisition order.

    This mirrors the deployment loop: the context at budget 3 contains the
    contexts at budgets 0 and 1.
    """

    x = np.asarray(task["x"])
    y = np.asarray(task["y"], dtype=np.float64)
    return [(budget, x[order[:budget]], y[order[:budget]]) for budget in budgets]


def train_meta(
    model: Any,
    tasks: Sequence[Mapping[str, np.ndarray]],
    *,
    seed: int,
    steps: int,
    episodes_per_step: int = 4,
    lr: float = 3e-4,
    grad_clip: float = 1.0,
    disjoint: bool = True,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    min_targets: int = 32,
    max_targets: int | None = None,
    scheduler: str | None = "cosine",
    log_every: int = 50,
    step_hook: Callable[[int, Any], None] | None = None,
) -> list[dict[str, float]]:
    """Drive one episodic meta-training run and return a loss history.

    The model must implement ``training_loss`` as in the module contract. The
    loop is intentionally thin: sampling here, model math in the model.
    """

    import torch

    if not tasks:
        raise ValueError("at least one task is required")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = None
    if scheduler == "cosine":
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    history: list[dict[str, float]] = []
    for step in range(steps):
        losses = []
        for _ in range(episodes_per_step):
            task = tasks[int(rng.integers(len(tasks)))]
            episode = sample_episode(
                task,
                rng,
                budgets=budgets,
                disjoint=disjoint,
                min_targets=min_targets,
                max_targets=max_targets,
            )
            losses.append(model.training_loss(*_episode_tensors(episode, device, torch)))
        loss = torch.stack([value if value.ndim == 0 else value.mean() for value in losses]).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if schedule is not None:
            schedule.step()
        if step_hook is not None:
            step_hook(step, model)
        if step % log_every == 0 or step == steps - 1:
            history.append({"step": float(step), "loss": float(loss.detach())})
    return history


def evaluate_task(
    model: Any,
    task: Mapping[str, np.ndarray],
    *,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    order: np.ndarray | None = None,
    seed: int = 0,
    predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    | None = None,
) -> dict[str, Any]:
    """Score one held-out task at every budget and aggregate with Brier ALC.

    ``order`` is the frozen acquisition order reused across seeds and folds.
    When omitted, a deterministic permutation of the given ``seed`` is used.
    ``predict_fn`` decouples the protocol from torch for tests and analysis.
    """

    x = np.asarray(task["x"])
    y = np.asarray(task["y"], dtype=np.float64)
    if order is None:
        order = np.random.default_rng(seed).permutation(len(y))
    predict = predict_fn if predict_fn is not None else _torch_predict(model)
    rows: dict[str, dict[str, float]] = {}
    scores: list[float] = []
    for budget, x_context, y_context in budget_contexts(task, order, budgets=budgets):
        probabilities, _ = predict(x_context, y_context, x)
        brier = brier_score(y.tolist(), probabilities.tolist())
        scores.append(brier)
        rows[str(budget)] = {
            "brier": brier,
            "ece": calibration_ece(y.tolist(), probabilities.tolist()),
            "mean_probability": float(np.mean(probabilities)),
        }
    return {"by_budget": rows, "brier_alc": brier_alc(scores)}


def _episode_tensors(episode: Episode, device: Any, torch: Any) -> tuple[Any, ...]:
    def tensor(values: np.ndarray) -> Any:
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    return (
        tensor(episode.x_context),
        tensor(episode.y_context),
        tensor(episode.x_target),
        tensor(episode.y_target),
    )


def _torch_predict(model: Any) -> Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[Any, Any]]:
    def predict(
        x_context: np.ndarray, y_context: np.ndarray, x_target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        device = next(model.parameters()).device
        with torch.no_grad():
            probabilities, variance = model.predict(
                torch.as_tensor(x_context, dtype=torch.float32, device=device),
                torch.as_tensor(y_context, dtype=torch.float32, device=device),
                torch.as_tensor(x_target, dtype=torch.float32, device=device),
            )
        return (
            probabilities.detach().cpu().numpy(),
            variance.detach().cpu().numpy(),
        )

    return predict
