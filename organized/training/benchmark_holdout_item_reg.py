"""Item-side regularisation and cold-target selection for NP training.

Benchmark holdout changes *what is measured*; it does not by itself change
what the model is free to learn.  The concern this module addresses is a
difficulty **lookup table** keyed by the item embedding:
``validation/fixed_bank/unseen_target_brier_score`` calls a target unseen
because the *(subject, item) cell* is unseen, not the item, so under a random
subject split that metric rewards the lookup.  The measured cost was a
fixed-bank 0.1272 against a leaderboard 0.1902 on the same checkpoint.

**Which protocol you are in decides whether any of this is needed.**

``benchmark_holdout=True``
    Every validation item is already cold: its whole benchmark is absent from
    training, which is why the trainer leaves ``held_i_arr`` empty
    (``codex_es_fixed_modal_cnp_train.py:795``).  The existing fixed bank is
    therefore *already* a cold-target bank and needs no replacement.
    ``cold_start_validation_bank`` still accepts this case and says so through
    ``cold_start_eligibility``, so a caller needs no protocol branch of its
    own, but it buys nothing here.

``benchmark_holdout=False`` (legacy random subject split)
    Items are shared between train and validation, 20% are reserved in
    ``held_i_arr``, and ``val_tasks`` has been through ``_warm``, which strips
    exactly those items (``:817``).  A cold-target bank is a genuine fix here,
    and it must be built from the **unfiltered** tasks -- warm-filtered tasks
    have no cold items at all, so the bank would be empty.

The three controls, cheapest first:

1. ``cold_start_validation_bank`` -- select checkpoints on held-out *items*
   rather than unseen cells.  Adds nothing to the model.
2. ``apply_item_noise`` -- perturb item embeddings during training only, so an
   exact embedding has no exact difficulty.
3. ``random_item_bottleneck`` -- fewer item dimensions, so less capacity for
   per-item storage.  Changes the inference contract; read its docstring.

Options 2 and 3 are capacity and smoothness knobs.  Neither *proves* the
lookup is gone; only a cold-target metric can show that.  Under
``modal_ridge_brier_check`` a 0.899 cross-arm cosine still cost +77% Brier, so
a geometric property of an embedding is not evidence about downstream score.
Measure, do not infer.
"""

from __future__ import annotations


DEFAULT_ITEM_NOISE_STD = 0.0
DEFAULT_ITEM_DROPOUT = 0.0
DEFAULT_ITEM_BOTTLENECK = 0          # 0 disables the projection
DEFAULT_MIN_COLD_TARGETS = 8
DEFAULT_BOTTLENECK_SEED = 20261018

#: ``cfg`` keys that must also enter ``resume_identity``.  Every one of these
#: changes the training distribution or the input geometry, so a checkpoint
#: must never resume across a change in any of them.
REGULARISATION_IDENTITY_KEYS = (
    "item_noise_std",
    "item_dropout",
    "item_bottleneck",
    "item_bottleneck_seed",
    "selection_bank",
)

_REGIME_ALL_COLD = "benchmark_holdout_all_items_cold"
_REGIME_HELD_ITEMS = "random_split_held_items"


def _validate(noise_std: float, dropout: float) -> None:
    if noise_std < 0.0:
        raise ValueError(f"item_noise_std must be >= 0, got {noise_std}")
    if not 0.0 <= dropout < 1.0:
        # p == 1 drops every dimension and the 1/(1-p) rescale divides by zero.
        raise ValueError(f"item_dropout must be in [0, 1), got {dropout}")


def _eligible(val_tasks, held_item_ids, max_context, min_cold_targets):
    """Classify the protocol and return the per-task context/target pools.

    Returns ``(regime, [(task_index, context_pool, target_pool), ...])`` with
    positional indices into each task's own ``items`` array.  ``context_pool``
    is ``None`` in the all-cold regime, where the carve is a single
    permutation rather than two disjoint pools.
    """
    import numpy as np

    held = (np.empty(0, dtype=np.int64) if held_item_ids is None
            else np.asarray(held_item_ids, dtype=np.int64))

    if held.size == 0:
        # Benchmark holdout: nothing is warm, so any carve has cold targets.
        # Require enough room that the largest ladder draw still leaves a
        # scoreable target set, rather than drawing k and hoping.
        regime, pools = _REGIME_ALL_COLD, []
        for index, task in enumerate(val_tasks):
            if len(task["items"]) >= max_context + min_cold_targets:
                pools.append((index, None, None))
        return regime, pools

    regime, pools = _REGIME_HELD_ITEMS, []
    for index, task in enumerate(val_tasks):
        is_cold = np.isin(task["items"], held)
        warm = np.flatnonzero(~is_cold)
        cold = np.flatnonzero(is_cold)
        if warm.size >= 1 and cold.size >= min_cold_targets:
            pools.append((index, warm, cold))
    return regime, pools


def cold_start_eligibility(val_tasks, held_item_ids=None, *, max_context,
                           min_cold_targets=DEFAULT_MIN_COLD_TARGETS):
    """Loggable counts for a cold-target bank, without building one.

    Call this before ``cold_start_validation_bank`` and log the result.  A bank
    that silently shrank to four episodes selects checkpoints on noise, and the
    only way to notice is to record how many subjects qualified.
    """
    regime, pools = _eligible(val_tasks, held_item_ids, max_context, min_cold_targets)
    stats = {
        "regime": regime,
        "n_validation_tasks": len(val_tasks),
        "n_eligible_tasks": len(pools),
        "min_cold_targets": min_cold_targets,
        "cold_target_selection_available": bool(pools),
    }
    if regime == _REGIME_HELD_ITEMS and pools:
        stats["cold_items_per_task_min"] = min(int(cold.size) for _, _, cold in pools)
        stats["warm_items_per_task_min"] = min(int(warm.size) for _, warm, _ in pools)
    return stats


def cold_start_validation_bank(val_tasks, rng, *, budget_sampler, max_context,
                               max_targets, n_episodes, held_item_ids=None,
                               min_cold_targets=DEFAULT_MIN_COLD_TARGETS):
    """Freeze held-out-subject episodes whose targets the model never trained on.

    Returns the same ``(task_index, context_index, target_index)`` triples as
    the trainer's ``_episode_indices``, with positional indices into each
    task's own ``items`` array, so ``_fixed_validation_metrics`` consumes this
    bank unchanged.

    ``held_item_ids`` empty or omitted selects the benchmark-holdout regime,
    where every item is already cold and the carve is one permutation cut at a
    ladder-drawn ``k``.  A non-empty ``held_item_ids`` selects the random-split
    regime: context from the subject's warm items, targets from its cold
    items, disjoint by construction rather than by a cut point.  In that
    regime ``val_tasks`` must be the **unfiltered** tasks; passing the
    trainer's ``val_tasks`` raises, because ``_warm`` has already removed every
    cold item.
    """
    import numpy as np

    regime, pools = _eligible(val_tasks, held_item_ids, max_context, min_cold_targets)
    if not pools:
        if regime == _REGIME_HELD_ITEMS:
            raise RuntimeError(
                "no validation subject has >=1 warm and "
                f">={min_cold_targets} cold items. The usual cause is passing the "
                "trainer's warm-filtered `val_tasks`, which `_warm` has already "
                "stripped of every held item; pass the unfiltered tasks for the "
                "validation subjects instead."
            )
        raise RuntimeError(
            f"no validation subject has >= max_context + min_cold_targets "
            f"({max_context} + {min_cold_targets}) items; lower min_cold_targets "
            "or max_context"
        )

    chosen = rng.choice(len(pools), size=min(n_episodes, len(pools)), replace=False)
    bank = []
    for pick in chosen:
        index, warm, cold = pools[int(pick)]
        if regime == _REGIME_ALL_COLD:
            n = len(val_tasks[index]["items"])
            k = budget_sampler(rng, max_context, n)
            permutation = rng.permutation(n)
            context, target = permutation[:k], permutation[k:k + max_targets]
        else:
            # The trainer's sampler reserves one observation for a target,
            # since there a single permutation is cut into context and target.
            # Here the target pool is already disjoint, so the whole warm set
            # is available for context and ``+ 1`` restores that point.
            k = budget_sampler(rng, max_context, warm.size + 1)
            context = rng.permutation(warm)[:k]
            target = rng.permutation(cold)[:max_targets]
        bank.append((index, context, target))
    return bank


def bank_coverage(bank, val_tasks, held_item_ids=None):
    """Loggable description of a realised bank: sizes, and how cold it is.

    ``cold_target_fraction`` is the point of the exercise, so it is reported
    rather than assumed.  It is 1.0 by construction under benchmark holdout
    and should also be 1.0 under the random split; anything less means warm
    items leaked into the scored set.
    """
    import numpy as np

    held = (np.empty(0, dtype=np.int64) if held_item_ids is None
            else np.asarray(held_item_ids, dtype=np.int64))
    context_sizes, target_sizes, scored, cold_hits, total = [], [], set(), 0, 0
    for task_index, context_index, target_index in bank:
        items = val_tasks[task_index]["items"]
        context_sizes.append(int(len(context_index)))
        target_sizes.append(int(len(target_index)))
        target_items = items[target_index]
        scored.update(int(i) for i in target_items)
        total += int(target_items.size)
        cold_hits += (int(target_items.size) if held.size == 0
                      else int(np.isin(target_items, held).sum()))
    return {
        "n_episodes": len(bank),
        "context_size_counts": {
            int(k): int(v) for k, v in zip(*np.unique(context_sizes, return_counts=True))
        } if context_sizes else {},
        "targets_per_episode_min": min(target_sizes) if target_sizes else 0,
        "targets_per_episode_mean": (round(float(np.mean(target_sizes)), 2)
                                     if target_sizes else 0.0),
        "distinct_items_scored": len(scored),
        "cold_target_fraction": round(cold_hits / total, 6) if total else 0.0,
    }


def apply_item_noise(x, *, noise_std=DEFAULT_ITEM_NOISE_STD,
                     dropout=DEFAULT_ITEM_DROPOUT, generator=None):
    """Perturb item embeddings, for training batches only.

    Never call this on a validation or evaluation batch: the frozen checkpoint
    is scored on exact store vectors, so noise at eval time measures a
    different model than the one being shipped.

    The result is renormalised because the store is L2-normalised and the
    model was fit to unit-norm inputs; perturbing without renormalising would
    change input scale as well as direction, confounding the regularisation
    with a gain change.  A row that dropout empties would normalise to the
    zero vector rather than a unit vector, so those rows fall back to the
    original embedding -- astronomically rare, but a zero item vector is a
    silently different input, not a noisy one.

    ``generator`` defaults to the global torch RNG, which the trainer seeds and
    saves in the resume checkpoint, so resumed runs stay reproducible.  A
    supplied generator must live on the same device as ``x``.
    """
    import torch

    _validate(noise_std, dropout)
    if noise_std == 0.0 and dropout == 0.0:
        return x

    out = x
    if dropout > 0.0:
        keep = torch.rand(out.shape, device=out.device, dtype=out.dtype,
                          generator=generator) >= dropout
        out = out * keep / (1.0 - dropout)
    if noise_std > 0.0:
        out = out + noise_std * torch.randn(
            out.shape, device=out.device, dtype=out.dtype, generator=generator
        )

    norms = out.norm(dim=-1, keepdim=True)
    degenerate = norms <= 1e-6
    if bool(degenerate.any()):
        out = torch.where(degenerate, x, out)
        norms = out.norm(dim=-1, keepdim=True)
    return out / norms.clamp_min(1e-12)


def random_item_bottleneck(emb, out_dim=DEFAULT_ITEM_BOTTLENECK, *,
                           seed=DEFAULT_BOTTLENECK_SEED):
    """Project the item embedding table down to ``out_dim``, renormalised.

    A fixed seeded Gaussian projection, so it is a data-side change rather
    than a model change and is reproducible from ``seed`` alone.  The trainer
    builds its model from ``emb.shape[1]``
    (``codex_es_fixed_modal_cnp_train.py:849``), so training needs no further
    flag once the table is projected.

    **This alters the inference contract, and both ways of getting it wrong
    are real.**  ``predictor.py`` builds the model with ``EMBEDDING_DIM`` taken
    from ``metadata.json`` and loads with ``strict=True``:

    - metadata left at 1024 -> ``item_to_key`` is ``Linear(64, width)`` in the
      checkpoint and ``Linear(1024, width)`` in the fresh model, so
      ``load_state_dict`` raises a size mismatch.  Loud, and therefore safe.
    - metadata set to 64 with no projection step -> the model builds correctly,
      but the predictor's ``vectors[:, :EMBEDDING_DIM]`` takes the first 64
      *Matryoshka* dimensions instead of applying this projection.  It loads
      cleanly and scores nonsense: the same silent failure class as swapping
      embedding arms under a frozen checkpoint.

    So a projected checkpoint must ship ``bottleneck_metadata`` and the
    predictor must apply the projection after encoding, before the MRL slice
    is bypassed.  ``bottleneck_metadata`` exists to make that record mandatory.

    Fewer dimensions is a capacity reduction, not a guarantee: 64 dimensions
    still separate 1.36M items comfortably.  Judge it on a cold-target metric,
    never on retained cosine.
    """
    import numpy as np

    if not out_dim:
        return emb
    if out_dim < 0:
        raise ValueError(f"item_bottleneck must be >= 0, got {out_dim}")
    if out_dim >= emb.shape[1]:
        return emb

    generator = np.random.default_rng(seed)
    projection = generator.normal(
        0.0, 1.0 / np.sqrt(out_dim), size=(emb.shape[1], out_dim)
    ).astype(np.float32)
    projected = np.asarray(emb, dtype=np.float32) @ projection
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    return projected / np.clip(norms, 1e-12, None)


def bottleneck_metadata(out_dim, source_dim, *, seed=DEFAULT_BOTTLENECK_SEED):
    """The embedding-contract record a projected checkpoint must ship.

    Merge into the submission's ``metadata.json`` embedding block.  ``method``
    is explicit so a reader cannot mistake the projection for the Matryoshka
    slice that ``output_dimension`` otherwise implies.
    """
    if not out_dim or out_dim >= source_dim:
        return {}
    return {
        "output_dimension": int(out_dim),
        "item_bottleneck": int(out_dim),
        "item_bottleneck_seed": int(seed),
        "item_bottleneck_source_dimension": int(source_dim),
        "item_bottleneck_method": "seeded_gaussian_projection_then_l2_normalise",
    }


def regularisation_config(*, item_noise_std=DEFAULT_ITEM_NOISE_STD,
                          item_dropout=DEFAULT_ITEM_DROPOUT,
                          item_bottleneck=DEFAULT_ITEM_BOTTLENECK,
                          bottleneck_seed=DEFAULT_BOTTLENECK_SEED,
                          cold_target_selection=False):
    """The ``cfg`` entries these controls must record, validated.

    Merge into the trainer's ``cfg`` and add ``REGULARISATION_IDENTITY_KEYS``
    to ``resume_identity``.  Every key here changes the training distribution
    or the input geometry, so resuming across a change in one would train a
    different model under the same checkpoint tag.
    """
    _validate(item_noise_std, item_dropout)
    if item_bottleneck < 0:
        raise ValueError(f"item_bottleneck must be >= 0, got {item_bottleneck}")
    return {
        "item_noise_std": float(item_noise_std),
        "item_dropout": float(item_dropout),
        "item_bottleneck": int(item_bottleneck),
        "item_bottleneck_seed": int(bottleneck_seed),
        "selection_bank": ("cold_target_items" if cold_target_selection
                           else "unseen_cells_warm_items"),
    }
