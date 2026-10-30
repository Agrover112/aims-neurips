"""Pure helpers for fixed-panel, streaming online evaluation.

The panel is built once from filtered benchmark-holdout tasks.  A predictor
only receives acquired labels; evaluation labels are touched only after each
budget's predictions have been emitted.  Keeping panel construction outside
the model runner makes all policies and checkpoints directly paired.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np


BUDGETS = (0, 1, 3, 7, 15, 31)
ALC_WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)


def stable_rank(*parts: object) -> int:
    """Stable, platform-independent order key for a frozen manifest."""
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def panel_id(*, split_seed: int, min_item_responders: int,
             min_items_per_subject: int, max_source_items: int) -> str:
    return (
        "benchmark-holdout-online-v1"
        f"-s{split_seed}-n{min_item_responders}"
        f"-m{min_items_per_subject}-cap{max_source_items}"
    )


def _task_key(subject: object, benchmark: object) -> str:
    return f"{subject}::{benchmark}"


def choose_balanced_tasks(responses, *, role: str, max_source_items: int,
                          min_source_items: int, split_seed: int):
    """Select complete tasks under an item cap without inspecting outcomes.

    A round-robin over benchmark IDs mimics the competition's stated preference
    for underrepresented benchmarks.  Ties are resolved from a stable hash.
    The result is a deterministic *panel*, not a claim about CodaBench's
    private sampler.
    """
    import pandas as pd

    if role not in {"validation", "test"}:
        raise ValueError("role must be validation or test")
    d = responses[responses["benchmark_split"] == role].copy()
    if d.empty:
        raise RuntimeError(f"no {role} responses in filtered benchmark split")
    tasks = (d.groupby(["subject_uid", "benchmark_id"], as_index=False)["item_uid"]
             .nunique().rename(columns={"item_uid": "n_source_items"}))
    tasks = tasks[tasks["n_source_items"] >= min_source_items].copy()
    if tasks.empty:
        raise RuntimeError(f"no {role} tasks have >= {min_source_items} source items")
    tasks["task_id"] = [
        _task_key(subject, benchmark)
        for subject, benchmark in zip(tasks["subject_uid"], tasks["benchmark_id"], strict=True)
    ]
    tasks["rank"] = [
        stable_rank("task", split_seed, role, task_id)
        for task_id in tasks["task_id"]
    ]

    by_benchmark = {
        str(benchmark): group.sort_values("rank").to_dict("records")
        for benchmark, group in tasks.groupby("benchmark_id", sort=True)
    }
    cursors = defaultdict(int)
    selected: list[dict] = []
    selected_items = 0
    # Continue until no complete eligible task can fit. Every selection uses a
    # current candidate only; labels and item text never affect this ordering.
    while True:
        available = []
        for benchmark, candidates in by_benchmark.items():
            index = cursors[benchmark]
            while index < len(candidates) and candidates[index]["n_source_items"] + selected_items > max_source_items:
                # This task cannot fit now or later, because the used item
                # count only grows. Mark it permanently skipped.
                index += 1
            cursors[benchmark] = index
            if index < len(candidates):
                available.append((sum(row["benchmark_id"] == benchmark for row in selected),
                                  candidates[index]["rank"], benchmark))
        if not available:
            break
        _, _, chosen_benchmark = min(available)
        chosen = by_benchmark[chosen_benchmark][cursors[chosen_benchmark]]
        selected.append(chosen)
        selected_items += int(chosen["n_source_items"])
        cursors[chosen_benchmark] += 1

    if not selected:
        raise RuntimeError(f"item cap {max_source_items} cannot fit any eligible {role} task")
    return pd.DataFrame(selected).drop(columns="rank"), selected_items


def make_panel(responses, *, split_seed: int, min_source_items: int,
               max_source_items: int, order_seeds: Sequence[int]):
    """Construct disjoint acquisition/evaluation pools and shared streams."""
    import pandas as pd

    if responses.duplicated(["subject_uid", "benchmark_id", "item_uid"]).any():
        raise ValueError("panel input must have one canonical response per task/item")
    task_frames, panel_rows, order_rows = [], [], []
    selected_by_role = {}
    for role in ("validation", "test"):
        selected, total_source_items = choose_balanced_tasks(
            responses, role=role, max_source_items=max_source_items,
            min_source_items=min_source_items, split_seed=split_seed,
        )
        selected_by_role[role] = {
            "tasks": int(len(selected)), "source_items": int(total_source_items),
            "benchmarks": int(selected["benchmark_id"].nunique()),
        }
        chosen = responses.merge(
            selected[["subject_uid", "benchmark_id", "task_id", "n_source_items"]],
            on=["subject_uid", "benchmark_id"], how="inner", validate="many_to_one",
        ).copy()
        for task_id, task in chosen.groupby("task_id", sort=True):
            task = task.copy()
            task["pool_rank"] = [
                stable_rank("pool", split_seed, task_id, item_uid)
                for item_uid in task["item_uid"].astype(str)
            ]
            task = task.sort_values(["pool_rank", "item_uid"])
            n_acquisition = (len(task) + 1) // 2  # CodaBench gives an odd extra item to acquisition.
            task["pool"] = "evaluation"
            task.iloc[:n_acquisition, task.columns.get_loc("pool")] = "acquisition"
            task_frames.append(task.drop(columns="pool_rank"))
            panel_rows.append({
                "task_id": task_id,
                "benchmark_split": role,
                "subject_uid": str(task["subject_uid"].iloc[0]),
                "benchmark_id": str(task["benchmark_id"].iloc[0]),
                "n_source_items": int(len(task)),
                "n_acquisition_items": int(n_acquisition),
                "n_evaluation_items": int(len(task) - n_acquisition),
            })
            acquisition = task[task["pool"] == "acquisition"]
            for order_seed in order_seeds:
                ordered = acquisition.assign(_rank=[
                    stable_rank("stream", split_seed, order_seed, task_id, item_uid)
                    for item_uid in acquisition["item_uid"].astype(str)
                ]).sort_values(["_rank", "item_uid"])
                for position, item_uid in enumerate(ordered["item_uid"].astype(str)):
                    order_rows.append({
                        "task_id": task_id,
                        "order_seed": int(order_seed),
                        "item_uid": item_uid,
                        "stream_position": int(position),
                    })
    items = pd.concat(task_frames, ignore_index=True)
    tasks = pd.DataFrame(panel_rows).sort_values("task_id").reset_index(drop=True)
    orders = pd.DataFrame(order_rows).sort_values(
        ["task_id", "order_seed", "stream_position"]
    ).reset_index(drop=True)
    if set(items["pool"].unique()) != {"acquisition", "evaluation"}:
        raise RuntimeError("panel did not create both acquisition and evaluation pools")
    return tasks, items, orders, selected_by_role


def brier_score(labels: Iterable[float], probabilities: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=np.float64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Brier score requires equally sized, nonempty labels and probabilities")
    return float(np.mean((np.clip(p, 0.0, 1.0) - y) ** 2))


def expected_calibration_error(labels: Iterable[float], probabilities: Iterable[float],
                               n_bins: int = 10) -> float:
    y = np.asarray(list(labels), dtype=np.float64)
    p = np.clip(np.asarray(list(probabilities), dtype=np.float64), 0.0, 1.0)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("ECE requires equally sized, nonempty labels and probabilities")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for bin_id in range(n_bins):
        mask = bins == bin_id
        if mask.any():
            total += mask.mean() * abs(float(y[mask].mean() - p[mask].mean()))
    return float(total)


def brier_alc(brier_by_budget: Sequence[float]) -> float:
    if len(brier_by_budget) != len(ALC_WEIGHTS):
        raise ValueError(f"expected {len(ALC_WEIGHTS)} budget scores")
    return float(np.dot(np.asarray(brier_by_budget), np.asarray(ALC_WEIGHTS)))


def beta_binomial_probability(labels: Sequence[int]) -> float:
    """Jeffreys-prior fallback used by the current strongest submission."""
    return float((sum(labels) + 0.5) / (len(labels) + 1.0))


def accept_random_prefix(*, labels_acquired: int, max_labels: int) -> bool:
    """Accept the current item in an already-frozen uniformly random stream."""
    return labels_acquired < max_labels


def accept_uncertainty(*, probability: float, labels_acquired: int,
                       max_labels: int, items_remaining: int,
                       threshold: float) -> bool:
    """Streaming uncertainty policy with the mandatory reserve rule."""
    labels_remaining = max_labels - labels_acquired
    if labels_remaining <= 0:
        return False
    if items_remaining <= labels_remaining:
        return True
    return abs(float(probability) - 0.5) <= threshold


def train_probe_sets(*, processed_dir=None, prefiltered_path=None,
                     split_seed: int, min_item_responders: int,
                     min_items_per_subject: int, count: int):
    """Draw disjoint reference and calibration probe item ids from TRAIN only.

    The scored acquisition policies need items whose embeddings the predictor
    can see but which are never evaluated. Training benchmarks are the only
    safe source: a reference set built from the task's own evaluation pool
    would let the policy steer toward the scored items, and probes drawn from
    it would leak the scored distribution into the label spend.

    Two sources are accepted because the two workspaces differ. A workspace
    holding the full ``datasets/full/processed`` tree can re-run the filter; a
    workspace holding only the exported subset cannot, and instead has the
    filter's own output with its ``benchmark_split`` column. Preferring the
    export keeps the scored policies runnable in both places -- without this
    they raise ``FileNotFoundError`` on the export-only workspace, which is
    where the cheap CPU budget lives.

    Returns ``(reference_uids, probe_uids)``, disjoint and deterministic in
    ``split_seed``.
    """
    import numpy as np
    import pandas as pd

    if prefiltered_path:
        responses = pd.read_parquet(prefiltered_path,
                                    columns=["item_uid", "benchmark_split"])
    elif processed_dir is not None:
        import benchmark_holdout_filters as filters

        responses, _ = filters.load_benchmark_holdout_filtered(
            processed_dir, min_item_responders=min_item_responders,
            min_items_per_subject=min_items_per_subject, split_seed=split_seed,
        )
    else:
        raise ValueError("one of prefiltered_path or processed_dir is required")

    train = np.sort(
        responses.loc[responses["benchmark_split"] == "train", "item_uid"]
        .astype(str).unique()
    )
    if len(train) < 2 * count:
        raise RuntimeError(
            f"only {len(train)} train items available for {count} reference "
            f"and {count} probe items"
        )
    draw = np.random.default_rng(split_seed).choice(len(train), size=2 * count,
                                                    replace=False)
    reference = [str(train[index]) for index in draw[:count]]
    probes = [str(train[index]) for index in draw[count:]]
    return reference, probes


def bernoulli_variance(probability: float) -> float:
    """Predictive Bernoulli variance ``p(1-p)``: the classic uncertainty score.

    This is a *score*, not the threshold rule in ``accept_uncertainty``. The
    threshold rule buys anything inside a fixed band of 0.5 and so its label
    spend depends on the predictor's calibration; a score paced by
    ``accept_by_rank`` spends the same budget whatever the calibration, which
    is what makes the two comparable. Ported unchanged from
    ``organized/training/anp_acquisition.py`` at commit db7bec01.
    """
    probability = float(probability)
    return probability * (1.0 - probability)


def expected_brier_reduction(predict, context, candidate, reference_baseline,
                             candidate_probability):
    """Expected squared change in the reference predictions from one purchase.

    ``sum_y P(y) * mean_j (p_j(context + (candidate, y)) - p_j(context))^2``
    over a fixed reference set. A label is valuable when learning it would move
    the predictor's beliefs about *other* items, which is the quantity the ALC
    actually rewards; Bernoulli variance instead rewards being unsure about the
    candidate itself, which may be irreducible noise.

    ``predict(context)`` must return the reference-set probabilities. Both
    outcomes are evaluated, so this costs two extra predictor calls per
    candidate -- the expensive policy of the set. No evaluation item and no
    future candidate is read, and the candidate's own label is never revealed.
    """
    import numpy as np

    baseline = np.asarray(reference_baseline, dtype=float)
    probability = float(candidate_probability)
    changes = []
    for outcome in (0, 1):
        updated = np.asarray(predict(list(context) + [(candidate, outcome)]), dtype=float)
        changes.append(float(np.mean((updated - baseline) ** 2)))
    return (1.0 - probability) * changes[0] + probability * changes[1]


def accept_by_rank(*, score_value: float, calibration, labels_acquired: int,
                   max_labels: int, items_remaining: int, uniform: float) -> bool:
    """Pace a score against train-only probes, keeping the reserve rule.

    A raw score cannot be thresholded without knowing its scale, and the scale
    moves with the context. Comparing against probe scores recomputed in the
    same context turns it into an upper-tail rank, and buying when that rank is
    within the remaining label fraction spends the budget evenly over the
    stream instead of exhausting it on the first few items.

    Random tie-breaking matters: a collapsed latent or an item-independent
    predictor gives every candidate the same score, and without the jitter the
    rank would be constant and the policy would buy everything at once.
    Ported from ``anp_acquisition.accept`` at commit db7bec01.
    """
    import numpy as np

    labels_remaining = max_labels - labels_acquired
    if labels_remaining <= 0:
        return False
    if items_remaining <= labels_remaining:
        return True  # reserve rule: too few candidates left to be choosy
    if calibration is None or len(calibration) == 0:
        return float(uniform) < labels_remaining / items_remaining
    probes = np.asarray(calibration, dtype=float)
    above = int(np.count_nonzero(probes > score_value))
    ties = int(np.count_nonzero(probes == score_value))
    tail = (above + float(uniform) * (1 + ties)) / (len(probes) + 1)
    return tail <= labels_remaining / items_remaining


def accept_confidence_extremes(*, probability: float, labels_acquired: int,
                               max_labels: int, items_remaining: int,
                               threshold: float) -> bool:
    """Streaming control that selects only high-confidence current items.

    This is deliberately a contrast to entropy acquisition, not an information
    claim. The identical reserve rule makes its label budget comparable.
    """
    labels_remaining = max_labels - labels_acquired
    if labels_remaining <= 0:
        return False
    if items_remaining <= labels_remaining:
        return True
    return abs(float(probability) - 0.5) >= threshold
