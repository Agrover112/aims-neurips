"""Paired, streaming offline replay for filtered benchmark-holdout checkpoints.

Build a panel once (CPU only):

  modal run --profile aims-test modal_benchmark_holdout_replay.py::main \
    --action build-panel

Evaluate on validation benchmarks while developing a predictor or policy:

  modal run --profile aims-test modal_benchmark_holdout_replay.py::main \
    --action replay --checkpoint NAME.pt --role validation --policy random_prefix

``--role test`` is deliberately explicit.  All methods receive the identical
frozen tasks, acquisition/evaluation partitions, and candidate streams.  The
runner never uses a future candidate while deciding whether to acquire the
current candidate, and it never exposes evaluation labels to the predictor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal


APP_NAME = "aims-neurips-benchmark-holdout-online-replay"
VOLUME_NAME = "aims-neurips-data"
DATASETS = Path("/data/datasets/full")
PROCESSED = DATASETS / "processed"
CHECKPOINTS = Path("/data/results/cnp/checkpoints")
REPLAY_ROOT = Path("/data/results/online-replay")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24",
).pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")

_HERE = Path(__file__).resolve().parent
if str(_HERE) == "/root":
    # Modal uploads this entrypoint and its support files flat into /root.
    _FILTERS = Path("/root/benchmark_holdout_filters.py")
    _ANP = Path("/root/anp_crossattn_blocks.py")
    _CNP = Path("/root/cnp_blocks.py")
    _HELPERS = Path("/root/benchmark_holdout_replay.py")
else:
    _TRAINING = _HERE.parents[1] / "training"
    _ORIGINAL_ANP128 = _HERE.parents[1] / "submission" / "assets" / "anp-sa1-128-qwen3-8b-seed20261018.pt"
    _FILTERS = _TRAINING / "benchmark_holdout_filters.py"
    _ANP = _TRAINING / "anp_crossattn_blocks.py"
    _CNP = _TRAINING / "cnp_blocks.py"
    _HELPERS = _HERE / "benchmark_holdout_replay.py"
    for local, remote in (
        (_FILTERS, "/root/benchmark_holdout_filters.py"),
        (_ANP, "/root/anp_crossattn_blocks.py"),
        (_CNP, "/root/cnp_blocks.py"),
        (_HELPERS, "/root/benchmark_holdout_replay.py"),
    ):
        if local.exists():
            image = image.add_local_file(local, remote)
    if _ORIGINAL_ANP128.exists():
        image = image.add_local_file(_ORIGINAL_ANP128, "/root/anp-sa1-128-qwen3-8b-seed20261018.pt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet(frame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_panel(panel: Path, manifest: dict, *, expected_order_seed_count: int | None = None) -> None:
    """Refuse a reused panel whose frozen data no longer matches its manifest."""
    if manifest.get("protocol_version") != "benchmark-holdout-online-replay-v1":
        raise ValueError(f"unsupported panel protocol: {manifest.get('protocol_version')!r}")
    for filename, expected_hash in manifest.get("files", {}).items():
        path = panel / filename
        if not path.is_file():
            raise FileNotFoundError(f"panel manifest lists missing file: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"panel file hash differs from frozen manifest: {path}")
    if expected_order_seed_count is not None:
        expected = list(range(expected_order_seed_count))
        actual = manifest.get("streaming", {}).get("order_seeds")
        if actual != expected:
            raise ValueError(
                "existing panel has different order seeds; choose a new panel configuration rather than "
                f"silently changing the paired protocol (existing={actual}, requested={expected})"
            )

@app.function(image=image, cpu=2, memory=16_384, volumes={"/data": volume},
              timeout=20 * 60)
def export_holdout_split_distribution(split_seed: int = 20261018,
                                      min_item_responders: int = 15,
                                      min_items_per_subject: int = 80) -> dict:
    """Summarise the *full* filtered benchmark split for a local PDF figure.

    This deliberately exports aggregate statistics only: it does not expose
    response rows or test labels to a model.  The BGE centroid comparison is a
    descriptive covariate-shift diagnostic, not a training input.
    """
    import numpy as np
    import pandas as pd
    import benchmark_holdout_filters as filters

    responses, audit = filters.load_benchmark_holdout_filtered(
        PROCESSED,
        min_item_responders=min_item_responders,
        min_items_per_subject=min_items_per_subject,
        split_seed=split_seed,
    )
    bins_probability = np.linspace(0.0, 1.0, 21)
    bins_log_support = np.linspace(0.0, 5.0, 21)
    summary: dict[str, object] = {
        "description": "full filtered benchmark-holdout data; before panel sampling",
        "parameters": audit["parameters"],
        "benchmark_split_map": audit["benchmark_split_map"],
        "roles": {},
    }
    centroids: dict[str, np.ndarray] = {}
    store = DATASETS / "allbench-embed-bge_m3_long-v2"
    embedded_ids = pd.read_parquet(store / "items.parquet", columns=["item_uid"])["item_uid"].astype(str)
    item_positions = pd.Series(np.arange(len(embedded_ids), dtype=np.int64), index=embedded_ids)
    embedded = np.load(store / "item_embeddings.npy", mmap_mode="r")
    for role in ("train", "validation", "test"):
        frame = responses[responses["benchmark_split"] == role]
        item = frame.groupby(["benchmark_id", "item_uid"], as_index=False)["label"].agg(
            pass_rate="mean", responders="size"
        )
        task = frame.groupby(["benchmark_id", "subject_uid"], as_index=False)["label"].mean()
        item_ids = item["item_uid"].astype(str).drop_duplicates()
        positions = item_positions.reindex(item_ids).dropna().to_numpy(dtype=np.int64)
        if len(positions) != len(item_ids):
            raise RuntimeError(f"{role}: missing BGE vectors for {len(item_ids) - len(positions)} filtered items")
        centroid = np.asarray(embedded[positions], dtype=np.float64).mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids[role] = centroid
        pass_hist, _ = np.histogram(item["pass_rate"], bins=bins_probability)
        task_hist, _ = np.histogram(task["label"], bins=bins_probability)
        support_hist, _ = np.histogram(np.log10(item["responders"]), bins=bins_log_support)
        summary["roles"][role] = {
            "benchmarks": sorted(frame["benchmark_id"].astype(str).unique().tolist()),
            "responses": int(len(frame)),
            "tasks": int(len(task)),
            "items": int(len(item)),
            "mean_label": float(frame["label"].mean()),
            "item_pass_rate": {
                "mean": float(item["pass_rate"].mean()),
                "std": float(item["pass_rate"].std()),
                "quantiles": {str(q): float(item["pass_rate"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                "histogram": pass_hist.astype(int).tolist(),
            },
            "task_pass_rate": {
                "mean": float(task["label"].mean()),
                "std": float(task["label"].std()),
                "histogram": task_hist.astype(int).tolist(),
            },
            "item_responders": {
                "median": float(item["responders"].median()),
                "q95": float(item["responders"].quantile(0.95)),
                "histogram_log10": support_hist.astype(int).tolist(),
            },
        }
    summary["histogram_edges"] = {
        "probability": bins_probability.tolist(),
        "log10_item_responders": bins_log_support.tolist(),
    }
    summary["bge_m3_long_centroid_cosine"] = {
        left: {right: float(np.dot(centroids[left], centroids[right]))
               for right in ("train", "validation", "test")}
        for left in ("train", "validation", "test")
    }
    destination = REPLAY_ROOT / "data-analysis" / "benchmark-holdout-split-distribution-v1.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary, destination)
    volume.commit()
    return {"path": str(destination), "roles": summary["roles"],
            "bge_m3_long_centroid_cosine": summary["bge_m3_long_centroid_cosine"]}

@app.function(image=image, cpu=8, memory=32_768, volumes={"/data": volume},
              timeout=2 * 60 * 60)
def build_panel(split_seed: int = 20261018, min_item_responders: int = 15,
                min_items_per_subject: int = 80, max_source_items: int = 1000,
                order_seed_count: int = 10) -> dict:
    """Freeze a Coda-shaped validation/test panel; never overwrite one."""
    import benchmark_holdout_filters as filters
    import benchmark_holdout_replay as replay

    if order_seed_count < 1:
        raise ValueError("order_seed_count must be positive")
    name = replay.panel_id(
        split_seed=split_seed, min_item_responders=min_item_responders,
        min_items_per_subject=min_items_per_subject, max_source_items=max_source_items,
    )
    destination = REPLAY_ROOT / "panels" / name
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_panel(destination, manifest, expected_order_seed_count=order_seed_count)
        return {"status": "reused", **manifest}
    if destination.exists():
        raise RuntimeError(f"incomplete panel directory exists: {destination}")

    responses, filter_audit = filters.load_benchmark_holdout_filtered(
        PROCESSED,
        min_item_responders=min_item_responders,
        min_items_per_subject=min_items_per_subject,
        split_seed=split_seed,
    )
    tasks, items, orders, selected = replay.make_panel(
        responses,
        split_seed=split_seed,
        min_source_items=80,
        max_source_items=max_source_items,
        order_seeds=tuple(range(order_seed_count)),
    )
    destination.mkdir(parents=True, exist_ok=False)
    _atomic_parquet(tasks, destination / "tasks.parquet")
    _atomic_parquet(items, destination / "items.parquet")
    _atomic_parquet(orders, destination / "candidate_orders.parquet")
    manifest = {
        "protocol_version": "benchmark-holdout-online-replay-v1",
        "panel_name": name,
        "split_seed": int(split_seed),
        "filtered_data": filter_audit,
        "panel_selection": {
            "mode": "balanced_complete_tasks_under_source_item_cap",
            "max_source_items_per_role": int(max_source_items),
            "minimum_source_items_per_task": 80,
            "roles": selected,
        },
        "pooling": {
            "unit": "canonical_source_item",
            "acquisition_fraction": "ceil(n_items / 2) / n_items",
            "evaluation_fraction": "floor(n_items / 2) / n_items",
            "disjoint": True,
        },
        "streaming": {
            "order_seeds": list(range(order_seed_count)),
            "budgets": list(replay.BUDGETS),
            "budget_weights": list(replay.ALC_WEIGHTS),
            "policy_can_see": "current candidate, current prediction, acquired labels, remaining counts",
            "policy_cannot_see": "future candidates or evaluation outcomes",
        },
        "files": {
            name: _sha256(destination / name)
            for name in ("tasks.parquet", "items.parquet", "candidate_orders.parquet")
        },
    }
    _atomic_json(manifest, manifest_path)
    volume.commit()
    print(json.dumps({"panel": name, "roles": selected}, sort_keys=True), flush=True)
    return {"status": "created", **manifest}


class _ANPCheckpointPredictor:
    """Inference-only ANP using the exact precomputed training embeddings."""

    def __init__(self, checkpoint_path: Path, arm: str) -> None:
        import numpy as np
        import pandas as pd
        import torch
        from anp_crossattn_blocks import AttentiveNeuralProcess

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = dict(checkpoint["config"])
        if config.get("model_family") != "anp_cross_attention":
            raise ValueError("offline runner currently supports anp_cross_attention checkpoints")
        if str(config.get("arm")) != arm:
            raise ValueError(f"checkpoint arm={config.get('arm')!r}, expected {arm!r}")
        store = DATASETS / f"allbench-embed-{arm}-v2"
        item_ids = pd.read_parquet(store / "items.parquet")["item_uid"].astype(str).to_numpy()
        self.position = {item_id: i for i, item_id in enumerate(item_ids)}
        self.embeddings = np.load(store / "item_embeddings.npy", mmap_mode="r")
        self.device = torch.device("cpu")
        self.model = AttentiveNeuralProcess(
            self.embeddings.shape[1],
            width=int(config["width"]),
            self_attention_blocks=int(config["n_blocks"]),
            cross_attention_heads=int(config["cross_attention_heads"]),
            use_latent=bool(config["use_latent"]),
            latent_dim=int(config["latent_dim"]),
            residual_proj=bool(config["residual_proj"]),
            dropout=float(config["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.max_context = int(config["max_context"])
        self.config = config
        self.checkpoint_sha256 = _sha256(checkpoint_path)

    def _vectors(self, item_uids: list[str]):
        import numpy as np
        import torch

        try:
            indices = [self.position[str(item_uid)] for item_uid in item_uids]
        except KeyError as exc:
            raise KeyError(f"item missing from embedding store: {exc.args[0]}") from exc
        return torch.from_numpy(np.asarray(self.embeddings[indices], dtype=np.float32)).to(self.device)

    def predict(self, context: list[tuple[str, int]], target_uids: list[str]) -> list[float]:
        import torch

        context = context[-self.max_context:]
        x_context = self._vectors([item_uid for item_uid, _ in context]) if context else torch.empty(
            (0, self.embeddings.shape[1]), dtype=torch.float32, device=self.device
        )
        y_context = torch.tensor([label for _, label in context], dtype=torch.float32, device=self.device)
        x_target = self._vectors(target_uids)
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model.logits(x_context, y_context, x_target))
        return probabilities.cpu().numpy().astype(float).tolist()


def _probabilities(predictor: _ANPCheckpointPredictor | None, *, mode: str,
                   context: list[tuple[str, int]], target_uids: list[str],
                   shrinkage_weight: float, neutral_at_zero: bool,
                   prior_mean: float = 0.5) -> list[float]:
    import numpy as np
    import benchmark_holdout_replay as replay

    labels = [label for _, label in context]
    beta = replay.beta_binomial_probability(labels)
    if mode == "beta_binomial":
        return [beta] * len(target_uids)
    if mode == "constant_prior":
        # The "knows nothing" floor: one number from the training benchmarks,
        # never updated by a label. Any model should beat this.
        return [prior_mean] * len(target_uids)
    if mode == "empirical_mean":
        # The subject's raw hit rate with no shrinkage: after one label it
        # predicts exactly 0 or 1. It isolates what beta-binomial's prior adds.
        return [sum(labels) / len(labels) if labels else prior_mean] * len(target_uids)
    if predictor is None:
        raise ValueError(f"prediction mode {mode!r} requires an ANP checkpoint")
    if mode == "anp_beta_gate" and len(context) < 2:
        # Same gate as the TabFM regressor, which cannot run on fewer than two
        # labels. Applying it to the ANP too puts every gated model on the
        # identical beta-binomial prediction at budgets 0 and 1.
        return [beta] * len(target_uids)
    anp = np.asarray(predictor.predict(context, target_uids), dtype=float)
    if mode in {"anp", "anp_beta_gate"}:
        return anp.tolist()
    if mode == "anp_beta_shrinkage":
        if not 0.0 <= shrinkage_weight <= 1.0:
            raise ValueError("shrinkage_weight must lie in [0, 1]")
        if neutral_at_zero and not context:
            return [0.5] * len(target_uids)
        return (shrinkage_weight * anp + (1.0 - shrinkage_weight) * beta).tolist()
    raise ValueError(
        "mode must be anp, beta_binomial, anp_beta_shrinkage, anp_beta_gate, "
        "constant_prior, or empirical_mean"
    )

@app.function(image=image, cpu=8, memory=32_768, volumes={"/data": volume},
              timeout=4 * 60 * 60)
def replay(panel_name: str, checkpoint: str = "", role: str = "validation",
           prediction_mode: str = "anp", policy: str = "random_prefix",
           uncertainty_threshold: float = 0.15, shrinkage_weight: float = 1.0,
           neutral_at_zero: bool = True, max_order_seeds: int = 10,
           run_name: str = "", probe_count: int = 32,
           prefiltered_responses: str = "") -> dict:
    """Replay one predictor/policy against the fixed panel without training."""
    import numpy as np
    import pandas as pd
    import benchmark_holdout_replay as helper

    if role not in {"validation", "test"}:
        raise ValueError("role must be validation or test")
    if policy not in {"random_prefix", "uncertainty", "confidence_extremes", "bernoulli_variance", "expected_brier_reduction"}:
        raise ValueError(
            "policy must be random_prefix, uncertainty, confidence_extremes, "
            "bernoulli_variance, or expected_brier_reduction"
        )
    panel = REPLAY_ROOT / "panels" / panel_name
    manifest_path = panel / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"panel missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_panel(panel, manifest)
    tasks = pd.read_parquet(panel / "tasks.parquet")
    items = pd.read_parquet(panel / "items.parquet")
    orders = pd.read_parquet(panel / "candidate_orders.parquet")
    tasks = tasks[tasks["benchmark_split"] == role].copy()
    if tasks.empty:
        raise RuntimeError(f"panel has no {role} tasks")
    task_ids = set(tasks["task_id"].astype(str))
    items = items[items["task_id"].astype(str).isin(task_ids)].copy()
    available_seeds = sorted(orders["order_seed"].unique())[:max_order_seeds]
    if not available_seeds:
        raise RuntimeError("panel has no candidate streams")
    orders = orders[
        orders["task_id"].astype(str).isin(task_ids) & orders["order_seed"].isin(available_seeds)
    ].copy()

    predictor = None
    checkpoint_path = None
    prior_mean = 0.5
    if prediction_mode in {"constant_prior", "empirical_mean"}:
        # Pooled label rate over TRAIN benchmarks only: a validation or test
        # mean would hand the baseline the answer it is meant not to have.
        import pandas as pd

        if prefiltered_responses:
            responses = pd.read_parquet(prefiltered_responses,
                                        columns=["label", "benchmark_split"])
        else:
            import benchmark_holdout_filters as filters

            parameters = manifest["filtered_data"]["parameters"]
            responses, _ = filters.load_benchmark_holdout_filtered(
                PROCESSED, split_seed=int(manifest["split_seed"]),
                min_item_responders=int(parameters["min_item_responders"]),
                min_items_per_subject=int(parameters["min_items_per_subject"]),
            )
        prior_mean = float(responses.loc[responses["benchmark_split"] == "train",
                                         "label"].mean())
        print(f"{prediction_mode}: training-benchmark label mean {prior_mean:.4f}",
              flush=True)
    if prediction_mode not in {"beta_binomial", "constant_prior", "empirical_mean"}:
        checkpoint_path = CHECKPOINTS / checkpoint
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint missing: {checkpoint_path}")
        checkpoint_payload = __import__("torch").load(checkpoint_path, map_location="cpu", weights_only=False)
        arm = str(checkpoint_payload["config"]["arm"])
        predictor = _ANPCheckpointPredictor(checkpoint_path, arm)

    # ``expected_brier_reduction`` needs a reference set to measure belief movement on,
    # and both scored policies need probes to pace against. Both are drawn from
    # TRAIN benchmarks only: a reference built from this task's own evaluation
    # items would let the policy steer toward the scored set, and probes drawn
    # from it would leak the scored distribution into the label spend.
    reference_uids: list[str] = []
    probe_uids: list[str] = []
    if policy in {"bernoulli_variance", "expected_brier_reduction"}:
        parameters = manifest["filtered_data"]["parameters"]
        reference_uids, probe_uids = helper.train_probe_sets(
            processed_dir=None if prefiltered_responses else PROCESSED,
            prefiltered_path=prefiltered_responses or None,
            split_seed=int(manifest["split_seed"]),
            min_item_responders=int(parameters["min_item_responders"]),
            min_items_per_subject=int(parameters["min_items_per_subject"]),
            count=probe_count,
        )
        print(f"policy {policy}: {len(reference_uids)} train reference items, "
              f"{len(probe_uids)} train probes", flush=True)

    item_by_task = {
        str(task_id): group.set_index(group["item_uid"].astype(str), drop=False)
        for task_id, group in items.groupby("task_id", sort=False)
    }
    prediction_rows, decision_rows, score_rows = [], [], []
    for task_id in sorted(task_ids):
        task_items = item_by_task[task_id]
        evaluation = task_items[task_items["pool"] == "evaluation"]
        evaluation_uids = evaluation["item_uid"].astype(str).tolist()
        if not evaluation_uids:
            raise RuntimeError(f"{task_id} has no evaluation items")
        for order_seed in available_seeds:
            stream = orders[
                (orders["task_id"].astype(str) == task_id)
                & (orders["order_seed"] == order_seed)
            ].sort_values("stream_position")
            context: list[tuple[str, int]] = []
            cursor = 0
            briers = []
            # Recomputed after every purchase, because a score's scale moves
            # with the context; ``None`` means "stale, recompute on next use".
            calibration = None
            for requested_budget in helper.BUDGETS:
                while len(context) < requested_budget and cursor < len(stream):
                    candidate_uid = str(stream.iloc[cursor]["item_uid"])
                    # The policy scores only this current candidate. It cannot
                    # inspect later candidates or any evaluation label.
                    candidate_probability = _probabilities(
                        predictor, mode=prediction_mode, context=context,
                        target_uids=[candidate_uid], shrinkage_weight=shrinkage_weight,
                        neutral_at_zero=neutral_at_zero, prior_mean=prior_mean,
                    )[0]
                    remaining = len(stream) - cursor
                    decision_score = None
                    if policy == "random_prefix":
                        accept = helper.accept_random_prefix(
                            labels_acquired=len(context), max_labels=max(helper.BUDGETS)
                        )
                    elif policy == "uncertainty":
                        accept = helper.accept_uncertainty(
                            probability=candidate_probability, labels_acquired=len(context),
                            max_labels=max(helper.BUDGETS), items_remaining=remaining,
                            threshold=uncertainty_threshold,
                        )
                    elif policy == "confidence_extremes":
                        accept = helper.accept_confidence_extremes(
                            probability=candidate_probability, labels_acquired=len(context),
                            max_labels=max(helper.BUDGETS), items_remaining=remaining,
                            threshold=uncertainty_threshold,
                        )
                    else:
                        def _predict(ctx, uids):
                            return _probabilities(
                                predictor, mode=prediction_mode, context=ctx,
                                target_uids=uids, shrinkage_weight=shrinkage_weight,
                                neutral_at_zero=neutral_at_zero, prior_mean=prior_mean,
                            )

                        if policy == "bernoulli_variance":
                            def _score(uid, probability):
                                return helper.bernoulli_variance(probability)
                        else:
                            baseline = _predict(context, reference_uids)

                            def _score(uid, probability):
                                return helper.expected_brier_reduction(
                                    lambda ctx: _predict(ctx, reference_uids),
                                    context, uid, baseline, probability,
                                )

                        if calibration is None:
                            probe_probabilities = _predict(context, probe_uids)
                            calibration = [
                                _score(uid, probability)
                                for uid, probability in zip(probe_uids, probe_probabilities)
                            ]
                        score_value = _score(candidate_uid, candidate_probability)
                        accept = helper.accept_by_rank(
                            score_value=score_value, calibration=calibration,
                            labels_acquired=len(context), max_labels=max(helper.BUDGETS),
                            items_remaining=remaining,
                            uniform=float(np.random.default_rng(
                                (int(order_seed), int(cursor), hash(task_id) & 0xFFFF)
                            ).random()),
                        )
                        decision_score = float(score_value)
                    decision = {
                        "task_id": task_id, "benchmark_split": role,
                        "order_seed": int(order_seed), "stream_position": int(cursor),
                        "item_uid": candidate_uid, "probability_before_decision": candidate_probability,
                        "accepted": bool(accept), "labels_acquired_before": int(len(context)),
                        "items_remaining_including_current": int(remaining),
                        "acquisition_score": decision_score,
                    }
                    if accept:
                        # The outcome becomes visible only now, after accepting.
                        label = int(task_items.loc[candidate_uid, "label"])
                        context.append((candidate_uid, label))
                        decision["revealed_label"] = label
                        calibration = None  # context changed; probe scores stale
                    else:
                        decision["revealed_label"] = None
                    decision_rows.append(decision)
                    cursor += 1

                probabilities = _probabilities(
                    predictor, mode=prediction_mode, context=context,
                    target_uids=evaluation_uids, shrinkage_weight=shrinkage_weight,
                    neutral_at_zero=neutral_at_zero, prior_mean=prior_mean,
                )
                # Evaluation outcomes are read only after all probabilities at
                # this checkpoint are committed to ``probabilities``.
                labels = evaluation["label"].astype(int).to_numpy()
                brier = helper.brier_score(labels, probabilities)
                ece = helper.expected_calibration_error(labels, probabilities)
                briers.append(brier)
                score_rows.append({
                    "task_id": task_id, "benchmark_split": role,
                    "benchmark_id": str(evaluation["benchmark_id"].iloc[0]),
                    "order_seed": int(order_seed), "requested_budget": int(requested_budget),
                    "labels_acquired": int(len(context)), "n_evaluation_items": int(len(evaluation)),
                    "brier": brier, "ece": ece,
                })
                for item_uid, label, probability in zip(evaluation_uids, labels, probabilities, strict=True):
                    prediction_rows.append({
                        "task_id": task_id, "benchmark_split": role,
                        "benchmark_id": str(evaluation["benchmark_id"].iloc[0]),
                        "order_seed": int(order_seed), "requested_budget": int(requested_budget),
                        "labels_acquired": int(len(context)), "item_uid": item_uid,
                        "label": int(label), "probability": float(probability),
                    })
            score_rows[-1]["brier_alc"] = helper.brier_alc(briers)

    scores = pd.DataFrame(score_rows)
    # The task is the independent unit: candidate-order replicates are first
    # averaged within each task before reporting the macro panel mean.
    task_budget = scores.groupby(["task_id", "requested_budget"], as_index=False)[["brier", "ece"]].mean()
    task_benchmarks = tasks[["task_id", "benchmark_id"]].drop_duplicates("task_id")
    task_budget = task_budget.merge(task_benchmarks, on="task_id", how="left", validate="many_to_one")
    task_alc = scores.groupby("task_id", as_index=False)["brier_alc"].mean()
    task_alc = task_alc.merge(task_benchmarks, on="task_id", how="left", validate="one_to_one")
    benchmark_budget = task_budget.groupby(["benchmark_id", "requested_budget"], as_index=False)[["brier", "ece"]].mean()
    benchmark_alc = task_alc.groupby("benchmark_id", as_index=False)["brier_alc"].mean()
    predictions = pd.DataFrame(prediction_rows)
    micro_by_budget = {}
    for budget in helper.BUDGETS:
        rows = predictions[predictions["requested_budget"] == budget]
        micro_by_budget[str(budget)] = {
            "brier": helper.brier_score(rows["label"], rows["probability"]),
            "ece": helper.expected_calibration_error(rows["label"], rows["probability"]),
        }
    summary = {
        "protocol_version": manifest["protocol_version"],
        "panel_name": panel_name,
        "role": role,
        "prediction_mode": prediction_mode,
        "policy": policy,
        "probe_count": int(probe_count),
        "reference_items": len(reference_uids),
        "calibration_probes": len(probe_uids),
        "calibration": (
            "context-specific train-only probe score upper-tail rank"
            if policy in {"bernoulli_variance", "expected_brier_reduction"} else "not applicable"
        ),
        "checkpoint": checkpoint or None,
        "checkpoint_sha256": predictor.checkpoint_sha256 if predictor is not None else None,
        "n_tasks": int(len(task_alc)),
        "n_order_seeds": int(len(available_seeds)),
        "macro_task_brier_alc": float(task_alc["brier_alc"].mean()),
        "macro_task_brier_by_budget": {
            str(budget): float(task_budget.loc[task_budget["requested_budget"] == budget, "brier"].mean())
            for budget in helper.BUDGETS
        },
        "macro_task_ece_by_budget": {
            str(budget): float(task_budget.loc[task_budget["requested_budget"] == budget, "ece"].mean())
            for budget in helper.BUDGETS
        },
        "micro_response_brier_alc": helper.brier_alc(
            [micro_by_budget[str(budget)]["brier"] for budget in helper.BUDGETS]
        ),
        "micro_response_by_budget": micro_by_budget,
        "per_benchmark": {
            str(benchmark): {
                "macro_task_brier_alc": float(
                    benchmark_alc.loc[benchmark_alc["benchmark_id"] == benchmark, "brier_alc"].iloc[0]
                ),
                "macro_task_by_budget": {
                    str(budget): {
                        "brier": float(benchmark_budget.loc[
                            (benchmark_budget["benchmark_id"] == benchmark)
                            & (benchmark_budget["requested_budget"] == budget), "brier"
                        ].iloc[0]),
                        "ece": float(benchmark_budget.loc[
                            (benchmark_budget["benchmark_id"] == benchmark)
                            & (benchmark_budget["requested_budget"] == budget), "ece"
                        ].iloc[0]),
                    }
                    for budget in helper.BUDGETS
                },
            }
            for benchmark in sorted(benchmark_alc["benchmark_id"].astype(str))
        },
        "development_guard": (
            "validation role is for model/policy selection; test role must be run only for a frozen choice"
        ),
    }
    if not run_name:
        checkpoint_tag = Path(checkpoint).stem if checkpoint else "beta-binomial"
        run_name = f"{role}-{prediction_mode}-{policy}-{checkpoint_tag}"
    run_name = "".join(char if char.isalnum() or char in "-_" else "-" for char in run_name)
    output = REPLAY_ROOT / "runs" / panel_name / run_name
    if output.exists():
        raise FileExistsError(f"replay output exists: {output}; choose a new run_name")
    output.mkdir(parents=True, exist_ok=False)
    _atomic_parquet(predictions, output / "predictions.parquet")
    _atomic_parquet(pd.DataFrame(decision_rows), output / "acquisition_decisions.parquet")
    _atomic_parquet(scores, output / "task_budget_scores.parquet")
    _atomic_parquet(task_alc, output / "task_alc.parquet")
    _atomic_parquet(benchmark_budget, output / "benchmark_budget_scores.parquet")
    _atomic_parquet(benchmark_alc, output / "benchmark_alc.parquet")
    _atomic_json(summary, output / "summary.json")
    volume.commit()
    print(json.dumps(summary, sort_keys=True), flush=True)
    return {"output": str(output), **summary}


@app.local_entrypoint()
def main(action: str = "build-panel", panel_name: str = "", checkpoint: str = "",
         role: str = "validation", prediction_mode: str = "anp",
         policy: str = "random_prefix", uncertainty_threshold: float = 0.15,
         shrinkage_weight: float = 1.0, neutral_at_zero: bool = True,
         max_order_seeds: int = 10, run_name: str = "", split_seed: int = 20261018,
         min_item_responders: int = 15, min_items_per_subject: int = 80,
         max_source_items: int = 1000, order_seed_count: int = 10,
         probe_count: int = 32, prefiltered_responses: str = "") -> None:
    if action == "export-holdout-split-distribution":
        print(export_holdout_split_distribution.remote(
            split_seed=split_seed, min_item_responders=min_item_responders,
            min_items_per_subject=min_items_per_subject,
        ))
    elif action == "build-panel":
        print(build_panel.remote(
            split_seed=split_seed, min_item_responders=min_item_responders,
            min_items_per_subject=min_items_per_subject, max_source_items=max_source_items,
            order_seed_count=order_seed_count,
        ))
    elif action == "replay":
        if not panel_name:
            raise ValueError("--panel-name is required for --action replay")
        # spawn, not remote: ``.remote`` blocks the local entrypoint until
        # the replay finishes, which makes a sweep of many arms serial. The
        # scored policies cost 17-33x a threshold policy, so serial execution
        # turns a sweep that could run concurrently into hours of wall clock.
        # Requires --detach, or the app tears down and kills the spawned call.
        call = replay.spawn(
            panel_name, checkpoint, role, prediction_mode, policy,
            uncertainty_threshold, shrinkage_weight, neutral_at_zero,
            max_order_seeds, run_name, probe_count, prefiltered_responses,
        )
        print(f"spawned replay call {call.object_id}")
    else:
        raise ValueError(
            "action must be export-holdout-split-distribution, build-panel, or replay"
        )
