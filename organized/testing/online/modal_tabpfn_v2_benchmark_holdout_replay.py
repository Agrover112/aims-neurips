"""Streaming TabPFN-v2 baseline on the frozen benchmark-holdout replay panel.

This is deliberately separate from ``modal_benchmark_holdout_replay.py``. Both
runners consume the same immutable panel and write the same
score/prediction/decision artifacts, so model x acquisition-policy effects are
paired rather than confounded by a different task sample.

The BGE-M3 store has 1,024 features.  TabPFN-v2's documented pre-training
limit is 500 input features, so ``prepare-transform`` fits a 500-D PCA *only
on training-benchmark item embeddings*.  That unsupervised transform is then
frozen and applied unchanged to validation/test candidates.  It neither reads
their outcomes nor adapts to their embeddings.

Typical use (do this once; it is CPU only):

  modal run --profile aims-test modal_tabpfn_v2_benchmark_holdout_replay.py::main \
    --action prepare-transform

Then compare the three streaming policies on validation:

  modal run --profile aims-test modal_tabpfn_v2_benchmark_holdout_replay.py::main \
    --action replay --policy random_prefix \
    --run-name validation-tabpfn-v2-pca500-random-v1

Use ``--role test`` only after freezing the model/policy choice on validation.
"""

from __future__ import annotations

import hashlib
import os
import json
from collections import Counter
from pathlib import Path

import modal


APP_NAME = "aims-neurips-tabpfn-v2-online-replay"
VOLUME_NAME = "aims-neurips-data"
DATASETS = Path("/data/datasets/full")
PROCESSED = DATASETS / "processed"
REPLAY_ROOT = Path("/data/results/online-replay")
BGE_ARM = "bge_m3_long"
TABPFN_VERSION = "2.2.1"
FEATURE_DIMENSION = 500
DEFAULT_PANEL = "benchmark-holdout-online-v1-s20261018-n15-m80-cap1000"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
# TabPFN's weights are cached on the volume (TABPFN_MODEL_CACHE_DIR), so no
# HuggingFace credential is attached. This is deliberately a constant, not a
# conditional on an environment variable: Modal evaluates this module twice,
# once locally and once inside the container, and a dependency list that
# differs between the two raises "Function has N dependencies but container
# got M object ids" and crash-loops at startup. A workspace without the cached
# weights fails with a clear download error instead.
_HF_SECRETS: list = []
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2,<3",
        "pandas>=2.2,<3",
        "pyarrow>=17,<24",
        "scikit-learn>=1.5,<2",
        f"tabpfn=={TABPFN_VERSION}",
    )
    .env(
        {
            "HF_HOME": "/data/huggingface-cache",
            "TABPFN_MODEL_CACHE_DIR": "/data/tabpfn-cache",
            "TABPFN_NO_BROWSER": "1",
        }
    )
)

_HERE = Path(__file__).resolve().parent
if str(_HERE) == "/root":
    _FILTERS = Path("/root/benchmark_holdout_filters.py")
    _HELPERS = Path("/root/benchmark_holdout_replay.py")
else:
    _TRAINING = _HERE.parents[1] / "training"
    _FILTERS = _TRAINING / "benchmark_holdout_filters.py"
    _HELPERS = _HERE / "benchmark_holdout_replay.py"
    for local, remote in (
        (_FILTERS, "/root/benchmark_holdout_filters.py"),
        (_HELPERS, "/root/benchmark_holdout_replay.py"),
    ):
        if local.exists():
            image = image.add_local_file(local, remote)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_parquet(frame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _validate_panel(panel: Path, manifest: dict) -> None:
    if manifest.get("protocol_version") != "benchmark-holdout-online-replay-v1":
        raise ValueError(f"unsupported panel protocol: {manifest.get('protocol_version')!r}")
    for filename, expected_hash in manifest.get("files", {}).items():
        path = panel / filename
        if not path.is_file():
            raise FileNotFoundError(f"panel manifest lists missing file: {path}")
        if _sha256(path) != expected_hash:
            raise RuntimeError(f"panel file hash differs from frozen manifest: {path}")


def _transform_name(*, split_seed: int, min_item_responders: int,
                    min_items_per_subject: int, n_components: int) -> str:
    return (
        f"tabpfn-v2-pca{n_components}-{BGE_ARM}"
        f"-s{split_seed}-n{min_item_responders}-m{min_items_per_subject}"
    )


def _transform_paths(*, split_seed: int, min_item_responders: int,
                     min_items_per_subject: int, n_components: int) -> tuple[Path, Path]:
    root = REPLAY_ROOT / "transforms"
    name = _transform_name(
        split_seed=split_seed, min_item_responders=min_item_responders,
        min_items_per_subject=min_items_per_subject, n_components=n_components,
    )
    return root / f"{name}.npz", root / f"{name}.json"


def _panel_and_manifest(panel_name: str):
    panel = REPLAY_ROOT / "panels" / panel_name
    manifest_path = panel / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"panel missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_panel(panel, manifest)
    return panel, manifest


@app.function(image=image, cpu=8, memory=32_768, volumes={"/data": volume},
              timeout=2 * 60 * 60)
def prepare_transform(panel_name: str = DEFAULT_PANEL,
                      n_components: int = FEATURE_DIMENSION,
                      prefiltered_responses: str = "") -> dict:
    """Fit and persist the train-only BGE-M3 PCA transform exactly once."""
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    import benchmark_holdout_filters as filters

    if n_components != FEATURE_DIMENSION:
        raise ValueError(
            f"n_components is fixed at {FEATURE_DIMENSION} for this registered TabPFN-v2 replay; "
            f"got {n_components}"
        )
    _, panel_manifest = _panel_and_manifest(panel_name)
    parameters = panel_manifest["filtered_data"]["parameters"]
    split_seed = int(parameters["split_seed"])
    min_item_responders = int(parameters["min_item_responders"])
    min_items_per_subject = int(parameters["min_items_per_subject"])
    destination, metadata_path = _transform_paths(
        split_seed=split_seed, min_item_responders=min_item_responders,
        min_items_per_subject=min_items_per_subject, n_components=n_components,
    )
    if destination.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _sha256(destination) != metadata.get("transform_sha256"):
            raise RuntimeError(f"cached PCA hash differs from metadata: {destination}")
        return {"status": "reused", "path": str(destination), "metadata": metadata}
    if destination.exists() or metadata_path.exists():
        raise RuntimeError("incomplete PCA artifact exists; inspect it instead of silently overwriting it")

    if prefiltered_responses:
        # A workspace holding only the exported subset has no
        # datasets/full/processed and cannot re-run the filter. The export
        # carries the filter's own output including benchmark_split, so the
        # train-only PCA fit is identical either way.
        responses = pd.read_parquet(prefiltered_responses)
        audit = {"source": "prefiltered_export", "path": str(prefiltered_responses)}
    else:
        responses, audit = filters.load_benchmark_holdout_filtered(
            PROCESSED,
            min_item_responders=min_item_responders,
            min_items_per_subject=min_items_per_subject,
            split_seed=split_seed,
        )
    train_item_uids = sorted(
        responses.loc[responses["benchmark_split"] == "train", "item_uid"].astype(str).unique()
    )
    if len(train_item_uids) < n_components:
        raise RuntimeError(
            f"train split has only {len(train_item_uids)} unique items, fewer than PCA components={n_components}"
        )
    store = DATASETS / f"allbench-embed-{BGE_ARM}-v2"
    item_ids = pd.read_parquet(store / "items.parquet", columns=["item_uid"])["item_uid"].astype(str)
    positions = pd.Series(np.arange(len(item_ids), dtype=np.int64), index=item_ids)
    selected = positions.reindex(train_item_uids)
    if selected.isna().any():
        missing = int(selected.isna().sum())
        raise RuntimeError(f"{missing} filtered train items have no BGE-M3 vector")
    embeddings = np.load(store / "item_embeddings.npy", mmap_mode="r")
    matrix = np.asarray(embeddings[selected.to_numpy(dtype=np.int64)], dtype=np.float32)
    if matrix.shape[1] < n_components:
        raise RuntimeError(f"BGE-M3 vectors have only {matrix.shape[1]} dimensions")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=split_seed)
    pca.fit(matrix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.savez_compressed(
            handle,
            mean=np.asarray(pca.mean_, dtype=np.float32),
            components=np.asarray(pca.components_, dtype=np.float32),
        )
    metadata = {
        "kind": "PCA",
        "model": "TabPFN-v2",
        "tabpfn_version": TABPFN_VERSION,
        "embedding_arm": BGE_ARM,
        "source_dimension": int(matrix.shape[1]),
        "output_dimension": int(n_components),
        "fit_scope": "unique item embeddings from filtered training benchmarks only",
        "fit_item_count": int(len(train_item_uids)),
        "uses_labels": False,
        "split_seed": split_seed,
        "min_item_responders": min_item_responders,
        "min_items_per_subject": min_items_per_subject,
        "filter_audit": audit,
        "panel_name_checked": panel_name,
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "transform_sha256": _sha256(destination),
    }
    _atomic_json(metadata, metadata_path)
    volume.commit()
    return {"status": "created", "path": str(destination), "metadata": metadata}


class _TabPFNV2StreamPredictor:
    """Context-cached TabPFN-v2 predictor with an explicit one-class fallback."""

    def __init__(self, *, transform_path: Path, n_estimators: int, seed: int) -> None:
        import numpy as np
        import pandas as pd
        import tabpfn
        from tabpfn import TabPFNClassifier

        if str(tabpfn.__version__) != TABPFN_VERSION:
            raise RuntimeError(
                f"expected tabpfn=={TABPFN_VERSION}, got {tabpfn.__version__}; "
                "version changes require a separately named replay"
            )
        if n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        with np.load(transform_path) as payload:
            self.mean = np.asarray(payload["mean"], dtype=np.float32)
            self.components = np.asarray(payload["components"], dtype=np.float32)
        if self.components.shape != (FEATURE_DIMENSION, self.mean.shape[0]):
            raise ValueError(
                "TabPFN transform shape must be (500, source_dimension); "
                f"got components={self.components.shape}, mean={self.mean.shape}"
            )
        store = DATASETS / f"allbench-embed-{BGE_ARM}-v2"
        item_ids = pd.read_parquet(store / "items.parquet", columns=["item_uid"])["item_uid"].astype(str)
        self.position = {item_uid: index for index, item_uid in enumerate(item_ids)}
        self.embeddings = np.load(store / "item_embeddings.npy", mmap_mode="r")
        if self.embeddings.shape[1] != self.mean.shape[0]:
            raise ValueError("transform source dimension differs from the BGE-M3 store")
        self.model = TabPFNClassifier(
            # The panel's contexts contain at most 31 labels. CPU inference is
            # adequate at this scale and avoids an unreliable GPU queue.
            device="cpu",
            n_estimators=n_estimators,
            fit_mode="fit_with_cache",
            random_state=seed,
            n_jobs=1,
            ignore_pretraining_limits=False,
        )
        self._fitted_context: tuple[tuple[str, int], ...] | None = None
        self.fit_count = 0
        self.predict_count = 0
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self.transform_sha256 = _sha256(transform_path)

    def _vectors(self, item_uids: list[str]):
        import numpy as np

        try:
            indices = [self.position[item_uid] for item_uid in item_uids]
        except KeyError as exc:
            raise KeyError(f"item missing from BGE-M3 store: {exc.args[0]}") from exc
        raw = np.asarray(self.embeddings[indices], dtype=np.float32)
        return (raw - self.mean) @ self.components.T

    @staticmethod
    def _fallback_probability(context: list[tuple[str, int]]) -> float:
        # Same explicit symmetric Beta(2,2) fallback as the existing TabPFN-v2
        # baseline. TabPFN cannot define a binary class mapping before both
        # classes have been observed.
        labels = [label for _, label in context]
        return float((2.0 + sum(labels)) / (4.0 + len(labels)))

    def predict(self, context: list[tuple[str, int]], target_uids: list[str]) -> tuple[list[float], str]:
        import numpy as np

        if not target_uids:
            return [], "empty_target"
        labels = [label for _, label in context]
        if len(set(labels)) < 2:
            return [self._fallback_probability(context)] * len(target_uids), "beta_2_2_fallback"
        signature = tuple(context)
        if signature != self._fitted_context:
            self.model.fit(self._vectors([item_uid for item_uid, _ in context]), np.asarray(labels, dtype=np.int64))
            self._fitted_context = signature
            self.fit_count += 1
        class_one = np.flatnonzero(self.model.classes_ == 1)
        if len(class_one) != 1:
            raise RuntimeError(f"TabPFN binary class mapping is invalid: {self.model.classes_!r}")
        probabilities = self.model.predict_proba(self._vectors(target_uids))[:, int(class_one[0])]
        self.predict_count += 1
        return np.asarray(probabilities, dtype=float).tolist(), "tabpfn_v2"


@app.function(
    image=image,
    cpu=8,
    memory=32_768,
    volumes={"/data": volume},
    secrets=_HF_SECRETS,
    timeout=4 * 60 * 60,
)
def replay(panel_name: str = DEFAULT_PANEL, role: str = "validation",
           policy: str = "random_prefix", uncertainty_threshold: float = 0.15,
           max_order_seeds: int = 10, n_estimators: int = 8,
           seed: int = 20261018, run_name: str = "", probe_count: int = 32,
           prefiltered_responses: str = "") -> dict:
    """Run one TabPFN-v2 x streaming-acquisition-policy replay."""
    import numpy as np
    import pandas as pd
    import benchmark_holdout_replay as helper

    if role not in {"validation", "test"}:
        raise ValueError("role must be validation or test")
    if policy not in {"random_prefix", "uncertainty", "confidence_extremes",
                      "bernoulli_variance", "expected_brier_reduction"}:
        raise ValueError(
            "policy must be random_prefix, uncertainty, confidence_extremes, "
            "bernoulli_variance, or expected_brier_reduction"
        )
    if not 0.0 <= uncertainty_threshold <= 0.5:
        raise ValueError("uncertainty_threshold must lie in [0, 0.5]")
    panel, panel_manifest = _panel_and_manifest(panel_name)
    parameters = panel_manifest["filtered_data"]["parameters"]
    transform_path, transform_metadata_path = _transform_paths(
        split_seed=int(parameters["split_seed"]),
        min_item_responders=int(parameters["min_item_responders"]),
        min_items_per_subject=int(parameters["min_items_per_subject"]),
        n_components=FEATURE_DIMENSION,
    )
    if not transform_path.is_file() or not transform_metadata_path.is_file():
        raise FileNotFoundError(
            "missing train-only TabPFN PCA transform; run --action prepare-transform first"
        )
    transform_metadata = json.loads(transform_metadata_path.read_text(encoding="utf-8"))
    if _sha256(transform_path) != transform_metadata.get("transform_sha256"):
        raise RuntimeError("PCA transform hash differs from metadata")

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
    predictor = _TabPFNV2StreamPredictor(
        transform_path=transform_path, n_estimators=n_estimators, seed=seed,
    )

    item_by_task = {
        str(task_id): group.set_index(group["item_uid"].astype(str), drop=False)
        for task_id, group in items.groupby("task_id", sort=False)
    }
    # Scored policies need a reference set to measure belief movement on and
    # probes to pace against, both drawn from TRAIN benchmarks only. The
    # predictor holds the whole embedding store, so it can score items that
    # never appear in the panel.
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

    prediction_rows: list[dict] = []
    decision_rows: list[dict] = []
    score_rows: list[dict] = []
    mode_counts: Counter[str] = Counter()
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
            briers: list[float] = []
            # Stale after every purchase: a score's scale moves with context.
            calibration = None
            for requested_budget in helper.BUDGETS:
                while len(context) < requested_budget and cursor < len(stream):
                    candidate_uid = str(stream.iloc[cursor]["item_uid"])
                    remaining = len(stream) - cursor
                    if policy == "random_prefix":
                        # Random prefix is intentionally model-free during
                        # acquisition.  Avoiding this needless TabPFN call
                        # keeps its inference cost honest.
                        candidate_probability = None
                        prediction_mode = "not_scored_random_prefix"
                        accept = helper.accept_random_prefix(
                            labels_acquired=len(context), max_labels=max(helper.BUDGETS)
                        )
                    else:
                        probabilities, prediction_mode = predictor.predict(context, [candidate_uid])
                        candidate_probability = float(probabilities[0])
                        mode_counts[prediction_mode] += 1
                        if policy == "uncertainty":
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
                            def _reference(ctx):
                                values, _ = predictor.predict(ctx, reference_uids)
                                return values

                            if policy == "bernoulli_variance":
                                def _score(uid, probability):
                                    return helper.bernoulli_variance(probability)
                            else:
                                baseline = _reference(context)

                                def _score(uid, probability):
                                    return helper.expected_brier_reduction(
                                        _reference, context, uid, baseline, probability,
                                    )

                            if calibration is None:
                                probe_values, _ = predictor.predict(context, probe_uids)
                                calibration = [
                                    _score(uid, float(value))
                                    for uid, value in zip(probe_uids, probe_values)
                                ]
                            accept = helper.accept_by_rank(
                                score_value=_score(candidate_uid, candidate_probability),
                                calibration=calibration,
                                labels_acquired=len(context),
                                max_labels=max(helper.BUDGETS),
                                items_remaining=remaining,
                                uniform=float(np.random.default_rng(
                                    (int(order_seed), int(cursor), hash(task_id) & 0xFFFF)
                                ).random()),
                            )
                    decision = {
                        "task_id": task_id,
                        "benchmark_split": role,
                        "order_seed": int(order_seed),
                        "stream_position": int(cursor),
                        "item_uid": candidate_uid,
                        "probability_before_decision": candidate_probability,
                        "prediction_mode": prediction_mode,
                        "accepted": bool(accept),
                        "labels_acquired_before": int(len(context)),
                        "items_remaining_including_current": int(remaining),
                    }
                    if accept:
                        label = int(task_items.loc[candidate_uid, "label"])
                        context.append((candidate_uid, label))
                        calibration = None  # context changed; probes are stale
                        decision["revealed_label"] = label
                    else:
                        decision["revealed_label"] = None
                    decision_rows.append(decision)
                    cursor += 1

                probabilities, prediction_mode = predictor.predict(context, evaluation_uids)
                mode_counts[prediction_mode] += 1
                labels = evaluation["label"].astype(int).to_numpy()
                brier = helper.brier_score(labels, probabilities)
                ece = helper.expected_calibration_error(labels, probabilities)
                briers.append(brier)
                score_rows.append({
                    "task_id": task_id,
                    "benchmark_split": role,
                    "benchmark_id": str(evaluation["benchmark_id"].iloc[0]),
                    "order_seed": int(order_seed),
                    "requested_budget": int(requested_budget),
                    "labels_acquired": int(len(context)),
                    "n_evaluation_items": int(len(evaluation)),
                    "brier": brier,
                    "ece": ece,
                    "prediction_mode": prediction_mode,
                })
                for item_uid, label, probability in zip(evaluation_uids, labels, probabilities, strict=True):
                    prediction_rows.append({
                        "task_id": task_id,
                        "benchmark_split": role,
                        "benchmark_id": str(evaluation["benchmark_id"].iloc[0]),
                        "order_seed": int(order_seed),
                        "requested_budget": int(requested_budget),
                        "labels_acquired": int(len(context)),
                        "item_uid": item_uid,
                        "label": int(label),
                        "probability": float(probability),
                        "prediction_mode": prediction_mode,
                    })
            score_rows[-1]["brier_alc"] = helper.brier_alc(briers)

    scores = pd.DataFrame(score_rows)
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
        "protocol_version": panel_manifest["protocol_version"],
        "panel_name": panel_name,
        "role": role,
        "model_id": "tabpfn_v2_pca500_bge_m3_long",
        "model_label": "TabPFN-v2 + train-only PCA-500 (BGE-M3)",
        "tabpfn_version": TABPFN_VERSION,
        "device": "CPU",
        "n_estimators": int(n_estimators),
        "seed": int(seed),
        "feature_transform": transform_metadata,
        "policy": policy,
        "uncertainty_threshold": float(uncertainty_threshold),
        "fallback": "symmetric Beta(2,2) when context has fewer than two classes",
        "prediction_mode_counts": dict(sorted(mode_counts.items())),
        "tabpfn_model_fit_count": int(predictor.fit_count),
        "tabpfn_prediction_call_count": int(predictor.predict_count),
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
        run_name = f"{role}-tabpfn-v2-pca500-{policy}-s{seed}"
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
def main(action: str = "prepare-transform", panel_name: str = DEFAULT_PANEL,
         role: str = "validation", policy: str = "random_prefix",
         uncertainty_threshold: float = 0.15, max_order_seeds: int = 10,
         n_estimators: int = 8, seed: int = 20261018, run_name: str = "",
         n_components: int = FEATURE_DIMENSION, probe_count: int = 32,
         prefiltered_responses: str = "") -> None:
    if action == "prepare-transform":
        print(prepare_transform.remote(panel_name, n_components,
                                       prefiltered_responses))
    elif action == "replay":
        # spawn, not remote: see the note in
        # modal_benchmark_holdout_replay.py. Requires --detach.
        call = replay.spawn(
            panel_name, role, policy, uncertainty_threshold, max_order_seeds,
            n_estimators, seed, run_name, probe_count, prefiltered_responses,
        )
        print(f"spawned replay call {call.object_id}")
    else:
        raise ValueError("action must be prepare-transform or replay")
