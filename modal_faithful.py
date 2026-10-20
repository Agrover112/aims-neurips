"""Three-seed, paper-faithful NP/ANP BeaverTails experiment on Modal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "aims-neurips-faithful"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "faithful-pilot-beavertails-v2-features"
RESULT_ROOT = REMOTE_ROOT / "results" / "faithful-np-beavertails-3seed-v3-residual"
CNP_RESULT_ROOT = REMOTE_ROOT / "results" / "cnp-airt-beavertails-3seed-v2-residual"
SEEDS = (20261018, 20261019, 20261020)
ACQUISITION_ORDER_SEEDS = tuple(range(10))

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=2,<3",
        "pandas>=2.2,<3",
        "pyarrow>=17,<24",
        "torch==2.5.1",
        "sentence-transformers==3.4.1",
    )
    .add_local_dir(PROJECT_ROOT / "src", "/opt/aims/src", copy=True)
    .env(
        {
            "PYTHONPATH": "/opt/aims/src",
            "HF_HOME": "/data/huggingface-cache",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
)


def _hash(*values: object) -> str:
    return hashlib.sha256("::".join(map(str, values)).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _training_signature() -> str:
    """Fingerprint training code and protocol so stale caches cannot be reused silently."""

    digest = hashlib.sha256()
    source = Path("/opt/aims/src/aims_starter/faithful_np.py")
    if not source.exists():
        source = PROJECT_ROOT / "src" / "aims_starter" / "faithful_np.py"
    digest.update(source.read_bytes() if source.exists() else b"source-unavailable")
    digest.update(
        json.dumps(
            {
                "corpus": CORPUS_ROOT.name,
                "seeds": list(SEEDS),
                "training_steps": 800,
                "episodes_per_step": 4,
                "n_z_samples_train": 4,
                "n_z_samples_evaluation": 256,
                "np_residual_scale": 0.0,
                "anp_residual_scale": 0.1,
                "np_map_prior_variance": 4.0,
                "budgets": [0, 1, 3, 7, 15, 31],
            },
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()[:16]


@app.function(image=image, cpu=2, memory=4096, timeout=5 * 60)
def smoke_symbols() -> dict[str, Any]:
    """Exercise the canonical NP/ANP and IRT interfaces without training."""

    import torch

    from aims_starter.faithful_np import CNPAIRT, TNPDAIRT, Amortized2PL, FaithfulNPAIRT

    torch.manual_seed(7)
    x_C = torch.randn(3, 8)
    y_C = torch.tensor([0.0, 1.0, 1.0])
    x_T = torch.randn(6, 8)
    y_T = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    checks = {}
    for variant in ("np", "anp", "stacked_mhsa_anp"):
        model = FaithfulNPAIRT(8, variant=variant, d_model=16, d_z=4, n_heads=4)
        loss, _ = model.training_loss(x_C, y_C, x_T, y_T, n_z_samples=2)
        p_y_T, var_p_y_T = model.predict(x_C, y_C, x_T, n_z_samples=3)
        checks[variant] = {
            "loss_finite": bool(torch.isfinite(loss)),
            "p_y_T_shape": list(p_y_T.shape),
            "var_p_y_T_shape": list(var_p_y_T.shape),
        }
    airt = Amortized2PL(8, 2, d_model=16)
    checks["amortized_2pl"] = {
        "logits_shape": list(airt.logits(x_T, torch.zeros(6, dtype=torch.long)).shape)
    }
    tnp = TNPDAIRT(8, d_model=16, n_heads=4, n_layers=1)
    checks["tnpd_airt"] = {
        "loss_finite": bool(torch.isfinite(tnp.training_loss(x_C, y_C, x_T, y_T))),
        "p_y_T_shape": list(tnp.predict(x_C, y_C, x_T).shape),
    }
    changed_subject = x_T.clone()
    changed_subject[:, 3:5] += 100.0
    invariant_model = Amortized2PL(8, 2, d_model=16, subject_feature_slice=(3, 5))
    task_indices = torch.zeros(len(x_T), dtype=torch.long)
    a, b = invariant_model.item_parameters(x_T)
    changed_a, changed_b = invariant_model.item_parameters(changed_subject)
    checks["subject_invariance"] = {
        "amortized_a_unchanged": bool(torch.allclose(a, changed_a)),
        "amortized_b_unchanged": bool(torch.allclose(b, changed_b)),
        "amortized_logits_unchanged": bool(
            torch.allclose(
                invariant_model.logits(x_T, task_indices),
                invariant_model.logits(changed_subject, task_indices),
            )
        ),
    }
    np_probe = FaithfulNPAIRT(
        8, variant="np", d_model=16, d_z=4, n_heads=4, subject_feature_slice=(3, 5)
    )
    z_probe = torch.randn(2, 4)
    np_logits = np_probe.decode_logits(
        z_probe,
        np_probe.encode_item_x(x_T),
        None,
        np_probe.map_theta(np_probe.encode_item_x(x_C), y_C),
    )
    changed_context = x_C.clone()
    changed_context[:, 3:5] += 100.0
    changed_np_logits = np_probe.decode_logits(
        z_probe,
        np_probe.encode_item_x(changed_subject),
        None,
        np_probe.map_theta(np_probe.encode_item_x(changed_context), y_C),
    )
    cnp_probe = CNPAIRT(8, d_model=16, subject_feature_slice=(3, 5))
    tnp_probe = TNPDAIRT(8, d_model=16, n_heads=4, n_layers=1, subject_feature_slice=(3, 5))
    checks["subject_invariance"].update(
        {
            "np_logits_unchanged": bool(torch.allclose(np_logits, changed_np_logits)),
            "cnp_logits_unchanged": bool(
                torch.allclose(
                    cnp_probe.logits(x_C, y_C, x_T),
                    cnp_probe.logits(changed_context, y_C, changed_subject),
                )
            ),
            "tnpd_logits_unchanged": bool(
                torch.allclose(
                    tnp_probe.logits(x_C, y_C, x_T),
                    tnp_probe.logits(changed_context, y_C, changed_subject),
                )
            ),
        }
    )
    return checks


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    volumes={str(REMOTE_ROOT): volume},
    timeout=5 * 60,
)
def inspect_feature_sources() -> dict[str, Any]:
    import pyarrow.parquet as pq

    processed = REMOTE_ROOT / "datasets" / "full" / "processed"
    result = {}
    for name in ("items", "subjects", "benchmarks", "observations"):
        parquet = pq.ParquetFile(processed / f"{name}.parquet")
        frame = parquet.read_row_group(0).slice(0, 3).to_pandas()
        result[name] = {
            "columns": parquet.schema_arrow.names,
            "sample": frame.astype(str).to_dict(orient="records"),
        }
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32_768,
    volumes={str(REMOTE_ROOT): volume},
    timeout=2 * 60 * 60,
)
def prepare_corpus(meta_task_count: int = 128, items_per_task: int = 128) -> dict[str, Any]:
    """Freeze meta-training tasks and leakage-safe item/subject/benchmark features."""

    import numpy as np
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    existing = _read_json(CORPUS_ROOT / "manifest.json")
    if existing is not None and "feature_slices" in existing:
        return {"status": "reused", **existing}
    processed = REMOTE_ROOT / "datasets" / "full" / "processed"
    episodes = REMOTE_ROOT / "datasets" / "full" / "episodes"
    tasks = pd.read_parquet(processed / "tasks.parquet")
    candidates = tasks[
        (tasks["split_by_benchmark"] == "train")
        & (tasks["split_by_task"] == "train")
        & (tasks["n_items"] >= 80)
        & (tasks["benchmark_id"] != "beavertails")
    ].copy()
    candidates["selection_key"] = [
        _hash(benchmark, task)
        for benchmark, task in zip(candidates["benchmark_id"], candidates["task_id"], strict=True)
    ]
    candidates = candidates.sort_values(["benchmark_id", "selection_key"])
    candidates = candidates.groupby("benchmark_id", sort=True).head(3)
    selected = candidates.sort_values("selection_key").head(meta_task_count)
    selected_task_ids = selected["task_id"].astype(str).tolist()
    print(
        f"Selected {len(selected_task_ids)} tasks from {selected['benchmark_id'].nunique()} benchmarks",
        flush=True,
    )
    meta = pd.read_parquet(
        processed / "observations.parquet",
        filters=[("task_id", "in", selected_task_ids)],
    )
    meta = (
        meta.sort_values("observation_id")
        .groupby(["task_id", "item_uid"], as_index=False, sort=True)
        .first()
    )
    meta["sample_key"] = [
        _hash(task, item) for task, item in zip(meta["task_id"], meta["item_uid"], strict=True)
    ]
    meta = (
        meta.sort_values(["task_id", "sample_key"])
        .groupby("task_id", sort=True)
        .head(items_per_task)
        .drop(columns="sample_key")
        .reset_index(drop=True)
    )
    beaver = pd.read_parquet(
        processed / "observations.parquet", filters=[("benchmark_id", "==", "beavertails")]
    )
    beaver_episodes = pd.read_parquet(
        episodes / "episode_items.parquet", filters=[("benchmark_id", "==", "beavertails")]
    )
    item_ids = sorted(set(meta["item_uid"].astype(str)) | set(beaver["item_uid"].astype(str)))
    items = pd.read_parquet(processed / "items.parquet", filters=[("item_uid", "in", item_ids)])
    items = items.drop_duplicates("item_uid").sort_values("item_uid").reset_index(drop=True)
    if len(items) != len(item_ids):
        raise RuntimeError(f"item join incomplete: expected {len(item_ids)}, received {len(items)}")
    subject_ids = sorted(
        set(meta["subject_uid"].astype(str)) | set(beaver["subject_uid"].astype(str))
    )
    subjects = pd.read_parquet(
        processed / "subjects.parquet", filters=[("subject_uid", "in", subject_ids)]
    )
    subjects = (
        subjects.drop_duplicates("subject_uid").sort_values("subject_uid").reset_index(drop=True)
    )
    benchmark_ids = sorted(set(meta["benchmark_id"].astype(str)) | {"beavertails"})
    benchmarks = pd.read_parquet(
        processed / "benchmarks.parquet", filters=[("benchmark_id", "in", benchmark_ids)]
    )
    benchmarks = (
        benchmarks.drop_duplicates("benchmark_id")
        .sort_values("benchmark_id")
        .reset_index(drop=True)
    )
    # Benchmark identity is represented once by the dedicated benchmark
    # embedding block below; do not duplicate it in the item text.
    item_texts = [
        f"item: {text}\nmetadata: {metadata}\ngrading: {grading}"
        for text, metadata, grading in zip(
            items["model_text"].fillna(""),
            items["item_features"].fillna(""),
            items["grading_criterion"].fillna(""),
            strict=True,
        )
    ]
    subject_columns = (
        "display_name",
        "normalized_name",
        "provider",
        "release_date",
        "harness",
        "reasoning_effort",
        "harness_version",
        "subject_features_extra",
    )
    subject_texts = [
        "\n".join(f"{column}: {getattr(row, column)}" for column in subject_columns)
        for row in subjects.fillna("").itertuples(index=False)
    ]
    benchmark_columns = (
        "name",
        "description",
        "one_line_description",
        "modality",
        "domain",
        "multi_single_turn",
        "benchmark_features",
    )
    benchmark_texts = [
        "\n".join(f"{column}: {getattr(row, column)}" for column in benchmark_columns)
        for row in benchmarks.fillna("").itertuples(index=False)
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    item_embeddings = encoder.encode(
        item_texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    subject_embeddings = encoder.encode(
        subject_texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    benchmark_embeddings = encoder.encode(
        benchmark_texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    item_numeric = np.column_stack(
        [
            np.log1p(items["content_chars"].fillna(0).to_numpy(dtype="float32")) / 10.0,
            items["content_missing"].fillna(True).to_numpy(dtype="float32"),
            items["item_features"].fillna("").ne("").to_numpy(dtype="float32"),
            items["grading_criterion"].fillna("").ne("").to_numpy(dtype="float32"),
            items["asset_manifest"].fillna("").ne("").to_numpy(dtype="float32"),
        ]
    ).astype("float32")
    CORPUS_ROOT.mkdir(parents=True, exist_ok=False)
    meta.to_parquet(CORPUS_ROOT / "meta_observations.parquet", index=False)
    beaver.to_parquet(CORPUS_ROOT / "beaver_observations.parquet", index=False)
    beaver_episodes.to_parquet(CORPUS_ROOT / "beaver_episode_items.parquet", index=False)
    items[["item_uid", "benchmark_id"]].to_parquet(CORPUS_ROOT / "items.parquet", index=False)
    subjects[["subject_uid"]].to_parquet(CORPUS_ROOT / "subjects.parquet", index=False)
    benchmarks[["benchmark_id"]].to_parquet(CORPUS_ROOT / "benchmarks.parquet", index=False)
    np.save(CORPUS_ROOT / "item_embeddings.npy", item_embeddings)
    np.save(CORPUS_ROOT / "item_numeric.npy", item_numeric)
    np.save(CORPUS_ROOT / "subject_embeddings.npy", subject_embeddings)
    np.save(CORPUS_ROOT / "benchmark_embeddings.npy", benchmark_embeddings)
    manifest = {
        "format_version": 2,
        "meta_tasks": int(meta["task_id"].nunique()),
        "meta_benchmarks": int(meta["benchmark_id"].nunique()),
        "meta_rows": len(meta),
        "beavertails_tasks": int(beaver["task_id"].nunique()),
        "beavertails_rows": len(beaver),
        "unique_items": len(items),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "item_embedding_dimension": int(item_embeddings.shape[1]),
        "subject_embedding_dimension": int(subject_embeddings.shape[1]),
        "benchmark_embedding_dimension": int(benchmark_embeddings.shape[1]),
        "numeric_item_features": int(item_numeric.shape[1]),
        "combined_x_dimension": int(
            item_embeddings.shape[1]
            + subject_embeddings.shape[1]
            + benchmark_embeddings.shape[1]
            + item_numeric.shape[1]
        ),
        "feature_slices": {
            "item": [0, int(item_embeddings.shape[1])],
            "numeric": [
                int(item_embeddings.shape[1]),
                int(item_embeddings.shape[1] + item_numeric.shape[1]),
            ],
            "subject": [
                int(item_embeddings.shape[1] + item_numeric.shape[1]),
                int(item_embeddings.shape[1] + item_numeric.shape[1] + subject_embeddings.shape[1]),
            ],
            "benchmark": [
                int(item_embeddings.shape[1] + item_numeric.shape[1] + subject_embeddings.shape[1]),
                int(
                    item_embeddings.shape[1]
                    + item_numeric.shape[1]
                    + subject_embeddings.shape[1]
                    + benchmark_embeddings.shape[1]
                ),
            ],
        },
        "normalized_embeddings": True,
        "selection": "up to three deterministic training tasks per benchmark; 128 items per task",
    }
    (CORPUS_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    volume.commit()
    return {"status": "created", **manifest}


def _x_for_rows(
    rows,
    item_to_index,
    subject_to_index,
    benchmark_to_index,
    item_embeddings,
    item_numeric,
    subject_embeddings,
    benchmark_embeddings,
):
    import numpy as np

    item_indices = np.asarray([item_to_index[str(value)] for value in rows["item_uid"]])
    subject_indices = np.asarray([subject_to_index[str(value)] for value in rows["subject_uid"]])
    benchmark_indices = np.asarray(
        [benchmark_to_index[str(value)] for value in rows["benchmark_id"]]
    )
    return np.concatenate(
        [
            item_embeddings[item_indices],
            item_numeric[item_indices],
            subject_embeddings[subject_indices],
            benchmark_embeddings[benchmark_indices],
        ],
        axis=1,
    ).astype("float32")


def _task_arrays(frame, feature_state):
    result = []
    for task_id, rows in frame.groupby("task_id", sort=True):
        result.append(
            {
                "task_id": str(task_id),
                "x": _x_for_rows(rows, **feature_state),
                "y": rows["label"].to_numpy(dtype="float32"),
            }
        )
    return result


def _evaluation_task(task_id, beaver, assignments, observation_lookup, feature_state):
    assignment = assignments[assignments["task_id"].astype(str) == task_id]
    acquisition_rows = assignment[assignment["pool"] == "acquisition"].sort_values(
        "acquisition_rank"
    )
    label_rows = observation_lookup.loc[
        acquisition_rows["label_observation_id"].astype(str).tolist()
    ]
    evaluation_items = set(
        assignment.loc[assignment["pool"] == "evaluation", "item_uid"].astype(str)
    )
    targets = beaver[
        (beaver["task_id"].astype(str) == task_id)
        & (beaver["item_uid"].astype(str).isin(evaluation_items))
    ]
    return {
        "x_C_pool": _x_for_rows(label_rows, **feature_state),
        "y_C_pool": label_rows["label"].to_numpy(dtype="float32"),
        "x_T": _x_for_rows(targets, **feature_state),
        "y_T": targets["label"].to_numpy(dtype="float32"),
    }


def _reorder_evaluation(evaluation, *, task_id: str, order_seed: int):
    """Deterministically reshuffle only the label-reveal order."""
    import hashlib

    import numpy as np

    digest = hashlib.sha256(f"{task_id}:{order_seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    order = rng.permutation(len(evaluation["y_C_pool"]))
    return {
        **evaluation,
        "x_C_pool": evaluation["x_C_pool"][order],
        "y_C_pool": evaluation["y_C_pool"][order],
    }


def _variant_model_kwargs(
    variant: str, subject_feature_slice: tuple[int, int]
) -> dict[str, Any]:
    """FaithfulNPAIRT kwargs shared by training and every rescore pass.

    The ANP rows must carry the same ``residual_scale`` used during training,
    otherwise the saved ``decoder_residual`` weights have nowhere to load.
    """

    kwargs: dict[str, Any] = {"subject_feature_slice": subject_feature_slice}
    if variant in {"anp", "stacked_mhsa_anp"}:
        # Give ANP's target-dependent representation r* an actual route into
        # the logits; otherwise this is only attention in the encoder and is
        # not an ANP decoder.
        kwargs["residual_scale"] = 0.1
    return kwargs


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32_768,
    volumes={str(REMOTE_ROOT): volume},
    timeout=4 * 60 * 60,
)
def run_seed(seed: int) -> dict[str, Any]:
    """Train all faithful variants for one seed and four held-out-subject folds."""

    import numpy as np
    import pandas as pd
    import torch

    from aims_starter.faithful_np import (
        score_amortized_2pl,
        score_model,
        score_tnpd,
        train_amortized_2pl,
        train_model,
        train_tnpd,
        write_metrics,
    )

    seed_root = RESULT_ROOT / f"seed-{seed}"
    signature = _training_signature()
    existing = _read_json(seed_root / "metrics.json")
    if existing is not None:
        if existing.get("protocol", {}).get("training_signature") != signature:
            raise RuntimeError(
                "cached metrics were produced by different training code or protocol; "
                f"bump RESULT_ROOT or remove {seed_root} before rerunning"
            )
        return {"status": "reused", **existing}
    manifest = _read_json(CORPUS_ROOT / "manifest.json")
    if manifest is None:
        raise FileNotFoundError("faithful corpus is missing")
    subject_feature_slice = tuple(manifest["feature_slices"]["subject"])
    items = pd.read_parquet(CORPUS_ROOT / "items.parquet")
    subjects = pd.read_parquet(CORPUS_ROOT / "subjects.parquet")
    benchmarks = pd.read_parquet(CORPUS_ROOT / "benchmarks.parquet")
    feature_state = {
        "item_to_index": {
            value: index for index, value in enumerate(items["item_uid"].astype(str))
        },
        "subject_to_index": {
            value: index for index, value in enumerate(subjects["subject_uid"].astype(str))
        },
        "benchmark_to_index": {
            value: index for index, value in enumerate(benchmarks["benchmark_id"].astype(str))
        },
        "item_embeddings": np.load(CORPUS_ROOT / "item_embeddings.npy"),
        "item_numeric": np.load(CORPUS_ROOT / "item_numeric.npy"),
        "subject_embeddings": np.load(CORPUS_ROOT / "subject_embeddings.npy"),
        "benchmark_embeddings": np.load(CORPUS_ROOT / "benchmark_embeddings.npy"),
    }
    meta = pd.read_parquet(CORPUS_ROOT / "meta_observations.parquet")
    beaver = pd.read_parquet(CORPUS_ROOT / "beaver_observations.parquet")
    assignments = pd.read_parquet(CORPUS_ROOT / "beaver_episode_items.parquet")
    observation_lookup = beaver.set_index("observation_id", drop=False)
    meta_tasks = _task_arrays(meta, feature_state)
    beaver_tasks = {task["task_id"]: task for task in _task_arrays(beaver, feature_state)}
    task_ids = sorted(beaver_tasks)
    variants = ("online_airt_2pl", "np", "anp", "stacked_mhsa_anp", "tnpd_airt")
    seed_root.mkdir(parents=True, exist_ok=False)
    folds = []
    for fold, heldout in enumerate(task_ids):
        print(f"seed={seed} fold={fold + 1}/4 heldout={heldout}", flush=True)
        training_tasks = meta_tasks + [beaver_tasks[task] for task in task_ids if task != heldout]
        evaluation = _evaluation_task(
            heldout,
            beaver,
            assignments,
            observation_lookup,
            feature_state,
        )
        fold_root = seed_root / f"fold-{fold}"
        fold_root.mkdir(parents=True, exist_ok=False)
        model_results = {}
        airt, history = train_amortized_2pl(
            training_tasks,
            seed=seed + fold * 1000,
            steps=800,
            subject_feature_slice=subject_feature_slice,
        )
        model_results["online_airt_2pl"] = score_amortized_2pl(airt, evaluation)
        torch.save(airt.state_dict(), fold_root / "online_airt_2pl.pt")
        write_metrics(fold_root / "online_airt_2pl-history.json", {"history": history})
        for offset, variant in enumerate(variants[1:], start=1):
            if variant == "tnpd_airt":
                continue
            print(f"seed={seed} fold={fold} training={variant}", flush=True)
            variant_kwargs = _variant_model_kwargs(variant, subject_feature_slice)
            model, history = train_model(
                training_tasks,
                variant=variant,
                seed=seed + fold * 1000 + offset,
                steps=800,
                episodes_per_step=4,
                model_kwargs=variant_kwargs,
            )
            model_results[variant] = score_model(
                model, evaluation, seed=seed + fold * 1000 + offset
            )
            torch.save(model.state_dict(), fold_root / f"{variant}.pt")
            write_metrics(fold_root / f"{variant}-history.json", {"history": history})
        print(f"seed={seed} fold={fold} training=tnpd_airt", flush=True)
        tnpd, history = train_tnpd(
            training_tasks,
            seed=seed + fold * 1000 + 4,
            steps=800,
            episodes_per_step=4,
            subject_feature_slice=(389, 773),
        )
        model_results["tnpd_airt"] = score_tnpd(tnpd, evaluation)
        torch.save(tnpd.state_dict(), fold_root / "tnpd_airt.pt")
        write_metrics(fold_root / "tnpd_airt-history.json", {"history": history})
        # Identical beta-binomial control, evaluated on this fold's fixed context order.
        y_T = np.asarray(evaluation["y_T"], dtype=float)
        beta = {}
        for budget in (0, 1, 3, 7, 15, 31):
            y_C = evaluation["y_C_pool"][:budget]
            probability = (2.0 + float(y_C.sum())) / (4.0 + len(y_C))
            beta[str(budget)] = {
                "brier": float(np.mean((probability - y_T) ** 2)),
                "mean_probability": float(probability),
                "mean_epistemic_variance": 0.0,
            }
        beta["brier_alc"] = float(
            sum(
                weight * beta[str(budget)]["brier"]
                for budget, weight in zip(
                    (0, 1, 3, 7, 15, 31), (0.1, 0.2, 0.2, 0.2, 0.2, 0.1), strict=True
                )
            )
        )
        model_results["beta_binomial"] = beta
        folds.append({"fold": fold, "heldout_task": heldout, "models": model_results})
    result = {
        "status": "faithful_three_seed_experiment",
        "seed": seed,
        "corpus": manifest,
        "protocol": {
            "outer_split": "four-fold leave-one-BeaverTails-subject-out",
            "features": "MiniLM item + subject + benchmark embeddings and five item metadata scalars",
            "training_signature": signature,
            "meta_training_tasks": len(meta_tasks),
            "training_steps": 800,
            "episodes_per_step": 4,
            "n_z_samples_train": 4,
            "n_z_samples_evaluation": 256,
            "np_context_likelihood_weight": 0.5,
            "np_target_likelihood_weight": 0.5,
            "np_kl_weight": 1.0,
            "np_residual_scale": 0.0,
            "anp_residual_scale": 0.1,
            "np_map_prior_variance": 4.0,
            "tnp_variant": "TfNP-D with exact Bernoulli likelihood and learned budget-zero prior token",
            "budgets": [0, 1, 3, 7, 15, 31],
        },
        "folds": folds,
    }
    write_metrics(seed_root / "metrics.json", result)
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32_768,
    volumes={str(REMOTE_ROOT): volume},
    timeout=2 * 60 * 60,
)
def run_cnp_seed(seed: int) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch

    from aims_starter.faithful_np import score_cnp, train_cnp, write_metrics

    output = CNP_RESULT_ROOT / f"seed-{seed}.json"
    signature = _training_signature()
    existing = _read_json(output)
    if existing is not None:
        if existing.get("training_signature") != signature:
            raise RuntimeError(
                "cached CNP metrics were produced by different training code or protocol; "
                f"bump CNP_RESULT_ROOT or remove {output} before rerunning"
            )
        return existing
    items = pd.read_parquet(CORPUS_ROOT / "items.parquet")
    subjects = pd.read_parquet(CORPUS_ROOT / "subjects.parquet")
    benchmarks = pd.read_parquet(CORPUS_ROOT / "benchmarks.parquet")
    feature_state = {
        "item_to_index": dict(zip(items["item_uid"].astype(str), range(len(items)), strict=True)),
        "subject_to_index": dict(
            zip(subjects["subject_uid"].astype(str), range(len(subjects)), strict=True)
        ),
        "benchmark_to_index": dict(
            zip(benchmarks["benchmark_id"].astype(str), range(len(benchmarks)), strict=True)
        ),
        "item_embeddings": np.load(CORPUS_ROOT / "item_embeddings.npy"),
        "item_numeric": np.load(CORPUS_ROOT / "item_numeric.npy"),
        "subject_embeddings": np.load(CORPUS_ROOT / "subject_embeddings.npy"),
        "benchmark_embeddings": np.load(CORPUS_ROOT / "benchmark_embeddings.npy"),
    }
    meta = pd.read_parquet(CORPUS_ROOT / "meta_observations.parquet")
    beaver = pd.read_parquet(CORPUS_ROOT / "beaver_observations.parquet")
    assignments = pd.read_parquet(CORPUS_ROOT / "beaver_episode_items.parquet")
    observation_lookup = beaver.set_index("observation_id", drop=False)
    meta_tasks = _task_arrays(meta, feature_state)
    beaver_tasks = {task["task_id"]: task for task in _task_arrays(beaver, feature_state)}
    folds = []
    for fold, heldout in enumerate(sorted(beaver_tasks)):
        training_tasks = meta_tasks + [task for key, task in beaver_tasks.items() if key != heldout]
        evaluation = _evaluation_task(
            heldout, beaver, assignments, observation_lookup, feature_state
        )
        model, history = train_cnp(
            training_tasks,
            seed=seed + fold * 1000,
            steps=800,
            episodes_per_step=4,
            subject_feature_slice=(389, 773),
        )
        folds.append(
            {
                "fold": fold,
                "heldout_task": heldout,
                "models": {"cnp_airt": score_cnp(model, evaluation)},
            }
        )
        checkpoint_root = CNP_RESULT_ROOT / f"seed-{seed}-checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_root / f"fold-{fold}.pt")
        write_metrics(checkpoint_root / f"fold-{fold}-history.json", {"history": history})
    result = {"seed": seed, "training_signature": signature, "folds": folds}
    CNP_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_metrics(output, result)
    volume.commit()
    return result


@app.function(image=image, cpu=2, memory=4096, volumes={str(REMOTE_ROOT): volume})
def aggregate_cnp() -> dict[str, Any]:
    import numpy as np

    results = [_read_json(CNP_RESULT_ROOT / f"seed-{seed}.json") for seed in SEEDS]
    budgets = (0, 1, 3, 7, 15, 31)
    by_budget = {}
    for budget in budgets:
        rows = [
            fold["models"]["cnp_airt"][str(budget)]
            for result in results
            for fold in result["folds"]
        ]
        by_budget[str(budget)] = {
            metric: float(np.mean([row[metric] for row in rows])) for metric in ("brier", "ece")
        }
    seed_alc = [
        np.mean([fold["models"]["cnp_airt"]["brier_alc"] for fold in result["folds"]])
        for result in results
    ]
    output = {
        "aggregate": {"cnp_airt": {"brier_alc": float(np.mean(seed_alc)), "by_budget": by_budget}}
    }
    CNP_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (CNP_RESULT_ROOT / "metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    volume.commit()
    return output


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32_768,
    volumes={str(REMOTE_ROOT): volume},
    timeout=60 * 60,
)
def rescore_seed_calibration(seed: int) -> dict[str, Any]:
    """Rescore saved checkpoints with ECE; do not retrain."""

    import numpy as np
    import pandas as pd
    import torch

    from aims_starter.faithful_np import (
        TNPDAIRT,
        Amortized2PL,
        FaithfulNPAIRT,
        score_amortized_2pl,
        score_model,
        score_tnpd,
        write_metrics,
    )
    from aims_starter.metrics import DEFAULT_BUDGETS, DEFAULT_WEIGHTS, calibration_ece

    seed_root = RESULT_ROOT / f"seed-{seed}"
    output = seed_root / "calibration.json"
    existing = _read_json(output)
    if existing is not None:
        return existing
    items = pd.read_parquet(CORPUS_ROOT / "items.parquet")
    subjects = pd.read_parquet(CORPUS_ROOT / "subjects.parquet")
    benchmarks = pd.read_parquet(CORPUS_ROOT / "benchmarks.parquet")
    feature_state = {
        "item_to_index": dict(zip(items["item_uid"].astype(str), range(len(items)), strict=True)),
        "subject_to_index": dict(
            zip(subjects["subject_uid"].astype(str), range(len(subjects)), strict=True)
        ),
        "benchmark_to_index": dict(
            zip(benchmarks["benchmark_id"].astype(str), range(len(benchmarks)), strict=True)
        ),
        "item_embeddings": np.load(CORPUS_ROOT / "item_embeddings.npy"),
        "item_numeric": np.load(CORPUS_ROOT / "item_numeric.npy"),
        "subject_embeddings": np.load(CORPUS_ROOT / "subject_embeddings.npy"),
        "benchmark_embeddings": np.load(CORPUS_ROOT / "benchmark_embeddings.npy"),
    }
    beaver = pd.read_parquet(CORPUS_ROOT / "beaver_observations.parquet")
    assignments = pd.read_parquet(CORPUS_ROOT / "beaver_episode_items.parquet")
    observation_lookup = beaver.set_index("observation_id", drop=False)
    task_ids = sorted(beaver["task_id"].astype(str).unique())
    x_dim = int(next(iter(feature_state["item_embeddings"])).shape[0] + 5 + 384 + 384)
    folds = []
    for fold, heldout in enumerate(task_ids):
        evaluation = _evaluation_task(
            heldout, beaver, assignments, observation_lookup, feature_state
        )
        fold_root = seed_root / f"fold-{fold}"
        models = {}
        state = torch.load(fold_root / "online_airt_2pl.pt", map_location="cuda", weights_only=True)
        airt = Amortized2PL(x_dim, len(state["theta"]), subject_feature_slice=(389, 773)).cuda()
        airt.load_state_dict(state)
        models["online_airt_2pl"] = score_amortized_2pl(airt, evaluation)
        for offset, variant in enumerate(("np", "anp", "stacked_mhsa_anp"), start=1):
            model = FaithfulNPAIRT(
                x_dim, variant=variant, **_variant_model_kwargs(variant, (389, 773))
            ).cuda()
            model.load_state_dict(
                torch.load(fold_root / f"{variant}.pt", map_location="cuda", weights_only=True)
            )
            models[variant] = score_model(model, evaluation, seed=seed + fold * 1000 + offset)
        tnpd = TNPDAIRT(x_dim, subject_feature_slice=(389, 773)).cuda()
        tnpd.load_state_dict(
            torch.load(fold_root / "tnpd_airt.pt", map_location="cuda", weights_only=True)
        )
        models["tnpd_airt"] = score_tnpd(tnpd, evaluation)
        y_T = np.asarray(evaluation["y_T"], dtype=float)
        beta = {}
        for budget in DEFAULT_BUDGETS:
            y_C = evaluation["y_C_pool"][:budget]
            probability = (2.0 + float(y_C.sum())) / (4.0 + len(y_C))
            p_y_T = np.full(len(y_T), probability)
            beta[str(budget)] = {
                "brier": float(np.mean((p_y_T - y_T) ** 2)),
                "ece": calibration_ece(y_T.tolist(), p_y_T.tolist()),
            }
        beta["brier_alc"] = float(
            sum(
                weight * beta[str(budget)]["brier"]
                for budget, weight in zip(DEFAULT_BUDGETS, DEFAULT_WEIGHTS, strict=True)
            )
        )
        models["beta_binomial"] = beta
        folds.append({"fold": fold, "heldout_task": heldout, "models": models})
    result = {"seed": seed, "folds": folds}
    write_metrics(output, result)
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    volumes={str(REMOTE_ROOT): volume},
    timeout=10 * 60,
)
def aggregate_calibration() -> dict[str, Any]:
    import numpy as np

    seed_results = [_read_json(RESULT_ROOT / f"seed-{seed}" / "calibration.json") for seed in SEEDS]
    if any(result is None for result in seed_results):
        raise FileNotFoundError("calibration rescore incomplete")
    models = seed_results[0]["folds"][0]["models"]
    aggregate = {}
    for model in models:
        by_budget = {}
        for budget in (0, 1, 3, 7, 15, 31):
            rows = [
                fold["models"][model][str(budget)]
                for result in seed_results
                for fold in result["folds"]
            ]
            by_budget[str(budget)] = {
                metric: float(np.mean([row[metric] for row in rows])) for metric in ("brier", "ece")
            }
            by_budget[str(budget)]["brier_se"] = float(
                np.std([row["brier"] for row in rows], ddof=1) / np.sqrt(len(rows))
            )
            by_budget[str(budget)]["ece_se"] = float(
                np.std([row["ece"] for row in rows], ddof=1) / np.sqrt(len(rows))
            )
        seed_alc = [
            np.mean([fold["models"][model]["brier_alc"] for fold in result["folds"]])
            for result in seed_results
        ]
        aggregate[model] = {
            "brier_alc": float(np.mean(seed_alc)),
            "brier_alc_seed_std": float(np.std(seed_alc, ddof=1)),
            "by_budget": by_budget,
        }
    result = {"status": "calibration_rescore", "aggregate": aggregate}
    (RESULT_ROOT / "metrics-calibration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32_768,
    volumes={str(REMOTE_ROOT): volume},
    timeout=2 * 60 * 60,
)
def rescore_prior_seed(seed: int) -> dict[str, Any]:
    """Rescore saved checkpoints with a training-task ability prior mean.

    The prior mean is estimated from the training tasks only: a benchmark-level
    mean ability plus a subject deviation regressed on the subject metadata
    embedding. It enters the MAP anchor of the 2PL ability and the initialization
    of the amortized baseline, so the trained item banks are untouched.
    """
    import numpy as np
    import pandas as pd
    import torch

    from aims_starter.adaptation import fit_ability_prior, map_ability
    from aims_starter.faithful_np import (
        CNPAIRT,
        TNPDAIRT,
        Amortized2PL,
        FaithfulNPAIRT,
        score_amortized_2pl,
        score_cnp,
        score_model,
        score_tnpd,
        write_metrics,
    )
    from aims_starter.metrics import DEFAULT_BUDGETS, DEFAULT_WEIGHTS, calibration_ece

    output = RESULT_ROOT / f"seed-{seed}" / "prior.json"
    existing = _read_json(output)
    if existing is not None:
        return existing
    manifest = _read_json(CORPUS_ROOT / "manifest.json")
    if manifest is None or "feature_slices" not in manifest:
        raise FileNotFoundError("faithful corpus manifest is missing feature slices")
    subject_feature_slice = tuple(manifest["feature_slices"]["subject"])
    items = pd.read_parquet(CORPUS_ROOT / "items.parquet")
    subjects = pd.read_parquet(CORPUS_ROOT / "subjects.parquet")
    benchmarks = pd.read_parquet(CORPUS_ROOT / "benchmarks.parquet")
    feature_state = {
        "item_to_index": dict(zip(items["item_uid"].astype(str), range(len(items)), strict=True)),
        "subject_to_index": dict(
            zip(subjects["subject_uid"].astype(str), range(len(subjects)), strict=True)
        ),
        "benchmark_to_index": dict(
            zip(benchmarks["benchmark_id"].astype(str), range(len(benchmarks)), strict=True)
        ),
        "item_embeddings": np.load(CORPUS_ROOT / "item_embeddings.npy"),
        "item_numeric": np.load(CORPUS_ROOT / "item_numeric.npy"),
        "subject_embeddings": np.load(CORPUS_ROOT / "subject_embeddings.npy"),
        "benchmark_embeddings": np.load(CORPUS_ROOT / "benchmark_embeddings.npy"),
    }
    meta = pd.read_parquet(CORPUS_ROOT / "meta_observations.parquet")
    beaver = pd.read_parquet(CORPUS_ROOT / "beaver_observations.parquet")
    assignments = pd.read_parquet(CORPUS_ROOT / "beaver_episode_items.parquet")
    observation_lookup = beaver.set_index("observation_id", drop=False)
    meta_tasks = _task_arrays(meta, feature_state)
    beaver_tasks = {task["task_id"]: task for task in _task_arrays(beaver, feature_state)}
    task_ids = sorted(beaver_tasks)
    subject_embeddings = feature_state["subject_embeddings"]
    subject_to_index = feature_state["subject_to_index"]
    x_dim = int(next(iter(feature_state["item_embeddings"])).shape[0] + 5 + 384 + 384)

    def split_task(task_id: str) -> tuple[str, str]:
        subject, benchmark = task_id.rsplit("::", 1)
        return subject, benchmark

    def task_ability(model: Any, task: dict[str, Any], *, amortized: bool) -> float:
        x = torch.as_tensor(task["x"], dtype=torch.float32, device="cuda")
        y = torch.as_tensor(task["y"], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            if amortized:
                a, b = model.item_parameters(x)
                return map_ability(
                    a.cpu().numpy(), b.cpu().numpy(), y.cpu().numpy(), prior_mean=0.0
                )
            return float(model.map_theta(model.encode_item_x(x), y, prior_mean=0.0))

    folds = []
    for fold, heldout in enumerate(task_ids):
        training_tasks = meta_tasks + [task for key, task in beaver_tasks.items() if key != heldout]
        evaluation = _evaluation_task(
            heldout, beaver, assignments, observation_lookup, feature_state
        )
        fold_root = RESULT_ROOT / f"seed-{seed}" / f"fold-{fold}"
        cnp_root = CNP_RESULT_ROOT / f"seed-{seed}-checkpoints"
        models: dict[str, tuple[Any, bool, int | None]] = {}
        state = torch.load(fold_root / "online_airt_2pl.pt", map_location="cuda", weights_only=True)
        airt = Amortized2PL(
            x_dim, len(state["theta"]), subject_feature_slice=subject_feature_slice
        ).cuda()
        airt.load_state_dict(state)
        models["online_airt_2pl"] = (airt, True, None)
        for offset, variant in enumerate(("np", "anp", "stacked_mhsa_anp"), start=1):
            model = FaithfulNPAIRT(
                x_dim, variant=variant, **_variant_model_kwargs(variant, subject_feature_slice)
            ).cuda()
            model.load_state_dict(
                torch.load(fold_root / f"{variant}.pt", map_location="cuda", weights_only=True)
            )
            models[variant] = (model, False, seed + fold * 1000 + offset)
        tnpd = TNPDAIRT(x_dim, subject_feature_slice=subject_feature_slice).cuda()
        tnpd.load_state_dict(
            torch.load(fold_root / "tnpd_airt.pt", map_location="cuda", weights_only=True)
        )
        models["tnpd_airt"] = (tnpd, False, None)
        cnp = CNPAIRT(x_dim, subject_feature_slice=subject_feature_slice).cuda()
        cnp.load_state_dict(
            torch.load(cnp_root / f"fold-{fold}.pt", map_location="cuda", weights_only=True)
        )
        models["cnp_airt"] = (cnp, False, None)

        heldout_subject, heldout_benchmark = split_task(heldout)
        embedding = subject_embeddings[subject_to_index[heldout_subject]]
        arms: dict[str, dict[str, dict[str, Any]]] = {"none": {}, "benchmark": {}, "subject": {}}
        diagnostics = {}
        for name, (model, amortized, score_seed) in models.items():
            abilities = np.asarray(
                [task_ability(model, task, amortized=amortized) for task in training_tasks]
            )
            prior = fit_ability_prior(
                abilities,
                [split_task(task["task_id"])[1] for task in training_tasks],
                [split_task(task["task_id"])[0] for task in training_tasks],
                subject_embeddings,
                subject_to_index,
            )
            prior_means = {
                "none": 0.0,
                "benchmark": prior.predict(heldout_benchmark, embedding, subject=False),
                "subject": prior.predict(heldout_benchmark, embedding, subject=True),
            }
            diagnostics[name] = {
                "benchmark_mean": prior_means["benchmark"],
                "subject_mean": prior_means["subject"],
                "heldout_benchmark_mean": prior.benchmark_mean.get(heldout_benchmark),
                "alpha": float(prior.alpha) if np.isfinite(prior.alpha) else None,
                "loo_mse": prior.loo_mse,
                "n_fit": prior.n_fit,
            }
            for arm, prior_mean in prior_means.items():
                if amortized:
                    arms[arm][name] = score_amortized_2pl(model, evaluation, prior_mean=prior_mean)
                elif name == "cnp_airt":
                    arms[arm][name] = score_cnp(model, evaluation, prior_mean=prior_mean)
                elif name == "tnpd_airt":
                    arms[arm][name] = score_tnpd(model, evaluation, prior_mean=prior_mean)
                else:
                    arms[arm][name] = score_model(
                        model, evaluation, seed=score_seed, prior_mean=prior_mean
                    )
        y_T = np.asarray(evaluation["y_T"], dtype=float)
        beta = {}
        for budget in DEFAULT_BUDGETS:
            y_C = evaluation["y_C_pool"][:budget]
            probability = (2.0 + float(y_C.sum())) / (4.0 + len(y_C))
            p_y_T = np.full(len(y_T), probability)
            beta[str(budget)] = {
                "brier": float(np.mean((p_y_T - y_T) ** 2)),
                "ece": calibration_ece(y_T.tolist(), p_y_T.tolist()),
            }
        beta["brier_alc"] = float(
            sum(
                weight * beta[str(budget)]["brier"]
                for budget, weight in zip(DEFAULT_BUDGETS, DEFAULT_WEIGHTS, strict=True)
            )
        )
        for arm in arms.values():
            arm["beta_binomial"] = beta
        folds.append(
            {
                "fold": fold,
                "heldout_task": heldout,
                "arms": arms,
                "priors": diagnostics,
            }
        )
    result = {"seed": seed, "status": "ability_prior_rescore", "folds": folds}
    write_metrics(output, result)
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    volumes={str(REMOTE_ROOT): volume},
    timeout=10 * 60,
)
def aggregate_prior() -> dict[str, Any]:
    """Aggregate the ability-prior rescore across seeds."""

    import numpy as np

    seed_results = [_read_json(RESULT_ROOT / f"seed-{seed}" / "prior.json") for seed in SEEDS]
    if any(result is None for result in seed_results):
        raise FileNotFoundError("ability prior rescore incomplete")
    models = list(seed_results[0]["folds"][0]["arms"]["none"])
    aggregate: dict[str, dict[str, Any]] = {}
    for arm in ("none", "benchmark", "subject"):
        aggregate[arm] = {}
        for model in models:
            by_budget = {}
            for budget in (0, 1, 3, 7, 15, 31):
                rows = [
                    fold["arms"][arm][model][str(budget)]
                    for result in seed_results
                    for fold in result["folds"]
                ]
                metrics = [metric for metric in ("brier", "ece", "auc") if metric in rows[0]]
                by_budget[str(budget)] = {
                    metric: float(np.mean([row[metric] for row in rows])) for metric in metrics
                }
                by_budget[str(budget)]["brier_se"] = float(
                    np.std([row["brier"] for row in rows], ddof=1) / np.sqrt(len(rows))
                )
            seed_alc = [
                np.mean([fold["arms"][arm][model]["brier_alc"] for fold in result["folds"]])
                for result in seed_results
            ]
            aggregate[arm][model] = {
                "brier_alc": float(np.mean(seed_alc)),
                "brier_alc_seed_std": float(np.std(seed_alc, ddof=1)),
                "by_budget": by_budget,
            }
    priors = [
        {
            "seed": result["seed"],
            "fold": fold["fold"],
            "heldout_task": fold["heldout_task"],
            **fold["priors"],
        }
        for result in seed_results
        for fold in result["folds"]
    ]
    artifact = {"status": "ability_prior_rescore", "aggregate": aggregate, "priors": priors}
    (RESULT_ROOT / "metrics-prior.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    volume.commit()
    return artifact


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32_768,
    volumes={str(REMOTE_ROOT): volume},
    timeout=2 * 60 * 60,
)
def rescore_acquisition_orders(seed: int) -> dict[str, Any]:
    """Rescore saved checkpoints over ten reveal orders without retraining."""
    import numpy as np
    import pandas as pd
    import torch

    from aims_starter.faithful_np import (
        TNPDAIRT,
        Amortized2PL,
        FaithfulNPAIRT,
        score_amortized_2pl,
        score_model,
        score_tnpd,
        write_metrics,
    )
    from aims_starter.metrics import DEFAULT_BUDGETS, DEFAULT_WEIGHTS

    seed_root = RESULT_ROOT / f"seed-{seed}"
    output = seed_root / "acquisition-orders.json"
    existing = _read_json(output)
    if existing is not None:
        return existing
    items = pd.read_parquet(CORPUS_ROOT / "items.parquet")
    subjects = pd.read_parquet(CORPUS_ROOT / "subjects.parquet")
    benchmarks = pd.read_parquet(CORPUS_ROOT / "benchmarks.parquet")
    feature_state = {
        "item_to_index": dict(zip(items["item_uid"].astype(str), range(len(items)), strict=True)),
        "subject_to_index": dict(
            zip(subjects["subject_uid"].astype(str), range(len(subjects)), strict=True)
        ),
        "benchmark_to_index": dict(
            zip(benchmarks["benchmark_id"].astype(str), range(len(benchmarks)), strict=True)
        ),
        "item_embeddings": np.load(CORPUS_ROOT / "item_embeddings.npy"),
        "item_numeric": np.load(CORPUS_ROOT / "item_numeric.npy"),
        "subject_embeddings": np.load(CORPUS_ROOT / "subject_embeddings.npy"),
        "benchmark_embeddings": np.load(CORPUS_ROOT / "benchmark_embeddings.npy"),
    }
    beaver = pd.read_parquet(CORPUS_ROOT / "beaver_observations.parquet")
    assignments = pd.read_parquet(CORPUS_ROOT / "beaver_episode_items.parquet")
    observation_lookup = beaver.set_index("observation_id", drop=False)
    task_ids = sorted(beaver["task_id"].astype(str).unique())
    x_dim = sum(
        array.shape[1]
        for array in (
            feature_state["item_embeddings"],
            feature_state["item_numeric"],
            feature_state["subject_embeddings"],
            feature_state["benchmark_embeddings"],
        )
    )
    folds = []
    for fold, heldout in enumerate(task_ids):
        base = _evaluation_task(heldout, beaver, assignments, observation_lookup, feature_state)
        fold_root = seed_root / f"fold-{fold}"
        state = torch.load(fold_root / "online_airt_2pl.pt", map_location="cuda", weights_only=True)
        models = {
            "online_airt_2pl": Amortized2PL(
                x_dim, len(state["theta"]), subject_feature_slice=(389, 773)
            ).cuda()
        }
        models["online_airt_2pl"].load_state_dict(state)
        for variant in ("np", "anp", "stacked_mhsa_anp"):
            models[variant] = FaithfulNPAIRT(
                x_dim, variant=variant, **_variant_model_kwargs(variant, (389, 773))
            ).cuda()
            models[variant].load_state_dict(
                torch.load(fold_root / f"{variant}.pt", map_location="cuda", weights_only=True)
            )
        models["tnpd_airt"] = TNPDAIRT(x_dim, subject_feature_slice=(389, 773)).cuda()
        models["tnpd_airt"].load_state_dict(
            torch.load(fold_root / "tnpd_airt.pt", map_location="cuda", weights_only=True)
        )
        orders = []
        for order_seed in ACQUISITION_ORDER_SEEDS:
            evaluation = _reorder_evaluation(base, task_id=heldout, order_seed=order_seed)
            scores = {
                "online_airt_2pl": score_amortized_2pl(models["online_airt_2pl"], evaluation),
                "tnpd_airt": score_tnpd(models["tnpd_airt"], evaluation),
            }
            for offset, variant in enumerate(("np", "anp", "stacked_mhsa_anp"), start=1):
                scores[variant] = score_model(
                    models[variant],
                    evaluation,
                    seed=seed + fold * 1000 + order_seed * 10 + offset,
                )
            y_T = np.asarray(evaluation["y_T"], dtype=float)
            beta = {}
            for budget in DEFAULT_BUDGETS:
                y_C = evaluation["y_C_pool"][:budget]
                probability = (2.0 + float(y_C.sum())) / (4.0 + len(y_C))
                beta[str(budget)] = {"brier": float(np.mean((probability - y_T) ** 2))}
            beta["brier_alc"] = float(
                sum(
                    weight * beta[str(budget)]["brier"]
                    for budget, weight in zip(DEFAULT_BUDGETS, DEFAULT_WEIGHTS, strict=True)
                )
            )
            scores["beta_binomial"] = beta
            orders.append({"order_seed": order_seed, "models": scores})
        folds.append({"fold": fold, "heldout_task": heldout, "orders": orders})
    result = {"seed": seed, "order_seeds": list(ACQUISITION_ORDER_SEEDS), "folds": folds}
    write_metrics(output, result)
    volume.commit()
    return result


@app.function(image=image, cpu=2, memory=4096, volumes={str(REMOTE_ROOT): volume})
def aggregate_acquisition_orders() -> dict[str, Any]:
    """Average each model over folds/seeds within each shared reveal order."""
    import numpy as np

    results = [
        _read_json(RESULT_ROOT / f"seed-{seed}" / "acquisition-orders.json") for seed in SEEDS
    ]
    if any(result is None for result in results):
        raise FileNotFoundError("acquisition-order rescoring is incomplete")
    models = results[0]["folds"][0]["orders"][0]["models"]
    aggregate = {}
    for model in models:
        order_alc = []
        by_budget = {}
        for order_seed in ACQUISITION_ORDER_SEEDS:
            rows = [
                order["models"][model]
                for result in results
                for fold in result["folds"]
                for order in fold["orders"]
                if order["order_seed"] == order_seed
            ]
            order_alc.append(float(np.mean([row["brier_alc"] for row in rows])))
        for budget in (0, 1, 3, 7, 15, 31):
            order_scores = []
            for order_seed in ACQUISITION_ORDER_SEEDS:
                values = [
                    order["models"][model][str(budget)]["brier"]
                    for result in results
                    for fold in result["folds"]
                    for order in fold["orders"]
                    if order["order_seed"] == order_seed
                ]
                order_scores.append(float(np.mean(values)))
            by_budget[str(budget)] = {
                "brier": float(np.mean(order_scores)),
                "brier_order_std": float(np.std(order_scores, ddof=1)),
            }
        aggregate[model] = {
            "brier_alc": float(np.mean(order_alc)),
            "brier_alc_order_std": float(np.std(order_alc, ddof=1)),
            "order_brier_alc": order_alc,
            "by_budget": by_budget,
        }
    result = {
        "status": "ten_shared_acquisition_order_rescore",
        "order_seeds": list(ACQUISITION_ORDER_SEEDS),
        "aggregate": aggregate,
    }
    (RESULT_ROOT / "metrics-acquisition-orders.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32_768,
    volumes={str(REMOTE_ROOT): volume},
    timeout=60 * 60,
)
def rescore_cnp_acquisition_orders(seed: int) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch

    from aims_starter.faithful_np import CNPAIRT, score_cnp, write_metrics

    output = CNP_RESULT_ROOT / f"seed-{seed}-acquisition-orders.json"
    existing = _read_json(output)
    if existing is not None:
        return existing
    items = pd.read_parquet(CORPUS_ROOT / "items.parquet")
    subjects = pd.read_parquet(CORPUS_ROOT / "subjects.parquet")
    benchmarks = pd.read_parquet(CORPUS_ROOT / "benchmarks.parquet")
    feature_state = {
        "item_to_index": dict(zip(items["item_uid"].astype(str), range(len(items)), strict=True)),
        "subject_to_index": dict(
            zip(subjects["subject_uid"].astype(str), range(len(subjects)), strict=True)
        ),
        "benchmark_to_index": dict(
            zip(benchmarks["benchmark_id"].astype(str), range(len(benchmarks)), strict=True)
        ),
        "item_embeddings": np.load(CORPUS_ROOT / "item_embeddings.npy"),
        "item_numeric": np.load(CORPUS_ROOT / "item_numeric.npy"),
        "subject_embeddings": np.load(CORPUS_ROOT / "subject_embeddings.npy"),
        "benchmark_embeddings": np.load(CORPUS_ROOT / "benchmark_embeddings.npy"),
    }
    x_dim = sum(
        array.shape[1]
        for array in (
            feature_state["item_embeddings"],
            feature_state["item_numeric"],
            feature_state["subject_embeddings"],
            feature_state["benchmark_embeddings"],
        )
    )
    beaver = pd.read_parquet(CORPUS_ROOT / "beaver_observations.parquet")
    assignments = pd.read_parquet(CORPUS_ROOT / "beaver_episode_items.parquet")
    lookup = beaver.set_index("observation_id", drop=False)
    folds = []
    for fold, heldout in enumerate(sorted(beaver["task_id"].astype(str).unique())):
        base = _evaluation_task(heldout, beaver, assignments, lookup, feature_state)
        model = CNPAIRT(x_dim, subject_feature_slice=(389, 773)).cuda()
        model.load_state_dict(
            torch.load(
                CNP_RESULT_ROOT / f"seed-{seed}-checkpoints" / f"fold-{fold}.pt",
                map_location="cuda",
                weights_only=True,
            )
        )
        orders = [
            {
                "order_seed": order_seed,
                "models": {
                    "cnp_airt": score_cnp(
                        model, _reorder_evaluation(base, task_id=heldout, order_seed=order_seed)
                    )
                },
            }
            for order_seed in ACQUISITION_ORDER_SEEDS
        ]
        folds.append({"fold": fold, "heldout_task": heldout, "orders": orders})
    result = {"seed": seed, "order_seeds": list(ACQUISITION_ORDER_SEEDS), "folds": folds}
    write_metrics(output, result)
    volume.commit()
    return result


@app.function(image=image, cpu=2, memory=4096, volumes={str(REMOTE_ROOT): volume})
def aggregate_cnp_acquisition_orders() -> dict[str, Any]:
    import numpy as np

    results = [
        _read_json(CNP_RESULT_ROOT / f"seed-{seed}-acquisition-orders.json") for seed in SEEDS
    ]
    if any(result is None for result in results):
        raise FileNotFoundError("CNP acquisition-order rescoring is incomplete")
    order_alc, by_budget = [], {}
    for order_seed in ACQUISITION_ORDER_SEEDS:
        rows = [
            order["models"]["cnp_airt"]
            for result in results
            for fold in result["folds"]
            for order in fold["orders"]
            if order["order_seed"] == order_seed
        ]
        order_alc.append(float(np.mean([row["brier_alc"] for row in rows])))
    for budget in (0, 1, 3, 7, 15, 31):
        scores = []
        for order_seed in ACQUISITION_ORDER_SEEDS:
            values = [
                order["models"]["cnp_airt"][str(budget)]["brier"]
                for result in results
                for fold in result["folds"]
                for order in fold["orders"]
                if order["order_seed"] == order_seed
            ]
            scores.append(float(np.mean(values)))
        by_budget[str(budget)] = {
            "brier": float(np.mean(scores)),
            "brier_order_std": float(np.std(scores, ddof=1)),
        }
    result = {
        "aggregate": {
            "cnp_airt": {
                "brier_alc": float(np.mean(order_alc)),
                "brier_alc_order_std": float(np.std(order_alc, ddof=1)),
                "order_brier_alc": order_alc,
                "by_budget": by_budget,
            }
        }
    }
    (CNP_RESULT_ROOT / "metrics-acquisition-orders.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    volumes={str(REMOTE_ROOT): volume},
    timeout=30 * 60,
)
def aggregate_results() -> dict[str, Any]:
    import numpy as np

    seed_results = [_read_json(RESULT_ROOT / f"seed-{seed}" / "metrics.json") for seed in SEEDS]
    if any(result is None for result in seed_results):
        raise FileNotFoundError("one or more seed results are missing")
    budgets = (0, 1, 3, 7, 15, 31)
    model_names = list(seed_results[0]["folds"][0]["models"])
    aggregate = {}
    for model in model_names:
        by_budget = {}
        for budget in budgets:
            values = [
                fold["models"][model][str(budget)]["brier"]
                for result in seed_results
                for fold in result["folds"]
            ]
            by_budget[str(budget)] = {
                "brier": float(np.mean(values)),
                "brier_std": float(np.std(values, ddof=1)),
                "brier_se": float(np.std(values, ddof=1) / np.sqrt(len(values))),
                "n": len(values),
            }
        seed_alc = []
        for result in seed_results:
            fold_alc = [fold["models"][model]["brier_alc"] for fold in result["folds"]]
            seed_alc.append(float(np.mean(fold_alc)))
        aggregate[model] = {
            "brier_alc": float(np.mean(seed_alc)),
            "brier_alc_seed_std": float(np.std(seed_alc, ddof=1)),
            "seed_brier_alc": seed_alc,
            "by_budget": by_budget,
        }
    result = {
        "status": "faithful_three_seed_experiment",
        "seeds": list(SEEDS),
        "folds_per_seed": 4,
        "aggregate": aggregate,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    smoke_only: bool = False,
    inspect_features: bool = False,
    calibration_only: bool = False,
    prior_only: bool = False,
    cnp_only: bool = False,
    acquisition_orders_only: bool = False,
    cnp_acquisition_orders_only: bool = False,
) -> None:
    if smoke_only:
        print(json.dumps(smoke_symbols.remote(), indent=2, sort_keys=True))
        return
    if inspect_features:
        print(json.dumps(inspect_feature_sources.remote(), indent=2, sort_keys=True))
        return
    if calibration_only:
        list(rescore_seed_calibration.map(SEEDS, order_outputs=True))
        print(json.dumps(aggregate_calibration.remote(), indent=2, sort_keys=True))
        return
    if prior_only:
        list(rescore_prior_seed.map(SEEDS, order_outputs=True))
        print(json.dumps(aggregate_prior.remote(), indent=2, sort_keys=True))
        return
    if cnp_only:
        list(run_cnp_seed.map(SEEDS, order_outputs=True))
        print(json.dumps(aggregate_cnp.remote(), indent=2, sort_keys=True))
        return
    if acquisition_orders_only:
        list(rescore_acquisition_orders.map(SEEDS, order_outputs=True))
        print(json.dumps(aggregate_acquisition_orders.remote(), indent=2, sort_keys=True))
        return
    if cnp_acquisition_orders_only:
        list(rescore_cnp_acquisition_orders.map(SEEDS, order_outputs=True))
        print(json.dumps(aggregate_cnp_acquisition_orders.remote(), indent=2, sort_keys=True))
        return
    print(json.dumps(prepare_corpus.remote(), indent=2, sort_keys=True))
    seed_results = list(run_seed.map(SEEDS, order_outputs=True))
    print(json.dumps({"completed_seeds": [result["seed"] for result in seed_results]}, indent=2))
    print(json.dumps(aggregate_results.remote(), indent=2, sort_keys=True))
