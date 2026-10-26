"""Small controlled ablation for NP latent collapse on one BeaverTails fold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "aims-neurips-np-collapse"
REMOTE_ROOT = Path("/data")
CORPUS_ROOT = REMOTE_ROOT / "datasets/full/faithful-pilot-beavertails-v2-features"
RESULT_ROOT = REMOTE_ROOT / "results/np-collapse-ablation-v1"
SEED = 20261019
CONFIGS = (
    {"name": "current", "d_z": 64, "kl_weight": 1.0, "anchor_scale": 1.0},
    {"name": "scalar", "d_z": 1, "kl_weight": 1.0, "anchor_scale": 1.0},
    {"name": "scalar_kl01", "d_z": 1, "kl_weight": 0.1, "anchor_scale": 1.0},
    {"name": "scalar_kl001", "d_z": 1, "kl_weight": 0.01, "anchor_scale": 1.0},
    {"name": "latent_only_kl01", "d_z": 1, "kl_weight": 0.1, "anchor_scale": 0.0},
    {"name": "latent_only_kl001", "d_z": 1, "kl_weight": 0.01, "anchor_scale": 0.0},
    {
        "name": "latent_only_no_subject_kl001",
        "d_z": 1,
        "kl_weight": 0.01,
        "anchor_scale": 0.0,
        "mask_subject": True,
    },
    {
        "name": "anchor_no_subject_kl001",
        "d_z": 1,
        "kl_weight": 0.01,
        "anchor_scale": 1.0,
        "mask_subject": True,
    },
    {
        "name": "latent_y10_kl001",
        "d_z": 1,
        "kl_weight": 0.01,
        "anchor_scale": 0.0,
        "label_scale": 10.0,
        "center_labels": True,
    },
    {
        "name": "latent_aux1_kl001",
        "d_z": 1,
        "kl_weight": 0.01,
        "anchor_scale": 0.0,
        "ability_aux_weight": 1.0,
    },
    {
        "name": "latent_y10_aux1_kl001",
        "d_z": 1,
        "kl_weight": 0.01,
        "anchor_scale": 0.0,
        "label_scale": 10.0,
        "center_labels": True,
        "ability_aux_weight": 1.0,
    },
    {
        "name": "anchor_y10_kl001",
        "d_z": 1,
        "kl_weight": 0.01,
        "anchor_scale": 1.0,
        "label_scale": 10.0,
        "center_labels": True,
    },
    {
        "name": "anchor_y10_aux01_kl001",
        "d_z": 1,
        "kl_weight": 0.01,
        "anchor_scale": 1.0,
        "label_scale": 10.0,
        "center_labels": True,
        "ability_aux_weight": 0.1,
    },
    {
        "name": "latent_per_item_dz64",
        "d_z": 64,
        "kl_weight": 0.01,
        "anchor_scale": 0.0,
        "latent_item_parameters": True,
    },
    {
        "name": "latent_per_item_dz64_disjoint",
        "d_z": 64,
        "kl_weight": 0.01,
        "anchor_scale": 0.0,
        "latent_item_parameters": True,
        "disjoint_context_target": True,
    },
    {
        "name": "latent_per_item_dz1_disjoint",
        "d_z": 1,
        "kl_weight": 0.01,
        "anchor_scale": 0.0,
        "latent_item_parameters": True,
        "disjoint_context_target": True,
    },
    {
        "name": "latent_per_item_dz64_transfer_v2",
        "d_z": 64,
        "kl_weight": 0.01,
        "anchor_scale": 0.0,
        "latent_item_parameters": True,
        "disjoint_context_target": True,
        "transfer_check": True,
    },
    {
        "name": "hybrid_anchor_per_item_v1",
        "d_z": 64,
        "kl_weight": 0.01,
        "anchor_scale": 1.0,
        "latent_item_parameters": True,
        "disjoint_context_target": True,
    },
    {
        "name": "hybrid_anchor_per_item_temperature_v1",
        "d_z": 64,
        "kl_weight": 0.01,
        "anchor_scale": 1.0,
        "latent_item_parameters": True,
        "disjoint_context_target": True,
        "fit_temperature": True,
    },
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("aims-neurips-data", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24", "torch==2.5.1")
    .add_local_dir(Path(__file__).resolve().parent / "src", "/opt/aims/src", copy=True)
    .env({"PYTHONPATH": "/opt/aims/src", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
)


def _features(rows, state):
    import numpy as np

    return np.concatenate(
        [
            state["item"][rows["item_uid"].astype(str).map(state["item_i"]).to_numpy()],
            state["numeric"][rows["item_uid"].astype(str).map(state["item_i"]).to_numpy()],
            state["subject"][rows["subject_uid"].astype(str).map(state["subject_i"]).to_numpy()],
            state["benchmark"][
                rows["benchmark_id"].astype(str).map(state["benchmark_i"]).to_numpy()
            ],
        ],
        axis=1,
    ).astype("float32")


def _load(*, mask_subject: bool = False):
    import numpy as np
    import pandas as pd

    items = pd.read_parquet(CORPUS_ROOT / "items.parquet")
    subjects = pd.read_parquet(CORPUS_ROOT / "subjects.parquet")
    benchmarks = pd.read_parquet(CORPUS_ROOT / "benchmarks.parquet")
    state = {
        "item_i": dict(zip(items.item_uid.astype(str), range(len(items)), strict=True)),
        "subject_i": dict(zip(subjects.subject_uid.astype(str), range(len(subjects)), strict=True)),
        "benchmark_i": dict(
            zip(benchmarks.benchmark_id.astype(str), range(len(benchmarks)), strict=True)
        ),
        "item": np.load(CORPUS_ROOT / "item_embeddings.npy"),
        "numeric": np.load(CORPUS_ROOT / "item_numeric.npy"),
        "subject": np.load(CORPUS_ROOT / "subject_embeddings.npy"),
        "benchmark": np.load(CORPUS_ROOT / "benchmark_embeddings.npy"),
    }
    meta = pd.read_parquet(CORPUS_ROOT / "meta_observations.parquet")
    beaver = pd.read_parquet(CORPUS_ROOT / "beaver_observations.parquet")
    assignments = pd.read_parquet(CORPUS_ROOT / "beaver_episode_items.parquet")

    def tasks(frame):
        result = [
            {
                "task_id": str(key),
                "x": _features(rows, state),
                "y": rows.label.to_numpy(dtype="float32"),
            }
            for key, rows in frame.groupby("task_id", sort=True)
        ]
        if mask_subject:
            for task in result:
                task["x"][:, 389:773] = 0.0
        return result

    beaver_tasks = {task["task_id"]: task for task in tasks(beaver)}
    heldout = sorted(beaver_tasks)[1]
    training = tasks(meta) + [task for key, task in beaver_tasks.items() if key != heldout]
    assignment = assignments[assignments.task_id.astype(str) == heldout]
    acquisition = assignment[assignment.pool == "acquisition"].sort_values("acquisition_rank")
    lookup = beaver.set_index("observation_id", drop=False)
    context = lookup.loc[acquisition.label_observation_id.astype(str).tolist()]
    evaluation_items = set(assignment.loc[assignment.pool == "evaluation", "item_uid"].astype(str))
    target = beaver[
        (beaver.task_id.astype(str) == heldout) & beaver.item_uid.astype(str).isin(evaluation_items)
    ]
    evaluation = {
        "x_C_pool": _features(context, state),
        "y_C_pool": context.label.to_numpy(dtype="float32"),
        "x_T": _features(target, state),
        "y_T": target.label.to_numpy(dtype="float32"),
    }
    if mask_subject:
        evaluation["x_C_pool"][:, 389:773] = 0.0
        evaluation["x_T"][:, 389:773] = 0.0
    in_task = training[0]
    split = min(31, max(1, len(in_task["y"]) // 3))
    in_eval = {
        "x_C_pool": in_task["x"][:split],
        "y_C_pool": in_task["y"][:split],
        "x_T": in_task["x"][split:],
        "y_T": in_task["y"][split:],
    }
    return training, evaluation, in_eval


def _latent_diagnostics(model, evaluation):
    import torch
    from torch.distributions import kl_divergence

    device = next(model.parameters()).device
    output = {}
    for budget in (0, 1, 3, 7, 31):
        x = torch.as_tensor(evaluation["x_C_pool"][:budget], dtype=torch.float32, device=device)
        y = torch.as_tensor(evaluation["y_C_pool"][:budget], dtype=torch.float32, device=device)
        encoded = model.encode_x(x)
        q = model.q_z(encoded, y)
        q_flip = model.q_z(encoded, 1.0 - y) if budget else q
        mu, sigma = q.base_dist.loc, q.base_dist.scale
        theta = model.theta_from_z(q.rsample((512,))).squeeze(-1)
        output[str(budget)] = {
            "mu_mean": float(mu.mean()),
            "mu_norm": float(mu.norm()),
            "sigma_mean": float(sigma.mean()),
            "sigma_min": float(sigma.min()),
            "theta_mean": float(theta.mean()),
            "theta_std": float(theta.std(unbiased=False)),
            "flip_mu_distance": float((mu - q_flip.base_dist.loc).norm()),
            "kl_flipped_labels": float(kl_divergence(q, q_flip)),
        }
        if model.latent_item_parameters and budget == 7:
            z_samples = q.rsample((128,))
            x_target = torch.as_tensor(evaluation["x_T"], dtype=torch.float32, device=device)
            logits = model.decode_logits(
                z_samples,
                model.encode_item_x(x_target),
                None,
                torch.zeros((), device=device),
            )
            singular_values = model.decoder_jacobian_singular_values(x_target, q.base_dist.loc)
            output[str(budget)].update(
                {
                    "per_item_logit_variance": float(logits.var(dim=0).mean()),
                    "jacobian_singular_values": singular_values.detach().cpu().tolist(),
                    "jacobian_rank": int((singular_values > 1e-5).sum()),
                }
            )
    return output


@app.function(
    image=image, gpu="L4", cpu=8, memory=32_768, volumes={str(REMOTE_ROOT): volume}, timeout=60 * 60
)
def run_config(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    from aims_starter.faithful_np import score_model, train_model, write_metrics

    output = RESULT_ROOT / f"{config['name']}.json"
    if output.exists():
        return json.loads(output.read_text())
    training, evaluation, in_eval = _load(mask_subject=config.get("mask_subject", False))
    model, history = train_model(
        training,
        variant="np",
        seed=SEED,
        steps=300,
        episodes_per_step=4,
        kl_weight=config["kl_weight"],
        ability_aux_weight=config.get("ability_aux_weight", 0.0),
        model_kwargs={
            "d_z": config["d_z"],
            "anchor_scale": config["anchor_scale"],
            "label_scale": config.get("label_scale", 1.0),
            "center_labels": config.get("center_labels", False),
            "subject_feature_slice": (389, 773),
            "latent_item_parameters": config.get("latent_item_parameters", False),
        },
        disjoint_context_target=config.get("disjoint_context_target", False),
    )
    score = score_model(model, evaluation, seed=SEED)
    if config.get("fit_temperature"):
        # Diagnostic post-hoc temperature fit on the held-out labels.
        import numpy as np

        probabilities = []
        for budget in (0, 1, 3, 7, 15, 31):
            x_c = torch.as_tensor(
                evaluation["x_C_pool"][:budget], dtype=torch.float32, device="cuda"
            )
            y_c = torch.as_tensor(
                evaluation["y_C_pool"][:budget], dtype=torch.float32, device="cuda"
            )
            probabilities.append(
                model.predict(
                    x_c, y_c, torch.as_tensor(evaluation["x_T"], dtype=torch.float32, device="cuda")
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
        logits = np.concatenate(
            [np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)) for p in probabilities]
        )
        labels = np.tile(evaluation["y_T"], len(probabilities))
        temperatures = np.linspace(0.25, 4.0, 76)
        losses = [np.mean((1 / (1 + np.exp(-logits / t)) - labels) ** 2) for t in temperatures]
        score["temperature"] = {
            "T": float(temperatures[int(np.argmin(losses))]),
            "brier": float(min(losses)),
        }
    if config.get("transfer_check"):
        score["in_distribution"] = score_model(model, in_eval, seed=SEED)
    y_true = evaluation["y_T"]
    x_target = torch.as_tensor(evaluation["x_T"], dtype=torch.float32, device="cuda")
    for budget in (0, 1, 3, 7, 15, 31):
        x_context = torch.as_tensor(
            evaluation["x_C_pool"][:budget], dtype=torch.float32, device="cuda"
        )
        y_context = torch.as_tensor(
            evaluation["y_C_pool"][:budget], dtype=torch.float32, device="cuda"
        )
        probability = model.predict(x_context, y_context, x_target)[0].detach().cpu().numpy()
        positive = y_true == 1
        negative = y_true == 0
        auc = (
            float(
                (probability[positive, None] > probability[negative]).mean()
                + 0.5 * (probability[positive, None] == probability[negative]).mean()
            )
            if positive.any() and negative.any()
            else float("nan")
        )
        score[str(budget)].update({"auc": auc, "observed_label_rate": float(y_true.mean())})
    result = {
        "config": config,
        "score": score,
        "latent": _latent_diagnostics(model, evaluation),
        "history": history,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_metrics(output, result)
    volume.commit()
    return result


@app.local_entrypoint()
def main() -> None:
    results = list(run_config.map(CONFIGS, order_outputs=True))
    print(
        json.dumps(
            {
                row["config"]["name"]: {
                    "brier_alc": row["score"]["brier_alc"],
                    "final": row["history"][-1],
                    "latent_b7": row["latent"]["7"],
                    "auc": {
                        str(b): row["score"].get(str(b), {}).get("auc")
                        for b in (0, 1, 3, 7, 15, 31)
                    },
                    "in_distribution_alc": row["score"].get("in_distribution", {}).get("brier_alc"),
                    "temperature": row["score"].get("temperature"),
                }
                for row in results
            },
            indent=2,
            sort_keys=True,
        )
    )
