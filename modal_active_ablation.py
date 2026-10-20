"""Offline active-acquisition ablations on the benchmark-held-out validation split.

Notation
--------
P_β   Beta-Binomial predictor, p = (2 + successes) / (4 + n).
P_C   CNP-AIRT predictor, masked subject block, seed 20261018, calibrated.
A_R   Random acquisition: reveal the first budget candidates in the stream.
A_U(P) Uncertainty acquisition using predictor P: |p_P - 0.5| <= 0.15.
A_U(β) Beta-mean uncertainty acquisition. This is the decoupled arm for P_C.
A_V(β) Beta posterior-variance acquisition: Var_beta >= 0.01.
A_D   Disagreement acquisition: |p_C - p_β| >= 0.15.

Entropy acquisition is not a separate arm. For fixed threshold, H(p) is a
monotone transform of |p - 0.5|, so it produces identical decisions to A_U.

The simulator uses validation tasks only. Candidate streams are random
permutations of each task's acquisition pool; every arm sees the same streams.
Evaluation targets are the disjoint evaluation pool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "aims-active-acquisition-ablation"
VOLUME_NAME = "aims-neurips-data"
REMOTE_ROOT = Path("/data")
PROJECT_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = REMOTE_ROOT / "datasets" / "full" / "allbench-features-v1"
RESULT_ROOT = REMOTE_ROOT / "results" / "allbench-active-ablation"
CHECKPOINT_ROOT = REMOTE_ROOT / "results" / "allbench-deterministic-2seed" / "seed-20261018-masked-checkpoints"
CALIBRATION_PATH = REMOTE_ROOT / "results" / "allbench-deterministic-2seed" / "calibration-20261018-masked.json"
SPLIT = "validation"
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
DISTANCE_THRESHOLD = 0.15
VARIANCE_THRESHOLD = 0.01
SUBJECT_SLICE = (384, 768)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=2,<3", "pandas>=2.2,<3", "pyarrow>=17,<24")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(PROJECT_ROOT / "src", "/opt/aims/src", copy=True)
    .env({"PYTHONPATH": "/opt/aims/src", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
)

ARMS: dict[str, dict[str, Any]] = {
    "β-R": {
        "predictor": "beta",
        "acquisition": "random",
        "label": "P_beta + A_R",
        "description": "Beta-Binomial predictor, random stream order.",
        "study": "Priority-1: acquisition rule, Beta predictor",
    },
    "β-Uβ": {
        "predictor": "beta",
        "acquisition": "uncertainty_beta",
        "label": "P_beta + A_U(beta)",
        "description": "Beta predictor, Beta-mean uncertainty.",
        "study": "Priority-1: acquisition rule, Beta predictor",
    },
    "β-Vβ": {
        "predictor": "beta",
        "acquisition": "variance_beta",
        "label": "P_beta + A_V(beta)",
        "description": "Beta predictor, Beta posterior variance.",
        "study": "Priority-1: acquisition rule, Beta predictor",
    },
    "C-R": {
        "predictor": "cnp",
        "acquisition": "random",
        "label": "P_C + A_R",
        "description": "CNP-AIRT predictor, random stream order.",
        "study": "Priority-1: acquisition rule, CNP predictor",
    },
    "C-UC": {
        "predictor": "cnp",
        "acquisition": "uncertainty_model",
        "label": "P_C + A_U(C)",
        "description": "CNP-AIRT predictor, CNP uncertainty. Deployed active policy.",
        "study": "Priority-1: acquisition rule, CNP predictor",
    },
    "C-Uβ": {
        "predictor": "cnp",
        "acquisition": "uncertainty_beta",
        "label": "P_C + A_U(beta)",
        "description": "CNP-AIRT predictor, Beta-mean acquisition. Priority-2 decoupled arm.",
        "study": "Priority-2: decoupled acquisition source",
    },
    "C-Vβ": {
        "predictor": "cnp",
        "acquisition": "variance_beta",
        "label": "P_C + A_V(beta)",
        "description": "CNP-AIRT predictor, Beta posterior variance acquisition.",
        "study": "Priority-2: decoupled acquisition source",
    },
    "C-D": {
        "predictor": "cnp",
        "acquisition": "disagreement",
        "label": "P_C + A_D",
        "description": "CNP-AIRT predictor, CNP/Beta disagreement acquisition.",
        "study": "Priority-2: decoupled acquisition source",
    },
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _calibrator():
    from aims_starter.calibration import AnchorCalibrator, BudgetCalibrator

    artifact = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    entries = artifact["fitted"]["cnp_airt"]
    fitted = {
        int(entry["budget"]): BudgetCalibrator(
            scale=float(entry["scale"]),
            bias=float(entry["bias"]),
            anchor=float(entry["anchor"]),
            samples=int(entry["samples"]),
            fitted=bool(entry["fitted"]),
        )
        for entry in entries
    }
    return AnchorCalibrator(budgets=tuple(sorted(fitted)), budgets_map=fitted)


def _beta_probability(successes: float, count: int) -> float:
    return (2.0 + successes) / (4.0 + count)


def _beta_variance(successes: float, count: int) -> float:
    alpha = 2.0 + successes
    beta = 2.0 + count - successes
    return alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))


def _sigmoid(value):
    import numpy as np

    return 1.0 / (1.0 + np.exp(-value))


def _context_theta(model, encoded_c, labels_c, device):
    import torch

    if encoded_c.shape[0] == 0:
        return 0.0
    with torch.no_grad():
        theta = model.map_theta(encoded_c, labels_c)
        label_value = labels_c[:, None]
        residual = model.r_encoder(torch.cat([encoded_c, label_value], dim=-1))
        theta = theta + model.theta_delta(residual.mean(dim=0)).squeeze(-1)
    return float(theta.detach().cpu())


def _predict_cnp(model, calibrator, encoded_target, a_target, b_target, encoded_c, labels_c, budget):

    theta = _context_theta(model, encoded_c, labels_c, encoded_target.device)
    logits = a_target * (theta - b_target)
    if calibrator is None:
        return _sigmoid(logits)
    return calibrator.transform(logits, budget, labels_c.detach().cpu().numpy())


def _first_choice(
    arm: str,
    *,
    pos: int,
    candidates_in_stream,
    labels_remaining: int,
    p_model,
    p_beta: float,
    var_beta: float,
) -> int:
    """Return the stream position of the next acquisition.

    The returned position is always at or after ``pos``. When the policy has
    no qualifying candidate before the reserve point, the reserve point is
    returned, matching the coordinator's must-query rule.
    """
    import numpy as np

    stream_length = len(candidates_in_stream)
    if arm.endswith("-R"):
        return pos
    if stream_length - pos <= labels_remaining:
        return pos

    acquisition = ARMS[arm]["acquisition"]
    if acquisition == "uncertainty_beta":
        condition = np.abs(p_beta - 0.5) <= DISTANCE_THRESHOLD
        if condition:
            return pos
    elif acquisition == "variance_beta":
        if var_beta >= VARIANCE_THRESHOLD:
            return pos
    elif acquisition == "uncertainty_model":
        condition = np.abs(p_model - 0.5) <= DISTANCE_THRESHOLD
        hits = np.flatnonzero(condition)
        if hits.size:
            return pos + int(hits[0])
    elif acquisition == "disagreement":
        condition = np.abs(p_model - p_beta) >= DISTANCE_THRESHOLD
        hits = np.flatnonzero(condition)
        if hits.size:
            return pos + int(hits[0])
    else:
        raise ValueError(f"unknown acquisition: {acquisition}")
    return max(pos, stream_length - labels_remaining)


@app.function(
    image=image,
    cpu=8,
    memory=65_536,
    volumes={str(REMOTE_ROOT): volume},
    timeout=4 * 60 * 60,
)
def run_ablation(limit_tasks: int = 0, streams: int = 10) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import torch

    from aims_starter.faithful_np import CNPAIRT
    from aims_starter.metrics import calibration_ece

    device = torch.device("cpu")
    item_embeddings = np.load(CORPUS_ROOT / "item_embeddings.npy")
    subject_embeddings = np.load(CORPUS_ROOT / "subject_embeddings.npy")
    evaluation_items = pd.read_parquet(CORPUS_ROOT / "eval_items.parquet")
    evaluation_items = evaluation_items[evaluation_items["split"] == SPLIT].copy()
    task_ids = sorted(evaluation_items["task_id"].unique())
    if limit_tasks:
        task_ids = task_ids[:limit_tasks]
    print(f"tasks={len(task_ids)} streams={streams}", flush=True)

    state = torch.load(CHECKPOINT_ROOT / "cnp_airt.pt", map_location=device, weights_only=True)
    model = CNPAIRT(item_embeddings.shape[1] + subject_embeddings.shape[1], subject_feature_slice=SUBJECT_SLICE)
    model.load_state_dict(state)
    model.eval()
    calibrator = _calibrator()

    def features(item_index, subject_index):
        return np.concatenate(
            [item_embeddings[item_index], subject_embeddings[subject_index]], axis=1
        ).astype("float32")

    def encode(x):
        with torch.no_grad():
            encoded = model.encode_item_x(torch.as_tensor(x, dtype=torch.float32, device=device))
            raw = model.irt_parameter_net(encoded)
            a = torch.nn.functional.softplus(raw[:, 0]) + 0.05
            b = raw[:, 1]
        return encoded, a.detach().cpu().numpy(), b.detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    for task_number, task_id in enumerate(task_ids):
        task_rows = evaluation_items[evaluation_items["task_id"] == task_id]
        target_rows = task_rows[task_rows["pool"] == "evaluation"].reset_index(drop=True)
        candidate_rows = task_rows[task_rows["pool"] == "acquisition"].reset_index(drop=True)
        if target_rows.empty or candidate_rows.empty:
            continue
        x_target = features(
            target_rows["item_index"].to_numpy(), target_rows["subject_index"].to_numpy()
        )
        x_candidate = features(
            candidate_rows["item_index"].to_numpy(), candidate_rows["subject_index"].to_numpy()
        )
        y_target = target_rows["label"].to_numpy(dtype="float64")
        y_candidate = candidate_rows["label"].to_numpy(dtype="float64")
        encoded_target, a_target, b_target = encode(x_target)
        encoded_candidate, a_candidate, b_candidate = encode(x_candidate)

        for stream_seed in range(streams):
            rng = np.random.default_rng([stream_seed, task_number])
            stream = rng.permutation(len(candidate_rows))
            for arm, arm_spec in ARMS.items():
                selected: list[int] = []
                selected_rank: list[float] = []
                pos = 0
                # The coordinator runs one acquisition trajectory to the maximum
                # budget and evaluates nested prefixes at 0, 1, 3, 7, 15, 31.
                # If the pool has fewer than 31 candidates, later caps use the
                # labels actually acquired.
                max_labels = min(BUDGETS[-1], len(stream))
                while len(selected) < max_labels:
                    if pos >= len(stream):
                        # The policy selected the last candidate before the cap.
                        # Later checkpoints use the labels actually acquired.
                        break
                    labels_remaining = BUDGETS[-1] - len(selected)
                    if not selected:
                        encoded_c = torch.zeros((0, encoded_target.shape[1]), dtype=torch.float32)
                        labels_c = torch.zeros((0,), dtype=torch.float32)
                    else:
                        encoded_c = encoded_candidate[selected]
                        labels_c = torch.as_tensor(y_candidate[selected], dtype=torch.float32)
                    successes = float(y_candidate[selected].sum()) if selected else 0.0
                    p_beta = _beta_probability(successes, len(selected))
                    var_beta = _beta_variance(successes, len(selected))

                    if arm_spec["predictor"] == "beta":
                        p_model = np.full(len(stream) - pos, p_beta)
                    else:
                        theta = _context_theta(model, encoded_c, labels_c, device)
                        remaining = stream[pos:]
                        logits = a_candidate[remaining] * (theta - b_candidate[remaining])
                        if calibrator is None:
                            p_model = _sigmoid(logits)
                        else:
                            p_model = calibrator.transform(
                                logits, len(selected), labels_c.detach().cpu().numpy()
                            )
                    chosen = _first_choice(
                        arm,
                        pos=pos,
                        candidates_in_stream=stream,
                        labels_remaining=labels_remaining,
                        p_model=p_model,
                        p_beta=p_beta,
                        var_beta=var_beta,
                    )
                    if chosen >= len(stream):
                        raise RuntimeError(
                            f"acquisition stream exhausted: arm={arm} task={task_id} "
                            f"seed={stream_seed} selected={len(selected)}"
                        )
                    selected.append(int(stream[chosen]))
                    pos = chosen + 1
                    selected_rank.append(float(chosen))

                for budget in BUDGETS:
                    prefix = selected[:budget]
                    actual = len(prefix)
                    calibration_budget = max(value for value in BUDGETS if value <= actual)
                    successes = float(y_candidate[prefix].sum())
                    p_beta = _beta_probability(successes, actual)
                    if arm_spec["predictor"] == "beta":
                        p_target = np.full(len(y_target), p_beta)
                    else:
                        encoded_c = encoded_candidate[prefix]
                        labels_c = torch.as_tensor(y_candidate[prefix], dtype=torch.float32)
                        p_target = _predict_cnp(
                            model,
                            calibrator,
                            encoded_target,
                            a_target,
                            b_target,
                            encoded_c,
                            labels_c,
                            calibration_budget,
                        )
                    brier = float(np.mean((p_target - y_target) ** 2))
                    ece = float(calibration_ece(y_target.tolist(), p_target.tolist()))
                    ranks = selected_rank[:budget]
                    rows.append(
                        {
                            "arm": arm,
                            "predictor": arm_spec["predictor"],
                            "acquisition": arm_spec["acquisition"],
                            "task_id": str(task_id),
                            "stream_seed": stream_seed,
                            "budget": budget,
                            "brier": brier,
                            "ece": ece,
                            "labels_acquired": actual,
                            "mean_selected_rank": float(np.mean(ranks)) if ranks else 0.0,
                            "stream_length": len(stream),
                        }
                    )
        print(f"task {task_number + 1}/{len(task_ids)} {task_id}", flush=True)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no ablation rows produced")
    frame["weighted_brier"] = frame["brier"] * frame["budget"].map(
        dict(zip(BUDGETS, WEIGHTS, strict=True))
    )
    frame["weighted_ece"] = frame["ece"] * frame["budget"].map(
        dict(zip(BUDGETS, WEIGHTS, strict=True))
    )
    per_run = (
        frame.groupby(["arm", "task_id", "stream_seed"], sort=False)
        .agg(brier_alc=("weighted_brier", "sum"), ece_alc=("weighted_ece", "sum"))
        .reset_index()
    )
    frame = frame.merge(per_run, on=["arm", "task_id", "stream_seed"], how="left")

    summary_rows = []
    for arm, group in per_run.groupby("arm", sort=False):
        values = group["brier_alc"].to_numpy(dtype=float)
        eces = group["ece_alc"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "arm": arm,
                "label": ARMS[arm]["label"],
                "description": ARMS[arm]["description"],
                "study": ARMS[arm]["study"],
                "runs": len(group),
                "brier_alc_mean": float(values.mean()),
                "brier_alc_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "brier_alc_se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
                "ece_alc_mean": float(eces.mean()),
                "ece_alc_sd": float(eces.std(ddof=1)) if len(eces) > 1 else 0.0,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("brier_alc_mean").reset_index(drop=True)

    by_budget_rows = []
    for (arm, budget), group in frame.groupby(["arm", "budget"], sort=False):
        by_budget_rows.append(
            {
                "arm": arm,
                "budget": int(budget),
                "brier_mean": float(group["brier"].mean()),
                "ece_mean": float(group["ece"].mean()),
                "mean_selected_rank": float(group["mean_selected_rank"].mean()),
                "mean_stream_length": float(group["stream_length"].mean()),
            }
        )

    deltas = []
    for arm, group in per_run.groupby("arm", sort=False):
        predictor = ARMS[arm]["predictor"]
        baseline_arm = "β-R" if predictor == "beta" else "C-R"
        baseline = per_run[per_run["arm"] == baseline_arm][
            ["task_id", "stream_seed", "brier_alc"]
        ].rename(columns={"brier_alc": "baseline_alc"})
        joined = group.merge(baseline, on=["task_id", "stream_seed"], how="inner")
        delta = (joined["brier_alc"] - joined["baseline_alc"]).to_numpy(dtype=float)
        deltas.append(
            {
                "arm": arm,
                "baseline": baseline_arm,
                "delta_mean": float(delta.mean()),
                "delta_sd": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                "delta_se": float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0.0,
                "wins": int((delta < 0).sum()),
                "paired_runs": len(delta),
            }
        )

    artifact = {
        "status": "active_acquisition_ablation",
        "split": SPLIT,
        "protocol": "pairwise random candidate streams; same stream per arm",
        "predictors": {
            "beta": "Laplace-smoothed Beta-Binomial, p=(2+successes)/(4+n)",
            "cnp": "CNP-AIRT masked seed 20261018, validation-fitted AnchorCalibrator applied",
        },
        "notation": {
            "P_beta": "Beta-Binomial predictor",
            "P_C": "CNP-AIRT predictor",
            "A_R": "random acquisition in stream order",
            "A_U(P)": f"|p_P - 0.5| <= {DISTANCE_THRESHOLD}",
            "A_V(beta)": f"Beta posterior variance >= {VARIANCE_THRESHOLD}",
            "A_D": f"|p_C - p_beta| >= {DISTANCE_THRESHOLD}",
            "A_H": "entropy, omitted because it is monotone-equivalent to A_U at fixed threshold",
        },
        "omitted_arms": {
            "P_beta + A_U(C)": "identical to P_beta + A_U(beta), since the Beta predictor has no item signal",
            "P_beta + A_D": "identically zero disagreement, since the Beta predictor has no item signal",
            "P_C + A_H": "monotone-equivalent to P_C + A_U(C) at the fixed threshold",
        },
        "tasks": len(task_ids),
        "streams": streams,
        "budgets": list(BUDGETS),
        "weights": list(WEIGHTS),
        "summary": summary.to_dict(orient="records"),
        "deltas_vs_random": deltas,
        "by_budget": by_budget_rows,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(RESULT_ROOT / "ablation.json", artifact)
    frame.to_parquet(RESULT_ROOT / "ablation_rows.parquet", index=False)
    volume.commit()
    print(json.dumps(artifact["summary"], indent=2), flush=True)
    return artifact


@app.local_entrypoint()
def main(limit_tasks: int = 0, streams: int = 10) -> None:
    run_ablation.remote(limit_tasks=limit_tasks, streams=streams)
