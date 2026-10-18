"""Rapid held-out-subject pilot for online AIRT and neural-process variants."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .metrics import DEFAULT_BUDGETS, DEFAULT_WEIGHTS
from .smoke import _hashed_item_features


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _brier(y, p) -> float:
    import numpy as np

    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def _ece(y, p, bins: int = 10) -> float:
    import numpy as np

    y = np.asarray(y)
    p = np.asarray(p)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & (p < edges[index + 1])
        if index == bins - 1:
            mask |= p == 1.0
        if mask.any():
            result += float(mask.mean()) * abs(float(y[mask].mean() - p[mask].mean()))
    return result


def _item_parameters(network, x):
    import torch.nn.functional as F

    raw = network(x)
    return F.softplus(raw[..., 0]) + 0.05, raw[..., 1]


def _make_item_net(dimension: int):
    from torch import nn

    return nn.Sequential(nn.Linear(dimension, 48), nn.Tanh(), nn.Linear(48, 2))


class _OnlineAIRT:
    def __init__(self, dimension: int, n_subjects: int):
        import torch

        self.irt_parameter_net = _make_item_net(dimension)
        self.theta = torch.nn.Parameter(torch.zeros(n_subjects))

    def parameters(self):
        return list(self.irt_parameter_net.parameters()) + [self.theta]

    def state_dict(self):
        return {
            "irt_parameter_net": self.irt_parameter_net.state_dict(),
            "theta": self.theta.detach().cpu(),
        }


class _NPAIRT:
    def __init__(
        self,
        dimension: int,
        *,
        attentive: bool = False,
        stacked_mhsa: bool = False,
    ):
        from torch import nn

        self.attentive = attentive
        self.stacked_mhsa = stacked_mhsa
        self.irt_parameter_net = _make_item_net(dimension)
        self.s_encoder = nn.Sequential(
            nn.Linear(dimension + 1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.q_z_mu = nn.Linear(64, 1)
        self.q_z_log_sigma = nn.Linear(64, 1)
        if attentive:
            self.x_T_query = nn.Linear(dimension, 64)
            heads = 8 if stacked_mhsa else 4
            self.r_cross_attention = nn.MultiheadAttention(64, heads, batch_first=True)
            self.decoder_residual = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        if stacked_mhsa:
            self.s_self_attention = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=64,
                        nhead=8,
                        dim_feedforward=128,
                        dropout=0.0,
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(2)
                ]
            )

    def parameters(self):
        modules = [self.irt_parameter_net, self.s_encoder, self.q_z_mu, self.q_z_log_sigma]
        if self.attentive:
            modules += [self.x_T_query, self.r_cross_attention, self.decoder_residual]
        if self.stacked_mhsa:
            modules += [self.s_self_attention]
        return [parameter for module in modules for parameter in module.parameters()]

    def state_dict(self):
        result = {
            "irt_parameter_net": self.irt_parameter_net.state_dict(),
            "s_encoder": self.s_encoder.state_dict(),
            "q_z_mu": self.q_z_mu.state_dict(),
            "q_z_log_sigma": self.q_z_log_sigma.state_dict(),
        }
        if self.attentive:
            result.update(
                x_T_query=self.x_T_query.state_dict(),
                r_cross_attention=self.r_cross_attention.state_dict(),
                decoder_residual=self.decoder_residual.state_dict(),
            )
        if self.stacked_mhsa:
            result["s_self_attention"] = self.s_self_attention.state_dict()
        return result

    def q_z_C(self, x_C, y_C):
        import torch

        if len(x_C) == 0:
            device = next(iter(self.irt_parameter_net.parameters())).device
            return torch.zeros(1, device=device), torch.zeros(1, device=device), None
        s_i = self.s_encoder(torch.cat([x_C, y_C[:, None]], dim=-1))
        if self.stacked_mhsa:
            s_i = s_i[None, :, :]
            for layer in self.s_self_attention:
                s_i = layer(s_i)
            s_i = s_i.squeeze(0)
        s_C = s_i.mean(dim=0)
        mu_z_C = self.q_z_mu(s_C).squeeze(-1)
        log_sigma_z_C = self.q_z_log_sigma(s_C).squeeze(-1).clamp(-4.0, 1.5)
        return mu_z_C.reshape(1), log_sigma_z_C.reshape(1), s_i

    def decode(self, z, x_T, s_i):
        import torch

        a_T, b_T = _item_parameters(self.irt_parameter_net, x_T)
        theta = z  # This archived pilot uses a one-dimensional identity theta_from_z map.
        logits = a_T[None, :] * (theta[:, None] - b_T[None, :])
        if self.attentive and s_i is not None:
            q_T = self.x_T_query(x_T)[None, :, :]
            s_i_batched = s_i[None, :, :]
            r_star, _ = self.r_cross_attention(
                q_T, s_i_batched, s_i_batched, need_weights=False
            )
            residual_T = self.decoder_residual(r_star.squeeze(0)).squeeze(-1)
            logits = logits + residual_T[None, :]
        return torch.sigmoid(logits)


def _train_airt(x_all, rows, item_to_index, subject_to_index, *, steps: int, seed: int):
    import torch
    import torch.nn.functional as F

    _seed_everything(seed)
    model = _OnlineAIRT(x_all.shape[1], len(subject_to_index))
    item_index = torch.tensor([item_to_index[str(v)] for v in rows["item_uid"]])
    subject_index = torch.tensor([subject_to_index[str(v)] for v in rows["task_id"]])
    y = torch.tensor(rows["label"].to_numpy(), dtype=torch.float32)
    x = torch.tensor(x_all, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    for _ in range(steps):
        a, b = _item_parameters(model.irt_parameter_net, x[item_index])
        logits = a * (model.theta[subject_index] - b)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss = loss + 0.002 * model.theta.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.theta -= model.theta.mean()
    return model


def _train_np(
    x_all,
    task_rows: dict[str, Any],
    item_to_index,
    *,
    attentive: bool,
    stacked_mhsa: bool = False,
    steps: int,
    seed: int,
):
    import numpy as np
    import torch
    import torch.nn.functional as F

    _seed_everything(seed)
    model = _NPAIRT(x_all.shape[1], attentive=attentive, stacked_mhsa=stacked_mhsa)
    x = torch.tensor(x_all, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    rng = np.random.default_rng(seed)
    tasks = sorted(task_rows)
    budgets = [0, 1, 3, 7, 15, 31]
    for _ in range(steps):
        task_id = tasks[int(rng.integers(len(tasks)))]
        rows = task_rows[task_id]
        indices = np.arange(len(rows))
        rng.shuffle(indices)
        budget = min(int(rng.choice(budgets)), max(0, len(indices) - 1))
        context_rows = rows.iloc[indices[:budget]]
        target_indices = indices[budget : budget + min(96, len(indices) - budget)]
        target_rows = rows.iloc[target_indices]
        x_C = x[[item_to_index[str(v)] for v in context_rows["item_uid"]]]
        y_C = torch.tensor(context_rows["label"].to_numpy(), dtype=torch.float32)
        x_T = x[[item_to_index[str(v)] for v in target_rows["item_uid"]]]
        y_T = torch.tensor(target_rows["label"].to_numpy(), dtype=torch.float32)
        mu_z_C, log_sigma_z_C, s_i = model.q_z_C(x_C, y_C)
        z = mu_z_C + torch.randn(4) * log_sigma_z_C.exp()
        p_y_T = model.decode(z, x_T, s_i).mean(dim=0)
        prediction_loss = F.binary_cross_entropy(p_y_T.clamp(1e-5, 1 - 1e-5), y_T)
        kl = 0.5 * (
            mu_z_C.square() + (2 * log_sigma_z_C).exp() - 1.0 - 2 * log_sigma_z_C
        ).mean()
        loss = prediction_loss + 0.01 * kl
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


def _predict_airt(model, x_all, context_rows, target_items):
    import torch
    import torch.nn.functional as F

    x = torch.tensor(x_all, dtype=torch.float32)
    item_to_index = target_items.attrs["item_to_index"]
    theta = torch.zeros(1, requires_grad=True)
    if len(context_rows):
        optimizer = torch.optim.Adam([theta], lr=0.08)
        context_index = torch.tensor([item_to_index[str(v)] for v in context_rows["item_uid"]])
        y_C = torch.tensor(context_rows["label"].to_numpy(), dtype=torch.float32)
        with torch.no_grad():
            a, b = _item_parameters(model.irt_parameter_net, x[context_index])
        for _ in range(120):
            loss = F.binary_cross_entropy_with_logits(a * (theta - b), y_C, reduction="sum")
            loss = loss + theta.square().sum() / 8.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    target_index = torch.tensor([item_to_index[str(v)] for v in target_items])
    with torch.no_grad():
        a, b = _item_parameters(model.irt_parameter_net, x[target_index])
        return torch.sigmoid(a * (theta.detach() - b)).cpu().numpy()


def _predict_np(model, x_all, context_rows, target_items, *, seed: int):
    import torch

    torch.manual_seed(seed)
    item_to_index = target_items.attrs["item_to_index"]
    x = torch.tensor(x_all, dtype=torch.float32)
    x_C = x[[item_to_index[str(v)] for v in context_rows["item_uid"]]]
    y_C = torch.tensor(context_rows["label"].to_numpy(), dtype=torch.float32)
    x_T = x[[item_to_index[str(v)] for v in target_items]]
    with torch.no_grad():
        mu_z_C, log_sigma_z_C, s_i = model.q_z_C(x_C, y_C)
        z = mu_z_C + torch.randn(128) * log_sigma_z_C.exp()
        return model.decode(z, x_T, s_i).mean(dim=0).cpu().numpy()


def run_neural_pilot(
    observations,
    episode_items,
    items,
    *,
    output_dir: str | Path,
    seed: int = 20261018,
    airt_steps: int = 500,
    np_steps: int = 900,
) -> dict[str, Any]:
    """Run four-fold held-out-subject curves on one multi-subject benchmark."""

    import numpy as np
    import torch

    output_dir = Path(output_dir)
    benchmark_ids = sorted(set(episode_items["benchmark_id"].astype(str)))
    if len(benchmark_ids) != 1:
        raise ValueError("pilot requires exactly one benchmark")
    benchmark_id = benchmark_ids[0]
    task_ids = sorted(set(episode_items["task_id"].astype(str)))
    item_ids = sorted(set(episode_items["item_uid"].astype(str)))
    benchmark_items = items[items["item_uid"].astype(str).isin(item_ids)].copy()
    benchmark_items = benchmark_items.sort_values("item_uid").reset_index(drop=True)
    x_all = _hashed_item_features(benchmark_items).astype("float32")
    item_to_index = {value: index for index, value in enumerate(benchmark_items["item_uid"].astype(str))}
    task_rows = {
        task_id: observations[observations["task_id"].astype(str) == task_id].reset_index(drop=True)
        for task_id in task_ids
    }
    lookup = observations.set_index("observation_id", drop=False)
    model_names = [
        "beta_binomial",
        "online_airt_2pl",
        "np_airt",
        "anp_airt",
        "stacked_mhsa_anp_airt",
    ]
    fold_results = []
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for fold, heldout in enumerate(task_ids):
        train_tasks = [task for task in task_ids if task != heldout]
        train_rows = observations[observations["task_id"].astype(str).isin(train_tasks)].copy()
        subject_to_index = {task: index for index, task in enumerate(train_tasks)}
        fold_seed = seed + fold * 100
        airt = _train_airt(
            x_all, train_rows, item_to_index, subject_to_index, steps=airt_steps, seed=fold_seed
        )
        np_model = _train_np(
            x_all,
            {task: task_rows[task] for task in train_tasks},
            item_to_index,
            attentive=False,
            steps=np_steps,
            seed=fold_seed + 1,
        )
        anp_model = _train_np(
            x_all,
            {task: task_rows[task] for task in train_tasks},
            item_to_index,
            attentive=True,
            steps=np_steps,
            seed=fold_seed + 2,
        )
        stacked_model = _train_np(
            x_all,
            {task: task_rows[task] for task in train_tasks},
            item_to_index,
            attentive=True,
            stacked_mhsa=True,
            steps=np_steps,
            seed=fold_seed + 3,
        )
        torch.save(airt.state_dict(), checkpoint_dir / f"fold{fold}-online-airt.pt")
        torch.save(np_model.state_dict(), checkpoint_dir / f"fold{fold}-np-airt.pt")
        torch.save(anp_model.state_dict(), checkpoint_dir / f"fold{fold}-anp-airt.pt")
        torch.save(
            stacked_model.state_dict(),
            checkpoint_dir / f"fold{fold}-stacked-mhsa-anp-airt.pt",
        )

        assignment = episode_items[episode_items["task_id"].astype(str) == heldout]
        acquisition = assignment[assignment["pool"] == "acquisition"].sort_values("acquisition_rank")
        label_ids = acquisition["label_observation_id"].astype(str).tolist()
        evaluation_items = set(
            assignment.loc[assignment["pool"] == "evaluation", "item_uid"].astype(str)
        )
        evaluation_rows = task_rows[heldout][task_rows[heldout]["item_uid"].astype(str).isin(evaluation_items)]
        target_items = evaluation_rows["item_uid"].astype(str).reset_index(drop=True)
        target_items.attrs["item_to_index"] = item_to_index
        truth = evaluation_rows["label"].to_numpy(dtype=float)
        by_model: dict[str, Any] = {name: {} for name in model_names}
        for budget in DEFAULT_BUDGETS:
            context = lookup.loc[label_ids[:budget]].reset_index(drop=True)
            beta_probability = (2.0 + float(context["label"].sum())) / (4.0 + len(context))
            predictions = {
                "beta_binomial": np.full(len(truth), beta_probability),
                "online_airt_2pl": _predict_airt(airt, x_all, context, target_items),
                "np_airt": _predict_np(np_model, x_all, context, target_items, seed=fold_seed + budget),
                "anp_airt": _predict_np(anp_model, x_all, context, target_items, seed=fold_seed + budget),
                "stacked_mhsa_anp_airt": _predict_np(
                    stacked_model,
                    x_all,
                    context,
                    target_items,
                    seed=fold_seed + budget,
                ),
            }
            for name, probability in predictions.items():
                by_model[name][str(budget)] = {
                    "brier": _brier(truth, probability),
                    "ece": _ece(truth, probability),
                    "mean_probability": float(np.mean(probability)),
                }
        fold_results.append({"fold": fold, "heldout_task": heldout, "models": by_model})

    aggregate: dict[str, Any] = {}
    for name in model_names:
        curve = {}
        scores = []
        for budget in DEFAULT_BUDGETS:
            values = [fold["models"][name][str(budget)]["brier"] for fold in fold_results]
            eces = [fold["models"][name][str(budget)]["ece"] for fold in fold_results]
            curve[str(budget)] = {
                "brier": float(np.mean(values)),
                "brier_std": float(np.std(values, ddof=1)),
                "ece": float(np.mean(eces)),
            }
            scores.append(float(np.mean(values)))
        aggregate[name] = {
            "brier_alc": float(sum(w * score for w, score in zip(DEFAULT_WEIGHTS, scores, strict=True))),
            "by_budget": curve,
        }

    result = {
        "status": "rapid_pilot_not_final_evidence",
        "benchmark_id": benchmark_id,
        "protocol": {
            "outer_split": "leave-one-subject-task-out (4 folds)",
            "offline_training": "all observed responses from the other three subjects",
            "online_context_budgets": list(DEFAULT_BUDGETS),
            "evaluation": "frozen untouched evaluation-item pool of held-out subject",
            "seed": seed,
            "airt_steps": airt_steps,
            "np_steps": np_steps,
        },
        "n_folds": len(fold_results),
        "n_items": len(item_ids),
        "aggregate": aggregate,
        "folds": fold_results,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
