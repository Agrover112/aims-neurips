"""Paper-faithful binary Neural Process variants with an IRT-anchored decoder."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal, kl_divergence

from .metrics import DEFAULT_BUDGETS, DEFAULT_WEIGHTS, calibration_ece


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class AttentionBlock(nn.Module):
    """Permutation-equivariant pre-norm self-attention block."""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
        self.norm_feedforward = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, 2 * hidden),
            nn.ReLU(),
            nn.Linear(2 * hidden, hidden),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attention(values)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        values = values + attended
        return values + self.feedforward(self.norm_feedforward(values))


class FaithfulNPAIRT(nn.Module):
    """NP/ANP latent model adapted to binary outcomes with a 2PL anchor."""

    def __init__(
        self,
        x_dim: int,
        *,
        variant: str,
        d_model: int = 128,
        d_z: int = 64,
        n_heads: int = 8,
        residual_scale: float = 0.0,
    ):
        super().__init__()
        if variant not in {"np", "anp", "stacked_mhsa_anp"}:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.d_model = d_model
        self.d_z = d_z
        self.residual_scale = residual_scale
        self.x_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.irt_parameter_net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2)
        )
        self.s_encoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        attention_depth = {"np": 0, "anp": 1, "stacked_mhsa_anp": 2}[variant]
        self.s_self_attention = nn.ModuleList(
            [AttentionBlock(d_model, n_heads) for _ in range(attention_depth)]
        )
        self.q_z_mu = nn.Linear(d_model, d_z)
        self.q_z_raw_sigma = nn.Linear(d_model, d_z)

        if variant != "np":
            self.r_encoder = nn.Sequential(
                nn.Linear(d_model + 1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            self.r_self_attention = nn.ModuleList(
                [AttentionBlock(d_model, n_heads) for _ in range(attention_depth)]
            )
            self.x_T_query = nn.Linear(d_model, d_model)
            self.x_C_key = nn.Linear(d_model, d_model)
            self.r_cross_attention = nn.MultiheadAttention(
                d_model, n_heads, batch_first=True, dropout=0.0
            )
            self.r_star_norm = nn.LayerNorm(d_model)

        self.theta_from_z = nn.Sequential(
            nn.Linear(d_z, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )
        d_r = d_model if variant != "np" else 0
        self.decoder_residual = None
        if residual_scale > 0.0:
            self.decoder_residual = nn.Sequential(
                nn.Linear(d_model + d_z + d_r, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )

    def encode_x(self, x: torch.Tensor) -> torch.Tensor:
        """Encode NP inputs ``x`` before the latent and deterministic paths."""

        return self.x_encoder(x)

    def q_z(
        self,
        x_encoded: torch.Tensor,
        y: torch.Tensor,
    ) -> Independent:
        """Return the latent-path distribution q(z | s), where s is set-aggregated."""

        if x_encoded.shape[0] == 0:
            mu_z = torch.zeros(self.d_z, device=x_encoded.device)
            sigma_z = torch.ones(self.d_z, device=x_encoded.device)
            return Independent(Normal(mu_z, sigma_z), 1)
        xy = torch.cat([x_encoded, y[:, None]], dim=-1)[None, :, :]
        s_i = self.s_encoder(xy)
        for layer in self.s_self_attention:
            s_i = layer(s_i)
        s_set = s_i.mean(dim=1).squeeze(0)
        mu_z = self.q_z_mu(s_set)
        sigma_z = 0.1 + 0.9 * torch.sigmoid(self.q_z_raw_sigma(s_set))
        return Independent(Normal(mu_z, sigma_z), 1)

    def r_star(
        self,
        x_C_encoded: torch.Tensor,
        y_C: torch.Tensor,
        x_T_encoded: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return the ANP target-specific deterministic representation r_* ."""

        if self.variant == "np":
            return None
        if x_C_encoded.shape[0] == 0:
            return torch.zeros(
                x_T_encoded.shape[0], self.d_model, device=x_T_encoded.device
            )
        xy_C = torch.cat([x_C_encoded, y_C[:, None]], dim=-1)[None, :, :]
        r_i = self.r_encoder(xy_C)
        for layer in self.r_self_attention:
            r_i = layer(r_i)
        q_T = self.x_T_query(x_T_encoded)[None, :, :]
        k_C = self.x_C_key(x_C_encoded)[None, :, :]
        r_star, _ = self.r_cross_attention(q_T, k_C, r_i, need_weights=False)
        return self.r_star_norm(r_star.squeeze(0))

    def decode_logits(
        self,
        z: torch.Tensor,
        x_T_encoded: torch.Tensor,
        r_star: torch.Tensor | None,
        theta_anchor: torch.Tensor,
    ) -> torch.Tensor:
        raw_irt_parameters = self.irt_parameter_net(x_T_encoded)
        a = torch.nn.functional.softplus(raw_irt_parameters[:, 0]) + 0.05
        b = raw_irt_parameters[:, 1]
        theta = theta_anchor + self.theta_from_z(z).squeeze(-1)
        irt_logit = a[None, :] * (theta[:, None] - b[None, :])
        if self.decoder_residual is None:
            return irt_logit
        n_z_samples = z.shape[0]
        n_T = x_T_encoded.shape[0]
        z_T = z[:, None, :].expand(-1, n_T, -1)
        x_T_batched = x_T_encoded[None, :, :].expand(n_z_samples, -1, -1)
        decoder_inputs = [x_T_batched, z_T]
        if r_star is not None:
            decoder_inputs.append(r_star[None, :, :].expand(n_z_samples, -1, -1))
        residual = self.decoder_residual(torch.cat(decoder_inputs, dim=-1)).squeeze(-1)
        return irt_logit + self.residual_scale * residual

    @torch.no_grad()
    def map_theta(
        self,
        x_C_encoded: torch.Tensor,
        y_C: torch.Tensor,
        *,
        prior_variance: float = 4.0,
        steps: int = 12,
    ) -> torch.Tensor:
        """Regularized 2PL MAP anchor computed from the acquired context."""

        theta = torch.zeros((), device=x_C_encoded.device)
        if x_C_encoded.shape[0] == 0:
            return theta
        raw_irt_parameters = self.irt_parameter_net(x_C_encoded)
        a_C = torch.nn.functional.softplus(raw_irt_parameters[:, 0]) + 0.05
        b_C = raw_irt_parameters[:, 1]
        for _ in range(steps):
            p_C = torch.sigmoid(a_C * (theta - b_C))
            gradient = torch.sum(a_C * (y_C - p_C)) - theta / prior_variance
            hessian = -torch.sum(a_C.square() * p_C * (1.0 - p_C)) - 1.0 / prior_variance
            theta = (theta - gradient / hessian).clamp(-4.0, 4.0)
        return theta

    def training_loss(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
        y_T: torch.Tensor,
        *,
        n_z_samples: int = 4,
        kl_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        x_C_encoded = self.encode_x(x_C)
        x_T_encoded = self.encode_x(x_T)
        q_z_C = self.q_z(x_C_encoded, y_C)
        # Following NP/ANP training, C is a subset of T and q(z | s_T) is the target posterior.
        q_z_T = self.q_z(x_T_encoded, y_T)
        z_T = q_z_T.rsample((n_z_samples,))
        z_C = q_z_C.rsample((n_z_samples,))
        r_star = self.r_star(x_C_encoded, y_C, x_T_encoded)
        theta_anchor = self.map_theta(x_C_encoded, y_C)
        logits_T = self.decode_logits(z_T, x_T_encoded, r_star, theta_anchor)
        logits_C = self.decode_logits(z_C, x_T_encoded, r_star, theta_anchor)
        y_T_batched = y_T[None, :].expand_as(logits_T)
        nll_T = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_T, y_T_batched, reduction="mean"
        )
        nll_C = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_C, y_T_batched, reduction="mean"
        )
        nll = 0.5 * (nll_T + nll_C)
        kl = kl_divergence(q_z_T, q_z_C)
        loss = nll + kl_weight * kl
        return loss, {
            "nll": float(nll.detach()),
            "nll_C": float(nll_C.detach()),
            "nll_T": float(nll_T.detach()),
            "kl": float(kl.detach()),
        }

    @torch.no_grad()
    def predict(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
        *,
        n_z_samples: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_C_encoded = self.encode_x(x_C)
        x_T_encoded = self.encode_x(x_T)
        q_z_C = self.q_z(x_C_encoded, y_C)
        z = q_z_C.rsample((n_z_samples,))
        r_star = self.r_star(x_C_encoded, y_C, x_T_encoded)
        theta_anchor = self.map_theta(x_C_encoded, y_C)
        p_y_T = torch.sigmoid(self.decode_logits(z, x_T_encoded, r_star, theta_anchor))
        return p_y_T.mean(dim=0), p_y_T.var(dim=0, unbiased=False)


class Amortized2PL(nn.Module):
    """Semantic item network with offline task abilities and online MAP adaptation."""

    def __init__(self, x_dim: int, n_tasks: int, d_model: int = 128):
        super().__init__()
        self.irt_parameter_net = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )
        self.theta = nn.Parameter(torch.zeros(n_tasks))

    def item_parameters(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return 2PL discrimination a(x) and difficulty b(x)."""

        raw = self.irt_parameter_net(x)
        return torch.nn.functional.softplus(raw[:, 0]) + 0.05, raw[:, 1]

    def logits(self, x: torch.Tensor, task_indices: torch.Tensor) -> torch.Tensor:
        a, b = self.item_parameters(x)
        return a * (self.theta[task_indices] - b)


def train_amortized_2pl(
    tasks: list[dict[str, np.ndarray]],
    *,
    seed: int,
    steps: int = 800,
) -> tuple[Amortized2PL, list[dict[str, float]]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = np.concatenate([task["x"] for task in tasks])
    y = np.concatenate([task["y"] for task in tasks])
    task_indices = np.concatenate(
        [np.full(len(task["y"]), index) for index, task in enumerate(tasks)]
    )
    model = Amortized2PL(x.shape[1], len(tasks)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        indices = rng.integers(0, len(y), size=min(2048, len(y)))
        x_batch = torch.as_tensor(x[indices], dtype=torch.float32, device=device)
        y_batch = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
        batch_tasks = torch.as_tensor(task_indices[indices], dtype=torch.long, device=device)
        logits = model.logits(x_batch, batch_tasks)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_batch)
        loss = loss + 1e-3 * model.theta.square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            model.theta -= model.theta.mean()
        if step % 50 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})
    return model, history


def score_amortized_2pl(
    model: Amortized2PL,
    task: dict[str, Any],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    x_T = torch.as_tensor(task["x_T"], dtype=torch.float32, device=device)
    y_T = np.asarray(task["y_T"], dtype=float)
    a_T, b_T = model.item_parameters(x_T)
    result = {}
    for budget in DEFAULT_BUDGETS:
        x_C = torch.as_tensor(task["x_C_pool"][:budget], dtype=torch.float32, device=device)
        y_C = torch.as_tensor(task["y_C_pool"][:budget], dtype=torch.float32, device=device)
        theta = torch.zeros(1, device=device, requires_grad=True)
        if budget:
            a_C, b_C = model.item_parameters(x_C)
            a_C = a_C.detach()
            b_C = b_C.detach()
            optimizer = torch.optim.Adam([theta], lr=0.05)
            for _ in range(200):
                logits = a_C * (theta - b_C)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y_C, reduction="sum"
                ) + theta.square().sum() / 8.0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            p_y_T = torch.sigmoid(a_T * (theta.detach() - b_T)).cpu().numpy()
        result[str(budget)] = {
            "brier": float(np.mean((p_y_T - y_T) ** 2)),
            "ece": calibration_ece(y_T.tolist(), p_y_T.tolist()),
            "mean_probability": float(p_y_T.mean()),
            "mean_epistemic_variance": 0.0,
        }
    result["brier_alc"] = float(
        sum(
            weight * result[str(budget)]["brier"]
            for budget, weight in zip(DEFAULT_BUDGETS, DEFAULT_WEIGHTS, strict=True)
        )
    )
    return result


class CNPAIRT(nn.Module):
    """Deterministic CNP with a regularized 2PL MAP ability anchor."""

    def __init__(self, x_dim: int, d_model: int = 128):
        super().__init__()
        self.x_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        self.r_encoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.theta_delta = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )
        self.irt_parameter_net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2)
        )

    @torch.no_grad()
    def map_theta(self, x_C_encoded: torch.Tensor, y_C: torch.Tensor) -> torch.Tensor:
        theta = torch.zeros((), device=x_C_encoded.device)
        if x_C_encoded.shape[0] == 0:
            return theta
        raw = self.irt_parameter_net(x_C_encoded)
        a_C = torch.nn.functional.softplus(raw[:, 0]) + 0.05
        b_C = raw[:, 1]
        for _ in range(12):
            p_C = torch.sigmoid(a_C * (theta - b_C))
            gradient = torch.sum(a_C * (y_C - p_C)) - theta / 4.0
            hessian = -torch.sum(a_C.square() * p_C * (1.0 - p_C)) - 0.25
            theta = (theta - gradient / hessian).clamp(-4.0, 4.0)
        return theta

    def logits(self, x_C: torch.Tensor, y_C: torch.Tensor, x_T: torch.Tensor) -> torch.Tensor:
        x_C_encoded = self.x_encoder(x_C)
        x_T_encoded = self.x_encoder(x_T)
        theta = self.map_theta(x_C_encoded, y_C)
        if x_C_encoded.shape[0] > 0:
            r_i = self.r_encoder(torch.cat([x_C_encoded, y_C[:, None]], dim=-1))
            theta = theta + self.theta_delta(r_i.mean(dim=0)).squeeze(-1)
        raw = self.irt_parameter_net(x_T_encoded)
        a_T = torch.nn.functional.softplus(raw[:, 0]) + 0.05
        b_T = raw[:, 1]
        return a_T * (theta - b_T)

    def training_loss(
        self, x_C: torch.Tensor, y_C: torch.Tensor, x_T: torch.Tensor, y_T: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            self.logits(x_C, y_C, x_T), y_T
        )

    @torch.no_grad()
    def predict(self, x_C: torch.Tensor, y_C: torch.Tensor, x_T: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x_C, y_C, x_T))


def train_cnp(
    tasks: list[dict[str, np.ndarray]], *, seed: int, steps: int = 800, episodes_per_step: int = 4
) -> tuple[CNPAIRT, list[dict[str, float]]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNPAIRT(tasks[0]["x"].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        loss = torch.stack(
            [
                model.training_loss(*_episode(tasks[int(rng.integers(len(tasks)))], rng, device))
                for _ in range(episodes_per_step)
            ]
        ).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})
    return model, history


def score_cnp(model: CNPAIRT, task: dict[str, Any]) -> dict[str, Any]:
    device = next(model.parameters()).device
    x_T = torch.as_tensor(task["x_T"], dtype=torch.float32, device=device)
    y_T = np.asarray(task["y_T"], dtype=float)
    result = {}
    for budget in DEFAULT_BUDGETS:
        x_C = torch.as_tensor(task["x_C_pool"][:budget], dtype=torch.float32, device=device)
        y_C = torch.as_tensor(task["y_C_pool"][:budget], dtype=torch.float32, device=device)
        p_y_T = model.predict(x_C, y_C, x_T).cpu().numpy()
        result[str(budget)] = {
            "brier": float(np.mean((p_y_T - y_T) ** 2)),
            "ece": calibration_ece(y_T.tolist(), p_y_T.tolist()),
        }
    result["brier_alc"] = float(
        sum(
            weight * result[str(budget)]["brier"]
            for budget, weight in zip(DEFAULT_BUDGETS, DEFAULT_WEIGHTS, strict=True)
        )
    )
    return result


class TNPDAIRT(nn.Module):
    """TfNP-D with an exact Bernoulli head and an IRT-anchored logit."""

    def __init__(
        self,
        x_dim: int,
        *,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        theta_delta_scale: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.theta_delta_scale = theta_delta_scale
        self.x_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        self.xy_encoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            2 * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.prior_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.irt_parameter_net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2)
        )
        self.theta_head = nn.Linear(d_model, 1)

    @torch.no_grad()
    def map_theta(
        self,
        x_C_encoded: torch.Tensor,
        y_C: torch.Tensor,
        *,
        prior_variance: float = 4.0,
        steps: int = 12,
    ) -> torch.Tensor:
        theta = torch.zeros((), device=x_C_encoded.device)
        if x_C_encoded.shape[0] == 0:
            return theta
        raw_irt_parameters = self.irt_parameter_net(x_C_encoded)
        a_C = torch.nn.functional.softplus(raw_irt_parameters[:, 0]) + 0.05
        b_C = raw_irt_parameters[:, 1]
        for _ in range(steps):
            p_C = torch.sigmoid(a_C * (theta - b_C))
            gradient = torch.sum(a_C * (y_C - p_C)) - theta / prior_variance
            hessian = -torch.sum(a_C.square() * p_C * (1.0 - p_C)) - 1.0 / prior_variance
            theta = (theta - gradient / hessian).clamp(-4.0, 4.0)
        return theta

    def logits(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        x_C_encoded = self.x_encoder(x_C)
        x_T_encoded = self.x_encoder(x_T)
        C_tokens = self.xy_encoder(
            torch.cat([x_C_encoded, y_C[:, None]], dim=-1)
        )
        T_tokens = self.xy_encoder(
            torch.cat(
                [x_T_encoded, torch.zeros(len(x_T_encoded), 1, device=x_T_encoded.device)],
                dim=-1,
            )
        )
        tokens = torch.cat(
            [self.prior_token.to(x_T_encoded.device), C_tokens[None], T_tokens[None]],
            dim=1,
        )
        n_C_with_prior = len(C_tokens) + 1
        n_tokens = tokens.shape[1]
        mask = torch.full(
            (n_tokens, n_tokens), float("-inf"), device=x_T_encoded.device
        )
        mask[:, :n_C_with_prior] = 0.0
        encoded = self.transformer(tokens, mask=mask)
        h_T = encoded[:, -len(x_T_encoded) :].squeeze(0)
        raw_irt_parameters = self.irt_parameter_net(x_T_encoded)
        a_T = torch.nn.functional.softplus(raw_irt_parameters[:, 0]) + 0.05
        b_T = raw_irt_parameters[:, 1]
        theta_anchor = self.map_theta(x_C_encoded, y_C)
        theta_T = theta_anchor + self.theta_delta_scale * self.theta_head(h_T).squeeze(-1)
        irt_logit_T = a_T * (theta_T - b_T)
        return irt_logit_T

    def training_loss(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
        y_T: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            self.logits(x_C, y_C, x_T), y_T
        )

    @torch.no_grad()
    def predict(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.logits(x_C, y_C, x_T))


def train_tnpd(
    tasks: list[dict[str, np.ndarray]],
    *,
    seed: int,
    steps: int = 800,
    episodes_per_step: int = 4,
) -> tuple[TNPDAIRT, list[dict[str, float]]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TNPDAIRT(tasks[0]["x"].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        losses = []
        for _ in range(episodes_per_step):
            task = tasks[int(rng.integers(len(tasks)))]
            losses.append(model.training_loss(*_episode(task, rng, device)))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})
    return model, history


def score_tnpd(model: TNPDAIRT, task: dict[str, Any]) -> dict[str, Any]:
    device = next(model.parameters()).device
    x_T = torch.as_tensor(task["x_T"], dtype=torch.float32, device=device)
    y_T = np.asarray(task["y_T"], dtype=float)
    result = {}
    for budget in DEFAULT_BUDGETS:
        x_C = torch.as_tensor(task["x_C_pool"][:budget], dtype=torch.float32, device=device)
        y_C = torch.as_tensor(task["y_C_pool"][:budget], dtype=torch.float32, device=device)
        p_y_T = model.predict(x_C, y_C, x_T).cpu().numpy()
        result[str(budget)] = {
            "brier": float(np.mean((p_y_T - y_T) ** 2)),
            "ece": calibration_ece(y_T.tolist(), p_y_T.tolist()),
            "mean_probability": float(p_y_T.mean()),
            "mean_epistemic_variance": 0.0,
            "mean_bernoulli_variance": float(np.mean(p_y_T * (1.0 - p_y_T))),
        }
    result["brier_alc"] = float(
        sum(
            weight * result[str(budget)]["brier"]
            for budget, weight in zip(DEFAULT_BUDGETS, DEFAULT_WEIGHTS, strict=True)
        )
    )
    return result


def _episode(task: dict[str, np.ndarray], rng: np.random.Generator, device: torch.device):
    n = len(task["y"])
    order = rng.permutation(n)
    n_C = min(int(rng.choice(DEFAULT_BUDGETS)), n - 1)
    n_T = min(n, max(64, n_C + 32))
    indices_C = order[:n_C]
    indices_T = np.concatenate([indices_C, order[n_C:n_T]])
    x = torch.as_tensor(task["x"], dtype=torch.float32, device=device)
    y = torch.as_tensor(task["y"], dtype=torch.float32, device=device)
    return (
        x[indices_C],
        y[indices_C],
        x[indices_T],
        y[indices_T],
    )


def train_model(
    tasks: list[dict[str, np.ndarray]],
    *,
    variant: str,
    seed: int,
    steps: int = 800,
    episodes_per_step: int = 4,
) -> tuple[FaithfulNPAIRT, list[dict[str, float]]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaithfulNPAIRT(tasks[0]["x"].shape[1], variant=variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        losses = []
        diagnostics = []
        for _ in range(episodes_per_step):
            task = tasks[int(rng.integers(len(tasks)))]
            episode = _episode(task, rng, device)
            loss, diagnostic = model.training_loss(*episode)
            losses.append(loss)
            diagnostics.append(diagnostic)
        batch_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == steps - 1:
            history.append(
                {
                    "step": step,
                    "loss": float(batch_loss.detach()),
                    "nll": float(np.mean([value["nll"] for value in diagnostics])),
                    "nll_C": float(np.mean([value["nll_C"] for value in diagnostics])),
                    "nll_T": float(np.mean([value["nll_T"] for value in diagnostics])),
                    "kl": float(np.mean([value["kl"] for value in diagnostics])),
                }
            )
    return model, history


def score_model(
    model: FaithfulNPAIRT,
    task: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    x_T = torch.as_tensor(task["x_T"], dtype=torch.float32, device=device)
    y_T = np.asarray(task["y_T"], dtype=float)
    result = {}
    for budget in DEFAULT_BUDGETS:
        seed_everything(seed + budget)
        x_C = torch.as_tensor(task["x_C_pool"][:budget], dtype=torch.float32, device=device)
        y_C = torch.as_tensor(task["y_C_pool"][:budget], dtype=torch.float32, device=device)
        p_y_T, var_p_y_T = model.predict(x_C, y_C, x_T)
        p_y_T_numpy = p_y_T.cpu().numpy()
        result[str(budget)] = {
            "brier": float(np.mean((p_y_T_numpy - y_T) ** 2)),
            "ece": calibration_ece(y_T.tolist(), p_y_T_numpy.tolist()),
            "mean_probability": float(p_y_T_numpy.mean()),
            "mean_epistemic_variance": float(var_p_y_T.mean().cpu()),
        }
    result["brier_alc"] = float(
        sum(
            weight * result[str(budget)]["brier"]
            for budget, weight in zip(DEFAULT_BUDGETS, DEFAULT_WEIGHTS, strict=True)
        )
    )
    return result


def write_metrics(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
