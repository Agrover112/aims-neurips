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


def binary_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Pairwise ROC-AUC without adding a sklearn dependency."""
    positive = np.asarray(y_true) == 1
    negative = np.asarray(y_true) == 0
    if not positive.any() or not negative.any():
        return float("nan")
    pair = probability[positive, None] - probability[negative]
    return float((pair > 0).mean() + 0.5 * (pair == 0).mean())


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
        anchor_scale: float = 1.0,
        label_scale: float = 1.0,
        center_labels: bool = False,
        subject_feature_slice: tuple[int, int] | None = None,
        latent_item_parameters: bool = False,
        direct_decoder: bool = False,
    ):
        super().__init__()
        if variant not in {"np", "anp", "stacked_mhsa_anp"}:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.d_model = d_model
        self.d_z = d_z
        self.residual_scale = residual_scale
        self.anchor_scale = anchor_scale
        self.label_scale = label_scale
        self.center_labels = center_labels
        self.subject_feature_slice = subject_feature_slice
        self.latent_item_parameters = latent_item_parameters
        self.direct_decoder = bool(direct_decoder)
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

        self.theta_from_z = nn.Sequential(nn.Linear(d_z, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.latent_item_net = None
        if latent_item_parameters:
            self.latent_item_net = nn.Sequential(
                nn.Linear(d_model + d_z, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 2),
            )
        d_r = d_model if variant != "np" else 0
        self.direct_logit_decoder = None
        if self.direct_decoder:
            self.direct_logit_decoder = nn.Sequential(
                nn.Linear(d_model + d_z + d_r, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )
        self.decoder_residual = None
        if residual_scale > 0.0 and not self.direct_decoder:
            self.decoder_residual = nn.Sequential(
                nn.Linear(d_model + d_z + d_r, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )

    def encode_x(self, x: torch.Tensor) -> torch.Tensor:
        """Encode item tokens without repeating subject identity in every token."""

        return self.encode_item_x(x)

    def encode_item_x(self, x: torch.Tensor) -> torch.Tensor:
        """Encode item inputs while excluding subject metadata from IRT parameters."""

        if self.subject_feature_slice is None:
            return self.x_encoder(x)
        start, stop = self.subject_feature_slice
        item_x = torch.cat(
            [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]], dim=-1
        )
        return self.x_encoder(item_x)

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
        label_value = 2.0 * y[:, None] - 1.0 if self.center_labels else y[:, None]
        xy = torch.cat([x_encoded, self.label_scale * label_value], dim=-1)[None, :, :]
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
            return torch.zeros(x_T_encoded.shape[0], self.d_model, device=x_T_encoded.device)
        label_value = 2.0 * y_C[:, None] - 1.0 if self.center_labels else y_C[:, None]
        xy_C = torch.cat([x_C_encoded, self.label_scale * label_value], dim=-1)[None, :, :]
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
        """Decode target logits from the latent, item features, and context.

        Channel structure, kept identical across the ladder:

        - Base logit is always the item-level IRT form ``a(x)(theta - b(x))``
          with item-only parameters, so subject identity cannot reach ``a`` or
          ``b`` and the MAP anchor stays identified.
        - ``r_star`` is ANP's per-target deterministic representation. It enters
          only through ``decoder_residual``, which is built when
          ``residual_scale > 0``. At scale 0 the ANP row degenerates to latent
          attention in the encoder and is not an ANP decoder.
        - The residual is a post-anchor correction. ``theta_anchor`` is fit on
          the base logits, so it is not the mode of the corrected logits. Do
          not include ``r_star`` in the anchor fit: for context items it
          attends to its own label.
        """

        n_z_samples = z.shape[0]
        n_T = x_T_encoded.shape[0]
        z_T = z[:, None, :].expand(-1, n_T, -1)
        x_T_batched = x_T_encoded[None, :, :].expand(n_z_samples, -1, -1)
        if self.direct_logit_decoder is not None:
            decoder_inputs = [x_T_batched, z_T]
            if r_star is not None:
                decoder_inputs.append(r_star[None, :, :].expand(n_z_samples, -1, -1))
            return self.direct_logit_decoder(
                torch.cat(decoder_inputs, dim=-1)
            ).squeeze(-1)

        theta = self.anchor_scale * theta_anchor + self.theta_from_z(z).squeeze(-1)
        raw_irt_parameters = self.irt_parameter_net(x_T_encoded)
        if self.latent_item_net is None:
            a = torch.nn.functional.softplus(raw_irt_parameters[:, 0]) + 0.05
            b = raw_irt_parameters[:, 1]
            irt_logit = a[None, :] * (theta[:, None] - b[None, :])
        else:
            n_z_samples = z.shape[0]
            n_T = x_T_encoded.shape[0]
            z_T = z[:, None, :].expand(-1, n_T, -1)
            x_T_batched = x_T_encoded[None, :, :].expand(n_z_samples, -1, -1)
            adjustment = self.latent_item_net(torch.cat([x_T_batched, z_T], dim=-1))
            raw_a = raw_irt_parameters[None, :, 0] + adjustment[..., 0]
            b = raw_irt_parameters[None, :, 1] + adjustment[..., 1]
            a = torch.nn.functional.softplus(raw_a) + 0.05
            irt_logit = a * (theta[:, None] - b)
        if self.decoder_residual is None:
            return irt_logit
        decoder_inputs = [x_T_batched, z_T]
        if r_star is not None:
            decoder_inputs.append(r_star[None, :, :].expand(n_z_samples, -1, -1))
        residual = self.decoder_residual(torch.cat(decoder_inputs, dim=-1)).squeeze(-1)
        # Preserve the MAP/IRT level: r* may reorder targets, but cannot
        # shift the target-set mean logit.
        residual = residual - residual.mean(dim=-1, keepdim=True)
        return irt_logit + self.residual_scale * residual

    def decoder_jacobian_singular_values(self, x_T: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return singular values of d logits(x_T, z) / d z for latent-rank audits."""

        x_T_encoded = self.encode_item_x(x_T)

        def logits(single_z: torch.Tensor) -> torch.Tensor:
            return self.decode_logits(
                single_z[None, :],
                x_T_encoded,
                None,
                torch.zeros((), device=x_T.device),
            ).squeeze(0)

        jacobian = torch.autograd.functional.jacobian(logits, z)
        return torch.linalg.svdvals(jacobian)

    @torch.no_grad()
    def map_theta(
        self,
        x_C_encoded: torch.Tensor,
        y_C: torch.Tensor,
        *,
        prior_mean: float = 0.0,
        prior_variance: float = 4.0,
        steps: int = 12,
    ) -> torch.Tensor:
        """Regularized 2PL MAP anchor computed from the acquired context."""

        theta = torch.as_tensor(float(prior_mean), device=x_C_encoded.device)
        if x_C_encoded.shape[0] == 0:
            return theta
        raw_irt_parameters = self.irt_parameter_net(x_C_encoded)
        a_C = torch.nn.functional.softplus(raw_irt_parameters[:, 0]) + 0.05
        b_C = raw_irt_parameters[:, 1]
        for _ in range(steps):
            p_C = torch.sigmoid(a_C * (theta - b_C))
            gradient = torch.sum(a_C * (y_C - p_C)) - (theta - prior_mean) / prior_variance
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
        ability_aux_weight: float = 0.0,
        prior_nll_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Negative ELBO per target.

        Two corrections against the original version of this function:

        1. **The KL is now divided by the target count.** The likelihood is a
           ``reduction="mean"`` over targets while ``kl_divergence`` is a sum
           over latent dimensions, so the old ``nll + kl_weight * kl`` actually
           minimised ``(1/n)*sum_nll + KL`` -- i.e. the true objective
           ``sum_nll + KL`` with the KL weighted **n_targets times too
           heavily** (64x at n=64, 1000x at n=1000). The cheapest way to pay
           down a KL that large is to make q(z|C u T) equal q(z|C), which is
           posterior collapse: the latent stops carrying any information about
           the targets. Correct per-target objective is ``mean_nll + KL/n``.

        2. **The prior-sample likelihood is now opt-in and off by default.**
           The old code used ``nll = 0.5*(nll_T + nll_C)`` where ``nll_C``
           scores the targets under a sample from the *prior* q(z|C). No ELBO
           contains such a term -- the bound is ``E_q[log p(y|z)]`` under the
           posterior alone. Worse, it masks the collapse above: it forces the
           prior path to predict well, so the model still looks healthy on the
           loss curve even when z is uninformative. Set
           ``prior_nll_weight > 0`` to restore the old heuristic for an
           ablation.
        """
        x_C_encoded = self.encode_x(x_C)
        x_T_encoded = self.encode_x(x_T)
        x_C_item_encoded = self.encode_item_x(x_C)
        x_T_item_encoded = self.encode_item_x(x_T)
        q_z_C = self.q_z(x_C_encoded, y_C)
        # Following NP/ANP training, C is a subset of T and q(z | s_T) is the target posterior.
        q_z_T = self.q_z(x_T_encoded, y_T)
        z_T = q_z_T.rsample((n_z_samples,))
        z_C = q_z_C.rsample((n_z_samples,))
        r_star = self.r_star(x_C_encoded, y_C, x_T_encoded)
        theta_anchor = self.map_theta(x_C_item_encoded, y_C)
        logits_T = self.decode_logits(z_T, x_T_item_encoded, r_star, theta_anchor)
        logits_C = self.decode_logits(z_C, x_T_item_encoded, r_star, theta_anchor)
        y_T_batched = y_T[None, :].expand_as(logits_T)
        nll_T = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_T, y_T_batched, reduction="mean"
        )
        nll_C = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_C, y_T_batched, reduction="mean"
        )
        # ELBO uses the posterior-sample likelihood only; the prior term is a
        # heuristic kept behind a flag (see the docstring).
        nll = (1.0 - prior_nll_weight) * nll_T + prior_nll_weight * nll_C
        # Per-target scale: nll is a mean over targets, so the summed KL has to
        # be divided by the same count or it is n_targets times too strong.
        kl = kl_divergence(q_z_T, q_z_C) / max(1, y_T.numel())
        ability_aux = torch.zeros((), device=x_C.device)
        if ability_aux_weight > 0.0:
            mu_theta_C = self.theta_from_z(q_z_C.base_dist.loc).squeeze()
            mu_theta_T = self.theta_from_z(q_z_T.base_dist.loc).squeeze()
            if y_C.numel():
                rate_C = (y_C.sum() + 2.0) / (y_C.numel() + 4.0)
                ability_aux = ability_aux + (mu_theta_C - torch.logit(rate_C)) ** 2
            rate_T = (y_T.sum() + 2.0) / (y_T.numel() + 4.0)
            ability_aux = ability_aux + (mu_theta_T - torch.logit(rate_T)) ** 2
        loss = nll + kl_weight * kl + ability_aux_weight * ability_aux
        return loss, {
            "nll": float(nll.detach()),
            "nll_C": float(nll_C.detach()),
            "nll_T": float(nll_T.detach()),
            "kl": float(kl.detach()),
            "ability_aux": float(ability_aux.detach()),
        }

    @torch.no_grad()
    def predict(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
        *,
        n_z_samples: int = 256,
        prior_mean: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_C_encoded = self.encode_x(x_C)
        x_T_encoded = self.encode_x(x_T)
        x_C_item_encoded = self.encode_item_x(x_C)
        x_T_item_encoded = self.encode_item_x(x_T)
        q_z_C = self.q_z(x_C_encoded, y_C)
        z = q_z_C.rsample((n_z_samples,))
        r_star = self.r_star(x_C_encoded, y_C, x_T_encoded)
        theta_anchor = self.map_theta(x_C_item_encoded, y_C, prior_mean=prior_mean)
        p_y_T = torch.sigmoid(self.decode_logits(z, x_T_item_encoded, r_star, theta_anchor))
        return p_y_T.mean(dim=0), p_y_T.var(dim=0, unbiased=False)


class Amortized1PL(nn.Module):
    """Semantic item network with unit discrimination and online MAP adaptation.

    The only free item parameter is difficulty ``b(x)``. Task abilities are
    learned offline for meta-training and replaced by the same penalized MAP
    solve used by the 2PL model at prediction time.
    """

    def __init__(
        self,
        x_dim: int,
        n_tasks: int,
        d_model: int = 128,
        subject_feature_slice: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.subject_feature_slice = subject_feature_slice
        self.irt_parameter_net = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.theta = nn.Parameter(torch.zeros(n_tasks))

    def item_difficulty(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 1PL item difficulty ``b(x)``."""

        if self.subject_feature_slice is not None:
            start, stop = self.subject_feature_slice
            x = torch.cat(
                [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]], dim=-1
            )
        return self.irt_parameter_net(x).squeeze(-1)

    def item_parameters(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unit discrimination and item difficulty for shared scoring code."""

        difficulty = self.item_difficulty(x)
        return torch.ones_like(difficulty), difficulty

    def logits(self, x: torch.Tensor, task_indices: torch.Tensor) -> torch.Tensor:
        return self.theta[task_indices] - self.item_difficulty(x)


class Amortized2PL(nn.Module):
    """Semantic item network with offline task abilities and online MAP adaptation."""

    def __init__(
        self,
        x_dim: int,
        n_tasks: int,
        d_model: int = 128,
        subject_feature_slice: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.subject_feature_slice = subject_feature_slice
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

        if self.subject_feature_slice is not None:
            start, stop = self.subject_feature_slice
            x = torch.cat(
                [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]], dim=-1
            )
        raw = self.irt_parameter_net(x)
        return torch.nn.functional.softplus(raw[:, 0]) + 0.05, raw[:, 1]

    def logits(self, x: torch.Tensor, task_indices: torch.Tensor) -> torch.Tensor:
        a, b = self.item_parameters(x)
        return a * (self.theta[task_indices] - b)


def task_row_weights(
    task_indices: np.ndarray, n_tasks: int, *, balanced: bool
) -> np.ndarray:
    """Per-row loss weights that give every task equal total influence.

    The amortized trainers below flatten all tasks into one row pool and draw
    uniformly from it, so a task's influence is proportional to its item count.
    Tasks are not comparable in size -- ``babilong`` carries 526k items and
    ``devbench`` 271 -- so uniform row sampling lets a handful of large
    benchmarks set the item network's parameters.

    Weighting each row by ``1 / rows_in_its_task`` makes every task contribute
    the same total mass regardless of size, which is also how the evaluation
    metric treats them: ALC is averaged within a task first, then equally
    across tasks. Weights are rescaled to mean one so the effective learning
    rate does not change when the flag is toggled.

    This matters only for the flat-batch amortized trainers. The episodic
    meta-training loops (``train_cnp``, ``train_model``) already sample one
    task uniformly per episode and average within it, so they are balanced by
    construction and need no weighting.

    Returns an array of ones when ``balanced`` is false.
    """

    weights = np.ones(len(task_indices), dtype=np.float32)
    if not balanced:
        return weights
    counts = np.bincount(task_indices, minlength=n_tasks).astype(np.float64)
    per_row = 1.0 / np.clip(counts[task_indices], 1.0, None)
    return (per_row / per_row.mean()).astype(np.float32)


def cap_task_items(
    tasks: list[dict[str, np.ndarray]],
    max_items_per_task: int | None,
    rng: np.random.Generator,
) -> list[dict[str, np.ndarray]]:
    """Subsample each task to at most ``max_items_per_task`` items.

    The cap is a training dial, not a property of the corpus. Embeddings are
    per-item and reusable, so the full corpus is encoded once and the cap is
    applied here, per run, instead of being baked into the feature store the
    way ``modal_allbench.TRAIN_ITEMS_PER_TASK`` does it.

    Sampling is random rather than ``head``: taking the first N by acquisition
    rank, as the store builder does, biases training toward early-ranked items.
    """

    if max_items_per_task is None:
        return tasks
    if max_items_per_task < 1:
        raise ValueError("max_items_per_task must be positive")
    capped = []
    for task in tasks:
        n = len(task["y"])
        if n <= max_items_per_task:
            capped.append(task)
            continue
        keep = np.sort(rng.permutation(n)[:max_items_per_task])
        capped.append({**task, "x": task["x"][keep], "y": task["y"][keep]})
    return capped


def train_amortized_1pl(
    tasks: list[dict[str, np.ndarray]],
    *,
    seed: int,
    steps: int = 800,
    subject_feature_slice: tuple[int, int] | None = None,
    task_balanced: bool = False,
    max_items_per_task: int | None = None,
) -> tuple[Amortized1PL, list[dict[str, float]]]:
    """Train a 1PL item network with one offline ability per training task."""

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = cap_task_items(tasks, max_items_per_task, np.random.default_rng(seed))
    x = np.concatenate([task["x"] for task in tasks])
    y = np.concatenate([task["y"] for task in tasks])
    task_indices = np.concatenate(
        [np.full(len(task["y"]), index) for index, task in enumerate(tasks)]
    )
    row_weights = task_row_weights(task_indices, len(tasks), balanced=task_balanced)
    model = Amortized1PL(x.shape[1], len(tasks), subject_feature_slice=subject_feature_slice).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        indices = rng.integers(0, len(y), size=min(2048, len(y)))
        x_batch = torch.as_tensor(x[indices], dtype=torch.float32, device=device)
        y_batch = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
        batch_tasks = torch.as_tensor(task_indices[indices], dtype=torch.long, device=device)
        batch_weights = torch.as_tensor(row_weights[indices], dtype=torch.float32, device=device)
        logits = model.logits(x_batch, batch_tasks)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_batch, weight=batch_weights
        )
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


def train_amortized_2pl(
    tasks: list[dict[str, np.ndarray]],
    *,
    seed: int,
    steps: int = 800,
    subject_feature_slice: tuple[int, int] | None = None,
    task_balanced: bool = False,
    max_items_per_task: int | None = None,
) -> tuple[Amortized2PL, list[dict[str, float]]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = cap_task_items(tasks, max_items_per_task, np.random.default_rng(seed))
    x = np.concatenate([task["x"] for task in tasks])
    y = np.concatenate([task["y"] for task in tasks])
    task_indices = np.concatenate(
        [np.full(len(task["y"]), index) for index, task in enumerate(tasks)]
    )
    row_weights = task_row_weights(task_indices, len(tasks), balanced=task_balanced)
    model = Amortized2PL(x.shape[1], len(tasks), subject_feature_slice=subject_feature_slice).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        indices = rng.integers(0, len(y), size=min(2048, len(y)))
        x_batch = torch.as_tensor(x[indices], dtype=torch.float32, device=device)
        y_batch = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
        batch_tasks = torch.as_tensor(task_indices[indices], dtype=torch.long, device=device)
        batch_weights = torch.as_tensor(row_weights[indices], dtype=torch.float32, device=device)
        logits = model.logits(x_batch, batch_tasks)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_batch, weight=batch_weights
        )
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


def map_theta_2pl(
    model: Amortized2PL,
    x_C: torch.Tensor,
    y_C: torch.Tensor,
    *,
    prior_mean: float = 0.0,
    steps: int = 20,
) -> torch.Tensor:
    """Exact 1-D Newton solve of the penalized context likelihood.

    Solves ``sum BCE(a_C (theta - b_C), y_C) + (theta - prior_mean)^2 / 8``.
    The objective is convex in theta, so Newton with a capped step converges in
    a handful of iterations. The earlier Adam loop overshot at budget 1, where a
    single label moved theta far enough to saturate the sigmoid.
    """

    device = next(model.parameters()).device
    theta = torch.as_tensor(float(prior_mean), dtype=torch.float32, device=device)
    if x_C.shape[0] == 0:
        return theta
    a_C, b_C = model.item_parameters(x_C)
    a_C = a_C.detach()
    b_C = b_C.detach()
    y_C = y_C.to(dtype=torch.float32)
    for _ in range(steps):
        probability = torch.sigmoid(a_C * (theta - b_C))
        gradient = torch.sum(a_C * (probability - y_C)) + (theta - prior_mean) / 4.0
        hessian = torch.sum(a_C.square() * probability * (1.0 - probability)) + 0.25
        theta = theta - (gradient / hessian).clamp(-1.0, 1.0)
    return theta.detach()


def map_theta_1pl(
    model: Amortized1PL,
    x_C: torch.Tensor,
    y_C: torch.Tensor,
    *,
    prior_mean: float = 0.0,
    steps: int = 20,
) -> torch.Tensor:
    """Exact 1-D Newton MAP for a 1PL ability, with unit discrimination."""

    device = next(model.parameters()).device
    theta = torch.as_tensor(float(prior_mean), dtype=torch.float32, device=device)
    if x_C.shape[0] == 0:
        return theta
    b_C = model.item_difficulty(x_C).detach()
    y_C = y_C.to(dtype=torch.float32)
    for _ in range(steps):
        probability = torch.sigmoid(theta - b_C)
        gradient = torch.sum(probability - y_C) + (theta - prior_mean) / 4.0
        hessian = torch.sum(probability * (1.0 - probability)) + 0.25
        theta = theta - (gradient / hessian).clamp(-1.0, 1.0)
    return theta.detach()


def score_amortized_1pl(
    model: Amortized1PL,
    task: dict[str, Any],
    *,
    prior_mean: float = 0.0,
) -> dict[str, Any]:
    """Score a 1PL item network with the same protocol as the 2PL model."""

    device = next(model.parameters()).device
    x_T = torch.as_tensor(task["x_T"], dtype=torch.float32, device=device)
    y_T = np.asarray(task["y_T"], dtype=float)
    b_T = model.item_difficulty(x_T)
    result = {}
    for budget in DEFAULT_BUDGETS:
        x_C = torch.as_tensor(task["x_C_pool"][:budget], dtype=torch.float32, device=device)
        y_C = torch.as_tensor(task["y_C_pool"][:budget], dtype=torch.float32, device=device)
        theta = map_theta_1pl(model, x_C, y_C, prior_mean=prior_mean)
        with torch.no_grad():
            p_y_T = torch.sigmoid(theta - b_T).cpu().numpy()
        result[str(budget)] = {
            "brier": float(np.mean((p_y_T - y_T) ** 2)),
            "ece": calibration_ece(y_T.tolist(), p_y_T.tolist()),
            "auc": binary_auc(y_T, p_y_T),
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


def score_amortized_2pl(
    model: Amortized2PL,
    task: dict[str, Any],
    *,
    prior_mean: float = 0.0,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    x_T = torch.as_tensor(task["x_T"], dtype=torch.float32, device=device)
    y_T = np.asarray(task["y_T"], dtype=float)
    a_T, b_T = model.item_parameters(x_T)
    result = {}
    for budget in DEFAULT_BUDGETS:
        x_C = torch.as_tensor(task["x_C_pool"][:budget], dtype=torch.float32, device=device)
        y_C = torch.as_tensor(task["y_C_pool"][:budget], dtype=torch.float32, device=device)
        theta = map_theta_2pl(model, x_C, y_C, prior_mean=prior_mean)
        with torch.no_grad():
            p_y_T = torch.sigmoid(a_T * (theta - b_T)).cpu().numpy()
        result[str(budget)] = {
            "brier": float(np.mean((p_y_T - y_T) ** 2)),
            "ece": calibration_ece(y_T.tolist(), p_y_T.tolist()),
            "auc": binary_auc(y_T, p_y_T),
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

    def __init__(
        self,
        x_dim: int,
        d_model: int = 128,
        subject_feature_slice: tuple[int, int] | None = None,
        label_scale: float = 1.0,
        center_labels: bool = False,
        use_theta_delta: bool = True,
    ):
        super().__init__()
        self.subject_feature_slice = subject_feature_slice
        self.label_scale = label_scale
        self.center_labels = center_labels
        self.use_theta_delta = use_theta_delta
        self.x_encoder = nn.Sequential(nn.Linear(x_dim, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.r_encoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.theta_delta = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )
        self.irt_parameter_net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2)
        )

    def encode_item_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.subject_feature_slice is None:
            return self.x_encoder(x)
        start, stop = self.subject_feature_slice
        item_x = torch.cat(
            [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]], dim=-1
        )
        return self.x_encoder(item_x)

    @torch.no_grad()
    def map_theta(
        self,
        x_C_encoded: torch.Tensor,
        y_C: torch.Tensor,
        *,
        prior_mean: float = 0.0,
    ) -> torch.Tensor:
        theta = torch.as_tensor(float(prior_mean), device=x_C_encoded.device)
        if x_C_encoded.shape[0] == 0:
            return theta
        raw = self.irt_parameter_net(x_C_encoded)
        a_C = torch.nn.functional.softplus(raw[:, 0]) + 0.05
        b_C = raw[:, 1]
        for _ in range(12):
            p_C = torch.sigmoid(a_C * (theta - b_C))
            gradient = torch.sum(a_C * (y_C - p_C)) - (theta - prior_mean) / 4.0
            hessian = -torch.sum(a_C.square() * p_C * (1.0 - p_C)) - 0.25
            theta = (theta - gradient / hessian).clamp(-4.0, 4.0)
        return theta

    def delta_theta(self, x_C_encoded: torch.Tensor, y_C: torch.Tensor) -> torch.Tensor:
        """Return the label-conditioned ability correction, or zero when disabled."""

        if not self.use_theta_delta or x_C_encoded.shape[0] == 0:
            return torch.zeros((), device=x_C_encoded.device)
        label_value = 2.0 * y_C[:, None] - 1.0 if self.center_labels else y_C[:, None]
        r_i = self.r_encoder(torch.cat([x_C_encoded, self.label_scale * label_value], dim=-1))
        return self.theta_delta(r_i.mean(dim=0)).squeeze(-1)

    def logits(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
        *,
        prior_mean: float = 0.0,
        delta_scale: float = 1.0,
    ) -> torch.Tensor:
        x_C_encoded = self.encode_item_x(x_C)
        x_T_item_encoded = self.encode_item_x(x_T)
        theta = self.map_theta(x_C_encoded, y_C, prior_mean=prior_mean)
        if delta_scale != 0.0:
            theta = theta + delta_scale * self.delta_theta(x_C_encoded, y_C)
        raw = self.irt_parameter_net(x_T_item_encoded)
        a_T = torch.nn.functional.softplus(raw[:, 0]) + 0.05
        b_T = raw[:, 1]
        return a_T * (theta - b_T)

    def training_loss(
        self, x_C: torch.Tensor, y_C: torch.Tensor, x_T: torch.Tensor, y_T: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x_C, y_C, x_T), y_T)

    @torch.no_grad()
    def predict(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
        *,
        prior_mean: float = 0.0,
        delta_scale: float = 1.0,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.logits(x_C, y_C, x_T, prior_mean=prior_mean, delta_scale=delta_scale)
        )


def train_cnp(
    tasks: list[dict[str, np.ndarray]],
    *,
    seed: int,
    steps: int = 800,
    episodes_per_step: int = 4,
    subject_feature_slice: tuple[int, int] | None = None,
    disjoint_context_target: bool = False,
    use_theta_delta: bool = True,
) -> tuple[CNPAIRT, list[dict[str, float]]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNPAIRT(
        tasks[0]["x"].shape[1],
        subject_feature_slice=subject_feature_slice,
        use_theta_delta=use_theta_delta,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        loss = torch.stack(
            [
                model.training_loss(
                    *_episode(
                        tasks[int(rng.integers(len(tasks)))],
                        rng,
                        device,
                        disjoint_context_target=disjoint_context_target,
                    )
                )
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


def score_cnp(
    model: CNPAIRT,
    task: dict[str, Any],
    *,
    prior_mean: float = 0.0,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    x_T = torch.as_tensor(task["x_T"], dtype=torch.float32, device=device)
    y_T = np.asarray(task["y_T"], dtype=float)
    result = {}
    for budget in DEFAULT_BUDGETS:
        x_C = torch.as_tensor(task["x_C_pool"][:budget], dtype=torch.float32, device=device)
        y_C = torch.as_tensor(task["y_C_pool"][:budget], dtype=torch.float32, device=device)
        p_y_T = model.predict(x_C, y_C, x_T, prior_mean=prior_mean).cpu().numpy()
        result[str(budget)] = {
            "brier": float(np.mean((p_y_T - y_T) ** 2)),
            "ece": calibration_ece(y_T.tolist(), p_y_T.tolist()),
            "auc": binary_auc(y_T, p_y_T),
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
        subject_feature_slice: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.theta_delta_scale = theta_delta_scale
        self.subject_feature_slice = subject_feature_slice
        self.x_encoder = nn.Sequential(nn.Linear(x_dim, d_model), nn.LayerNorm(d_model), nn.ReLU())
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

    def encode_item_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.subject_feature_slice is None:
            return self.x_encoder(x)
        start, stop = self.subject_feature_slice
        item_x = torch.cat(
            [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]], dim=-1
        )
        return self.x_encoder(item_x)

    @torch.no_grad()
    def map_theta(
        self,
        x_C_encoded: torch.Tensor,
        y_C: torch.Tensor,
        *,
        prior_mean: float = 0.0,
        prior_variance: float = 4.0,
        steps: int = 12,
    ) -> torch.Tensor:
        theta = torch.as_tensor(float(prior_mean), device=x_C_encoded.device)
        if x_C_encoded.shape[0] == 0:
            return theta
        raw_irt_parameters = self.irt_parameter_net(x_C_encoded)
        a_C = torch.nn.functional.softplus(raw_irt_parameters[:, 0]) + 0.05
        b_C = raw_irt_parameters[:, 1]
        for _ in range(steps):
            p_C = torch.sigmoid(a_C * (theta - b_C))
            gradient = torch.sum(a_C * (y_C - p_C)) - (theta - prior_mean) / prior_variance
            hessian = -torch.sum(a_C.square() * p_C * (1.0 - p_C)) - 1.0 / prior_variance
            theta = (theta - gradient / hessian).clamp(-4.0, 4.0)
        return theta

    def logits(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
        *,
        prior_mean: float = 0.0,
    ) -> torch.Tensor:
        x_C_encoded = self.encode_item_x(x_C)
        x_T_encoded = self.encode_item_x(x_T)
        C_tokens = self.xy_encoder(torch.cat([x_C_encoded, y_C[:, None]], dim=-1))
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
        mask = torch.full((n_tokens, n_tokens), float("-inf"), device=x_T_encoded.device)
        mask[:, :n_C_with_prior] = 0.0
        encoded = self.transformer(tokens, mask=mask)
        h_T = encoded[:, -len(x_T_encoded) :].squeeze(0)
        raw_irt_parameters = self.irt_parameter_net(x_T_encoded)
        a_T = torch.nn.functional.softplus(raw_irt_parameters[:, 0]) + 0.05
        b_T = raw_irt_parameters[:, 1]
        theta_anchor = self.map_theta(x_C_encoded, y_C, prior_mean=prior_mean)
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
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x_C, y_C, x_T), y_T)

    @torch.no_grad()
    def predict(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
        *,
        prior_mean: float = 0.0,
    ) -> torch.Tensor:
        return torch.sigmoid(self.logits(x_C, y_C, x_T, prior_mean=prior_mean))


def train_tnpd(
    tasks: list[dict[str, np.ndarray]],
    *,
    seed: int,
    steps: int = 800,
    episodes_per_step: int = 4,
    subject_feature_slice: tuple[int, int] | None = None,
) -> tuple[TNPDAIRT, list[dict[str, float]]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TNPDAIRT(tasks[0]["x"].shape[1], subject_feature_slice=subject_feature_slice).to(device)
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


def score_tnpd(
    model: TNPDAIRT,
    task: dict[str, Any],
    *,
    prior_mean: float = 0.0,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    x_T = torch.as_tensor(task["x_T"], dtype=torch.float32, device=device)
    y_T = np.asarray(task["y_T"], dtype=float)
    result = {}
    for budget in DEFAULT_BUDGETS:
        x_C = torch.as_tensor(task["x_C_pool"][:budget], dtype=torch.float32, device=device)
        y_C = torch.as_tensor(task["y_C_pool"][:budget], dtype=torch.float32, device=device)
        p_y_T = model.predict(x_C, y_C, x_T, prior_mean=prior_mean).cpu().numpy()
        result[str(budget)] = {
            "brier": float(np.mean((p_y_T - y_T) ** 2)),
            "ece": calibration_ece(y_T.tolist(), p_y_T.tolist()),
            "auc": binary_auc(y_T, p_y_T),
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


def _episode(
    task: dict[str, np.ndarray],
    rng: np.random.Generator,
    device: torch.device,
    *,
    disjoint_context_target: bool = False,
):
    n = len(task["y"])
    order = rng.permutation(n)
    n_C = min(int(rng.choice(DEFAULT_BUDGETS)), n - 1)
    n_T = min(n, max(64, n_C + 32))
    indices_C = order[:n_C]
    indices_T = (
        order[n_C:n_T] if disjoint_context_target else np.concatenate([indices_C, order[n_C:n_T]])
    )
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
    kl_weight: float = 1.0,
    ability_aux_weight: float = 0.0,
    model_kwargs: dict[str, Any] | None = None,
    disjoint_context_target: bool = False,
) -> tuple[FaithfulNPAIRT, list[dict[str, float]]]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaithfulNPAIRT(tasks[0]["x"].shape[1], variant=variant, **(model_kwargs or {})).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        losses = []
        diagnostics = []
        for _ in range(episodes_per_step):
            task = tasks[int(rng.integers(len(tasks)))]
            episode = _episode(task, rng, device, disjoint_context_target=disjoint_context_target)
            loss, diagnostic = model.training_loss(
                *episode,
                kl_weight=kl_weight,
                ability_aux_weight=ability_aux_weight,
            )
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
                    "ability_aux": float(np.mean([value["ability_aux"] for value in diagnostics])),
                }
            )
    return model, history


def score_model(
    model: FaithfulNPAIRT,
    task: dict[str, Any],
    *,
    seed: int,
    prior_mean: float = 0.0,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    x_T = torch.as_tensor(task["x_T"], dtype=torch.float32, device=device)
    y_T = np.asarray(task["y_T"], dtype=float)
    result = {}
    for budget in DEFAULT_BUDGETS:
        seed_everything(seed + budget)
        x_C = torch.as_tensor(task["x_C_pool"][:budget], dtype=torch.float32, device=device)
        y_C = torch.as_tensor(task["y_C_pool"][:budget], dtype=torch.float32, device=device)
        p_y_T, var_p_y_T = model.predict(x_C, y_C, x_T, prior_mean=prior_mean)
        p_y_T_numpy = p_y_T.cpu().numpy()
        result[str(budget)] = {
            "brier": float(np.mean((p_y_T_numpy - y_T) ** 2)),
            "ece": calibration_ece(y_T.tolist(), p_y_T_numpy.tolist()),
            "auc": binary_auc(y_T, p_y_T_numpy),
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
