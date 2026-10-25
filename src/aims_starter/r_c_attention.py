"""Isolated deterministic ANP-attention rung for the r_C ablation.

This module is intentionally separate from the production CNP/ANP classes in
``faithful_np.py``.  It implements the missing target-specific context
representation

    r_{C,t} = Attn(query=h_t, key=h_c, value=r_c)

with no latent variable ``z`` and no IRT anchor.  The decoder emits Bernoulli
logits directly, matching the paper's ``phi_t = g(x_t, r)`` form.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# Probit approximation constant: E_z~N(m, v)[sigmoid(z)] ~= sigmoid(m / sqrt(1 + pi/8 * v)).
PROBIT_SCALE = math.pi / 8.0
PRIOR_THETA_VARIANCE = 4.0


def marginalize_logits(mean_logits, logit_variance):
    """Integrate a Bernoulli logit over a Gaussian ability posterior.

    The models solve a Newton MAP for theta and then plug that point estimate
    into the response function. Under a proper scoring rule that is dominated
    by the posterior mean of the probability,

        E_{theta ~ q}[sigmoid(a (theta - b))],

    which is what ``scripts/information_ceiling.py`` evaluates by quadrature
    when it computes the Bayes-risk floor. The plug-in estimate is
    systematically over-confident wherever the posterior is wide -- that is,
    at the small budgets, which carry half the ALC weight.

    The probit approximation gives the integral in closed form, so no
    quadrature is needed at inference: the Newton solve already produces the
    Hessian, and its inverse is the Laplace posterior variance.
    """

    return mean_logits * torch.rsqrt(1.0 + PROBIT_SCALE * logit_variance)


class ContextSelfAttention(nn.Module):
    """Permutation-invariant self-attention over label-conditioned context reps."""

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Self-attend over context tokens.

        ``mask`` is a boolean valid-token mask.  A 2-D input keeps the original
        single-task behavior; a 3-D input is a padded batch.  Rows with no valid
        context return zeros instead of producing NaNs from an all-masked
        attention call.
        """

        if context.shape[0] == 0:
            return context
        single = context.dim() == 2
        if single:
            context = context.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        if context.shape[1] == 0:
            return context.squeeze(0) if single else context
        if mask is None:
            mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
        empty = ~mask.any(dim=1)
        safe_mask = mask
        if bool(empty.any()):
            safe_mask = mask.clone()
            safe_mask[empty, 0] = True
        attended, _ = self.attention(
            context,
            context,
            context,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        output = self.norm(context + attended)
        if bool(empty.any()):
            output = output.clone()
            output[empty] = 0.0
        return output.squeeze(0) if single else output


class TargetContextAttention(nn.Module):
    """Target-specific context representation r_{C,t}."""

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        target: torch.Tensor,
        context_item: torch.Tensor,
        context_value: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attend targets to context items.

        ``context_mask`` marks valid context tokens.  Padded targets are allowed
        and are simply ignored by the caller's loss mask.  A task with an empty
        context gets a zero correction.
        """

        if context_item.shape[0] == 0:
            return torch.zeros_like(target)
        single = target.dim() == 2
        if single:
            target = target.unsqueeze(0)
            context_item = context_item.unsqueeze(0)
            context_value = context_value.unsqueeze(0)
            if context_mask is not None:
                context_mask = context_mask.unsqueeze(0)
        if target.shape[1] == 0:
            return target.squeeze(0) if single else target
        if context_item.shape[1] == 0:
            return torch.zeros_like(target).squeeze(0) if single else torch.zeros_like(target)
        if context_mask is None:
            context_mask = torch.ones(
                context_item.shape[:2],
                dtype=torch.bool,
                device=context_item.device,
            )
        empty = ~context_mask.any(dim=1)
        safe_mask = context_mask
        if bool(empty.any()):
            safe_mask = context_mask.clone()
            safe_mask[empty, 0] = True
        query = self.query(target)
        key = self.key(context_item)
        value = self.value(context_value)
        attended, _ = self.attention(
            query,
            key,
            value,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        output = self.norm(target + attended)
        if bool(empty.any()):
            output = output.clone()
            output[empty] = 0.0
        return output.squeeze(0) if single else output


class DeterministicANPAttention(nn.Module):
    """Deterministic ANP-style rung: attention without latent ``z``.

    Context and target items are encoded separately.  Context values are
    label-conditioned and self-attended; each target then cross-attends to the
    context.  The decoder maps ``[h_t, r_{C,t}]`` to one Bernoulli logit.
    """

    def __init__(
        self,
        x_dim: int,
        *,
        d_model: int = 128,
        n_heads: int = 8,
        n_context_layers: int = 2,
        subject_feature_slice: tuple[int, int] | None = None,
        activation: str = "relu",
        activation_all: bool = False,
    ):
        super().__init__()
        if activation not in ("relu", "gelu"):
            raise ValueError("activation must be 'relu' or 'gelu'")
        self.subject_feature_slice = subject_feature_slice
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        act_all = nn.GELU() if activation_all else nn.ReLU()
        self.item_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.LayerNorm(d_model),
            act,
            nn.Linear(d_model, d_model),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            act_all,
            nn.Linear(d_model, d_model),
        )
        self.context_layers = nn.ModuleList(
            [ContextSelfAttention(d_model, n_heads) for _ in range(n_context_layers)]
        )
        self.cross_attention = TargetContextAttention(d_model, n_heads)
        self.decoder = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            act_all,
            nn.Linear(d_model, 1),
        )

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        if self.subject_feature_slice is not None:
            start, stop = self.subject_feature_slice
            x = torch.cat(
                [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]],
                dim=-1,
            )
        return self.item_encoder(x)

    def forward(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        h_C = self.encode_item(x_C)
        h_T = self.encode_item(x_T)
        if h_C.shape[0] == 0:
            r_C = h_C.new_zeros((0, h_C.shape[-1]))
        else:
            r_C = self.context_encoder(torch.cat([h_C, y_C[:, None]], dim=-1))
            for layer in self.context_layers:
                r_C = layer(r_C)
        r_CT = self.cross_attention(h_T, h_C, r_C)
        return self.decoder(torch.cat([h_T, r_CT], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_C, y_C, x_T))


class _MultidimContextPosterior(nn.Module):
    """Shared native CNP/ANP head with a Gaussian posterior over theta.

    The context set produces ``q(theta | C) = N(mu_C, diag(sigma_C^2))`` with
    prior ``N(0, prior_variance * I)``.  Item parameters are item-only:
    ``a_t`` is a positive D-vector and ``d_t`` is a scalar intercept.  The
    response is the compensatory MIRT logit ``a_t . theta - d_t``, marginalized
    over the posterior with the probit approximation.  Keeping the item head
    item-only prevents the context from bypassing theta through the item
    parameters.
    """

    def __init__(
        self,
        x_dim: int,
        *,
        ability_dim: int = 4,
        d_model: int = 128,
        n_heads: int = 8,
        subject_feature_slice: tuple[int, int] | None = None,
        prior_variance: float = 1.0,
        max_kl_weight: float = 0.01,
    ):
        super().__init__()
        if ability_dim < 1:
            raise ValueError("ability_dim must be positive")
        if prior_variance <= 0.0:
            raise ValueError("prior_variance must be positive")
        self.ability_dim = int(ability_dim)
        self.subject_feature_slice = subject_feature_slice
        self.prior_variance = float(prior_variance)
        self.prior_logvar = math.log(self.prior_variance)
        self.max_kl_weight = float(max_kl_weight)
        self.kl_weight = 0.0
        self._aux_loss = None

        self.item_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.theta_mean = nn.Linear(d_model, self.ability_dim)
        self.theta_logvar = nn.Linear(d_model, self.ability_dim)
        nn.init.zeros_(self.theta_mean.weight)
        nn.init.zeros_(self.theta_mean.bias)
        nn.init.zeros_(self.theta_logvar.weight)
        nn.init.zeros_(self.theta_logvar.bias)
        self.item_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.ability_dim + 1),
        )

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        if self.subject_feature_slice is not None:
            start, stop = self.subject_feature_slice
            x = torch.cat(
                [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]],
                dim=-1,
            )
        return self.item_encoder(x)

    def _context_tokens(self, x_C: torch.Tensor, y_C: torch.Tensor) -> torch.Tensor:
        h_C = self.encode_item(x_C)
        if h_C.shape[0] == 0:
            return h_C
        return self.context_encoder(torch.cat([h_C, y_C[:, None]], dim=-1))

    def _pool_context(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _posterior(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.shape[0] == 0:
            return (
                tokens.new_zeros((self.ability_dim,)),
                tokens.new_full((self.ability_dim,), self.prior_logvar),
            )
        pooled = self._pool_context(tokens)
        return (
            self.theta_mean(pooled),
            self.theta_logvar(pooled).clamp(-6.0, 2.0),
        )

    def _kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        variance = torch.exp(logvar)
        return (
            0.5
            * (
                variance / self.prior_variance
                + mu.square() / self.prior_variance
                - 1.0
                - logvar
                + self.prior_logvar
            ).sum()
        )

    def _item_parameters(self, h_T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.item_head(h_T)
        a = torch.nn.functional.softplus(raw[..., : self.ability_dim]) + 0.05
        return a, raw[..., self.ability_dim]

    def forward(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self._context_tokens(x_C, y_C)
        mu, logvar = self._posterior(tokens)
        a, d = self._item_parameters(self.encode_item(x_T))
        mean_logits = a @ mu - d
        logit_variance = (a.square() * torch.exp(logvar)).sum(dim=-1)
        logits = marginalize_logits(mean_logits, logit_variance)
        self._aux_loss = self.kl_weight * self._kl(mu, logvar)
        return logits

    def aux_loss(self) -> torch.Tensor:
        if self._aux_loss is None:
            return next(self.parameters()).new_zeros(())
        return self._aux_loss

    def set_training_progress(self, step: int, total_steps: int) -> None:
        warmup = max(1, int(total_steps) // 4)
        self.kl_weight = self.max_kl_weight * min(1.0, float(step) / float(warmup))

    @torch.no_grad()
    def predict_proba(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_C, y_C, x_T))


class MultidimCNP(_MultidimContextPosterior):
    """Native CNP with a D-dimensional Gaussian posterior over subject theta."""

    def _pool_context(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=0)


class MultidimANP(_MultidimContextPosterior):
    """Native ANP with self-attention context pooling and a D-dimensional theta.

    The target cross-attention of the deterministic ANP is intentionally not
    used here: theta is a subject-level quantity, so the context is pooled to
    one posterior rather than one representation per target.
    """

    def __init__(
        self,
        x_dim: int,
        *,
        ability_dim: int = 4,
        d_model: int = 128,
        n_heads: int = 8,
        n_context_layers: int = 2,
        subject_feature_slice: tuple[int, int] | None = None,
        prior_variance: float = 1.0,
        max_kl_weight: float = 0.01,
    ):
        super().__init__(
            x_dim,
            ability_dim=ability_dim,
            d_model=d_model,
            n_heads=n_heads,
            subject_feature_slice=subject_feature_slice,
            prior_variance=prior_variance,
            max_kl_weight=max_kl_weight,
        )
        self.context_layers = nn.ModuleList(
            [ContextSelfAttention(d_model, n_heads) for _ in range(n_context_layers)]
        )
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pool_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
        self.pool_norm = nn.LayerNorm(d_model)

    def _pool_context(self, tokens: torch.Tensor) -> torch.Tensor:
        for layer in self.context_layers:
            tokens = layer(tokens)
        query = self.pool_query.expand(1, 1, -1)
        pooled, _ = self.pool_attention(
            query,
            tokens.unsqueeze(0),
            tokens.unsqueeze(0),
            need_weights=False,
        )
        return self.pool_norm(pooled.squeeze(0).squeeze(0))


class BayesianCNPIRT(nn.Module):
    """Bayesian CNP-IRT with amortized Gaussian posteriors.

    The context set produces ``q(theta | C) = N(mu_C, diag(sigma_C^2))`` with
    prior ``N(0, prior_theta_variance * I)``.  The item encoder produces a
    diagonal Gaussian posterior over each item's ``(log a_i, d_i)``, with
    ``a_i = exp(log a_i)`` and prior ``N(0, prior_item_variance * I)``.  The
    response is the compensatory MIRT logit ``a_i . theta - d_i``, marginalized
    over both posteriors with a probit approximation.  The item posterior is
    amortized per item, not per task.
    """

    def __init__(
        self,
        x_dim: int,
        *,
        ability_dim: int = 4,
        d_model: int = 128,
        subject_feature_slice: tuple[int, int] | None = None,
        prior_theta_variance: float = 1.0,
        prior_item_variance: float = 0.25,
        max_kl_weight: float = 0.01,
        input_norm: bool = False,
        activation: str = "relu",
        activation_all: bool = False,
    ):
        super().__init__()
        if ability_dim < 1:
            raise ValueError("ability_dim must be positive")
        if prior_theta_variance <= 0.0 or prior_item_variance <= 0.0:
            raise ValueError("prior variances must be positive")
        self.ability_dim = int(ability_dim)
        self.subject_feature_slice = subject_feature_slice
        self.prior_theta_variance = float(prior_theta_variance)
        self.prior_item_variance = float(prior_item_variance)
        self.prior_variance = self.prior_theta_variance
        self.prior_theta_logvar = math.log(self.prior_theta_variance)
        self.prior_item_logvar = math.log(self.prior_item_variance)
        self.max_kl_weight = float(max_kl_weight)
        self.kl_weight = 0.0
        self._aux_loss = None
        if activation not in ("relu", "gelu"):
            raise ValueError("activation must be 'relu' or 'gelu'")
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        act_all = nn.GELU() if activation_all else nn.ReLU()

        self.context_encoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            act_all,
            nn.Linear(d_model, d_model),
        )
        self.theta_mean = nn.Linear(d_model, self.ability_dim)
        self.theta_logvar = nn.Linear(d_model, self.ability_dim)
        self.item_mean = nn.Linear(d_model, self.ability_dim + 1)
        self.item_logvar = nn.Linear(d_model, self.ability_dim + 1)
        for head in (self.theta_mean, self.item_mean):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for head, bias in (
            (self.theta_logvar, self.prior_theta_logvar),
            (self.item_logvar, self.prior_item_logvar),
        ):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, bias)

        self.input_norm = None
        if input_norm:
            norm_dim = subject_feature_slice[0] if subject_feature_slice is not None else x_dim
            self.input_norm = nn.LayerNorm(norm_dim)
        self.item_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.LayerNorm(d_model),
            act,
            nn.Linear(d_model, d_model),
        )

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        if self.subject_feature_slice is not None:
            start, stop = self.subject_feature_slice
            x = torch.cat(
                [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]],
                dim=-1,
            )
            if self.input_norm is not None:
                x = torch.cat([self.input_norm(x[..., :start]), x[..., start:]], dim=-1)
        elif self.input_norm is not None:
            x = self.input_norm(x)
        return self.item_encoder(x)

    def _context_tokens(self, x_C: torch.Tensor, y_C: torch.Tensor) -> torch.Tensor:
        h_C = self.encode_item(x_C)
        if h_C.shape[0] == 0:
            return h_C
        return self.context_encoder(torch.cat([h_C, y_C[:, None]], dim=-1))

    def _theta_posterior(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.shape[0] == 0:
            return (
                tokens.new_zeros((self.ability_dim,)),
                tokens.new_full((self.ability_dim,), self.prior_theta_logvar),
            )
        pooled = tokens.mean(dim=0)
        return (
            self.theta_mean(pooled),
            self.theta_logvar(pooled).clamp(-6.0, 2.0),
        )

    def _item_posterior(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.item_mean(h),
            self.item_logvar(h).clamp(-6.0, 2.0),
        )

    @staticmethod
    def _diag_kl(mu: torch.Tensor, logvar: torch.Tensor, prior_variance: float) -> torch.Tensor:
        prior_logvar = math.log(prior_variance)
        return 0.5 * (
            torch.exp(logvar) / prior_variance
            + mu.square() / prior_variance
            - 1.0
            - logvar
            + prior_logvar
        ).sum(dim=-1)

    @staticmethod
    def _item_moments(
        mu: torch.Tensor, logvar: torch.Tensor, ability_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_loga = mu[..., :ability_dim]
        var_loga = torch.exp(logvar[..., :ability_dim])
        mean_a = torch.exp(mu_loga + 0.5 * var_loga)
        var_a = (torch.exp(var_loga) - 1.0) * torch.exp(2.0 * mu_loga + var_loga)
        return mean_a, var_a, mu[..., ability_dim], torch.exp(logvar[..., ability_dim])

    def forward(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self._context_tokens(x_C, y_C)
        theta_mu, theta_logvar = self._theta_posterior(tokens)
        item_mu, item_logvar = self._item_posterior(self.encode_item(x_T))
        mean_a, var_a, mu_d, var_d = self._item_moments(item_mu, item_logvar, self.ability_dim)
        theta_var = torch.exp(theta_logvar)
        mean_logits = mean_a @ theta_mu - mu_d
        logit_variance = (
            (mean_a.square() * theta_var).sum(dim=-1)
            + (var_a * (theta_mu.square() + theta_var)).sum(dim=-1)
            + var_d
        )
        logits = marginalize_logits(mean_logits, logit_variance)
        kl_theta = self._diag_kl(theta_mu, theta_logvar, self.prior_theta_variance).sum()
        if item_mu.shape[0] == 0:
            kl_item = item_mu.new_zeros(())
        else:
            kl_item = self._diag_kl(item_mu, item_logvar, self.prior_item_variance).mean()
        self._aux_loss = self.kl_weight * (kl_theta + kl_item)
        return logits

    def aux_loss(self) -> torch.Tensor:
        if self._aux_loss is None:
            return next(self.parameters()).new_zeros(())
        return self._aux_loss

    def set_training_progress(self, step: int, total_steps: int) -> None:
        warmup = max(1, int(total_steps) // 4)
        self.kl_weight = self.max_kl_weight * min(1.0, float(step) / float(warmup))

    @torch.no_grad()
    def predict_proba(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_C, y_C, x_T))


class DeterministicCNP(nn.Module):
    """Pure CNP decoder for the reverse ladder: phi_t = g([h_t, r_C])."""

    def __init__(
        self,
        x_dim: int,
        *,
        d_model: int = 128,
        subject_feature_slice: tuple[int, int] | None = None,
        input_norm: bool = False,
        activation: str = "relu",
        activation_all: bool = False,
    ):
        super().__init__()
        if activation not in ("relu", "gelu"):
            raise ValueError("activation must be 'relu' or 'gelu'")
        self.subject_feature_slice = subject_feature_slice
        self.input_norm = None
        if input_norm:
            norm_dim = subject_feature_slice[0] if subject_feature_slice is not None else x_dim
            self.input_norm = nn.LayerNorm(norm_dim)
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        act_all = nn.GELU() if activation_all else nn.ReLU()
        self.item_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.LayerNorm(d_model),
            act,
            nn.Linear(d_model, d_model),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            act_all,
            nn.Linear(d_model, d_model),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            act_all,
            nn.Linear(d_model, 1),
        )

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        if self.subject_feature_slice is not None:
            start, stop = self.subject_feature_slice
            x = torch.cat(
                [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]],
                dim=-1,
            )
            if self.input_norm is not None:
                x = torch.cat([self.input_norm(x[..., :start]), x[..., start:]], dim=-1)
        elif self.input_norm is not None:
            x = self.input_norm(x)
        return self.item_encoder(x)

    def forward(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        h_C = self.encode_item(x_C)
        h_T = self.encode_item(x_T)
        if h_C.shape[0] == 0:
            r_C = h_T.new_zeros((h_T.shape[-1],))
        else:
            r_C = self.context_encoder(torch.cat([h_C, y_C[:, None]], dim=-1)).mean(dim=0)
        r_CT = r_C.unsqueeze(0).expand(h_T.shape[0], -1)
        return self.decoder(torch.cat([h_T, r_CT], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_C, y_C, x_T))


class ContextConditionedIRT(nn.Module):
    """Isolated reverse-ladder rungs that keep the 2PL/MAP anchor.

    Modes
    -----
    ``2pl``          no context representation; item head + MAP theta.
    ``scalar``       pooled r_C -> one scalar delta theta.
    ``residual``     pooled r_C -> mean-centered per-target logit residual.
    ``additive_ab``  item head plus pooled-r_C offsets on target (a, b);
                     the ability anchor uses only the item-only base head.
    ``direct_ab``    pooled r_C directly parameterizes (a, b).
    """

    MODES = ("rasch", "2pl", "scalar", "residual", "additive_ab", "direct_ab")

    def __init__(
        self,
        x_dim: int,
        *,
        mode: str,
        d_model: int = 128,
        subject_feature_slice: tuple[int, int] | None = None,
        prior_variance: float = PRIOR_THETA_VARIANCE,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode}")
        if prior_variance <= 0.0:
            raise ValueError("prior_variance must be positive")
        self.mode = mode
        self.prior_variance = float(prior_variance)
        self.subject_feature_slice = subject_feature_slice
        self.item_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.item_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )
        if mode not in ("rasch", "2pl"):
            self.context_encoder = nn.Sequential(
                nn.Linear(d_model + 1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            if mode == "scalar":
                self.scalar_head = nn.Sequential(
                    nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
                )
                nn.init.zeros_(self.scalar_head[-1].weight)
                nn.init.zeros_(self.scalar_head[-1].bias)
            elif mode == "residual":
                self.residual_head = nn.Sequential(
                    nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
                )
                nn.init.zeros_(self.residual_head[-1].weight)
                nn.init.zeros_(self.residual_head[-1].bias)
            elif mode == "additive_ab":
                self.ab_head = nn.Sequential(
                    nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2)
                )
                nn.init.zeros_(self.ab_head[-1].weight)
                nn.init.zeros_(self.ab_head[-1].bias)
            elif mode == "direct_ab":
                self.ab_head = nn.Sequential(
                    nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2)
                )

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        if self.subject_feature_slice is not None:
            start, stop = self.subject_feature_slice
            x = torch.cat(
                [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]],
                dim=-1,
            )
        return self.item_encoder(x)

    def _split_parameters(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Discrimination and difficulty from the item head's two outputs.

        Rasch pins discrimination to one, so the item network only controls
        difficulty and every item carries the same weight in the ability
        likelihood. That is the identifying constraint of the Rasch family,
        not a simplification of 2PL.
        """

        if self.mode == "rasch":
            return torch.ones_like(raw[..., 1]), raw[..., 1]
        return torch.nn.functional.softplus(raw[..., 0]) + 0.05, raw[..., 1]

    def _context_repr(
        self, x_C: torch.Tensor, y_C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_C = self.encode_item(x_C)
        if self.mode in ("rasch", "2pl"):
            return h_C, h_C.new_zeros((h_C.shape[-1],))
        if h_C.shape[0] == 0:
            return h_C, h_C.new_zeros((h_C.shape[-1],))
        context = self.context_encoder(torch.cat([h_C, y_C[:, None]], dim=-1))
        return h_C, context.mean(dim=0)

    def _context_parameters(
        self, h_C: torch.Tensor, r_C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.item_head(h_C)
        if self.mode in ("rasch", "2pl", "scalar", "residual"):
            return self._split_parameters(base)
        enriched = torch.cat([h_C, r_C.unsqueeze(0).expand(h_C.shape[0], -1)], dim=-1)
        offset = self.ab_head(enriched)
        if self.mode == "additive_ab":
            return self._split_parameters(base + offset)
        return self._split_parameters(offset)

    def _anchor_parameters(self, h_C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Item-only 2PL parameters for the ability MAP.

        Context-conditioned ``a``/``b`` offsets are target-side corrections.
        Feeding them back into the ability estimate makes the additive rung
        circular, so the anchor deliberately ignores ``r_C``.
        """
        return self._split_parameters(self.item_head(h_C))

    def _target_parameters(
        self, h_T: torch.Tensor, r_C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.item_head(h_T)
        if self.mode in ("rasch", "2pl", "scalar", "residual"):
            return self._split_parameters(base)
        enriched = torch.cat([h_T, r_C.unsqueeze(0).expand(h_T.shape[0], -1)], dim=-1)
        offset = self.ab_head(enriched)
        if self.mode == "additive_ab":
            return self._split_parameters(base + offset)
        return self._split_parameters(offset)

    @staticmethod
    def map_theta(
        a_C: torch.Tensor,
        b_C: torch.Tensor,
        y_C: torch.Tensor,
        *,
        prior_mean: float | torch.Tensor = 0.0,
        steps: int = 12,
        prior_variance: float = PRIOR_THETA_VARIANCE,
    ) -> torch.Tensor:
        if isinstance(prior_mean, torch.Tensor):
            prior = prior_mean.to(device=a_C.device, dtype=a_C.dtype).reshape(())
        else:
            prior = a_C.new_tensor(float(prior_mean))
        theta = prior
        if a_C.shape[0] == 0:
            return theta
        a_C = a_C.detach()
        b_C = b_C.detach()
        y_C = y_C.to(dtype=a_C.dtype)
        for _ in range(steps):
            probability = torch.sigmoid(a_C * (theta - b_C))
            gradient = torch.sum(a_C * (probability - y_C)) + (theta - prior) / prior_variance
            hessian = (
                torch.sum(a_C.square() * probability * (1.0 - probability)) + 1.0 / prior_variance
            )
            # The box is a numerical guard at +-2 standard deviations of the
            # prior, not an absolute constant: at prior_variance = 4 that is
            # +-4, matching every previously published run exactly, so this
            # generalization changes nothing for tau^2 = 4. A fixed +-4 box
            # would silently crush a wider prior -- at tau^2 = 9 it is only
            # 1.33 sigma, discarding real posterior mass instead of guarding
            # against numerical blowup.
            box = 2.0 * math.sqrt(prior_variance)
            theta = (theta - (gradient / hessian).clamp(-1.0, 1.0)).clamp(-box, box)
        return theta

    def forward(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        h_C, r_C = self._context_repr(x_C, y_C)
        if self.mode == "additive_ab":
            a_C, b_C = self._anchor_parameters(h_C)
        else:
            a_C, b_C = self._context_parameters(h_C, r_C)
        theta = self.map_theta(a_C, b_C, y_C, prior_variance=self.prior_variance)
        h_T = self.encode_item(x_T)
        a_T, b_T = self._target_parameters(h_T, r_C)
        if self.mode == "scalar":
            theta = theta + self.scalar_head(r_C).squeeze(-1)
        logits = a_T * (theta - b_T)
        if self.mode == "residual":
            enriched = torch.cat([h_T, r_C.unsqueeze(0).expand(h_T.shape[0], -1)], dim=-1)
            residual = self.residual_head(enriched).squeeze(-1)
            logits = logits + residual - residual.mean()
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_C, y_C, x_T))


class AutoregressiveContextIRT(nn.Module):
    """Soft-AR target decoder with a parameter-matched no-carry control.

    Targets are processed in sequence order.  Each step receives the previous
    *predicted* probability (never the target label), so training and inference
    use the same input distribution.  ``ar=False`` resets the recurrent state
    and the previous probability at every target; the parameters are identical.

    Modes
    -----
    ``additive_ab``  item head plus context/state offsets on (a, b).
    ``direct_ab``    context/state directly parameterizes (a, b).
    """

    MODES = ("additive_ab", "direct_ab")

    def __init__(
        self,
        x_dim: int,
        *,
        mode: str,
        ar: bool = True,
        d_model: int = 128,
        subject_feature_slice: tuple[int, int] | None = None,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.ar = bool(ar)
        self.subject_feature_slice = subject_feature_slice
        self.item_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.item_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.ab_head = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )
        self.ar_cell = nn.GRUCell(2 * d_model + 1, d_model)
        if mode == "additive_ab":
            nn.init.zeros_(self.ab_head[-1].weight)
            nn.init.zeros_(self.ab_head[-1].bias)

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        if self.subject_feature_slice is not None:
            start, stop = self.subject_feature_slice
            x = torch.cat(
                [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]],
                dim=-1,
            )
        return self.item_encoder(x)

    @staticmethod
    def _split(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.nn.functional.softplus(raw[..., 0]) + 0.05, raw[..., 1]

    def _context_repr(
        self, x_C: torch.Tensor, y_C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_C = self.encode_item(x_C)
        if h_C.shape[0] == 0:
            return h_C, h_C.new_zeros((h_C.shape[-1],))
        context = self.context_encoder(torch.cat([h_C, y_C[:, None]], dim=-1))
        return h_C, context.mean(dim=0)

    def _item_parameters(
        self, h: torch.Tensor, r_C: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.item_head(h)
        enriched = torch.cat([h, r_C.unsqueeze(0).expand(h.shape[0], -1), state], dim=-1)
        offset = self.ab_head(enriched)
        if self.mode == "additive_ab":
            return self._split(base + offset)
        return self._split(offset)

    def forward(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        h_C, r_C = self._context_repr(x_C, y_C)
        context_state = h_C.new_zeros((h_C.shape[0], h_C.shape[-1]))
        a_C, b_C = self._item_parameters(h_C, r_C, context_state)
        theta = ContextConditionedIRT.map_theta(a_C, b_C, y_C)
        h_T = self.encode_item(x_T)
        if h_T.shape[0] == 0:
            return h_T.new_zeros((0,))
        state = h_T.new_zeros((h_T.shape[-1],))
        previous = h_T.new_tensor(0.5)
        output = []
        for index in range(h_T.shape[0]):
            cell_input = torch.cat([h_T[index], r_C, previous.reshape(1)])
            if self.ar:
                state = self.ar_cell(cell_input.unsqueeze(0), state.unsqueeze(0)).squeeze(0)
            else:
                state = self.ar_cell(
                    cell_input.unsqueeze(0), torch.zeros_like(state).unsqueeze(0)
                ).squeeze(0)
            a_T, b_T = self._item_parameters(h_T[index : index + 1], r_C, state.unsqueeze(0))
            logit = a_T * (theta - b_T)
            output.append(logit.squeeze(0))
            if self.ar:
                previous = torch.sigmoid(logit).squeeze(0)
        return torch.stack(output)

    @torch.no_grad()
    def predict_proba(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_C, y_C, x_T))


class LoopedContextIRT(nn.Module):
    """Loop a shared transformer block over the target set.

    The base decoder produces ``z_t^0`` from the MAP ability and additive or
    direct item parameters.  Each loop refines all target logits in parallel:

        ``z_t^{k+1} = z_t^k + B([h_t, r_C, z_t^k, p_t^k])``

    ``B`` is one pre-norm transformer block reused for every loop.  No causal
    mask and no target order: this is set refinement, not AR.  ``loops=0`` is
    the parameter-matched base control.
    """

    MODES = ("additive_ab", "direct_ab")

    def __init__(
        self,
        x_dim: int,
        *,
        mode: str,
        loops: int = 1,
        d_model: int = 128,
        n_heads: int = 8,
        subject_feature_slice: tuple[int, int] | None = None,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.loops = int(loops)
        self.subject_feature_slice = subject_feature_slice
        self.item_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.item_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.ab_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )
        self.token_proj = nn.Linear(2 * d_model + 2, d_model)
        self.block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.loop_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.loop_head.weight)
        nn.init.zeros_(self.loop_head.bias)
        if mode == "additive_ab":
            nn.init.zeros_(self.ab_head[-1].weight)
            nn.init.zeros_(self.ab_head[-1].bias)

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        if self.subject_feature_slice is not None:
            start, stop = self.subject_feature_slice
            x = torch.cat(
                [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]],
                dim=-1,
            )
        return self.item_encoder(x)

    @staticmethod
    def _split(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.nn.functional.softplus(raw[..., 0]) + 0.05, raw[..., 1]

    def _context_repr(
        self, x_C: torch.Tensor, y_C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_C = self.encode_item(x_C)
        if h_C.shape[0] == 0:
            return h_C, h_C.new_zeros((h_C.shape[-1],))
        context = self.context_encoder(torch.cat([h_C, y_C[:, None]], dim=-1))
        return h_C, context.mean(dim=0)

    def _item_parameters(
        self, h: torch.Tensor, r_C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.item_head(h)
        enriched = torch.cat([h, r_C.unsqueeze(0).expand(h.shape[0], -1)], dim=-1)
        offset = self.ab_head(enriched)
        if self.mode == "additive_ab":
            return self._split(base + offset)
        return self._split(offset)

    def forward(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        h_C, r_C = self._context_repr(x_C, y_C)
        a_C, b_C = self._item_parameters(h_C, r_C)
        theta = ContextConditionedIRT.map_theta(a_C, b_C, y_C)
        h_T = self.encode_item(x_T)
        if h_T.shape[0] == 0:
            return h_T.new_zeros((0,))
        a_T, b_T = self._item_parameters(h_T, r_C)
        logits = a_T * (theta - b_T)
        if self.loops <= 0:
            return logits
        probabilities = torch.sigmoid(logits)
        for _ in range(self.loops):
            tokens = torch.cat(
                [
                    h_T,
                    r_C.unsqueeze(0).expand(h_T.shape[0], -1),
                    logits[:, None],
                    probabilities[:, None],
                ],
                dim=-1,
            )
            refined = self.block(self.token_proj(tokens).unsqueeze(0)).squeeze(0)
            logits = logits + self.loop_head(refined).squeeze(-1)
            probabilities = torch.sigmoid(logits)
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_C, y_C, x_T))


class NeuralResidualIRT(nn.Module):
    """2PL anchor plus a neural residual item curve.

    ``nr`` uses only the item representation and ability basis:

        ``z_i(theta) = a_i (theta - b_i) + gamma g(h_i, phi(theta))``

    ``u_nr`` adds a context representation:

        ``z_i(theta) = a_i (theta - b_i) + gamma g(h_i, r_C, phi(theta))``

    The 2PL anchor owns ability: ``theta`` is the anchor MAP, not a residual
    output. The residual head is zero-initialized, so training starts at 2PL.
    """

    MODES = ("nr", "u_nr")

    def __init__(
        self,
        x_dim: int,
        *,
        mode: str,
        d_model: int = 128,
        basis_size: int = 8,
        lengthscale: float = 1.0,
        residual_l2: float = 1e-3,
        subject_feature_slice: tuple[int, int] | None = None,
        subject_prior_slice: tuple[int, int] | None = None,
        use_subject_prior: bool = False,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode}")
        if use_subject_prior and subject_prior_slice is None:
            raise ValueError("subject_prior_slice is required for a subject prior")
        self.mode = mode
        self.lengthscale = float(lengthscale)
        self.residual_l2 = float(residual_l2)
        self.subject_feature_slice = subject_feature_slice
        self.subject_prior_slice = subject_prior_slice
        self.use_subject_prior = bool(use_subject_prior)
        self.item_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.item_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )
        if mode == "u_nr":
            self.context_encoder = nn.Sequential(
                nn.Linear(d_model + 1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
        residual_dim = d_model + basis_size
        if mode == "u_nr":
            residual_dim += d_model
        self.residual_head = nn.Sequential(
            nn.Linear(residual_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        if use_subject_prior:
            start, stop = subject_prior_slice
            self.subject_prior_head = nn.Sequential(
                nn.Linear(stop - start, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
            )
            nn.init.zeros_(self.subject_prior_head[-1].weight)
            nn.init.zeros_(self.subject_prior_head[-1].bias)
        centers = torch.linspace(-3.0, 3.0, basis_size)
        self.register_buffer("centers", centers)
        self._aux_loss = centers.new_zeros(())

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        if self.subject_feature_slice is not None:
            start, stop = self.subject_feature_slice
            x = torch.cat(
                [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]],
                dim=-1,
            )
        return self.item_encoder(x)

    @staticmethod
    def _split(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.nn.functional.softplus(raw[..., 0]) + 0.05, raw[..., 1]

    def _item_parameters(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split(self.item_head(h))

    def _context_repr(
        self, x_C: torch.Tensor, y_C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_C = self.encode_item(x_C)
        if self.mode == "nr" or h_C.shape[0] == 0:
            return h_C, h_C.new_zeros((h_C.shape[-1],))
        context = self.context_encoder(torch.cat([h_C, y_C[:, None]], dim=-1))
        return h_C, context.mean(dim=0)

    def _subject_prior(self, x_C: torch.Tensor, x_T: torch.Tensor) -> torch.Tensor | None:
        if not self.use_subject_prior:
            return None
        reference = x_T if x_T.shape[0] else x_C
        if reference.shape[0] == 0:
            return None
        start, stop = self.subject_prior_slice
        subject = reference[:, start:stop].mean(dim=0)
        return torch.tanh(self.subject_prior_head(subject)).reshape(())

    def _basis(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(theta - self.centers).square() / (2.0 * self.lengthscale**2))

    def aux_loss(self) -> torch.Tensor:
        return self._aux_loss

    def forward(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        h_C, r_C = self._context_repr(x_C, y_C)
        a_C, b_C = self._item_parameters(h_C)
        prior_mean = self._subject_prior(x_C, x_T)
        theta = ContextConditionedIRT.map_theta(
            a_C,
            b_C,
            y_C,
            prior_mean=0.0 if prior_mean is None else prior_mean,
        )
        h_T = self.encode_item(x_T)
        if h_T.shape[0] == 0:
            self._aux_loss = h_T.sum() * 0.0
            return h_T.new_zeros((0,))
        a_T, b_T = self._item_parameters(h_T)
        phi = self._basis(theta)
        features = [h_T, phi.unsqueeze(0).expand(h_T.shape[0], -1)]
        if self.mode == "u_nr":
            features.insert(1, r_C.unsqueeze(0).expand(h_T.shape[0], -1))
        residual = self.gamma * self.residual_head(torch.cat(features, dim=-1)).squeeze(-1)
        residual = residual - residual.mean()
        logits = a_T * (theta - b_T) + residual
        self._aux_loss = self.residual_l2 * residual.square().mean()
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_C, y_C, x_T))


class GPContextIRT(nn.Module):
    """Amortized finite-basis GP-IRT.

    The base 2PL logit is augmented with an item-specific Gaussian-process
    residual over ability:

        ``z_t(theta) = a_t (theta - b_t) + c_t^T phi(theta)``

    ``phi`` is a fixed RBF basis over ``theta`` and ``c_t`` is predicted from
    the item embedding (plus ``r_C`` in additive mode).  ``c_t`` has a GP prior
    ``N(0, K)``; the training loss includes ``lambda c^T K^{-1} c``.  Zero
    initialization makes the model start exactly at the base 2PL.
    """

    MODES = ("2pl", "additive_ab")

    def __init__(
        self,
        x_dim: int,
        *,
        mode: str,
        d_model: int = 128,
        basis_size: int = 8,
        lengthscale: float = 1.0,
        gp_lambda: float = 1e-3,
        subject_feature_slice: tuple[int, int] | None = None,
        subject_prior_slice: tuple[int, int] | None = None,
        use_subject_prior: bool = False,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode}")
        if use_subject_prior and subject_prior_slice is None:
            raise ValueError("subject_prior_slice is required for a subject prior")
        self.mode = mode
        self.gp_lambda = float(gp_lambda)
        self.lengthscale = float(lengthscale)
        self.subject_feature_slice = subject_feature_slice
        self.subject_prior_slice = subject_prior_slice
        self.use_subject_prior = bool(use_subject_prior)
        if use_subject_prior:
            start, stop = subject_prior_slice
            self.subject_prior_head = nn.Sequential(
                nn.Linear(stop - start, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
            )
            nn.init.zeros_(self.subject_prior_head[-1].weight)
            nn.init.zeros_(self.subject_prior_head[-1].bias)
        self.item_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.item_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )
        if mode == "additive_ab":
            self.context_encoder = nn.Sequential(
                nn.Linear(d_model + 1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            self.ab_head = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 2),
            )
            nn.init.zeros_(self.ab_head[-1].weight)
            nn.init.zeros_(self.ab_head[-1].bias)
        coefficient_dim = d_model if mode == "2pl" else 2 * d_model
        self.coefficient_head = nn.Sequential(
            nn.Linear(coefficient_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, basis_size),
        )
        nn.init.zeros_(self.coefficient_head[-1].weight)
        nn.init.zeros_(self.coefficient_head[-1].bias)
        centers = torch.linspace(-3.0, 3.0, basis_size)
        self.register_buffer("centers", centers)
        delta = centers[:, None] - centers[None, :]
        kernel = torch.exp(-delta.square() / (2.0 * lengthscale**2))
        kernel = kernel + 1e-4 * torch.eye(basis_size)
        self.register_buffer("kernel_inv", torch.linalg.inv(kernel))
        self._gp_penalty = centers.new_zeros(())

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        if self.subject_feature_slice is not None:
            start, stop = self.subject_feature_slice
            x = torch.cat(
                [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]],
                dim=-1,
            )
        return self.item_encoder(x)

    @staticmethod
    def _split(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.nn.functional.softplus(raw[..., 0]) + 0.05, raw[..., 1]

    def _context_repr(
        self, x_C: torch.Tensor, y_C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_C = self.encode_item(x_C)
        if self.mode == "2pl" or h_C.shape[0] == 0:
            return h_C, h_C.new_zeros((h_C.shape[-1],))
        context = self.context_encoder(torch.cat([h_C, y_C[:, None]], dim=-1))
        return h_C, context.mean(dim=0)

    def _features(self, h: torch.Tensor, r_C: torch.Tensor) -> torch.Tensor:
        if self.mode == "2pl":
            return h
        return torch.cat([h, r_C.unsqueeze(0).expand(h.shape[0], -1)], dim=-1)

    def _item_parameters(
        self, h: torch.Tensor, r_C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.item_head(h)
        if self.mode == "2pl":
            return self._split(base)
        enriched = torch.cat([h, r_C.unsqueeze(0).expand(h.shape[0], -1)], dim=-1)
        return self._split(base + self.ab_head(enriched))

    def _coefficients(self, h: torch.Tensor, r_C: torch.Tensor) -> torch.Tensor:
        return self.coefficient_head(self._features(h, r_C))

    def _subject_prior(self, x_C: torch.Tensor, x_T: torch.Tensor) -> torch.Tensor | None:
        if not self.use_subject_prior:
            return None
        reference = x_T if x_T.shape[0] else x_C
        if reference.shape[0] == 0:
            return None
        start, stop = self.subject_prior_slice
        subject = reference[:, start:stop].mean(dim=0)
        return torch.tanh(self.subject_prior_head(subject)).reshape(())

    @staticmethod
    def map_theta(
        a_C: torch.Tensor,
        b_C: torch.Tensor,
        c_C: torch.Tensor,
        y_C: torch.Tensor,
        centers: torch.Tensor,
        lengthscale: float,
        *,
        prior_mean: float | torch.Tensor = 0.0,
        steps: int = 12,
    ) -> torch.Tensor:
        if isinstance(prior_mean, torch.Tensor):
            prior = prior_mean.to(device=a_C.device, dtype=a_C.dtype).reshape(())
        else:
            prior = a_C.new_tensor(float(prior_mean))
        theta = prior
        if a_C.shape[0] == 0:
            return theta
        a_C = a_C.detach()
        b_C = b_C.detach()
        c_C = c_C.detach()
        y_C = y_C.to(dtype=a_C.dtype)
        for _ in range(steps):
            delta = theta - centers
            phi = torch.exp(-delta.square() / (2.0 * lengthscale**2))
            dphi = phi * (-delta / lengthscale**2)
            ddphi = phi * (delta.square() / lengthscale**4 - 1.0 / lengthscale**2)
            value = a_C * (theta - b_C) + c_C @ phi
            slope = a_C + c_C @ dphi
            curvature = c_C @ ddphi
            probability = torch.sigmoid(value)
            gradient = torch.sum((probability - y_C) * slope) + (theta - prior) / 4.0
            hessian = (
                torch.sum(
                    probability * (1.0 - probability) * slope.square()
                    + (probability - y_C) * curvature
                )
                + 0.25
            )
            theta = (theta - (gradient / hessian.clamp_min(1e-6)).clamp(-1.0, 1.0)).clamp(-4.0, 4.0)
        return theta

    def _set_penalty(self, c_C: torch.Tensor, c_T: torch.Tensor) -> None:
        coefficients = torch.cat([c_C, c_T], dim=0)
        if coefficients.shape[0] == 0:
            self._gp_penalty = coefficients.sum() * 0.0
            return
        quadratic = torch.einsum("nm,mk,nk->n", coefficients, self.kernel_inv, coefficients)
        self._gp_penalty = self.gp_lambda * quadratic.mean()

    def gp_penalty(self) -> torch.Tensor:
        return self._gp_penalty

    def forward(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        h_C, r_C = self._context_repr(x_C, y_C)
        a_C, b_C = self._item_parameters(h_C, r_C)
        c_C = self._coefficients(h_C, r_C)
        prior_mean = self._subject_prior(x_C, x_T)
        theta = self.map_theta(
            a_C,
            b_C,
            c_C,
            y_C,
            self.centers,
            self.lengthscale,
            prior_mean=0.0 if prior_mean is None else prior_mean,
        )
        h_T = self.encode_item(x_T)
        if h_T.shape[0] == 0:
            self._set_penalty(c_C, h_T.new_zeros((0, self.centers.shape[0])))
            return h_T.new_zeros((0,))
        a_T, b_T = self._item_parameters(h_T, r_C)
        c_T = self._coefficients(h_T, r_C)
        phi = torch.exp(-(theta - self.centers).square() / (2.0 * self.lengthscale**2))
        logits = a_T * (theta - b_T) + c_T @ phi
        self._set_penalty(c_C, c_T)
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_C, y_C, x_T))


class MultidimContextIRT(nn.Module):
    """Multidimensional-ability counterparts of the three best scalar rungs.

    The scalar ladder gives every subject one ability number, so two subjects
    with the same overall accuracy are indistinguishable even when one is
    strong at code and weak at multilingual knowledge and the other is the
    reverse. With ``ability_dim = D`` the item head emits a discrimination
    *vector* ``a_i`` in R^D and an intercept ``d_i``, and the response model
    becomes the compensatory MIRT logit

        ``z_i = a_i . theta - d_i``,

    with ``theta`` in R^D solved by a Newton MAP on the acquired context, the
    same way the scalar rungs solve for a single ability.

    ``D = 1`` is a reparametrization of the scalar model (``a theta - d``
    rather than ``a (theta - b)``), not a bit-identical copy of it, so the
    D = 1 arm is a control for the parametrization and the D > 1 arms test
    whether extra ability dimensions buy anything.

    ``mode`` and ``loops`` mirror the scalar ladder so the comparison is
    like-for-like:

    ``scalar``       pooled r_C adds one offset vector to theta.
    ``additive_ab``  pooled r_C offsets the target (a, d); the ability anchor
                     deliberately ignores r_C, as in ContextConditionedIRT.
    ``direct_ab``    pooled r_C parameterizes (a, d) outright.

    ``loops`` repeats one shared pre-norm transformer block over the target
    set, refining all target logits in parallel; ``loops = 0`` is the
    parameter-matched control.
    """

    MODES = ("scalar", "additive_ab", "direct_ab")
    PRIOR_VARIANCE = 4.0

    def __init__(
        self,
        x_dim: int,
        *,
        mode: str,
        ability_dim: int = 2,
        loops: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        subject_feature_slice: tuple[int, int] | None = None,
        prior_variance: float = PRIOR_THETA_VARIANCE,
        difficulty_adapter_dim: int = 0,
        adapter_prior_variance: float = 1.0,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode}")
        if prior_variance <= 0.0:
            raise ValueError("prior_variance must be positive")
        if difficulty_adapter_dim < 0:
            raise ValueError("difficulty_adapter_dim must be non-negative")
        if adapter_prior_variance <= 0.0:
            raise ValueError("adapter_prior_variance must be positive")
        # Instance-level so this matches ContextConditionedIRT's interface and
        # the tau sweep applies to both families. The default is unchanged, so
        # every existing checkpoint and published result is unaffected.
        self.prior_variance = float(prior_variance)
        self.difficulty_adapter_dim = int(difficulty_adapter_dim)
        self.adapter_prior_variance = float(adapter_prior_variance)
        if ability_dim < 1:
            raise ValueError("ability_dim must be positive")
        if loops < 0:
            raise ValueError("loops must be non-negative")
        self.mode = mode
        self.ability_dim = int(ability_dim)
        self.loops = int(loops)
        self.subject_feature_slice = subject_feature_slice
        out_dim = self.ability_dim + 1

        self.item_encoder = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.item_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        if self.difficulty_adapter_dim:
            self.difficulty_adapter_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, self.difficulty_adapter_dim, bias=False),
            )
        if mode == "scalar":
            self.theta_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, self.ability_dim)
            )
            nn.init.zeros_(self.theta_head[-1].weight)
            nn.init.zeros_(self.theta_head[-1].bias)
        else:
            self.ab_head = nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, out_dim)
            )
            if mode == "additive_ab":
                nn.init.zeros_(self.ab_head[-1].weight)
                nn.init.zeros_(self.ab_head[-1].bias)

        if self.loops:
            self.token_proj = nn.Linear(2 * d_model + 2, d_model)
            self.block = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.loop_head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.loop_head.weight)
            nn.init.zeros_(self.loop_head.bias)

    def encode_item(self, x: torch.Tensor) -> torch.Tensor:
        if self.subject_feature_slice is not None:
            start, stop = self.subject_feature_slice
            x = torch.cat(
                [x[..., :start], torch.zeros_like(x[..., start:stop]), x[..., stop:]],
                dim=-1,
            )
        return self.item_encoder(x)

    def _split(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Discrimination vector (positive) and scalar intercept."""

        a = torch.nn.functional.softplus(raw[..., : self.ability_dim]) + 0.05
        return a, raw[..., self.ability_dim]

    def _context_repr(
        self, x_C: torch.Tensor, y_C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_C = self.encode_item(x_C)
        if h_C.shape[0] == 0:
            return h_C, h_C.new_zeros((h_C.shape[-1],))
        context = self.context_encoder(torch.cat([h_C, y_C[:, None]], dim=-1))
        return h_C, context.mean(dim=0)

    def _anchor_parameters(self, h_C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Item-only parameters for the ability MAP.

        Context-conditioned offsets are target-side corrections; feeding them
        back into the ability estimate would make the additive rung circular.
        """

        return self._split(self.item_head(h_C))

    def _head_parameters(
        self, h: torch.Tensor, r_C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.item_head(h)
        if self.mode == "scalar":
            return self._split(base)
        enriched = torch.cat([h, r_C.unsqueeze(0).expand(h.shape[0], -1)], dim=-1)
        offset = self.ab_head(enriched)
        if self.mode == "additive_ab":
            return self._split(base + offset)
        return self._split(offset)

    def _difficulty_loadings(self, h: torch.Tensor) -> torch.Tensor:
        """Unit-scale item loadings for the online difficulty adapter."""

        if not self.difficulty_adapter_dim:
            return h.new_zeros((h.shape[0], 0))
        raw = self.difficulty_adapter_head(h)
        return torch.nn.functional.normalize(raw, dim=-1, eps=1e-6)

    @staticmethod
    def center_difficulty_loadings(
        context: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remove the context mean so the adapter cannot duplicate theta.

        The same offset is removed from context and target loadings. With one
        acquired label the centered context loading is zero, so that label can
        update ability but cannot spuriously identify a benchmark-specific
        ranking direction.
        """

        if context.shape[0] == 0:
            return context, target
        offset = context.mean(dim=0, keepdim=True)
        return context - offset, target - offset

    @staticmethod
    def map_theta(
        a_C: torch.Tensor,
        d_C: torch.Tensor,
        y_C: torch.Tensor,
        *,
        prior_mean: torch.Tensor | None = None,
        steps: int = 12,
        prior_variance: float = PRIOR_VARIANCE,
    ) -> torch.Tensor:
        """Newton MAP for a D-dimensional ability.

        Mirrors the scalar solve in ``ContextConditionedIRT.map_theta``: the
        same prior variance, the same step count, the same trust-region clamp
        on the update and the same box clamp on theta, generalized to a full
        Hessian. The context parameters are detached for the same reason as
        the scalar version -- the ability estimate is an inference step, not a
        path the item network should be trained through.
        """

        dim = a_C.shape[-1]
        theta = (
            a_C.new_zeros(dim)
            if prior_mean is None
            else prior_mean.to(device=a_C.device, dtype=a_C.dtype).reshape(dim)
        )
        prior = theta
        if a_C.shape[0] == 0:
            return theta
        a_C = a_C.detach()
        d_C = d_C.detach()
        y_C = y_C.to(dtype=a_C.dtype)
        eye = torch.eye(dim, device=a_C.device, dtype=a_C.dtype)
        for _ in range(steps):
            probability = torch.sigmoid(a_C @ theta - d_C)
            weight = probability * (1.0 - probability)
            gradient = a_C.transpose(0, 1) @ (probability - y_C) + (theta - prior) / prior_variance
            hessian = (a_C.transpose(0, 1) * weight) @ a_C + eye / prior_variance
            # Jitter keeps the solve well posed when every context item loads
            # on the same few dimensions, which is common at small budgets.
            step = torch.linalg.solve(hessian + 1e-4 * eye, gradient)
            norm = step.norm().clamp_min(1e-12)
            step = step * (norm.clamp(max=1.0) / norm)
            box = 2.0 * math.sqrt(prior_variance)
            theta = (theta - step).clamp(-box, box)
        return theta

    @staticmethod
    def map_theta_and_difficulty(
        a_C: torch.Tensor,
        d_C: torch.Tensor,
        u_C: torch.Tensor,
        y_C: torch.Tensor,
        *,
        steps: int = 12,
        theta_prior_variance: float = PRIOR_VARIANCE,
        adapter_prior_variance: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint MAP for ability and a centered low-rank difficulty adapter.

        Context logits are ``a_i @ theta - d_i - u_i @ z``. The adapter
        coefficients ``z`` modify item ranking while the centered loadings
        prevent a constant benchmark offset from competing with ``theta``.
        """

        theta_dim = a_C.shape[-1]
        adapter_dim = u_C.shape[-1]
        theta = a_C.new_zeros(theta_dim)
        adapter = a_C.new_zeros(adapter_dim)
        if a_C.shape[0] == 0:
            return theta, adapter
        if adapter_dim == 0:
            return (
                MultidimContextIRT.map_theta(
                    a_C,
                    d_C,
                    y_C,
                    steps=steps,
                    prior_variance=theta_prior_variance,
                ),
                adapter,
            )

        design = torch.cat([a_C.detach(), -u_C.detach()], dim=-1)
        d_C = d_C.detach()
        y_C = y_C.to(dtype=a_C.dtype)
        state = torch.cat([theta, adapter])
        prior_precision = torch.cat(
            [
                state.new_full((theta_dim,), 1.0 / theta_prior_variance),
                state.new_full((adapter_dim,), 1.0 / adapter_prior_variance),
            ]
        )
        eye = torch.eye(state.shape[0], device=state.device, dtype=state.dtype)
        for _ in range(steps):
            probability = torch.sigmoid(design @ state - d_C)
            weight = probability * (1.0 - probability)
            gradient = design.transpose(0, 1) @ (probability - y_C) + prior_precision * state
            hessian = (design.transpose(0, 1) * weight) @ design
            hessian = hessian + torch.diag(prior_precision)
            update = torch.linalg.solve(hessian + 1e-4 * eye, gradient)
            norm = update.norm().clamp_min(1e-12)
            state = state - update * (norm.clamp(max=1.0) / norm)
            theta_box = 2.0 * math.sqrt(theta_prior_variance)
            adapter_box = 2.0 * math.sqrt(adapter_prior_variance)
            state = torch.cat(
                [
                    state[:theta_dim].clamp(-theta_box, theta_box),
                    state[theta_dim:].clamp(-adapter_box, adapter_box),
                ]
            )
        return state[:theta_dim], state[theta_dim:]

    @staticmethod
    def laplace_covariance(
        a_C: torch.Tensor,
        d_C: torch.Tensor,
        theta: torch.Tensor,
        *,
        prior_variance: float = PRIOR_VARIANCE,
    ) -> torch.Tensor:
        """Laplace covariance at a multidimensional ability MAP estimate.

        The returned matrix is the inverse Hessian of the same regularized
        context likelihood optimized by :meth:`map_theta`. With no labels it
        is exactly the Gaussian prior covariance.
        """

        dim = theta.shape[-1]
        eye = torch.eye(dim, device=theta.device, dtype=theta.dtype)
        if a_C.shape[0] == 0:
            return eye * float(prior_variance)
        a_C = a_C.detach()
        d_C = d_C.detach()
        probability = torch.sigmoid(a_C @ theta.detach() - d_C)
        weight = probability * (1.0 - probability)
        hessian = (a_C.transpose(0, 1) * weight) @ a_C + eye / prior_variance
        covariance = torch.linalg.inv(hessian + 1e-4 * eye)
        return 0.5 * (covariance + covariance.transpose(0, 1))

    def forward(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
        *,
        marginalize_theta: bool = False,
    ) -> torch.Tensor:
        h_C, r_C = self._context_repr(x_C, y_C)
        if self.mode == "additive_ab":
            a_C, d_C = self._anchor_parameters(h_C)
        else:
            a_C, d_C = self._head_parameters(h_C, r_C)
        h_T = self.encode_item(x_T)
        a_T, d_T = self._head_parameters(h_T, r_C)
        u_C = self._difficulty_loadings(h_C)
        u_T = self._difficulty_loadings(h_T)
        u_C, u_T = self.center_difficulty_loadings(u_C, u_T)
        if self.difficulty_adapter_dim:
            theta, difficulty_state = self.map_theta_and_difficulty(
                a_C,
                d_C,
                u_C,
                y_C,
                theta_prior_variance=self.prior_variance,
                adapter_prior_variance=self.adapter_prior_variance,
            )
        else:
            theta = self.map_theta(a_C, d_C, y_C, prior_variance=self.prior_variance)
            difficulty_state = a_C.new_zeros((0,))
        theta_map = theta
        if self.mode == "scalar":
            theta = theta + self.theta_head(r_C)

        logits = a_T @ theta - d_T
        if self.difficulty_adapter_dim:
            logits = logits - u_T @ difficulty_state
        if marginalize_theta:
            if self.difficulty_adapter_dim:
                context_design = torch.cat([a_C, -u_C], dim=-1)
                target_design = torch.cat([a_T, -u_T], dim=-1)
                prior = torch.cat(
                    [
                        a_C.new_full((self.ability_dim,), self.prior_variance),
                        a_C.new_full((self.difficulty_adapter_dim,), self.adapter_prior_variance),
                    ]
                )
                state = torch.cat([theta_map, difficulty_state])
                probability = torch.sigmoid(context_design @ state - d_C)
                weight = probability * (1.0 - probability)
                precision = (context_design.transpose(0, 1) * weight) @ context_design
                precision = precision + torch.diag(prior.reciprocal())
                covariance = torch.linalg.inv(
                    precision
                    + 1e-4
                    * torch.eye(precision.shape[0], device=precision.device, dtype=precision.dtype)
                )
                logit_variance = torch.einsum(
                    "ij,jk,ik->i", target_design, covariance, target_design
                )
            else:
                covariance = self.laplace_covariance(
                    a_C,
                    d_C,
                    theta_map,
                    prior_variance=self.prior_variance,
                )
                logit_variance = torch.einsum("ij,jk,ik->i", a_T, covariance, a_T)
            logits = marginalize_logits(logits, logit_variance)
        if not self.loops:
            return logits

        expanded = r_C.unsqueeze(0).expand(h_T.shape[0], -1)
        for _ in range(self.loops):
            probability = torch.sigmoid(logits)
            tokens = self.token_proj(
                torch.cat([h_T, expanded, logits.unsqueeze(-1), probability.unsqueeze(-1)], dim=-1)
            )
            refined = self.block(tokens.unsqueeze(0)).squeeze(0)
            logits = logits + self.loop_head(refined).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba_posterior(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate ``E[p(y=1 | theta)]`` under the Laplace posterior."""

        return torch.sigmoid(self.forward(x_C, y_C, x_T, marginalize_theta=True))

    @torch.no_grad()
    def predict_proba(
        self,
        x_C: torch.Tensor,
        y_C: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_C, y_C, x_T))
