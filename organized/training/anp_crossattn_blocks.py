"""Target-to-context cross-attention blocks for Attentive Neural Processes.

The deterministic path is intentionally unlike ``cnp_blocks.NeuralProcess``:
each target item receives its own context representation through cross-attention.
The optional latent path remains global and mean-pooled, as in Kim et al. (2019).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:  # Package import during local tests; flat import in the Modal container.
    from .cnp_blocks import Projection
except ImportError:  # pragma: no cover - exercised by the remote container.
    from cnp_blocks import Projection

# Shrinkage constants of the shipped beta-binomial submission, copied from
# ``submission/aims_predictor/predictor.py`` so the anchored model starts from
# exactly that estimator rather than an approximation of it.
PRIOR_MEAN = 0.5
PRIOR_STRENGTH = 4.0
SUBJECT_STRENGTH = 8.0


class _SelfAttentionBlock(nn.Module):
    """One permutation-equivariant context self-attention block."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.norm_attention = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )
        self.norm_ff = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm_attention(x + attended)
        return self.norm_ff(x + self.ff(x))


class AttentiveNeuralProcess(nn.Module):
    """ANP with target-to-context cross-attention and an optional global latent.

    The deterministic representation for target ``t`` is

    ``r_t = Attention(q(x_t), k(x_C), v(h(x_C, y_C)))``.

    There is deliberately no deterministic mean pool. With ``use_latent=True``,
    mean pooling is used only to parameterise the global latent distribution.
    """

    def __init__(self, emb_dim: int, *, width: int = 64,
                 self_attention_blocks: int = 2,
                 cross_attention_heads: int = 4,
                 use_latent: bool = True, latent_dim: int = 32,
                 residual_proj: bool = False, dropout: float = 0.0,
                 prior_anchor: bool = False) -> None:
        super().__init__()
        if width % cross_attention_heads:
            raise ValueError(
                f"width={width} must divide evenly across "
                f"cross_attention_heads={cross_attention_heads}"
            )
        if self_attention_blocks < 0:
            raise ValueError("self_attention_blocks must be non-negative")

        self.width = width
        self.use_latent = use_latent
        self.prior_anchor = prior_anchor
        self.latent_dim = latent_dim
        self.self_attention_blocks = self_attention_blocks
        self.cross_attention_heads = cross_attention_heads

        # Keys and queries depend on item identity only. Values also receive
        # the observed binary response, which is what the target attends to.
        self.item_to_key = Projection(emb_dim, width, residual_proj)
        self.item_to_query = Projection(emb_dim, width, residual_proj)
        self.item_to_value = Projection(emb_dim, width, residual_proj)
        self.label_to_value = nn.Linear(1, width)
        self.context_blocks = nn.ModuleList([
            _SelfAttentionBlock(width, cross_attention_heads, dropout)
            for _ in range(self_attention_blocks)
        ])
        self.cross_attention = nn.MultiheadAttention(
            width, cross_attention_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(width)

        if use_latent:
            self.to_latent = nn.Linear(width, 2 * latent_dim)
            decoder_input = emb_dim + width + latent_dim
        else:
            decoder_input = emb_dim + width
        self.decoder = nn.Sequential(
            Projection(decoder_input, width, residual_proj),
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, 1),
        )

        if prior_anchor:
            # Zero the read-out so the network contributes exactly nothing at
            # initialisation and ``logits`` is the closed-form prior alone.
            # Without this the anchor would be buried under random noise of
            # order one, which is the whole failure it exists to prevent.
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)

    def _context_values(self, x_c: torch.Tensor, y_c: torch.Tensor) -> torch.Tensor:
        """Return self-attended context values with shape ``(1, |C|, d)``."""
        values = self.item_to_value(x_c) + self.label_to_value(2.0 * y_c[:, None] - 1.0)
        values = values.unsqueeze(0)
        for block in self.context_blocks:
            values = block(values)
        return values

    def _cross_representation(self, x_c: torch.Tensor, y_c: torch.Tensor,
                              x_t: torch.Tensor) -> torch.Tensor:
        """Give every target a different representation of its relevant context."""
        if len(x_c) == 0:
            return x_t.new_zeros((len(x_t), self.width))
        keys = self.item_to_key(x_c).unsqueeze(0)
        values = self._context_values(x_c, y_c)
        queries = self.item_to_query(x_t).unsqueeze(0)
        attended, _ = self.cross_attention(queries, keys, values, need_weights=False)
        return self.cross_norm(attended + queries).squeeze(0)

    def latent_params(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(x) == 0:
            pooled = x.new_zeros(self.width)
        else:
            pooled = self._context_values(x, y).mean(dim=1).squeeze(0)
        mu, logvar = self.to_latent(pooled).chunk(2)
        return mu, logvar.clamp(-8.0, 8.0)

    def _prior_logit(self, y_c: torch.Tensor) -> torch.Tensor:
        """Nested beta-binomial shrinkage over the context labels, as a logit.

        Context points share the subject and the benchmark, so the shipped
        predictor's subject-level and group-level tallies coincide; both stages
        are kept so this stays a copy of that math rather than a
        simplification. With no context both stages collapse to
        ``PRIOR_MEAN``, giving a zero logit.

        The result depends only on observed labels, never on a parameter, so
        no gradient flows through it. It is an offset, not a learned term.
        """
        count = float(len(y_c))
        successes = y_c.sum() if count else y_c.new_zeros(())
        subject_mean = (PRIOR_STRENGTH * PRIOR_MEAN + successes) / (PRIOR_STRENGTH + count)
        estimate = (SUBJECT_STRENGTH * subject_mean + successes) / (SUBJECT_STRENGTH + count)
        # Strictly interior by construction; clamped only against float error.
        estimate = estimate.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(estimate) - torch.log1p(-estimate)

    def logits(self, x_c: torch.Tensor, y_c: torch.Tensor, x_t: torch.Tensor,
               z: torch.Tensor | None = None) -> torch.Tensor:
        r_t = self._cross_representation(x_c, y_c, x_t)
        if self.use_latent:
            if z is None:
                z, _ = self.latent_params(x_c, y_c)
            z_t = z.unsqueeze(0).expand(len(x_t), -1)
            decoder_input = torch.cat([x_t, r_t, z_t], dim=-1)
        else:
            decoder_input = torch.cat([x_t, r_t], dim=-1)
        residual = self.decoder(decoder_input).squeeze(-1)
        if not self.prior_anchor:
            return residual
        # logit p = logit(shrinkage(y_c)) + f_theta(x_t, context). The model
        # can only move away from the shipped baseline by learning something,
        # so the baseline is a floor rather than a target.
        return self._prior_logit(y_c) + residual

    def loss(self, x_c: torch.Tensor, y_c: torch.Tensor, x_t: torch.Tensor,
             y_t: torch.Tensor, kl_weight: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.use_latent:
            nll = F.binary_cross_entropy_with_logits(self.logits(x_c, y_c, x_t), y_t)
            return nll, {}

        mu_prior, lv_prior = self.latent_params(x_c, y_c)
        x_all = torch.cat([x_c, x_t], dim=0)
        y_all = torch.cat([y_c, y_t], dim=0)
        mu_post, lv_post = self.latent_params(x_all, y_all)
        z = mu_post + torch.randn_like(mu_post) * (0.5 * lv_post).exp()
        nll = F.binary_cross_entropy_with_logits(self.logits(x_c, y_c, x_t, z=z), y_t)
        kl = 0.5 * (
            lv_prior - lv_post
            + (lv_post.exp() + (mu_post - mu_prior).square()) / lv_prior.exp()
            - 1.0
        ).sum()
        loss = nll + kl_weight * kl / max(len(y_t), 1)
        return loss, {"nll": float(nll), "kl": float(kl)}
