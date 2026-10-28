"""Set-Transformer latent Neural Process from the AEA benchmarking paper.

The model follows the architecture in Appendix B of "Active Evaluation
Acquisition for Efficient LLM Benchmarking":

* an induced Set Transformer encodes labelled context prompt pairs;
* learned multi-head pooling parameterises a global Gaussian latent;
* every target prompt cross-attends to the encoded context; and
* induced Set Transformer decoder layers let target predictions exchange
  information before the Bernoulli output head.

It is intentionally separate from the generic CNP and from the ANP module:
the architecture has a specific, testable meaning and should not become a
mixture of unrelated flags.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:  # Package import locally; flat import inside the Modal image.
    from .cnp_blocks import ISABBlock, MHSAPool
except ImportError:  # pragma: no cover - exercised in the Modal container.
    from cnp_blocks import ISABBlock, MHSAPool


class ActiveEvaluationNeuralProcess(nn.Module):
    """AEA-NP for binary item responses.

    ``width`` is the Set Transformer feature dimension. ``item_dim`` and
    ``label_dim`` are the two input representations before concatenation. The
    default 16/16/32 matches the paper's small-model configuration.
    """

    def __init__(self, emb_dim: int, *, width: int = 32, item_dim: int = 16,
                 label_dim: int = 16, encoder_blocks: int = 2,
                 decoder_blocks: int = 2, attention_heads: int = 8,
                 n_inducing: int = 16, latent_dim: int = 32,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if width % attention_heads:
            raise ValueError(
                f"width={width} must divide evenly across attention_heads="
                f"{attention_heads}"
            )
        if encoder_blocks < 0 or decoder_blocks < 0:
            raise ValueError("encoder_blocks and decoder_blocks must be non-negative")
        if n_inducing <= 0:
            raise ValueError("n_inducing must be positive")

        self.width = width
        self.item_dim = item_dim
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks
        self.attention_heads = attention_heads
        self.n_inducing = n_inducing

        # Appendix B: reduce prompt embeddings, expand scores, concatenate.
        self.item_encoder = nn.Linear(emb_dim, item_dim)
        self.label_encoder = nn.Linear(1, label_dim)
        self.pair_encoder = nn.Linear(item_dim + label_dim, width)

        # One shared network parameterises both p(z | context) and
        # q(z | context, target), exactly as described in the paper.
        self.context_blocks = nn.ModuleList([
            ISABBlock(width, heads=attention_heads, n_inducing=n_inducing,
                      dropout=dropout)
            for _ in range(encoder_blocks)
        ])
        self.pool = MHSAPool(width, attention_heads)
        self.to_latent = nn.Linear(width, 2 * latent_dim)

        # Queries and keys use prompt identity; values carry scored prompt
        # representations. PyTorch's attention supplies separate q/k/v maps.
        self.item_to_attention = nn.Linear(item_dim, width)
        self.cross_attention = nn.MultiheadAttention(
            width, attention_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(width)

        # AEA-NP differs from a standard ANP here: target rows then pass
        # through set layers, so they can communicate before prediction.
        self.decoder_input = nn.Linear(item_dim + width + latent_dim, width)
        self.decoder_set_blocks = nn.ModuleList([
            ISABBlock(width, heads=attention_heads, n_inducing=n_inducing,
                      dropout=dropout)
            for _ in range(decoder_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, 1),
        )

    def _item_repr(self, x: torch.Tensor) -> torch.Tensor:
        return self.item_encoder(x)

    def _context_values(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Encode labelled pairs into one contextual vector per pair."""
        if len(x) == 0:
            return x.new_zeros((0, self.width))
        item = self._item_repr(x)
        # Centre labels: absence of a response is not confused with a 0 label.
        label = self.label_encoder(2.0 * y[:, None] - 1.0)
        values = self.pair_encoder(torch.cat([item, label], dim=-1)).unsqueeze(0)
        for block in self.context_blocks:
            values = block(values)
        return values.squeeze(0)

    def latent_params(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self._context_values(x, y)
        pooled = self.pool(values.unsqueeze(0)).squeeze(0)
        mu, logvar = self.to_latent(pooled).chunk(2)
        return mu, logvar.clamp(-8.0, 8.0)

    def _cross_representation(self, x_c: torch.Tensor, y_c: torch.Tensor,
                              x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return target-specific context representations and target items."""
        target_item = self._item_repr(x_t)
        queries = self.item_to_attention(target_item).unsqueeze(0)
        if len(x_c) == 0:
            return target_item, x_t.new_zeros((len(x_t), self.width))
        context_item = self._item_repr(x_c)
        keys = self.item_to_attention(context_item).unsqueeze(0)
        values = self._context_values(x_c, y_c).unsqueeze(0)
        attended, _ = self.cross_attention(queries, keys, values, need_weights=False)
        return target_item, self.cross_norm(attended + queries).squeeze(0)

    def logits(self, x_c: torch.Tensor, y_c: torch.Tensor, x_t: torch.Tensor,
               z: torch.Tensor | None = None) -> torch.Tensor:
        target_item, target_context = self._cross_representation(x_c, y_c, x_t)
        if z is None:
            z, _ = self.latent_params(x_c, y_c)
        z_per_target = z.unsqueeze(0).expand(len(x_t), -1)
        decoded = self.decoder_input(torch.cat(
            [target_item, target_context, z_per_target], dim=-1
        )).unsqueeze(0)
        for block in self.decoder_set_blocks:
            decoded = block(decoded)
        return self.head(decoded.squeeze(0)).squeeze(-1)

    def loss(self, x_c: torch.Tensor, y_c: torch.Tensor, x_t: torch.Tensor,
             y_t: torch.Tensor, kl_weight: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
        """Negative ELBO with q(z | context plus target) and p(z | context)."""
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
        return nll + kl_weight * kl / max(len(y_t), 1), {
            "nll": float(nll), "kl": float(kl),
        }
