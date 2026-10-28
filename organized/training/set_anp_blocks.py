"""Set-ANP: the one-block induced-set-attention ablation of our ANP.

This module deliberately inherits every path from ``AttentiveNeuralProcess``.
The only learned computation it replaces is the context-to-context mixer:

* ANP: one multi-head self-attention block; and
* Set-ANP: one induced Set Attention Block (ISAB) with ``m`` inducing points.

The shared labelled-context value encoder, target-to-context cross-attention,
global latent path, decoder, loss, and checkpoint interface are unchanged.
That makes this a clean MHSA-versus-ISAB ablation, rather than the broader
AEA-NP architecture.
"""

from __future__ import annotations

from torch import nn

try:  # Package imports in tests; flat imports inside the Modal container.
    from .anp_crossattn_blocks import AttentiveNeuralProcess
    from .cnp_blocks import ISABBlock
except ImportError:  # pragma: no cover - exercised in the remote container.
    from anp_crossattn_blocks import AttentiveNeuralProcess
    from cnp_blocks import ISABBlock


class SetAttentiveNeuralProcess(AttentiveNeuralProcess):
    """ANP whose context mixer is an ISAB instead of self-attention.

    ``self_attention_blocks`` retains the ANP flag name because it counts
    context-mixing blocks.  The clean comparison uses exactly one block in
    both models.  ``n_inducing`` is the Set Transformer-specific knob.
    """

    def __init__(self, emb_dim: int, *, width: int = 64,
                 self_attention_blocks: int = 1,
                 cross_attention_heads: int = 4,
                 n_inducing: int = 16, use_latent: bool = True,
                 latent_dim: int = 32, residual_proj: bool = False,
                 dropout: float = 0.0) -> None:
        if n_inducing <= 0:
            raise ValueError("n_inducing must be positive")
        super().__init__(
            emb_dim, width=width, self_attention_blocks=self_attention_blocks,
            cross_attention_heads=cross_attention_heads, use_latent=use_latent,
            latent_dim=latent_dim, residual_proj=residual_proj, dropout=dropout,
        )
        self.n_inducing = n_inducing
        self.context_blocks = nn.ModuleList([
            ISABBlock(width, heads=cross_attention_heads,
                      n_inducing=n_inducing, dropout=dropout)
            for _ in range(self_attention_blocks)
        ])
