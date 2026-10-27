"""Set-encoder building blocks for the CNP family, kept importable and testable.

Separated from the Modal job so the architecture can be unit-tested on CPU with
no container, no volume and no GPU.

Design constraints that come from the task rather than from taste:

* **Permutation invariance over the context.** A subject's answered items are a
  set, not a sequence. Every encoder here is permutation *equivariant* and every
  pooling is permutation *invariant*, so the composition is invariant as the CNP
  paper's eq. (2) requires. ``tests/test_cnp_blocks.py`` checks this numerically
  rather than trusting the construction.
* **Small widths.** "Active Evaluation Acquisition for Efficient LLM
  Benchmarking" (Li et al.) solves essentially this task and uses x/y
  representations of **16** and set features of **32**, explicitly because a
  set-transformer over a few hundred test-takers "can easily overfit". Our
  subject counts are the same order (254-4,278 per benchmark), so the default
  here is 64, not the 256 a vision-scale intuition would suggest.
* **The label needs its own projection.** Concatenating a scalar label onto a
  1024-d embedding leaves it at 0.1% of input dims and ~1.25% of input
  magnitude; measured, flipping every context label then moved the output logit
  by 0.0013. The same paper independently does the same thing: "the evaluation
  scores are passed through a linear layer to increase their dimensionality".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# projections
# ---------------------------------------------------------------------------
class Projection(nn.Module):
    """``in_dim -> out_dim`` with an optional ResNet-style shortcut.

    A plain ``x + f(x)`` is impossible across a width change, so ``residual``
    adds a bias-free linear shortcut (the standard projection-shortcut trick).
    Off by default: on a wide entrance it is not a free parameter cost.
    """

    def __init__(self, in_dim: int, out_dim: int, residual: bool = False):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.shortcut = nn.Linear(in_dim, out_dim, bias=False) if residual else None

    def forward(self, x):
        h = self.main(x)
        return h + self.shortcut(x) if self.shortcut is not None else h


# ---------------------------------------------------------------------------
# permutation-EQUIVARIANT set layers
# ---------------------------------------------------------------------------
class LinearBlock(nn.Module):
    """Pointwise residual MLP. Equivariant trivially: no element interaction.

    This is the right default on small data -- the Active-Evaluation paper
    falls back to plain linear layers exactly when set-transformer layers
    overfit their smaller benchmarks.
    """

    def __init__(self, width: int, residual: bool = True, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, width))
        self.residual = residual

    def forward(self, x, **_):
        h = self.net(x)
        return x + h if self.residual else h


class MHSABlock(nn.Module):
    """Multi-head self-attention over set elements, pre-norm.

    Equivariant because attention has no positional encoding -- permuting the
    inputs permutes the outputs identically. Cost is O(n^2) in context size.
    """

    def __init__(self, width: int, heads: int = 4, residual: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(),
            nn.Linear(2 * width, width))
        self.residual = residual

    def forward(self, x, **_):
        h = self.norm(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a if self.residual else a
        f = self.ff(x)
        return x + f if self.residual else f


class ISABBlock(nn.Module):
    """Induced Set Attention Block (Set Transformer, Lee et al. 2018).

    Attends the set through ``m`` learned inducing points instead of to itself,
    dropping the cost from O(n^2) to O(nm). This is what the Active-Evaluation
    paper uses ("replace the self-attention layer with a more memory-efficient
    Set Transformer layer"), with 8-16 inducing points.
    """

    def __init__(self, width: int, heads: int = 4, n_inducing: int = 16,
                 residual: bool = True, dropout: float = 0.0):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, width) * 0.02)
        self.norm_in = nn.LayerNorm(width)
        self.mab1 = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.mab2 = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(),
            nn.Linear(2 * width, width))
        self.residual = residual

    def forward(self, x, **_):
        h = self.norm_in(x)
        ind = self.inducing.expand(x.shape[0], -1, -1)
        # inducing points attend to the set, then the set attends back
        z, _ = self.mab1(ind, h, h, need_weights=False)
        out, _ = self.mab2(h, z, z, need_weights=False)
        x = x + out if self.residual else out
        f = self.ff(x)
        return x + f if self.residual else f


class SGFormerBlock(nn.Module):
    """SGFormer's simple global attention (Wu et al., NeurIPS 2023).

    SGFormer has two branches: a GNN branch that consumes a graph, and a global
    attention branch that consumes **features only**. This is the latter, used
    standalone — no graph required, which is why it drops into a set encoder
    cleanly. The paper's point is that a *single* layer of this suffices, so
    ``n_blocks=1`` is the intended setting.

    The attention drops softmax for a normalised linear form:

        q~ = Q/||Q||,  k~ = K/||K||
        z_i = (v_i + (1/N) sum_j <q~_i, k~_j> v_j)
              / (1 + (1/N) sum_j <q~_i, k~_j>)

    Both sums factor through ``sum_j k~_j v_j^T`` (d x d) and ``sum_j k~_j``
    (d), so the cost is O(N d^2) rather than MHSA's O(N^2 d). Permutation
    equivariant: every term is either pointwise or a sum over the set.
    """

    def __init__(self, width: int, heads: int = 1, residual: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(),
            nn.Linear(2 * width, width))
        self.residual = residual

    def forward(self, x, **_):
        h = self.norm(x)                                   # (B, N, D)
        n = h.shape[1]
        q = F.normalize(self.q(h), dim=-1)
        k = F.normalize(self.k(h), dim=-1)
        v = self.v(h)
        # linear-attention factorisation: accumulate over the set once
        kv = torch.einsum("bnd,bne->bde", k, v)            # (B, D, D)
        num = v + torch.einsum("bnd,bde->bne", q, kv) / n
        ksum = k.sum(dim=1)                                # (B, D)
        den = 1.0 + torch.einsum("bnd,bd->bn", q, ksum).unsqueeze(-1) / n
        out = self.drop(num / den.clamp(min=1e-6))
        x = x + out if self.residual else out
        f = self.ff(x)
        return x + f if self.residual else f


ENCODERS = {"linear": LinearBlock, "mhsa": MHSABlock,
            "set_transformer": ISABBlock, "sgformer": SGFormerBlock}


def make_encoder(kind: str, width: int, n_blocks: int, **kw) -> nn.Module:
    if kind not in ENCODERS:
        raise ValueError(f"unknown encoder {kind!r}; options: {sorted(ENCODERS)}")
    cls = ENCODERS[kind]
    if kind == "sgformer":
        # the paper's claim is that one layer is enough; honour that default
        # but still allow stacking if an ablation wants it.
        kw.pop("n_inducing", None)
    return nn.ModuleList([cls(width, **kw) for _ in range(n_blocks)])


# ---------------------------------------------------------------------------
# permutation-INVARIANT pooling
# ---------------------------------------------------------------------------
class MeanPool(nn.Module):
    """The CNP paper's default aggregator.

    Worth keeping as the default for a reason beyond simplicity: a mean is
    computable in O(1) from the running mean, so acquisition can stream one
    label at a time with no recomputation -- exactly the competition's setting.
    """

    def forward(self, x, mask=None):
        if x.shape[1] == 0:
            return x.new_zeros(x.shape[0], x.shape[2])
        return x.mean(dim=1)


class AttentionPool(nn.Module):
    """Pooling by Multihead Attention (PMA) -- Set Transformer's learned pooling.

    A single learned seed query attends over the set. Strictly more expressive
    than a mean, but it gives up the O(1) streaming update above.
    """

    def __init__(self, width: int, heads: int = 4):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)

    def forward(self, x, mask=None):
        if x.shape[1] == 0:
            return x.new_zeros(x.shape[0], x.shape[2])
        h = self.norm(x)
        q = self.seed.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, h, h, need_weights=False)
        return out.squeeze(1)


def make_pool(kind: str, width: int, heads: int = 4) -> nn.Module:
    if kind == "mean":
        return MeanPool()
    if kind in ("learned", "pma", "attention"):
        return AttentionPool(width, heads)
    raise ValueError(f"unknown pooling {kind!r}; options: mean | learned")


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class NeuralProcess(nn.Module):
    """CNP (deterministic) or NP (latent), configurable encoder and pooling.

    Deterministic path is Garnelo et al. eq. (1)-(3):
        r_i = h(x_i, y_i);  r = pool(r_i);  phi = g(x_T, r)

    With ``use_latent`` it becomes the latent-variable NP the Active-Evaluation
    paper uses: the pooled representation parameterises a Gaussian ``z``, the
    prior is conditioned on the context and the posterior on context+target, and
    training maximises the ELBO. The latent is what supplies the predictive
    uncertainty an acquisition policy would need -- the deterministic CNP has
    none beyond the Bernoulli mean.

    Output is one Bernoulli logit per target: the paper's classification case
    with C=2. The factorisation over targets is IRT's local-independence
    assumption under another name.
    """

    def __init__(self, emb_dim: int, width: int = 64, n_blocks: int = 2,
                 encoder: str = "linear", pooling: str = "mean",
                 use_latent: bool = False, latent_dim: int = 32,
                 heads: int = 4, residual_proj: bool = False,
                 residual_blocks: bool = True, dropout: float = 0.0,
                 n_inducing: int = 16):
        super().__init__()
        self.width, self.use_latent, self.latent_dim = width, use_latent, latent_dim
        self.encoder_kind, self.pooling_kind = encoder, pooling

        kw = {"residual": residual_blocks, "dropout": dropout}
        if encoder == "mhsa":
            kw["heads"] = heads
        if encoder == "set_transformer":
            kw["heads"], kw["n_inducing"] = heads, n_inducing

        # item down, label UP -- see module docstring
        self.item_proj = Projection(emb_dim, width, residual_proj)
        self.label_proj = nn.Linear(1, width)
        self.enc_blocks = make_encoder(encoder, width, n_blocks, **kw)
        self.pool = make_pool(pooling, width, heads)

        if use_latent:
            self.to_latent = nn.Linear(width, 2 * latent_dim)   # mu, logvar
            dec_in = emb_dim + latent_dim
        else:
            dec_in = emb_dim + width
        self.dec_proj = Projection(dec_in, width, residual_proj)
        self.dec_blocks = make_encoder(encoder, width, n_blocks, **kw)
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.GELU(),
            nn.Linear(width // 2, 1))

    # -- eq (1) + (2) --------------------------------------------------------
    def _set_repr(self, x, y):
        """(n, emb) + (n,) -> (width,). Empty set -> zeros, i.e. the prior."""
        if x.shape[0] == 0:
            return x.new_zeros(self.width)
        # centre to {-1,+1}: with raw 0/1 a wrong answer and an absent
        # observation both contribute zero signal.
        h = self.item_proj(x) + self.label_proj(2.0 * y[:, None] - 1.0)
        h = h.unsqueeze(0)                       # (1, n, width)
        for blk in self.enc_blocks:
            h = blk(h)
        return self.pool(h).squeeze(0)

    def latent_params(self, x, y):
        raw = self.to_latent(self._set_repr(x, y))
        mu, logvar = raw[:self.latent_dim], raw[self.latent_dim:]
        return mu, logvar.clamp(-8.0, 8.0)

    # -- eq (3) --------------------------------------------------------------
    def logits(self, x_c, y_c, x_t, z=None):
        if self.use_latent:
            if z is None:
                mu, logvar = self.latent_params(x_c, y_c)
                z = mu                                    # deterministic at eval
            cond = z
        else:
            cond = self._set_repr(x_c, y_c)
        h = self.dec_proj(torch.cat([x_t, cond.unsqueeze(0).expand(x_t.shape[0], -1)], dim=-1))
        h = h.unsqueeze(0)
        for blk in self.dec_blocks:
            h = blk(h)
        return self.head(h.squeeze(0)).squeeze(-1)

    # -- training objective --------------------------------------------------
    def loss(self, x_c, y_c, x_t, y_t, kl_weight: float = 1.0):
        """BCE (CNP eq. 4) or the negative ELBO when latent.

        Latent case follows the Active-Evaluation paper: prior from the context
        alone, posterior from context+target, reconstruct the target under a
        posterior sample, penalise KL(posterior || prior).
        """
        if not self.use_latent:
            return F.binary_cross_entropy_with_logits(self.logits(x_c, y_c, x_t), y_t), {}

        x_all = torch.cat([x_c, x_t], dim=0)
        y_all = torch.cat([y_c, y_t], dim=0)
        mu_p, lv_p = self.latent_params(x_c, y_c)       # prior  p(z | O)
        mu_q, lv_q = self.latent_params(x_all, y_all)   # posterior q(z | O u T)
        z = mu_q + torch.randn_like(mu_q) * (0.5 * lv_q).exp()
        nll = F.binary_cross_entropy_with_logits(self.logits(x_c, y_c, x_t, z=z), y_t)
        kl = 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1.0).sum()
        return nll + kl_weight * kl / max(len(y_t), 1), {"nll": float(nll), "kl": float(kl)}
