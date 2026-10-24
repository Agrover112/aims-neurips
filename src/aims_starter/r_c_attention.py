"""Isolated deterministic ANP-attention rung for the r_C ablation.

This module is intentionally separate from the production CNP/ANP classes in
``faithful_np.py``.  It implements the missing target-specific context
representation

    r_{C,t} = Attn(query=h_t, key=h_c, value=r_c)

with no latent variable ``z`` and no IRT anchor.  The decoder emits Bernoulli
logits directly, matching the paper's ``phi_t = g(x_t, r)`` form.
"""

from __future__ import annotations

import torch
from torch import nn


class ContextSelfAttention(nn.Module):
    """Permutation-invariant self-attention over label-conditioned context reps."""

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.0
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, context: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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
            mask = torch.ones(
                context.shape[:2], dtype=torch.bool, device=context.device
            )
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
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.0
        )
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
    ):
        super().__init__()
        self.subject_feature_slice = subject_feature_slice
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
        self.context_layers = nn.ModuleList(
            [ContextSelfAttention(d_model, n_heads) for _ in range(n_context_layers)]
        )
        self.cross_attention = TargetContextAttention(d_model, n_heads)
        self.decoder = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
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


class DeterministicCNP(nn.Module):
    """Pure CNP decoder for the reverse ladder: phi_t = g([h_t, r_C])."""

    def __init__(
        self,
        x_dim: int,
        *,
        d_model: int = 128,
        subject_feature_slice: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.subject_feature_slice = subject_feature_slice
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
        self.decoder = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
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

    MODES = ("2pl", "scalar", "residual", "additive_ab", "direct_ab")

    def __init__(
        self,
        x_dim: int,
        *,
        mode: str,
        d_model: int = 128,
        subject_feature_slice: tuple[int, int] | None = None,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
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
        if mode != "2pl":
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

    @staticmethod
    def _split_parameters(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.nn.functional.softplus(raw[..., 0]) + 0.05, raw[..., 1]

    def _context_repr(self, x_C: torch.Tensor, y_C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_C = self.encode_item(x_C)
        if self.mode == "2pl":
            return h_C, h_C.new_zeros((h_C.shape[-1],))
        if h_C.shape[0] == 0:
            return h_C, h_C.new_zeros((h_C.shape[-1],))
        context = self.context_encoder(torch.cat([h_C, y_C[:, None]], dim=-1))
        return h_C, context.mean(dim=0)

    def _context_parameters(
        self, h_C: torch.Tensor, r_C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.item_head(h_C)
        if self.mode in ("2pl", "scalar", "residual"):
            return self._split_parameters(base)
        enriched = torch.cat([h_C, r_C.unsqueeze(0).expand(h_C.shape[0], -1)], dim=-1)
        offset = self.ab_head(enriched)
        if self.mode == "additive_ab":
            return self._split_parameters(base + offset)
        return self._split_parameters(offset)

    def _anchor_parameters(
        self, h_C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        if self.mode in ("2pl", "scalar", "residual"):
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
            gradient = torch.sum(a_C * (probability - y_C)) + (theta - prior) / 4.0
            hessian = torch.sum(a_C.square() * probability * (1.0 - probability)) + 0.25
            theta = (theta - (gradient / hessian).clamp(-1.0, 1.0)).clamp(-4.0, 4.0)
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
        theta = self.map_theta(a_C, b_C, y_C)
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
        enriched = torch.cat(
            [h, r_C.unsqueeze(0).expand(h.shape[0], -1), state], dim=-1
        )
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
            a_T, b_T = self._item_parameters(
                h_T[index : index + 1], r_C, state.unsqueeze(0)
            )
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
        enriched = torch.cat(
            [h, r_C.unsqueeze(0).expand(h.shape[0], -1)], dim=-1
        )
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

    def _subject_prior(
        self, x_C: torch.Tensor, x_T: torch.Tensor
    ) -> torch.Tensor | None:
        if not self.use_subject_prior:
            return None
        reference = x_T if x_T.shape[0] else x_C
        if reference.shape[0] == 0:
            return None
        start, stop = self.subject_prior_slice
        subject = reference[:, start:stop].mean(dim=0)
        return torch.tanh(self.subject_prior_head(subject)).reshape(())

    def _basis(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -(theta - self.centers).square() / (2.0 * self.lengthscale**2)
        )

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
        enriched = torch.cat(
            [h, r_C.unsqueeze(0).expand(h.shape[0], -1)], dim=-1
        )
        return self._split(base + self.ab_head(enriched))

    def _coefficients(self, h: torch.Tensor, r_C: torch.Tensor) -> torch.Tensor:
        return self.coefficient_head(self._features(h, r_C))

    def _subject_prior(
        self, x_C: torch.Tensor, x_T: torch.Tensor
    ) -> torch.Tensor | None:
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
            gradient = torch.sum((probability - y_C) * slope) + (
                theta - prior
            ) / 4.0
            hessian = torch.sum(
                probability * (1.0 - probability) * slope.square()
                + (probability - y_C) * curvature
            ) + 0.25
            theta = (
                theta - (gradient / hessian.clamp_min(1e-6)).clamp(-1.0, 1.0)
            ).clamp(-4.0, 4.0)
        return theta

    def _set_penalty(self, c_C: torch.Tensor, c_T: torch.Tensor) -> None:
        coefficients = torch.cat([c_C, c_T], dim=0)
        if coefficients.shape[0] == 0:
            self._gp_penalty = coefficients.sum() * 0.0
            return
        quadratic = torch.einsum(
            "nm,mk,nk->n", coefficients, self.kernel_inv, coefficients
        )
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
        phi = torch.exp(
            -(theta - self.centers).square() / (2.0 * self.lengthscale**2)
        )
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
