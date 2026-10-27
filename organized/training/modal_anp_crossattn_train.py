"""Launch the target-to-context Attentive Neural Process trainer.

This is intentionally a separate entry point from the CNP trainer. It reuses
the filtered-data loop, split, fixed validation bank, checkpoint policy, W&B
panels, and coverage accounting from ``codex_es_fixed_modal_cnp_train.py``.

It drops CNP's ``encoder`` and ``pooling`` flags. The deterministic ANP path
always uses target-to-context cross-attention. ``self_attention_blocks``
controls context-to-context attention before those values are read by targets.

Example:
  .venv/bin/modal run --detach --profile agrover112 \
    organized/training/modal_anp_crossattn_train.py::main \
    --seed 20261018 --width 64 --self-attention-blocks 2 \
    --cross-attention-heads 4 --steps 3000 --run-label anp-smoke
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "aims-neurips-anp-crossattn-launcher"
_BASE_PATH = Path(__file__).with_name("codex_es_fixed_modal_cnp_train.py")
_SPEC = importlib.util.spec_from_file_location("_filtered_np_trainer", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import-time guard.
    raise RuntimeError(f"could not load shared trainer from {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

app = modal.App(APP_NAME)


@app.local_entrypoint()
def main(seed: int = 20261018, arm: str = "qwen3_8b", width: int = 64,
         self_attention_blocks: int = 2, cross_attention_heads: int = 4,
         use_latent: bool = True, latent_dim: int = 32,
         residual_proj: bool = False, dropout: float = 0.0, steps: int = 3000,
         wandb_mode: str = "online", run_label: str = "") -> None:
    """Run one ANP configuration. Use a distinct label for every rerun."""
    if not run_label:
        run_label = datetime.now(timezone.utc).strftime("anp-%Y%m%dT%H%M%SZ")
    handle = _BASE.run.spawn(
        seed=seed,
        arm=arm,
        use_latent=use_latent,
        width=width,
        n_blocks=self_attention_blocks,
        steps=steps,
        residual_proj=residual_proj,
        wandb_mode=wandb_mode,
        run_label=run_label,
        model_family="anp_cross_attention",
        cross_attention_heads=cross_attention_heads,
        latent_dim=latent_dim,
        dropout=dropout,
    )
    print(handle.get())
