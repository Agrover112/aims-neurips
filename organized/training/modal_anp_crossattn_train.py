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

import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "aims-neurips-anp-crossattn-launcher"
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from codex_es_fixed_modal_cnp_train import app as _BASE_APP, run as _BASE_RUN

app = modal.App(APP_NAME)
# ``run`` belongs to the shared trainer's app. Including it hydrates the
# function in this launcher app, which is required before Modal can create a
# GPU-specific ``with_options`` variant.
app.include(_BASE_APP)


@app.local_entrypoint()
def main(seed: int = 20261018, arm: str = "qwen3_8b", width: int = 64,
         self_attention_blocks: int = 2, cross_attention_heads: int = 4,
         use_latent: bool = True, latent_dim: int = 32,
         residual_proj: bool = False, dropout: float = 0.0, steps: int = 3000,
         wandb_mode: str = "online", run_label: str = "", gpu: str = "",
         cpu: int = 8, max_context: int = 64,
         wandb_secret: str = "wandb", benchmark_holdout: bool = False,
         benchmark_min_item_responders: int = 15,
         benchmark_min_items_per_subject: int = 80,
         prior_anchor: bool = False, sam_rho: float = 0.0,
         prefiltered_responses: str = "", alc_weighted: bool = True,
         dense_prefix_training: bool = False, item_independent: bool = False) -> None:
    """Run one ANP configuration. Use a distinct label for every rerun."""
    if not run_label:
        run_label = datetime.now(timezone.utc).strftime("anp-%Y%m%dT%H%M%SZ")
    requested_accelerator = gpu or f"cpu-{cpu}"
    retry_policy = modal.Retries(max_retries=3, initial_delay=30.0, max_delay=60.0)
    runner = _BASE_RUN.with_options(
        cpu=cpu, memory=32_768, timeout=12 * 60 * 60, retries=retry_policy
    )
    if gpu:
        # GPU jobs use less host capacity than the CPU-only trainer. The
        # requested accelerator is also recorded in W&B.
        runner = runner.with_options(gpu=gpu, cpu=4, memory=16_384,
                                     timeout=12 * 60 * 60, retries=retry_policy)
    if wandb_mode != "disabled":
        # The shared trainer only reads WANDB_API_KEY from the environment; no
        # function in this file attached a secret before, so every run logged
        # to the volume only. Mirrors modal_set_anp_train.py. The workspace
        # must hold a `wandb` secret or Modal fails the spawn by name, which is
        # better than silently losing the run's history.
        runner = runner.with_options(
            secrets=[modal.Secret.from_name(wandb_secret,
                                            required_keys=["WANDB_API_KEY"])]
        )

    call = runner.spawn(
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
        max_context=max_context,
        cross_attention_heads=cross_attention_heads,
        latent_dim=latent_dim,
        dropout=dropout,
        requested_accelerator=requested_accelerator,
        benchmark_holdout=benchmark_holdout,
        benchmark_min_item_responders=benchmark_min_item_responders,
        benchmark_min_items_per_subject=benchmark_min_items_per_subject,
        prior_anchor=prior_anchor,
        sam_rho=sam_rho,
        prefiltered_responses=prefiltered_responses,
        alc_weighted=alc_weighted,
        dense_prefix_training=dense_prefix_training,
        item_independent=item_independent,
    )
    print(f"spawned Modal call {call.object_id}")
