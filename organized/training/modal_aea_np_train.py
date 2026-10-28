"""Launch AEA-NP, the Set-Transformer latent NP from the AEA paper.

This architecture is separate from CNP and ANP. It reuses only the filtered
data stream, fixed validation bank, exact coverage diagnostics, W&B naming,
and checkpoint policy from the shared trainer.

Example:
  .venv/bin/modal run --detach --profile agrover112 \
    organized/training/modal_aea_np_train.py::main \
    --seed 20261018 --width 32 --encoder-blocks 2 --decoder-blocks 2 \
    --attention-heads 8 --n-inducing 16 --latent-dim 32 --steps 3000 \
    --run-label aea-np-smoke
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "aims-neurips-aea-np-launcher"
_BASE_PATH = Path(__file__).with_name("codex_es_fixed_modal_cnp_train.py")
_SPEC = importlib.util.spec_from_file_location("_filtered_np_trainer", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import-time guard.
    raise RuntimeError(f"could not load shared trainer from {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

app = modal.App(APP_NAME)
app.include(_BASE.app)


@app.local_entrypoint()
def main(seed: int = 20261018, arm: str = "qwen3_8b", width: int = 32,
         item_dim: int = 16, label_dim: int = 16, encoder_blocks: int = 2,
         decoder_blocks: int = 2, attention_heads: int = 8,
         n_inducing: int = 16, latent_dim: int = 32, dropout: float = 0.0,
         steps: int = 3000, wandb_mode: str = "online", run_label: str = "",
         gpu: str = "", wandb_secret: str = "wandb") -> None:
    """Run one AEA-NP configuration. AEA-NP always uses its Gaussian latent."""
    if not run_label:
        run_label = datetime.now(timezone.utc).strftime("aea-np-%Y%m%dT%H%M%SZ")
    requested_accelerator = gpu or "cpu"
    runner = _BASE.run
    if wandb_mode != "disabled":
        # The shared trainer only reads WANDB_API_KEY from the environment, so
        # without this the run trains for hours and logs nowhere. required_keys
        # turns a misnamed key into an immediate spawn failure instead.
        runner = runner.with_options(
            secrets=[modal.Secret.from_name(wandb_secret,
                                            required_keys=["WANDB_API_KEY"])]
        )

    if gpu:
        runner = runner.with_options(gpu=gpu, cpu=4, memory=16_384,
                                     timeout=6 * 60 * 60)
    handle = runner.spawn(
        seed=seed,
        arm=arm,
        use_latent=True,
        width=width,
        n_blocks=encoder_blocks,
        decoder_blocks=decoder_blocks,
        n_inducing=n_inducing,
        item_dim=item_dim,
        label_dim=label_dim,
        steps=steps,
        residual_proj=False,
        wandb_mode=wandb_mode,
        run_label=run_label,
        model_family="aea_np",
        cross_attention_heads=attention_heads,
        latent_dim=latent_dim,
        dropout=dropout,
        requested_accelerator=requested_accelerator,
    )
    print(handle.get())
