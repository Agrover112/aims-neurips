"""Launch the matched latent-CNP baseline on CPU or an explicit GPU.

The model stays intentionally simple: pooled context representation plus a
global Gaussian latent. Use it against ANP to isolate target-specific
cross-attention, holding the data, seed, width, latent size, and training
budget fixed.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "aims-neurips-latent-cnp-launcher"
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from codex_es_fixed_modal_cnp_train import app as _BASE_APP, run as _BASE_RUN

app = modal.App(APP_NAME)
app.include(_BASE_APP)


@app.local_entrypoint()
def main(seed: int = 20261018, arm: str = "qwen3_8b", width: int = 128,
         latent_dim: int = 128, n_blocks: int = 2, dropout: float = 0.0,
         steps: int = 100000, wandb_mode: str = "online", run_label: str = "",
         gpu: str = "", cpu: int = 8,
         wandb_secret: str = "wandb") -> None:
    """Run one latent-CNP baseline with a pooled linear context encoder."""
    if not run_label:
        run_label = datetime.now(timezone.utc).strftime("latent-cnp-%Y%m%dT%H%M%SZ")
    requested_accelerator = gpu or f"cpu-{cpu}"
    # ``spawn`` below hands the invocation to Modal and returns immediately.
    # The retry policy then restarts preempted/transiently failed containers
    # from the trainer's atomic volume checkpoint, without a local client
    # remaining attached to the input.
    retry_policy = modal.Retries(max_retries=3, initial_delay=30.0, max_delay=60.0)
    runner = _BASE_RUN.with_options(
        cpu=cpu, memory=32_768, timeout=12 * 60 * 60, retries=retry_policy
    )
    if gpu:
        runner = runner.with_options(gpu=gpu, cpu=4, memory=16_384,
                                     timeout=12 * 60 * 60, retries=retry_policy)
    if wandb_mode != "disabled":
        # The shared trainer only reads WANDB_API_KEY from the environment, so
        # without this the run trains for hours and logs nowhere. required_keys
        # turns a misnamed key into an immediate spawn failure instead.
        runner = runner.with_options(
            secrets=[modal.Secret.from_name(wandb_secret,
                                            required_keys=["WANDB_API_KEY"])]
        )

    # Do not use ``remote`` here: a disappearing local client cancels its
    # input. ``spawn`` durably queues the job, so Modal—not this CLI process—
    # owns its lifetime and retry policy.
    call = runner.spawn(
        seed=seed,
        arm=arm,
        encoder="linear",
        pooling="mean",
        use_latent=True,
        width=width,
        n_blocks=n_blocks,
        latent_dim=latent_dim,
        dropout=dropout,
        steps=steps,
        wandb_mode=wandb_mode,
        run_label=run_label,
        model_family="cnp",
        requested_accelerator=requested_accelerator,
    )
    print(f"spawned Modal call {call.object_id}")
