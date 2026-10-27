"""Copy the aims-neurips-data Volume into another workspace, cloud to cloud.

Runs in the destination workspace, mounts its Volume, and pulls each top-level
folder from the source workspace with the Modal CLI, authenticated by the
`src-modal-token` secret (SRC_MODAL_TOKEN_ID / SRC_MODAL_TOKEN_SECRET).

    MODAL_PROFILE=aims-test modal run --detach scripts/modal_copy_volume_in.py
"""

import os
import subprocess

import modal

VOLUME_NAME = "aims-neurips-data"
FOLDERS = ["datasets", "cache", "results", "huggingface-cache"]

app = modal.App("copy-volume-in")
image = modal.Image.debian_slim().pip_install("modal==1.6.2")
dst = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


@app.function(image=image, cpu=1, memory=2048, timeout=6 * 3600,
              volumes={"/dst": dst}, secrets=[modal.Secret.from_name("src-modal-token")])
def copy():
    # The CLI subprocess must authenticate as the source workspace, not via this container's task auth.
    env = {k: v for k, v in os.environ.items() if not k.startswith("MODAL_")}
    env["MODAL_TOKEN_ID"] = os.environ["SRC_MODAL_TOKEN_ID"]
    env["MODAL_TOKEN_SECRET"] = os.environ["SRC_MODAL_TOKEN_SECRET"]
    for folder in FOLDERS:
        subprocess.run(["modal", "volume", "get", "--force", VOLUME_NAME, f"/{folder}", "/dst/"],
                       env=env, check=True, stdout=subprocess.DEVNULL)
        dst.commit()
        print("done", folder, flush=True)
