# Collaboration setup

Each collaborator needs a separate GitHub, Modal, and Weights & Biases account. Do not share API keys or copy credentials into Git, chat, or email. Separate accounts keep access revocable and make runs attributable to the person who launched them.

## GitHub

The repository owner invites collaborators by GitHub username. No email address is needed. After accepting the invitation, clone and install the project:

```bash
git clone https://github.com/Agrover112/aims-neurips.git
cd aims-neurips
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev,experiment,modal]'
```

Create branches for changes and merge them through pull requests. The repository's CI checks tests, submission packaging, and the official smoke test.

## Local environment

`.env.example` contains shared names and blank secret fields. It is safe to commit. A populated `.env` is ignored by Git and must stay local.

```bash
cp .env.example .env
```

The scripts do not load `.env` automatically. Either export the needed values in your shell or use a local environment loader. In most cases, `wandb login` and `modal setup` are simpler and safer than storing tokens in `.env`.

## Modal

The owner must invite the collaborator to the `agrover112` Modal workspace from **Workspace settings → Members**. Modal accepts an email invite or a reusable invite link. Add the collaborator as a **Member**, not Manager or Owner, unless they need billing or membership administration.

After accepting the invitation, the collaborator runs:

```bash
.venv/bin/modal setup
.venv/bin/modal profile current
.venv/bin/modal run modal_app.py --predictions examples/predictions.json
```

The active profile must point to the shared `agrover112` workspace. Each person should use a token created through their own Modal account. Never copy the owner's token from `~/.modal.toml` or environment variables.

Modal Members can deploy and modify apps in ordinary environments. Runs consume the shared workspace's compute budget, so agree on a spending limit before launching sweeps.

## Weights & Biases

The current W&B entity is `agrgnn-kth-royal-institute-of-technology`, and the project is `aims-predictive-evaluation`. Invite the collaborator's own W&B account to that team. If the project is Restricted, add the account to the project's **Users** tab as well.

The collaborator then runs:

```bash
.venv/bin/wandb login
WANDB_ENTITY=agrgnn-kth-royal-institute-of-technology \
WANDB_PROJECT=aims-predictive-evaluation \
.venv/bin/python scripts/run_experiment.py examples/predictions.json \
  --config configs/wandb-online.example.toml
```

Do not share the owner's W&B API key. Runs logged with separate accounts still appear in the shared project, while W&B records who created each run.

## Removing access

Remove a departing collaborator separately from GitHub, the Modal workspace, and the W&B team or restricted project. Rotate a token only if it was exposed or shared. Removing an account does not remove credentials copied from another person, which is another reason not to share tokens.
