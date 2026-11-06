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

## Continuing as Person-B on the offline-vs-online paper

Read `Final-plan.md` (work packages and the split) and `paper/eandd/PROGRESS.md` (what Person-A has done and found)
first. Person-B owns WP2 (Beta-Binomial grid), WP4a (ID panel), WP7 (item-difficulty probes) and WP9 (analysis and
figures), and writes the paper sections listed in the plan.

### 1. Get the code
```bash
git clone https://github.com/Agrover112/aims-neurips.git && cd aims-neurips
git checkout -b eandd-analysis            # Person-B's branch; Person-A uses eandd-models
uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev,experiment,modal]'
```
Merge through pull requests. Person-B edits only the files the plan assigns (`bb_grid.py`, `build_id_panel.py`,
`probes.py`, `analysis.py`, `plot_*.py`, all under `organized/testing/eandd/`); the replay and fit code is Person-A's.

### 2. Get the data (not in git: competition data and files over GitHub's 100 MB limit)
Ask Person-A for a private copy (`scp`, `rsync` or a shared drive), or fetch it from the Modal workspace after being
invited as described above. Never put these in git (`CONTRIBUTING.md`).

| What | Size | Put it at (or set the variable) |
|---|---|---|
| Responses, `filtered-responses.parquet` | 175 MB | `~/aims-noitem-run/filtered-responses.parquet` (`AIMS_RESPONSES`) |
| Panel: `items.parquet`, `candidate_orders.parquet` | 4 MB | `~/aims-replay-cache/panel/` (`AIMS_REPLAY_CACHE`) |
| BGE-M3 item embeddings: `item_embeddings.npy`, `items.parquet` | 504 MB | `~/aims-noitem-run/datasets/full/allbench-embed-bge_m3_long-v2/` (`AIMS_EMBEDDINGS`) |
| Frozen draws | 13 KB | already in git: `cp paper/eandd/data/draws.parquet ~/aims-replay-cache/paper-eandd/` (or set `AIMS_EANDD_DATA`) |

A copy of the responses also sits on the Modal volume `aims-neurips-data` (`/data/export/bhold-bge_m3_long-n15-m80-s20261018/`,
as `organized/testing/online/modal_hier_beta.py` uses it). I have **not** checked where the panel and the embeddings
sit on Modal, so ask Person-A for those two rather than searching. The embeddings can be regenerated with
`organized/training/modal_embed_v2.py`, but that needs a GPU and Modal credit, which Person-A must approve first.

### 3. Check that your setup reproduces Person-A's numbers
```bash
.venv/bin/python organized/testing/eandd/run_replay.py --run b-check --shape competition --role test \
    --models jeffreys,rasch-old,rasch-retuned,kernel-old --no-preds
```
About 20 s. The table must read: `rasch-retuned` -0.0032, `kernel-old` -0.0024, `rasch-old` -0.0011, `jeffreys` ALC
0.2499. If it does not, stop and compare your inputs with Person-A's before doing anything else. Run names must be new
every time: `save_run` refuses to overwrite.

### 4. Where your outputs go and how they reach Person-A
Follow the schemas in `organized/testing/eandd/common.py` (`scores/`, `preds/`, `fits/`, `runs/`). Small summary CSVs
go in `paper/eandd/data/`; because `data/` is git-ignored, commit them with `git add -f`. Do not edit the draws: they
are frozen. Figures only read saved files and follow `organized/utils/plotstyle.py` (see the plan's style rules).

Handoffs: send Person-A the held-out subject list and ID panel (WP4a) first, since WP4b waits on it, then the chosen
Beta-Binomial priors (WP2). Until Person-A's real `scores/` and `preds/` arrive, build the analysis on the files from
step 3 (same columns).

### 4b. Sharing results through Modal (agreed route)
Person-B writes results to the shared **`aims-neurips`** workspace, volume `aims-neurips-data`, under
`/results/eandd-analysis/<run>/` (for example `wp7-v1/`). Person-A pulls the small files with:
```bash
MODAL_PROFILE=aims-neurips modal volume ls  aims-neurips-data /results/eandd-analysis
MODAL_PROFILE=aims-neurips modal volume get aims-neurips-data /results/eandd-analysis/wp7-v1/summary.csv received_files/
```
Listing and downloading small files costs no credit. Moving subject lists, replay panels or per-draw scores into a
collaborator-owned workspace needs explicit approval from the owner first (an automatic review blocked one such transfer).

### 5. What is not available to you
The paper source (`paper/neurips_2026/`, kept out of the repository on purpose) and the compiled drafts: ask
Person-A through a private channel. The leaderboard token and any API keys stay with their owners.

## Removing access

Remove a departing collaborator separately from GitHub, the Modal workspace, and the W&B team or restricted project. Rotate a token only if it was exposed or shared. Removing an account does not remove credentials copied from another person, which is another reason not to share tokens.
