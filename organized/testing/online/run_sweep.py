#!/usr/bin/env python3
"""Drive the models x acquisition-policy sweep over one frozen replay panel.

Three things make this a script rather than a shell loop.

**Not every cell exists.** ``bernoulli_variance`` and
``expected_brier_reduction`` are implemented only in the ANP/beta runner;
``modal_tabpfn_v2_benchmark_holdout_replay.py`` and its TabFM twin validate
``policy`` against the three original rules. A driver that emitted all 30
commands would fail four of them with a ``ValueError`` after paying for the
container. Unsupported cells are reported as skipped, with the reason.

**Order matters for cost.** ``expected_brier_reduction`` evaluates both
outcomes over 32 reference items and recomputes probe calibration after every
purchase, so it costs roughly 33x a threshold policy. Arms run cheapest-first,
so a budget that runs out leaves the informative cheap cells already done
rather than half of one expensive cell.

**Reruns must not duplicate work.** Each arm writes to a run directory named
from its own parameters; an arm whose ``summary.json`` already exists is
skipped unless ``--force`` is passed.

Usage::

    # see the plan without spending anything
    python organized/testing/online/run_sweep.py --dry-run

    # run it
    python organized/testing/online/run_sweep.py --profile iclrgram

    # collect finished arms into a markdown table
    python organized/testing/online/run_sweep.py --collect --profile iclrgram
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODAL = str(ROOT / ".venv" / "bin" / "modal")

ANP_RUNNER = "organized/testing/online/modal_benchmark_holdout_replay.py"
TABPFN_RUNNER = "organized/testing/online/modal_tabpfn_v2_benchmark_holdout_replay.py"
TABFM_RUNNER = "organized/testing/online/modal_tabfm_benchmark_holdout_replay.py"

# cap100000 selects 82 validation / 45 test tasks. The earlier cap19000
# panel selects only 24 validation tasks, which resolves about 0.015 and
# leaves every arm "tied"; the cap number is a summed item budget, not a
# task count, so it is easy to mistake one panel for the other.
README = HERE / "README.md"
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

DEFAULT_PANEL = "benchmark-holdout-online-v1-s20261018-n15-m80-cap100000"
VOLUME = "aims-neurips-data"
RUNS_ROOT = "results/online-replay/runs"

CKPT_W64_FILTERED = (
    "anp-crossattn-latent-w64-sa1-h4-z32-benchmark-heldout-es-fixed-"
    "anp-sa1-w64-bgem3-filtered-benchmarkholdout-n15-20261029-bge_m3_long-seed20261018.pt"
)
CKPT_W64_BUDGETS = (
    "anp-crossattn-latent-w64-sa1-h4-z32-benchmark-heldout-es-fixed-"
    "anp-sa1-w64-bgem3-bhold-budgetladder-n15-20261029-bge_m3_long-seed20261018.pt"
)
CKPT_W64_CONTAM = (
    "anp-crossattn-latent-w64-sa1-h4-z32-es-fixed-"
    "anp-sa1-w64-bgem3-20261028-bge_m3_long-seed20261018.pt"
)
CKPT_W64_DENSE = (
    # Frozen copy of the best checkpoint (step 4550, fixed-bank Brier 0.1700)
    # from the still-running dense-prefix job; the live file is overwritten
    # whenever that job improves, which would make this row irreproducible.
    "anp-crossattn-latent-w64-sa1-h4-z32-denseprefix-unweighted-best-step4550-"
    "bge_m3_long-seed20261018.pt"
)
CKPT_W64_SAM = (
    "anp-crossattn-latent-w64-sa1-h4-z32-benchmark-heldout-es-fixed-"
    "anp-sa1-w64-bgem3-bhold-sam005-unweighted-n15-cpu-20261029-bge_m3_long-seed20261018.pt"
)
CKPT_W64_ANCHOR = (
    "anp-crossattn-latent-w64-sa1-h4-z32-benchmark-heldout-es-fixed-"
    "anp-sa1-w64-bgem3-bhold-anchor-unweighted-n15-cpu-20261029-bge_m3_long-seed20261018.pt"
)
CKPT_W128 = (
    "anp-crossattn-latent-w128-sa1-h4-z128-es-fixed-"
    "anp128-bgem3-l40s-100k-s20261018-bge_m3_long-seed20261018.pt"
)

# ``runner`` None means the model is served by the ANP/beta runner.
MODELS = [
    # The gated rows share one beta-binomial prediction at budgets 0 and 1, so
    # they differ from `beta` only from budget 3 up. TabFM's regressor cannot
    # run on fewer than two labels, so it has no ungated row.
    {"key": "beta", "label": "beta-binomial",
     "runner": ANP_RUNNER, "mode": "beta_binomial", "checkpoint": ""},
    {"key": "const-mean", "label": "Constant (training mean)",
     "runner": ANP_RUNNER, "mode": "constant_prior", "checkpoint": ""},
    {"key": "emp-mean", "label": "Empirical mean (no shrinkage)",
     "runner": ANP_RUNNER, "mode": "empirical_mean", "checkpoint": ""},
    {"key": "w64-filtered", "label": "ANP w64 filtered (alone)",
     "runner": ANP_RUNNER, "mode": "anp", "checkpoint": CKPT_W64_FILTERED},
    {"key": "w64-filtered-gate", "label": "ANP w64 filtered + beta at B0/B1",
     "runner": ANP_RUNNER, "mode": "anp_beta_gate", "checkpoint": CKPT_W64_FILTERED},
    {"key": "w64-dense", "label": "ANP w64 prefix training (alone)",
     "runner": ANP_RUNNER, "mode": "anp", "checkpoint": CKPT_W64_DENSE},
    {"key": "w128", "label": "ANP w128 (contaminated)",
     "runner": ANP_RUNNER, "mode": "anp", "checkpoint": CKPT_W128,
     "note": "trained on the random-subject split; it saw these benchmarks"},
    {"key": "w64-contam", "label": "ANP w64 (contaminated)",
     "runner": ANP_RUNNER, "mode": "anp", "checkpoint": CKPT_W64_CONTAM,
     "note": "random-subject split: it trained on this panel's benchmarks. "
             "Paired with w64-filtered this measures the leakage, it is not a result"},
    {"key": "w64-sam", "label": "ANP w64 SAM rho=0.05",
     "runner": ANP_RUNNER, "mode": "anp", "checkpoint": CKPT_W64_SAM,
     "note": "validation bank differs from w64-filtered; compare only on this panel"},
    {"key": "w64-anchor", "label": "ANP + prior anchor",
     "runner": ANP_RUNNER, "mode": "anp", "checkpoint": CKPT_W64_ANCHOR},
]

# Relative cost, used only to order the queue.
POLICIES = [
    {"key": "random_prefix", "cost": 1},
    {"key": "uncertainty", "cost": 1},
    {"key": "confidence_extremes", "cost": 1},
    {"key": "bernoulli_variance", "cost": 17},
    {"key": "expected_brier_reduction", "cost": 33},
]

# All three runners now accept all five policies; the two scored ones were
# ported into the tabular runners alongside the ANP/beta one.
TABULAR_POLICIES = {"random_prefix", "uncertainty", "confidence_extremes",
                    "bernoulli_variance", "expected_brier_reduction"}


def run_name(model_key: str, policy_key: str, role: str, seeds: int) -> str:
    return f"{role}-{model_key}-{policy_key}-s{seeds}-v2"


def arm_is_supported(model: dict, policy: dict) -> str | None:
    """Return a skip reason, or None when the cell can be run."""
    if model["runner"] in (TABPFN_RUNNER, TABFM_RUNNER):
        if policy["key"] not in TABULAR_POLICIES:
            return (f"{model['label']} runner validates policy against "
                    f"{sorted(TABULAR_POLICIES)}; {policy['key']} is only in the "
                    "ANP/beta runner")
    return None


def build_command(model: dict, policy: dict, *, panel: str, role: str,
                  seeds: int, profile: str, threshold: float,
                  probe_count: int, prefiltered_responses: str,
                  tabfm_gpu: str = "") -> list[str]:
    name = run_name(model["key"], policy["key"], role, seeds)
    command = [MODAL, "run", "--detach", "--profile", profile,
               f"{model['runner']}::main",
               "--action", "replay",
               "--panel-name", panel,
               "--role", role,
               "--policy", policy["key"],
               "--max-order-seeds", str(seeds),
               "--uncertainty-threshold", str(threshold),
               "--run-name", name]
    if model["runner"] == ANP_RUNNER:
        command += ["--prediction-mode", model["mode"]]
        if model["checkpoint"]:
            command += ["--checkpoint", model["checkpoint"]]
    command += list(model.get("extra", []))
    if model["runner"] == TABFM_RUNNER and tabfm_gpu:
        command += ["--gpu", tabfm_gpu]
    if prefiltered_responses and model.get("mode") in {"constant_prior", "empirical_mean"} \
            and policy["key"] not in {"bernoulli_variance", "expected_brier_reduction"}:
        command += ["--prefiltered-responses", prefiltered_responses]
    if policy["key"] in {"bernoulli_variance", "expected_brier_reduction"}:
        command += ["--probe-count", str(probe_count)]
        if prefiltered_responses:
            # A workspace holding only the exported subset has no
            # datasets/full/processed, so the train probe source must be the
            # filter's own output instead.
            command += ["--prefiltered-responses", prefiltered_responses]
    return command


def existing_runs(panel: str, profile: str) -> set[str]:
    """Run directories already on the volume, so a rerun can resume."""
    result = subprocess.run(
        [MODAL, "volume", "ls", VOLUME, f"{RUNS_ROOT}/{panel}", "--profile", profile],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return set()
    names = set()
    for line in result.stdout.splitlines():
        line = line.strip().strip("│ ").strip()
        if "/" in line:
            names.add(line.rstrip("/").split("/")[-1])
    return names


def prepare_transforms(models, *, panel, profile, dry_run,
                       prefiltered_responses="") -> None:
    """The tabular runners need their train-only PCA fitted once per panel."""
    for model in models:
        if model["runner"] not in (TABPFN_RUNNER, TABFM_RUNNER):
            continue
        command = [MODAL, "run", "--profile", profile,
                   f"{model['runner']}::main",
                   "--action", "prepare-transform", "--panel-name", panel]
        if prefiltered_responses:
            command += ["--prefiltered-responses", prefiltered_responses]
        print(f"\n# PCA transform for {model['label']}")
        print("  " + " ".join(shlex.quote(part) for part in command))
        if not dry_run:
            subprocess.run(command, check=False)


def _paired_table(per_task: dict, baseline_cell: tuple[str, str], out: list) -> None:
    """Paired difference against one reference arm, with a CI.

    Each arm's own standard deviation is dominated by *which subjects were
    drawn* -- about 0.069 here -- so reporting mean +- sd per arm would show
    every arm as indistinguishable and hide a real 0.005 gap. The arms share
    tasks, so the informative quantity is the per-task difference, whose
    variance has the subject effect cancelled out.
    """
    import statistics as stats

    reference = per_task.get(baseline_cell)
    if reference is None:
        out.append("\n(no per-task data for the reference arm; paired table skipped)")
        return
    label = dict((m["key"], m["label"]) for m in MODELS)[baseline_cell[0]]
    out.append(f"\n### Paired difference vs {label} x `{baseline_cell[1]}`\n")
    out.append("Positive means worse than the reference. Each arm's own standard "
               "deviation is dominated by which subjects were drawn, so the paired "
               "per-task difference is the informative quantity.\n")
    out.append("| arm | mean diff | sd of diff | 95% CI | n | verdict |")
    out.append("|---|---|---|---|---|---|")
    for cell, values in sorted(per_task.items()):
        if cell == baseline_cell:
            continue
        shared = sorted(set(values) & set(reference))
        if len(shared) < 3:
            continue
        diffs = [values[task] - reference[task] for task in shared]
        mean = stats.fmean(diffs)
        sd = stats.stdev(diffs)
        half = 1.96 * sd / len(diffs) ** 0.5
        if mean - half > 0:
            verdict = "worse"
        elif mean + half < 0:
            verdict = "**better**"
        else:
            verdict = "tied"
        out.append(f"| {cell[0]} x `{cell[1]}` | {mean:+.5f} | {sd:.5f} | "
                   f"[{mean - half:+.5f}, {mean + half:+.5f}] | {len(diffs)} | {verdict} |")


def collect(panel: str, profile: str, readme: Path | None = None,
            role: str = "validation") -> None:
    """Download every finished arm's summary and render the markdown tables.

    With ``readme`` the tables replace the block between the RESULTS markers,
    so the committed table is regenerated from the volume rather than
    hand-edited. Hand-editing is how a results table drifts from the runs it
    claims to describe.
    """
    done = existing_runs(panel, profile)
    if not done:
        print(f"no run directories under {RUNS_ROOT}/{panel}", file=sys.stderr)
        return
    # Build the expected name for every cell and look it up, rather than
    # parsing names back into keys: model and policy keys both contain
    # hyphens and underscores, so any split-based recovery is ambiguous.
    results: dict[tuple[str, str], dict] = {}
    per_task: dict[tuple[str, str], dict[str, float]] = {}
    budget_frames: list = []
    wanted = {}
    for model in MODELS:
        for policy in POLICIES:
            # One role per table. Validation and test cells share model and
            # policy keys, so collecting both at once would let one silently
            # overwrite the other in the same cell.
            for seeds in range(1, 11):
                wanted[run_name(model["key"], policy["key"], role, seeds)] = (
                    model["key"], policy["key"]
                )
    with tempfile.TemporaryDirectory() as workspace:
        for name in sorted(done):
            cell = wanted.get(name)
            if cell is None:
                print(f"  (ignoring unrecognised run directory {name})", file=sys.stderr)
                continue
            remote = f"{RUNS_ROOT}/{panel}/{name}/summary.json"
            local = Path(workspace) / f"{name}.json"
            fetched = subprocess.run(
                [MODAL, "volume", "get", VOLUME, remote, str(local),
                 "--force", "--profile", profile],
                capture_output=True, text=True,
            )
            if fetched.returncode != 0 or not local.exists():
                print(f"  (no summary.json yet for {name})", file=sys.stderr)
                continue
            results[cell] = json.loads(local.read_text())
            # Per-task values, for the paired comparison. Absent on an arm
            # that was interrupted after summary.json but before this file.
            task_remote = f"{RUNS_ROOT}/{panel}/{name}/task_alc.parquet"
            task_local = Path(workspace) / f"{name}-task_alc.parquet"
            got = subprocess.run(
                [MODAL, "volume", "get", VOLUME, task_remote, str(task_local),
                 "--force", "--profile", profile],
                capture_output=True, text=True,
            )
            if got.returncode == 0 and task_local.exists():
                import pandas as pd

                frame = pd.read_parquet(task_local)
                per_task[cell] = dict(
                    zip(frame["task_id"].astype(str), frame["brier_alc"].astype(float))
                )
            # The per-budget curve data: one row per task, seed and budget.
            # ALC collapses these six numbers into one, which hides whether a
            # model is better early, late, or only at one budget.
            budget_remote = f"{RUNS_ROOT}/{panel}/{name}/task_budget_scores.parquet"
            budget_local = Path(workspace) / f"{name}-task_budget_scores.parquet"
            got_budget = subprocess.run(
                [MODAL, "volume", "get", VOLUME, budget_remote, str(budget_local),
                 "--force", "--profile", profile],
                capture_output=True, text=True,
            )
            if got_budget.returncode == 0 and budget_local.exists():
                import pandas as pd

                frame = pd.read_parquet(budget_local)
                frame.insert(0, "policy", cell[1])
                frame.insert(0, "model", cell[0])
                budget_frames.append(frame)
    budget_table = None
    if budget_frames:
        import pandas as pd

        budget_table = pd.concat(budget_frames, ignore_index=True)
        scores_dir = HERE / "results"
        scores_dir.mkdir(exist_ok=True)
        budget_table.to_parquet(
            scores_dir / f"budget-scores-{panel}-{role}.parquet", index=False
        )
    out: list[str] = []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out.append(f"<!-- generated by run_sweep.py --write-readme on {stamp}; "
               "do not hand-edit -->")
    out.append(f"\nPanel `{panel}`, role **{role}**. Lower is better. An empty cell "
               "has not been run; `n/a` cannot be run.")
    for metric, title in (("macro_task_brier_alc", "Macro-task Brier ALC"),
                          ("micro_response_brier_alc", "Micro-response Brier ALC")):
        out.append(f"\n### {title}\n")
        out.append("| model | " + " | ".join(f"`{p['key']}`" for p in POLICIES) + " |")
        out.append("|---" * (len(POLICIES) + 1) + "|")
        # Mark the best cell per metric, so the table is readable at a glance
        # without implying the lead is significant; the paired table below is
        # what says whether it is.
        values = [(summary[metric], cell) for cell, summary in results.items()]
        best = min(values)[1] if values else None
        for model in MODELS:
            cells = []
            for policy in POLICIES:
                cell = (model["key"], policy["key"])
                summary = results.get(cell)
                if summary is None:
                    cells.append("n/a" if arm_is_supported(model, policy) else "")
                elif cell == best:
                    cells.append(f"**{summary[metric]:.5f}**")
                else:
                    cells.append(f"{summary[metric]:.5f}")
            out.append(f"| {model['label']} | " + " | ".join(cells) + " |")
    if budget_table is not None:
        # Average over order seeds within a task first, so each task counts
        # once, then over tasks: the same macro-task aggregation as the ALC.
        seed_mean = (budget_table.groupby(["model", "policy", "task_id", "requested_budget"],
                                          as_index=False)["brier"].mean())
        by_budget = seed_mean.groupby(["model", "policy", "requested_budget"])["brier"].mean()
        budgets = sorted(int(b) for b in seed_mean["requested_budget"].unique())
        weights = dict(zip((0, 1, 3, 7, 15, 31), (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)))
        out.append("\n## Brier by number of labels acquired\n")
        out.append("Macro-task Brier at each budget, lower is better. The ALC above is "
                   "the weighted sum of these columns "
                   "(weights " + ", ".join(f"B{b}={weights.get(b, 0)}" for b in budgets) + ").")
        for policy in POLICIES:
            rows = []
            for model in MODELS:
                if (model["key"], policy["key"]) not in results:
                    continue
                values = [by_budget.get((model["key"], policy["key"], b)) for b in budgets]
                rows.append((model["label"], values))
            if not rows:
                continue
            out.append(f"\n### `{policy['key']}`\n")
            out.append("| model | " + " | ".join(f"B{b}" for b in budgets) + " |")
            out.append("|---" * (len(budgets) + 1) + "|")
            for label, values in rows:
                out.append(f"| {label} | " + " | ".join(
                    "" if v is None else f"{v:.4f}" for v in values) + " |")
    if per_task:
        reference = ("beta", "random_prefix")
        if reference not in per_task:
            reference = sorted(per_task)[0]
        _paired_table(per_task, reference, out)
    out.append(f"\n{len(results)} of {len(MODELS) * len(POLICIES)} cells collected; "
               f"{len(per_task)} with per-task data.")
    rendered = "\n".join(out)
    print(rendered)
    if readme is not None:
        text = readme.read_text()
        start, end = ((START, END) if role == "validation"
                      else ("<!-- RESULTS-TEST:START -->", "<!-- RESULTS-TEST:END -->"))
        if start not in text or end not in text:
            raise RuntimeError(
                f"{readme} is missing the {start} / {end} markers; add them "
                "around the results section first"
            )
        head, _, rest = text.partition(start)
        _, _, tail = rest.partition(end)
        readme.write_text(f"{head}{start}\n{rendered}\n{end}{tail}")
        print(f"\nwrote {len(out)} lines into {readme}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument("--profile", default="iclrgram")
    parser.add_argument("--role", default="validation", choices=["validation", "test"])
    parser.add_argument("--seeds", type=int, default=3,
                        help="order seeds per arm (default 3)")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--probe-count", type=int, default=32)
    parser.add_argument("--tabfm-gpu", default="",
                        help="GPU type for the TabFM cells, e.g. T4. Needs a "
                             "workspace that can allocate GPUs")
    parser.add_argument("--prefiltered-responses", default="",
                        help="filtered-responses.parquet on the volume; required "
                             "for the scored policies on a workspace without "
                             "datasets/full/processed")
    parser.add_argument("--models", default="",
                        help="comma-separated model keys; default all")
    parser.add_argument("--policies", default="",
                        help="comma-separated policy keys; default all")
    parser.add_argument("--max-cost", type=int, default=0,
                        help="skip policies above this relative cost (0 = no limit)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="rerun arms whose run directory already exists")
    parser.add_argument("--skip-transforms", action="store_true")
    parser.add_argument("--collect", action="store_true",
                        help="only gather finished arms into a table")
    parser.add_argument("--write-readme", action="store_true",
                        help="with --collect, also rewrite the results block in "
                             "README.md between the RESULTS markers")
    args = parser.parse_args()

    if args.role == "test":
        print("WARNING: the test role is for one frozen choice, never for "
              "selection. Validation cells must already be decided.\n",
              file=sys.stderr)

    models = [m for m in MODELS
              if not args.models or m["key"] in args.models.split(",")]
    policies = [p for p in POLICIES
                if not args.policies or p["key"] in args.policies.split(",")]
    if args.max_cost:
        policies = [p for p in policies if p["cost"] <= args.max_cost]

    if args.collect:
        collect(args.panel, args.profile,
                readme=README if args.write_readme else None, role=args.role)
        return 0

    done = set() if args.force else existing_runs(args.panel, args.profile)
    if done:
        print(f"# {len(done)} run directories already on the volume\n")

    queue, skipped = [], []
    for policy in sorted(policies, key=lambda p: p["cost"]):
        for model in models:
            reason = arm_is_supported(model, policy)
            if reason:
                skipped.append((model, policy, reason))
                continue
            name = run_name(model["key"], policy["key"], args.role, args.seeds)
            if name in done:
                skipped.append((model, policy, "already on the volume"))
                continue
            queue.append((model, policy, name))

    print(f"# {len(queue)} arms to run, {len(skipped)} skipped")
    print(f"# panel {args.panel}  role {args.role}  seeds {args.seeds}  "
          f"profile {args.profile}")
    for model, policy, reason in skipped:
        print(f"  SKIP  {model['key']:<14} {policy['key']:<26} {reason}")
    for model in models:
        if model.get("note"):
            print(f"  NOTE  {model['key']:<14} {model['note']}")

    if not args.skip_transforms:
        prepare_transforms(models, panel=args.panel, profile=args.profile,
                           dry_run=args.dry_run,
                           prefiltered_responses=args.prefiltered_responses)

    failures = []
    for index, (model, policy, name) in enumerate(queue, start=1):
        command = build_command(model, policy, panel=args.panel, role=args.role,
                                seeds=args.seeds, profile=args.profile,
                                threshold=args.threshold,
                                probe_count=args.probe_count,
                                prefiltered_responses=args.prefiltered_responses,
                                tabfm_gpu=args.tabfm_gpu)
        print(f"\n# [{index}/{len(queue)}] {model['label']} x {policy['key']}"
              f"  (cost ~{policy['cost']}x)")
        print("  " + " ".join(shlex.quote(part) for part in command))
        if args.dry_run:
            continue
        completed = subprocess.run(command, capture_output=True, text=True)
        tail = (completed.stdout + completed.stderr).strip().splitlines()[-3:]
        for line in tail:
            print(f"    {line}")
        if completed.returncode != 0:
            failures.append(name)

    if args.dry_run:
        print(f"\n# dry run: nothing launched")
    elif failures:
        print(f"\n# {len(failures)} arms failed to launch: {failures}")
        return 1
    else:
        print(f"\n# launched {len(queue)} arms; "
              f"collect with --collect once they finish")
    return 0


if __name__ == "__main__":
    sys.exit(main())
