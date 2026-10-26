#!/usr/bin/env python3
"""Orchestrate full offline evaluation pipeline with flags.

Launches Modal jobs for IRT/MIRT/Bayesian MIRT evaluation across multiple:
  - Seeds (20261018, 20261019, 20261020)
  - Filtering thresholds (n≥15, n≥30)
  - Model types and variants
  - Evaluation protocols (filter_test, retention, generalization, etc.)

Usage:
  python run_evaluation_pipeline.py --seeds 20261018 20261019 --models classic_irt --dry-run
  python run_evaluation_pipeline.py --all                      # everything
  python run_evaluation_pipeline.py --models mirt --filters 30  # MIRT n≥30 only
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
TESTING_DIR = REPO_ROOT / "organized" / "testing" / "offline"

SEEDS = [20261018, 20261019, 20261020]
FILTERS = [15, 30]  # minimum subjects per item

MODELS = {
    "classic_irt": [
        ("modal_classic_irt_filter_test.py", {"seeds": SEEDS, "min_subjects": FILTERS}),
        ("modal_filter_retention.py", {"seeds": SEEDS}),
        ("modal_classic_irt_generalization.py", {"seeds": SEEDS}),
        ("modal_classic_irt_generalization_lbfgs.py", {"seeds": SEEDS}),
    ],
    "mirt": [
        ("modal_mirt_d2_d4_filtered.py", {"seeds": SEEDS}),
        ("modal_mirt_d2_d4_generalization.py", {"seeds": SEEDS}),
    ],
    "bayesian_mirt": [
        ("modal_bayesian_mirt_d1_d2_d4_filtered.py", {"seeds": SEEDS}),
        ("modal_bayesian_mirt_d1_d2_d4_generalization.py", {"seeds": SEEDS}),
    ],
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrate offline evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run: show what would be launched
  python run_evaluation_pipeline.py --dry-run

  # Everything (warning: ~600+ jobs)
  python run_evaluation_pipeline.py --all

  # Just classic IRT with n≥30
  python run_evaluation_pipeline.py --models classic_irt --filters 30

  # MIRT/Bayesian MIRT only, seeds 20261018-20261019
  python run_evaluation_pipeline.py --models mirt bayesian_mirt --seeds 20261018 20261019

  # Just generalization tests (no filter/retention)
  python run_evaluation_pipeline.py --protocols generalization

  # Run and follow logs
  python run_evaluation_pipeline.py --all --follow-logs
        """
    )

    parser.add_argument(
        "--all", action="store_true",
        help="Run all models, seeds, filters, protocols (default: nothing)"
    )

    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS.keys()),
        help="Model types to run (classic_irt, mirt, bayesian_mirt)"
    )

    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=SEEDS,
        help="Seeds to use (default: all three)"
    )

    parser.add_argument(
        "--filters", nargs="+", type=int, choices=FILTERS,
        help="Filtering thresholds: n≥{15,30} (default: all)"
    )

    parser.add_argument(
        "--protocols", nargs="+",
        choices=["filter_test", "retention", "generalization", "generalization_lbfgs"],
        help="Evaluation protocols (default: all available)"
    )

    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be launched without actually running"
    )

    parser.add_argument(
        "--detach", action="store_true", default=True,
        help="Run jobs detached (default: True)"
    )

    parser.add_argument(
        "--follow-logs", action="store_true",
        help="Follow Modal logs after launching (requires manual Ctrl+C)"
    )

    parser.add_argument(
        "--sequential", action="store_true",
        help="Launch jobs one at a time instead of in parallel"
    )

    return parser.parse_args()


def get_jobs(args):
    """Build list of (script, launch_args) tuples to run."""

    # Determine which models to run
    if args.all:
        model_list = list(MODELS.keys())
    elif args.models:
        model_list = args.models
    else:
        print("No models selected. Use --all or --models {classic_irt,mirt,bayesian_mirt}")
        return []

    # Determine which seeds
    seeds = args.seeds if args.seeds else SEEDS

    # Determine which filters
    filters = args.filters if args.filters else FILTERS

    jobs = []

    for model_type in model_list:
        for script_path, config in MODELS[model_type]:
            # Check if protocols filter applies
            script_name = Path(script_path).stem
            if args.protocols:
                # Only include if script matches one of requested protocols
                matching = False
                for protocol in args.protocols:
                    if protocol in script_name:
                        matching = True
                        break
                if not matching:
                    continue

            script_full = TESTING_DIR / script_path

            if not script_full.exists():
                print(f"Warning: {script_full} not found, skipping")
                continue

            # For scripts that take seed as argument (spawned with .spawn(seed))
            if "filter_test" in script_name:
                # Special case: filter_test takes min_subjects as well
                for filter_thresh in filters:
                    for seed in seeds:
                        jobs.append({
                            "script": script_full,
                            "model_type": model_type,
                            "seed": seed,
                            "filter": filter_thresh,
                            "cmd": f"modal run --detach {script_full}::main {filter_thresh}"
                        })
            else:
                # Other scripts spawn with seeds internally, just run once
                jobs.append({
                    "script": script_full,
                    "model_type": model_type,
                    "cmd": f"modal run --detach {script_full}"
                })

    return jobs


def launch_jobs(jobs, args):
    """Launch Modal jobs."""

    if not jobs:
        print("No jobs to launch.")
        return 0

    print(f"\n{'='*70}")
    print(f"Launching {len(jobs)} Modal jobs")
    print(f"{'='*70}\n")

    for idx, job in enumerate(jobs, 1):
        print(f"[{idx}/{len(jobs)}] {job['script'].name}", end="")
        if "seed" in job:
            print(f" (seed={job['seed']}, filter={job['filter']})", end="")
        print()

        if args.dry_run:
            print(f"  Would run: {job['cmd']}")
            continue

        # Launch the job
        try:
            result = subprocess.run(
                job['cmd'], shell=True, cwd=REPO_ROOT,
                capture_output=False, timeout=30
            )
            if result.returncode == 0:
                print(f"  ✓ Launched")
            else:
                print(f"  ✗ Failed with code {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout launching")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        if args.sequential and idx < len(jobs):
            print()

    print(f"\n{'='*70}")
    print(f"Launched {len([j for j in jobs if not args.dry_run])} jobs")
    print(f"{'='*70}\n")

    if not args.dry_run and args.follow_logs:
        print("Tailing Modal logs (Ctrl+C to exit)...")
        try:
            subprocess.run("modal logs", shell=True)
        except KeyboardInterrupt:
            print("\nLog following stopped.")

    return 0


def main():
    args = parse_args()

    # Validate
    if not args.all and not args.models:
        print("Error: Specify --all or --models {classic_irt,mirt,bayesian_mirt}")
        sys.exit(1)

    jobs = get_jobs(args)

    if args.dry_run:
        print(f"\nDRY RUN: Would launch {len(jobs)} jobs\n")
        if len(jobs) <= 10:
            for job in jobs:
                print(f"  {job['cmd']}")

    return launch_jobs(jobs, args)


if __name__ == "__main__":
    sys.exit(main())
