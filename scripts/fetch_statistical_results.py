"""Fetch lightweight result JSONs; no model weights or credentials are printed."""
import json
from pathlib import Path
import modal
from modal.config import config

MODELS = ("beta_binomial", "rasch", "feature_rasch", "linear_regression", "bayesian_rbf")
GROUP = "statistical-benchmark-20261029-v1"
OUTPUT = Path(__file__).resolve().parents[1] / "reports" / GROUP
REQUIRED = (
    "diagnostic/heldout_subjects/same_subject_split_half/brier_score",
    "diagnostic/training_subjects/same_subject_split_half/brier_score",
    "diagnostic/training_subjects/same_subject_response_holdout/roc_auc",
    "diagnostic/training_subjects/same_subject_split_half/roc_auc",
)
client = modal.Client.from_credentials(config.get("token_id", profile="arshiaswe", use_env=False),
                                       config.get("token_secret", profile="arshiaswe", use_env=False))
volume = modal.Volume.from_name("aims-neurips-data").hydrate(client)

def fetch(remote, local):
    data = b"".join(volume.read_file(remote))
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    return json.loads(data)

summary = []
cohort = None
reference_alc = None
for kind in MODELS:
    remote = f"/results/statistical-baselines/{GROUP}/{kind}"
    result = {name: fetch(remote + "/" + name + ".json", OUTPUT / kind / (name + ".json"))
              for name in ("completed", "curves", "final-validation", "final-test")}
    curves, test = result["curves"], result["final-test"]
    reference_alc = test["fixed_beta_2_2_brier_alc"]
    assert len(curves) == 5
    assert all(all(key in point for key in REQUIRED) for point in curves)
    actual = [(row["subject"], row["benchmark"]) for row in test["tasks"]]
    if cohort is None:
        cohort = actual
    assert cohort == actual, "Models were not scored on identical tasks"
    best = min(curves, key=lambda r: r["validation/brier_alc"])
    summary.append({"model": kind, "curve_points": len(curves), "all_requested_keys": True,
                    "selected_training_fraction": best["training_fraction"],
                    "selection_validation_alc": result["completed"]["best_validation_alc"],
                    "full_validation_alc": result["final-validation"]["brier_alc"],
                    "test_alc": test["brier_alc"], "test_benchmark_macro_alc": test["benchmark_macro_alc"],
                    "test_tasks": len(test["tasks"]), "test_benchmarks": sorted(test["by_benchmark"]),
                    "wandb_url": result["completed"]["wandb_url"]})
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
neural = fetch("/results/tabpfn-lora/tabpfn-v2-lora-bge64-benchmark-20261029-v1/final-test.json",
               OUTPUT / "tabpfn-lora-final-test.json")
assert cohort == [(r["subject"], r["benchmark"]) for r in neural["tasks"]]
assert abs(neural["beta_brier_alc"] - reference_alc) < 1e-6, "Shared reference differs from TabPFN episodes"
comparison = {"statistical_models": summary, "tabpfn_lora_test_alc": neural["brier_alc"],
              "fixed_beta_2_2_test_alc": reference_alc, "tabpfn_cohort_and_reference_verified": True}
(OUTPUT / "comparison-with-tabpfn.json").write_text(json.dumps(comparison, indent=2))
print(json.dumps(summary, indent=2))
print(json.dumps({k:v for k,v in comparison.items() if k != "statistical_models"}))
