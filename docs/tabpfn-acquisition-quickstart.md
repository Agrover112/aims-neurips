# Expected Brier acquisition with your TabPFN

Branch: `codex/tabpfn-lora-benchmark`.

1. `git fetch origin` then `git switch codex/tabpfn-lora-benchmark`.
2. Reusable model-independent code: `organized/acquisition/expected_brier.py` (NumPy only).
3. Implement `predict(x_context, y_context, x_query) -> probabilities`, returning shape `(len(x_query),)` for success class 1. Use **your frozen TabPFN weights and your own fitted feature transform**. Every call must fit/condition on exactly the supplied context, without retaining a previous hypothetical fit.
4. Choose fixed **training-only** reference/probe feature banks; start with 32 each. Use the same encoder/PCA/scaler as your TabPFN. Do not reuse our raw 1024D BGE bank if your model expects different features.
5. Compute probe scores after each acquired label, then score the current candidate and apply streaming pacing:

```python
from organized.acquisition.expected_brier import (
    expected_brier_scores, streaming_accept, stable_uniform,
)

# xc: (k,d), yc: (k,), x: (1,d); refs/probes: (32,d).
# Cache calibration using a key that includes the current pair and evidence.
calibration = expected_brier_scores(predict, xc, yc, probes, refs)
value = expected_brier_scores(predict, xc, yc, x, refs)[0]
buy = streaming_accept(value, calibration,
    context["labels_remaining"], context["items_remaining"],
    stable_uniform(input, context))
```

Recompute calibration only when the supplied acquired context changes. Return a Python `bool` from `labeling.py::acquisition_function`. Reconstruct xc/yc from the platform's `labeled`, filtered to the current subject and benchmark. Check zero budget / reserve-forced purchases before running the model. Never inspect future candidates or evaluation labels. Cache embeddings, not outcome-dependent predictions without an evidence key.

Our raw TabPFN wrapper, if useful: `organized/training/tabpfn_lora_core.py::predict`. It uses a Beta(2,2) fallback for empty/single-class contexts; use the same fallback when making hypothetical predictions. A torch bridge is:

```python
import torch
from organized.training.tabpfn_lora_core import predict as raw_predict
def predict(xc, yc, xt):
    with torch.inference_mode():
        return raw_predict(model,
            torch.as_tensor(xc, dtype=torch.float32, device=device),
            torch.as_tensor(yc, dtype=torch.float32, device=device),
            torch.as_tensor(xt, dtype=torch.float32, device=device)
        ).detach().cpu().numpy()
```

This bridge is for our raw TabPFN architecture wrapper, not automatically the sklearn `TabPFNClassifier` API. For your own wrapper, adapt the callback and explicitly map class 1 in `predict_proba`. Hypothetical labels must not modify the frozen weights or leak into the next context.

Compare against random on identical 50/50 item splits, streaming order and nested budgets 0/1/3/7/15/31. Report Brier ALC, each budget, runtime and forced purchases. The standard Beta-Binomial fallback gives all candidates equal scores until the predictor begins using item features; use randomized ties. Probe calibration costs 2 hypothetical context updates per probe, so benchmark TabPFN latency before full evaluation. A smaller bank changes the tested policy and should be reported.

ANP submission example: `organized/submission/expected_brier_labeling.py`. It is ANP-specific; do not copy it unchanged for TabPFN. Evaluator: `organized/training/modal_anp_acquisition.py`; original score math: `organized/training/anp_acquisition.py`. Canonical split/filter: `organized/training/benchmark_holdout_filters.py`.

Completed ANP comparison: `docs/completed-experiment-results.md`. Expected Brier ALC 0.19055 vs random 0.19374 on 390 validation tasks x 3 orders. This is an ANP result, not evidence it improves TabPFN. Equal benchmark weighting slightly favors random; test transfer on your model.

Use your own training-only reference/probe vectors. Our ANP bank is not committed; authorized users can regenerate it using `organized/training/modal_export_acquisition_bank.py`. Checkpoints, detailed reports and full embedding stores remain local/in Modal; the generic helper needs neither our workspace nor W&B credentials. Submission builders are in `scripts/build_*submission.py`; generated ZIPs are not committed.
