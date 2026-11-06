# Plan: cognitive-scale item features for IRT (adapted from Scales++)

Status: **plan only, nothing run.** Source: Bean et al., *Scales++* (arXiv:2510.26384, `literature/Scales++.pdf`), which builds on Zhou et al. (2025), *General Scales Unlock AI Evaluation with Explanatory and Predictive Power*.

## Question
Do LLM-annotated cognitive scales of an item (what the item demands) predict item difficulty on **unseen benchmarks** better than BGE-M3 item embeddings, and can they serve as item features in pooled/kernel Rasch? Same protocol and same pre-set criterion as the entropy test, so the result is comparable.

## What the paper does (facts, from the PDF)
- Each item gets **16 cognitive scales**, each rated **0-5** by GPT-4o using expert-validated prompts from Zhou et al. (one call per dimension).
- *Scales++ Lite* distils this: 8,000 queries from Tulu3-SFT-mixture labelled by GPT-4o; features = **Qwen2.5-7B-Instruct, layer-14 token embeddings, mean-pooled**; kNN graph (top-10 cosine neighbours); 3 graph-convolution layers, cross-entropy over the 0-5 classes per dimension; annotates 28,659 items in under 20 minutes.
- **In the Scales++ appendix (B):** the names of all 16 dimensions (Attention and scan; Calibrating knowns and unknowns; Conceptualisation learning abstraction; Critical thinking processes; Identifying relevant information; Knowledge applied science / customary / formal science / natural science / social science; Logical reasoning; Mind modelling and social cognition; Quantitative reasoning; Spatial reasoning and navigation; Verbal comprehension; Verbal expression) and **one** full rubric, Attention and Scan (definition, levels 0 to 5+ with examples, `{{instance}}` slot) as the prompt template.
- **Not in the paper** (needs Zhou et al.): the other **15 rubrics**. Zhou et al. have 18 rubrics; Scales++ uses 16.

## Steps
| # | Step | Where / cost | Notes |
|---|---|---|---|
| 0 | Get the 15 missing rubrics from Zhou et al. (2025); the 16 names and the Attention and Scan template are already in Scales++ appendix B | reading | Blocker for step 2. Use their wording unchanged; cite and disclose. |
| 1 | Pick the labelling set: **about 1,500-2,000 items** (not 8,000: free tiers cap requests per day), stratified over the 17 benchmarks, train benchmarks plus a held-out slice of unseen ones | local | Keep val/test benchmarks out of fitting. |
| 2 | Label with a **free-tier** judge (see "Judge model" below): one call per item that returns all 16 ratings as JSON, temperature 0, no truncation of input or output | free API, days of wall-clock because of daily request caps | **You create the key** (env file); I never read or print it. First check on about 30 items that the one-call ratings agree with one-call-per-dimension ratings; if they disagree, the sample must shrink further. |
| 3 | Distil: predict the 16 ratings from the **BGE-M3 embeddings we already have** (ridge first, then a small network or the paper's GNN) | local CPU | Departure from the paper (Qwen2.5-7B-Instruct is too heavy for local CPU). Report it as a departure. |
| 4 | Validate the distilled scales: fit on training benchmarks, correlate with the API's ratings on **unseen** benchmarks, per dimension and per benchmark | local | If this fails OOD, stop: the scales do not transfer. |
| 5 | Probe: ridge from the 16 scales (centred within benchmark) to observed pass rate, ID and OOD, with `entropy_probe.py`'s protocol | local | Compare on the **same items** as the embedding probe (OOD rank correlation 0.073 on the 3,282-item sample). |
| 6 | Criterion fixed in advance: OOD rank correlation at least 0.05 above the embedding probe and positive on at least 4 of 6 unseen benchmarks | - | Otherwise report a negative result; it still extends the item-difficulty table. |
| 7 | Only if step 6 passes: explanatory-IRT prior in pooled/kernel Rasch, $b_i = w^\top c_i + u_i$ with $w$ fitted on training benchmarks only; same 150 test draws, paired with kernel Rasch, 95% CIs | local CPU | Needs scales for the ~13,000 draw items: predicted by the distilled model, no new API calls. |
| 8 | Paper: one row and a paragraph in the item-difficulty analysis, protocol in the appendix, limitations updated | edit | No claim beyond what the probe and draws show. |

## Judge model (no paid API) -- all limits below are unverified aggregator figures; check the provider's official page before relying on them
Measured on our data (405,790 items of the 17 benchmarks): median item about 1,000 characters, 99th percentile about 5,800, **longest 34,375 characters (roughly 10,000 tokens, my estimate at 3.5 characters per token)**. The one rubric we have is about 1-2k tokens. So the judge needs a context window of **at least 16k tokens, comfortably 32k+**, and the output is only about 100 tokens (16 numbers as JSON).

| Option | Context | Fit |
|---|---|---|
| Gemini free tier (Flash class, via Google AI Studio) | about 1M tokens | **Recommended first try.** Context is far more than needed; the free request cap is roughly 500-1,500 per day, so 1,500-2,000 items take 1-4 days. Items from harmful-content benchmarks (harmbench, alignment_faking) may be blocked by safety filters: log these, do not drop them silently. |
| Mistral free tier (Mistral Small, 128k) | 128k | Fallback; check whether the free tier requires opting in to data training. |
| Groq free tier (Llama 3.1 8B, 128k) | 128k | Not recommended: per-minute token caps are too small for long items. |
| Local open model on this CPU (e.g. Qwen2.5-7B-Instruct, 4-bit) | 32k native | **Not feasible**: about 2k prompt tokens per call at CPU speed is tens of seconds per call, so thousands of items would take weeks. |
| vLLM on a Modal GPU (Qwen2.5-7B or Qwen3-8B Instruct) | 32k+ | Possible, but it spends Modal credit: ask twice before any GPU spawn. Not the default. |

No-truncation rules for the labelling script:
- Count the input tokens with the provider's token counter and **abort the item if the prompt exceeds the model's window**; never cut the item text.
- Set the output limit well above the 16-number JSON (at least 1,024 tokens) and use a non-thinking mode so the hidden reasoning cannot eat the budget.
- Record the finish reason for every call and treat anything other than a normal stop (length cut-off, safety block) as a failure to be re-run or logged, never as a rating.
- Parse strictly: exactly 16 integers in 0-5, otherwise the item is a failure.

## Rules for this work
- No GPU or Modal spend without asking twice; this plan is CPU plus one API labelling job.
- No leakage: nothing tuned on test; scales model and $w$ fitted on training benchmarks.
- Disclose the annotating LLM and the LLM coding assistant in the report.
- At scoring time the distilled model needs only BGE-M3 (already in the ZIP), so no API is called.

## Risks
- The distilled scales come from the same BGE-M3 embeddings that already failed to transfer difficulty, so the gain over them may be small.
- LLM ratings of non-English items (cmmlu is Chinese; 31 languages overall) may be unreliable.
- Free-tier daily limits mean labelling takes days, and the label set is small (1,500-2,000 items), so the distilled predictor in step 3 may be weak.
- One-call-for-16-dimensions is a shortcut from the paper's one-call-per-dimension procedure and is untested.

## Decisions needed from you
1. Which free tier to try first (Gemini recommended), and who creates the key.
2. Whether to read Zhou et al. first (step 0) before anything else.
