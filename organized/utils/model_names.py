"""The one set of display names used by every final-results figure and table.

Internal keys (run tags, replay model keys, cache columns) stay as they are;
only what a reader sees comes from here. All benchmark-holdout ANPs share one
architecture (width 64, one self-attention block, 4 heads, 32-d latent, BGE-M3
item embeddings), so a plain "ANP" is that model trained with uniform context
sizes, and every other ANP name says only what differs from it.
"""

BETA = "Beta-binomial"

NAME = {
    # Baselines
    "const-mean": "Constant (training mean)",
    "constant": "Constant (training mean)",
    "emp-mean": "Raw hit rate",
    "beta": BETA,
    "jeffreys": BETA,
    # Benchmark-holdout ANPs
    "w64-filtered": "ANP",
    "w64-ladder": "ANP, budget-ladder training",
    "w64-sam": "ANP + SAM",
    "w64-anchor": "ANP + prior anchor",
    "w64-sam-ladder": "ANP + SAM, budget-ladder training",
    "w64-dense": "ANP, dense-prefix training",
    "w64-noitem": "ANP without item embeddings",
    # Random-subject split (trained on some test-set benchmarks)
    "w64-contam": "ANP, random-subject split",
    "w128": "ANP, random-subject split (width 128)",
    # ANP combined with the beta-binomial
    "gate": "ANP, beta-binomial below 2 labels",
    "blend": "ANP/beta-binomial blend",
    "blend-budget": "ANP/beta-binomial blend, per budget",
    "aipw": "ANP + residual correction",
    "beta-aipw": "Beta-binomial + residual correction",
    # Hierarchical beta-binomial with a subject-metadata prior
    "hier, no metadata": "Hierarchical beta-binomial",
    "hier + provider": "Hierarchical beta-binomial + provider",
    "hier + name + provider": "Hierarchical beta-binomial + model name, provider",
    "hier + provider + date": "Hierarchical beta-binomial + provider, release date",
    "hier + name + provider + date": "Hierarchical beta-binomial + model name, provider, release date",
    # Pooled Rasch (uses every pair's labels)
    "pooled-rasch": "Pooled Rasch",
    "pooled-rasch-no-item": "Pooled Rasch without item difficulty",
    "combined-provider": "Pooled Rasch + provider prior",
    # ANP decomposition
    "full": "Full ANP",
    "level": "ANP level only",
    "order": "Beta-binomial level + ANP item ordering",
}

POLICY = {
    "random_prefix": "Random",
    "uncertainty": "Uncertainty",
    "confidence_extremes": "Confidence extremes",
    "bernoulli_variance": "Bernoulli variance",
    "expected_brier_reduction": "Expected Brier reduction",
}
