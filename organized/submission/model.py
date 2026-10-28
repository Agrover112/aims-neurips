"""CodaBench entry point for the staged latent-CNP-128 predictor."""

from aims_predictor.predictor import predict_probability


def predict(input: list, labeled: list | None = None) -> float:
    """Return a stateless probability conditioned only on ``labeled``."""

    return predict_probability(input, labeled)
