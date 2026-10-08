"""Competition entry point. Keep this file at the root of the submission ZIP."""

from aims_predictor import predict_probability


def predict(input: list, labeled: list | None = None) -> float:
    """Return the probability that the supplied subject answers the item correctly."""

    return predict_probability(input, labeled)

