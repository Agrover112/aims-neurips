"""CodaBench entry point for the TabFM streaming baseline."""

from aims_predictor.tabfm_predictor import predict_probability


def predict(input: list, labeled: list | None = None) -> float:
    return predict_probability(input, labeled)
