"""CodaBench entry point for the TabPFN-v2 streaming baseline."""

from aims_predictor.tabpfn_predictor import predict_probability


def predict(input: list, labeled: list | None = None) -> float:
    return predict_probability(input, labeled)
