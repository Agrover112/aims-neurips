"""Streaming acquisition policy for competition-approved test-time adaptation."""


def acquisition_function(
    input: list[dict],
    prediction: float | None = None,
    labeled: list | None = None,
    context: dict | None = None,
) -> bool | float:
    """Purchase uncertain outcomes while reserving enough candidates to fill the budget."""

    if context is None:
        content = str(input[1].get("item_content", ""))
        return float(-abs(len(content) - 1200))
    if prediction is None:
        raise ValueError("the streaming policy requires the current prediction")
    remaining = int(context["labels_remaining"])
    candidates = int(context["items_remaining"])
    if remaining <= 0:
        return False
    return bool(candidates <= remaining or abs(float(prediction) - 0.5) <= 0.15)
