from typing import Optional, Union, Dict, Any


def get_metric(
    metric_name: Optional[str] = None, get_all_metrics: bool = False
) -> Union[Dict[str, Any], Any]:
    """
    Return a specific metric value or all available metrics based on
    the given metric_name.

    Args:
        metric_name (Optional[str]): The name of a specific metric to retrieve.
            If provided, the value of the specified metric is returned.
        get_all_metrics (bool): If True, returns a dictionary of all available metrics.
            Overrides the metric_name argument.

    Returns:
        Union[Dict[str, int], int]: A dictionary of all available metrics or the value
            of a specified metric.
    """
    metrics = {
        "ndcg": 0,
        "recip_rank": 0,
        "P_5": 0,
        "P_10": 0,
        "P_20": 0,
        "recall_5": 0,
        "recall_10": 0,
        "recall_20": 0,
    }

    if get_all_metrics:
        return metrics
    if metric_name is not None:
        return metrics.get(metric_name, 0)
