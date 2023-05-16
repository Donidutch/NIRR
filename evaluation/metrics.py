def get_metric(metric_name=None):
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

    return metrics.get(metric_name, 0) if metric_name else metrics
