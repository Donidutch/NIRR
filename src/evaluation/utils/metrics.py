from typing import Optional, Union, Dict, Any


from typing import Optional, Dict

class Metrics:
    def __init__(self):
        self.metrics = {
            "ndcg": 0,
            "ndcg_cut_10": 0,
            "map":0,
            "recip_rank": 0,
            "P_5": 0,
            "P_10": 0,
            "P_20": 0,
            "recall_5": 0,
            "recall_10": 0,
            "recall_20": 0,
        }

    def get_all_metrics(self) -> Dict[str, int]:
        return self.metrics

    def get_metric(self, metric_name: Optional[str]) -> Optional[int]:
        return self.metrics.get(metric_name)
    