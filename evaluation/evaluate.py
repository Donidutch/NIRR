import pytrec_eval
from typing import Dict, Union
import numpy as np


def create_dic(qrels_df, index=42):
    qrels_dict = {}
    for _, row in qrels_df.iterrows():
        qid, _, docno, rel = row
        qid = str(qid)
        docno = str(docno)
        if qid not in qrels_dict:
            qrels_dict[qid] = {}
        qrels_dict[qid][docno] = int(rel)

    return qrels_dict


def evaluate_run(run, qrels_df, metric: Union[str, set] = None) -> Dict[str, float]:
    """
    Evaluate a run using the specified relevance judgments.

    Args:
        run (List[Dict[str, Dict[str, int]]]): List containing the document
        rankings for each query.
        qrels (List[Dict[str, Dict[str, int]]]): List containing the relevance
        judgments for each query.

    Returns:
        Dictionary containing the evaluation measures.
    """
    if metric is None:
        metric = {"ndcg_cut_10"}
    elif isinstance(metric, str):
        metric = {metric}

    qrels = create_dic(qrels_df) if type(qrels_df) is not dict else qrels_df

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metric)  # type: ignore

    results = evaluator.evaluate(run)

    metric_values = {}
    for measure in sorted(metric):  # type: ignore
        res = pytrec_eval.compute_aggregated_measure(
            measure, [query_measures[measure] for query_measures in results.values()]
        )
        metric_values[measure] = np.round(res, 3)
    return metric_values
