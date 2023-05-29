import pytrec_eval
from typing import Dict, Union
import numpy as np
import pandas as pd


def create_dic(qrels_df: pd.DataFrame, index: int = 42) -> Dict[str, Dict[str, int]]:
    """
    Create a dict representation of the relevance judgments from the given DataFrame.

    Args:
        qrels_df (pd.DataFrame): A DataFrame containing relevance judgments,
        with columns "qid", "ignore", "docno", and "rel".
        index (int, optional): The index to use for the dictionary. Defaults to 42.

    Returns:
        Dict[str, Dict[str, int]]: A dictionary representing relevance judgments,
            where each query ID is mapped to a dictionary
            containing document IDs as keys and relevance judgments as values.
    """
    qrels_dict = {}
    for _, row in qrels_df.iterrows():
        qid, _, docno, rel = row
        qid = str(qid)
        docno = str(docno)
        if qid not in qrels_dict:
            qrels_dict[qid] = {}
        qrels_dict[qid][docno] = int(rel)

    return qrels_dict


def evaluate_run(
    run: Dict[str, Dict[str, int]],
    qrels_df: Union[Dict[str, Dict[str, int]], pd.DataFrame],
    metric: Union[str, set],
) -> Dict[str, int]:
    """
    Evaluate a run using the specified relevance judgments.

    Args:
        run (List[Dict[str, Dict[str, int]]]): List containing the document rankings
            for each query.
        qrels_df (Union[Dict[str, Dict[str, int]], pd.DataFrame]): Dictionary or
            DataFrame containing the relevance judgments for each query.
        metric (Union[str, set], optional): The evaluation metric(s) to be used.
            Defaults to None.

    Returns:
        Dictionary containing the evaluation measures.
    """

    if metric is None:
        print("hi")
        metric = {"ndcg_cut_10"}
    elif isinstance(metric, str):
        metric = {metric}
    qrels = create_dic(qrels_df) if isinstance(qrels_df, pd.DataFrame) else qrels_df

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metric)
    print("metriiics", metric)
    results = evaluator.evaluate(run)

    metric_values = {}
    for measure in sorted(metric):
        res = pytrec_eval.compute_aggregated_measure(
            measure, [query_measures[measure] for query_measures in results.values()]
        )

        metric_values[measure] = res  # np.round(res, 5)
    # print("metricss",metric_values)
    return metric_values
