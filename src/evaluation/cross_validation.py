from typing import Dict, Union
import pandas as pd

from evaluation.kfold_preparation import generate_folds
from evaluation.model_evaluation import run_model_on_folds
from evaluation.metrics import get_metric
from evaluation.models import Model


def run_cross_validation(
    queries_file: str,
    qrels_file: str,
    index_path: str,
    corpus_path: str,
    kfolds: int,
    model_type: str = "bm25",
    tuning_measure: str = "ndcg_cut_10",
    expansion_method: str = None,
) -> Dict[str, Union[str, Dict[str, float], pd.DataFrame, float]]:
    """
    Run k-fold cross-validation using the given queries and relevance judgements files.

    Parameters:
    queries_file (str): The path to the queries file
        (CSV format with columns: qid, query)
    qrels_file (str): The path to the relevance judgements file
        (CSV format with columns: qid, Q0, docid, rel)
    index_path (str): The path to the search index directory
    kfolds (int): The number of folds to use in cross-validation
    model_type (str, optional): The retrieval model to use (default: bm25)
    tuning_measure (str, optional): The measure used to tune
        the model (default: ndcg_cut_10)

    Returns:
    dict: A dictionary containing:
         - best_config: The configuration that resulted in the highest score
         - metrics: A dictionary of evaluation metrics
         - mean_response_time: The mean response time across all folds
         - results_df: The results DataFrame
    #"""
    test = True
    if test:
        queries = pd.read_csv(queries_file, sep=",")
        qrels = pd.read_csv(qrels_file, sep=",")
        qrels.rename(columns={"label": "rel"}, inplace=True)
        qrels.rename(columns={"iteration": "Q0"}, inplace=True)
        qrels.rename(columns={"docno": "docid"}, inplace=True)
        new_cols = ["qid", "Q0", "docid", "rel"]

        qrels = qrels.reindex(columns=new_cols)
    else:
        queries = pd.read_csv(queries_file, sep=" ", names=["qid", "query"])
        qrels = pd.read_csv(qrels_file, sep=" ", names=["qid", "Q0", "docid", "rel"])
    unique_qids = pd.Series(queries["qid"].unique())

    groups = generate_folds(unique_qids, kfolds)
    results_df = pd.DataFrame(columns=["fold", "model_type", "params", "score"])
    model = Model(index_path)

    metric = tuning_measure
    metrics = get_metric(get_all_metrics=True)

    metrics, response_times, results_df, best_config = run_model_on_folds(
        kfolds,
        groups,
        queries,
        qrels,
        model,
        model_type,
        metric,
        results_df,
        metrics,
        unique_qids,
        index_path,
        corpus_path,
        expansion_method,
    )

    for metric in metrics:
        metrics[metric] /= kfolds
    mean_response_time = sum(response_times) / kfolds

    return {
        "best_config": best_config,
        "metrics": metrics,
        "mean_response_time": mean_response_time,
        "results_df": results_df,
    }
