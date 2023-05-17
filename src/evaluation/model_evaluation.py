import time
from typing import List, Dict, Tuple, Union, Any
import pandas as pd
from tqdm import tqdm
from evaluation.evaluate import evaluate_run
from evaluation.kfold_preparation import generate_folds
from evaluation.metrics import get_metric
from evaluation.models import Model
import importlib

def create_run(
    model: Model, queries: List[str], qids: List[str]
) -> Dict[str, Dict[str, int]]:
    """
    Create a run dictionary using the provided model, queries, and query IDs.

    Args:
        model Model: The model used for searching.
        queries (List[str]): A list of queries.
        qids (List[int]): A list of query IDs.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary representing the run, where each
            query ID is mapped to a dictionary containing document IDs
            as keys and corresponding scores as values.
    """
    if isinstance(qids, list) and isinstance(qids[0], int):
        qids = [str(qid) for qid in qids]
    batch_search_output = model.search(queries, qids)
    run = {}
    for qid, search_results in batch_search_output.items():
        run[qid] = {result.docid: result.score for result in search_results}  # type: ignore  # noqa: E501
    return run


def run_and_evaluate_model(
    model,
    train_queries: List[str],
    train_qids_str: List[str],
    training_qrels: pd.DataFrame,
    metrics: Dict[str, float],
) -> Tuple[float, Dict[str, int]]:
    start_time = time.time()

    run = create_run(model, train_queries, train_qids_str)
    response_time = (time.time() - start_time) / len(train_qids_str)

    # We're passing the keys of the metrics dictionary as a set to the function
    measures = evaluate_run(run, training_qrels, set(metrics.keys()))

    return response_time, measures


def run_model_on_folds(
    num_folds: int,
    groups: Dict[int, List[str]],
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    model: Any,
    model_type: str,
    metric: str,
    results_df: pd.DataFrame,
    metrics: Dict[str, float],
    unique_qids: pd.Series,
) -> Tuple[Dict[str, float], List[float], pd.DataFrame, str]:
    """
    Runs the specified model on the given data using k-fold cross-validation and returns
        evaluation metrics,
    response times, updated results DataFrame, and best configuration as a string.

    Args:
        num_folds (int): Number of folds for cross-validation.
        groups (Dict[int, List[str]]): Dictionary with fold number as key and list of
            query IDs as value.
        queries (pd.DataFrame): DataFrame containing query information.
        qrels (pd.DataFrame): DataFrame containing relevance information.
        model (Any): Model to be used for evaluation.
        model_type (str): Type of the model (e.g., "bm25", "lm").
        metric (str): Evaluation metric to use for tuning.
        results_df (pd.DataFrame): DataFrame to store evaluation results.
        metrics (Dict[str, float]): Dictionary to store aggregated evaluation metrics.
        unique_qids (pd.Series): Series containing unique query IDs.

    Returns:
        Tuple[Dict[str, float], List[float], pd.DataFrame, str]: Tuple containing the
            updated metrics,
        response times, updated results DataFrame, and best configuration as a string.
    """

    response_times = []
    best_config = None

    for i in tqdm(range(num_folds), desc="Folds", total=num_folds):
        kfold_preparation = importlib.import_module("evaluation.kfold_preparation")
        get_training_data = kfold_preparation.get_training_data
        
        train_queries, train_qids, training_qrels, train_qids_str = get_training_data(
            groups, i, queries, qrels, unique_qids
        )
        
        model_tuning = importlib.import_module("evaluation.model_tuning")
        tune_and_set_parameters = model_tuning.tune_and_set_parameters
        
        fold_params_performance, best_config = tune_and_set_parameters(
            model,
            train_queries,
            train_qids,
            training_qrels,
            i + 1,
            model_type,
            metric,
            results_df,
        )

        results_df = pd.concat([results_df, fold_params_performance], ignore_index=True)
        
        model_evaluation = importlib.import_module("evaluation.model_evaluation")
        run_and_evaluate_model = model_evaluation.run_and_evaluate_model
        update_metrics = model_evaluation.update_metrics
        
        response_time, measures = run_and_evaluate_model(
            model, train_queries, train_qids_str, training_qrels, metrics
        )

        response_times.append(response_time)
        metrics = update_metrics(metrics, measures)

    return metrics, response_times, results_df, str(best_config)

def run_cross_validation(
    queries_file: str,
    qrels_file: str,
    index_path: str,
    kfolds: int,
    model_type: str = "bm25",
    tuning_measure: str = "ndcg_cut_10",
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
    """
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


def update_metrics(metrics, measures):
    for metric in metrics:
        metrics[metric] += measures[metric]
    return metrics
