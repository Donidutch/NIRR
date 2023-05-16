import numpy as np
import pandas as pd
import itertools
from evaluation.models import Model
from .evaluate import evaluate_run
from .utils import load_queries_and_qrels
import time
from typing import Dict, List, Any, Union
from tqdm import tqdm


def create_run(
    model: Model, queries: List[str], qids: List[int]
) -> Dict[str, Dict[str, float]]:
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
    qids = [str(item) for item in qids]
    batch_search_output = model.search(queries, qids)
    run = {}
    for qid, search_results in batch_search_output.items():
        run[qid] = {result.docid: result.score for result in search_results}
    return run


def tune_parameters(
    model: Any,
    train_queries: List[str],
    train_qids: List[int],
    qrels: Dict[str, Dict[str, int]],
    fold: int,
    model_type: str = "bm25",
    tuning_measure: str = "ndcg_cut_10",
) -> pd.DataFrame:
    """
    Tune the parameters of the model using the given training queries, query IDs,
    relevance judgments, and fold.

    Args:
        model (Any): The model to be tuned.
        train_queries (List[str]): A list of training queries.
        train_qids (List[int]): A list of training query IDs.
        qrels (Dict[str, Dict[str, int]]): A dictionary representing
            relevance judgments, where each query ID is mapped
            to a dictionary containing document IDs as keys
            and relevance judgments as values.
        fold (int): The fold number or identifier for the tuning.
        model_type (str, optional): The type of model to tune. Defaults to "bm25".
        tuning_measure (str, optional): The evaluation measure used for tuning.
        Defaults to "ndcg_cut_10".

    Returns:
        pd.DataFrame: A DataFrame containing the performance scores
            for different parameter combinations, including the fold number,
            model type, parameters, and score.
    """
    if model_type == "bm25":
        tuning_params = {"k1": [0.9, 1.0, 1.1], "b": [0.6, 0.7, 0.8]}
        tuning_params = {"k1": [0.9], "b": [0.6]}

        param_combinations = list(
            itertools.product(tuning_params["k1"], tuning_params["b"])
        )
        print(param_combinations)
    elif model_type == "lm":
        tuning_params = {"mu": [1000, 1500, 2000]}
        param_combinations = list(itertools.product(tuning_params["mu"]))

    params_performance = pd.DataFrame(columns=["fold", "model_type", "params", "score"])

    for params in param_combinations:
        if model_type == "bm25":
            k1, b = params
            model.set_bm25_parameters(k1, b)
            params_str = f"k1: {k1}, b: {b}"
        elif model_type == "lm":
            mu = params[0]
            model.set_qld_parameters(mu)
            params_str = f"mu: {mu}"

        run = create_run(model, train_queries, train_qids)
        measures = evaluate_run(run, qrels, metric=tuning_measure)

        new_row = pd.DataFrame(
            {
                "fold": [fold],
                "model_type": [model_type],
                "params": [params_str],
                "score": [measures[tuning_measure]],
            }
        )

        params_performance = pd.concat([params_performance, new_row], ignore_index=True)

    return params_performance


def run_cross_validation(
    queries_file: str,
    qrels_file: str,
    index_path: str,
    kfolds: int,
    model_type: str = "bm25",
    tuning_measure: str = "ndcg_cut_10",
) -> Dict[str, Union[str, Dict[str, float], pd.DataFrame, float]]:
    """
    Cross-validate a retrieval model using k-folds cross-validation.

    Params:
        queries_file: str
            Path to the queries file
        qrels_file: str
            Path to the qrels file
        index_path: str
            Path to the index directory
        kfolds: int
            Number of folds to use in cross-validation
        model_type: str
            Type of retrieval model to use ("bm25" or "lm"). Default is "bm25".
        tuning_measure: str
            The measure to use for tuning ("ndcg_cut_10" or "map").
            Default is "ndcg_cut_10".

    Returns:
        Dict[str, Union[str, Dict[str,float], pd.DataFrame, float]]:
            A dictionary containing the following keys:
            - best_config: The best configuration of the model as a string
            - metrics: A dict of mean evaluation metrics for the best configuration
            - mean_response_time: The mean response time for query processing in seconds
            - results_df: A DataFrame containing each of the tuning parameter
                combinations tested, along with their associated performance
                score for the given tuning measure.

    """
    queries, qrels = load_queries_and_qrels(queries_file, qrels_file)
    qrels = pd.read_csv(qrels_file, sep=" ", names=["qid", "Q0", "docid", "rel"])
    seed = 42
    num_folds = kfolds

    rng = np.random.default_rng(seed)
    unique_qids = queries["qid"].unique()
    choice = rng.choice(range(num_folds), size=len(unique_qids))
    groups = {}
    for c, qid in zip(choice, unique_qids):
        if c not in groups:
            groups[c] = [qid]
        else:
            groups[c].append(qid)
    results_df = pd.DataFrame(columns=["fold", "model_type", "params", "score"])
    model = Model(index_path)

    K = 10
    metric = f"ndcg_cut_{K}"

    for i in tqdm(range(num_folds), desc="Folds", total=num_folds):
        validation_set = groups[i]
        validation_qrels = qrels.loc[qrels["qid"].isin(validation_set)]  # noqa: F841

        training_set = set(unique_qids).difference(set(validation_set))
        training_qrels = qrels.loc[qrels["qid"].isin(training_set)]
        train_queries = [
            queries.loc[queries["qid"] == qid]["query"].values[0]
            for qid in training_set
        ]
        train_qids = list(training_set)

        fold_params_performance = tune_parameters(
            model,
            train_queries,
            train_qids,
            training_qrels,
            fold=i + 1,
            model_type=model_type,
            tuning_measure=metric,
        )

        results_df = pd.concat([results_df, fold_params_performance], ignore_index=True)

    # Find the best configuration
    best_config = results_df.groupby("params")["score"].mean().idxmax()
    results_df["best_config"] = best_config

    # Calculate evaluation metrics and mean response time for the best configuration
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
    response_times = []
    for i in range(num_folds):
        # Set the best configuration
        if model_type == "bm25":
            k1, b = [float(x.split(": ")[1]) for x in best_config.split(", ")]
            model.set_bm25_parameters(k1, b)
        elif model_type == "lm":
            mu = float(best_config[4:-1])
            model.set_qld_parameters(mu)

        # Calculate the search response time
        start_time = time.time()
        run = create_run(model, train_queries, train_qids)
        response_time = (time.time() - start_time) / len(train_qids)
        response_times.append(response_time)

    # Calculate the metrics for this fold
    measures = evaluate_run(run, training_qrels, metric=list(metrics.keys()))
    for metric in metrics:
        metrics[metric] += measures[metric]

    # Calculate the mean metrics and response time
    for metric in metrics:
        metrics[metric] /= num_folds
        mean_response_time = sum(response_times) / num_folds
    return {
        "best_config": best_config,
        "metrics": metrics,
        "mean_response_time": mean_response_time,
        "results_df": results_df,
    }
