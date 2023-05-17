import ast
import itertools
import json
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from evaluation.metrics import get_metric
from evaluation.models import Model

from .evaluate import evaluate_run


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


def tune_parameters(
    model: Any,
    train_queries: List[str],
    train_qids: List[int],
    qrels: Union[pd.DataFrame, Dict[str, Dict[str, int]]],
    fold: int,
    model_type: str = "bm25",
    tuning_measure: str = "ndcg_cut_10",
    config_file: str = "evaluation/gridsearch_params.json",
) -> pd.DataFrame:
    """
    Tune the parameters of the model using the given training queries, query IDs,
    relevance judgments, and fold.
    """

    with open(config_file) as f:
        tuning_params = json.load(f)[model_type]

    param_combinations = list(itertools.product(*tuning_params.values()))

    params_performance = pd.DataFrame(columns=["fold", "model_type", "params", "score"])

    for params in param_combinations:
        model.set_parameters(model_type, dict(zip(tuning_params.keys(), params)))

        qids = [str(qid) for qid in train_qids]
        run = create_run(model, train_queries, qids)
        measures = evaluate_run(run, qrels, metric=tuning_measure)
        params_str = str(dict(zip(tuning_params.keys(), params)))

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


def set_model_config(model, model_type: str, best_config):
    if model_type == "bm25":
        if isinstance(best_config, tuple) and len(best_config) == 2:
            k1, b = best_config
        else:
            raise ValueError("Invalid value for best_config")
        model.set_bm25_parameters(k1, b)
    elif model_type == "lm":
        if isinstance(best_config, int):
            mu = best_config
        else:
            raise ValueError("Invalid value for best_config")
        model.set_qld_parameters(mu)


def generate_folds(
    unique_qids: pd.Series, kfolds: int, seed: int = 42
) -> Dict[int, List[str]]:
    rng = np.random.default_rng(seed)
    choice = rng.choice(range(kfolds), size=len(unique_qids))
    groups = {c: [] for c in range(kfolds)}
    for c, qid in zip(choice, unique_qids):
        groups[c].append(qid)
    return groups


def get_training_data(groups, fold, queries, qrels, unique_qids):
    validation_set = groups[fold]
    training_set = set(unique_qids).difference(set(validation_set))

    if isinstance(qrels, dict):
        qrels = pd.read_csv(
            "data/proc_data/train_sample/sample_qrels.tsv",
            sep=" ",
            names=["qid", "Q0", "docid", "rel"],
        )

    training_qrels = qrels.loc[qrels["qid"].isin(training_set)]
    train_queries = [
        queries.loc[queries["qid"] == qid]["query"].values[0] for qid in training_set
    ]
    train_qids = list(training_set)
    train_qids_str = [str(qid) for qid in train_qids]
    return train_queries, train_qids, training_qrels, train_qids_str


def tune_and_set_parameters(
    model,
    train_queries: List[str],
    train_qids: List[int],
    training_qrels: pd.DataFrame,
    fold: int,
    model_type: str,
    metric: str,
    results_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, str]:
    fold_params_performance = tune_parameters(
        model,
        train_queries,
        train_qids,
        training_qrels,
        fold=fold,
        model_type=model_type,
        tuning_measure=metric,
    )

    best_config = fold_params_performance.groupby("params")["score"].mean().idxmax()

    # Convert the string back into a dictionary
    if isinstance(best_config, str):
        best_config_dict = ast.literal_eval(best_config)
    else:
        best_config_dict = ast.literal_eval(str(best_config))
    print(best_config_dict)
    # Now you can access the values as you would in a regular dictionary

    if model_type == "bm25":
        best_config = (best_config_dict["k1"], best_config_dict["b"])

    elif model_type == "lm":
        best_config = best_config_dict["mu"]

    set_model_config(model, model_type, best_config)

    return fold_params_performance, str(best_config)


def run_model_on_folds(
    num_folds: int,
    groups: Dict[int, List[str]],
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    model,
    model_type: str,
    metric: str,
    results_df: pd.DataFrame,
    metrics: Dict[str, float],
    unique_qids: pd.Series,
) -> Tuple[Dict[str, float], List[float], pd.DataFrame, str]:
    response_times = []
    best_config = None

    for i in tqdm(range(num_folds), desc="Folds", total=num_folds):
        train_queries, train_qids, training_qrels, train_qids_str = get_training_data(
            groups, i, queries, qrels, unique_qids
        )

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

        response_time, measures = run_and_evaluate_model(
            model, train_queries, train_qids_str, training_qrels, metrics
        )
        response_times.append(response_time)

        for metric in metrics:
            metrics[metric] += measures[metric]

    return metrics, response_times, results_df, str(best_config)


def run_cross_validation(
    queries_file: str,
    qrels_file: str,
    index_path: str,
    kfolds: int,
    model_type: str = "bm25",
    tuning_measure: str = "ndcg_cut_10",
) -> Dict[str, Union[str, Dict[str, float], pd.DataFrame, float]]:
    queries = pd.read_csv(queries_file, sep=" ", names=["qid", "query"])
    qrels = pd.read_csv(qrels_file, sep=" ", names=["qid", "Q0", "docid", "rel"])
    unique_qids = pd.Series(queries["qid"].unique())

    groups = generate_folds(unique_qids, kfolds)
    results_df = pd.DataFrame(columns=["fold", "model_type", "params", "score"])
    model = Model(index_path)

    K = 10
    metric = f"ndcg_cut_{K}"
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
