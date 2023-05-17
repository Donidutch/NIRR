import itertools
import time
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from evaluation.metrics import get_metric

from evaluation.models import Model

from .evaluate import evaluate_run
from .utils import load_queries_and_qrels


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


def tune_parameters(
    model: Any,
    train_queries: List[str],
    train_qids: List[int],
    qrels: Union[pd.DataFrame, Dict[str, Dict[str, int]]],
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

    param_combinations = []
    if model_type == "bm25":
        tuning_params = {"k1": [0.9, 1.0, 1.1], "b": [0.6, 0.7, 0.8]}
        param_combinations = list(
            itertools.product(tuning_params["k1"], tuning_params["b"])
        )

    elif model_type == "lm":
        tuning_params = {"mu": [1000, 1500, 2000]}
        param_combinations = list(itertools.product(tuning_params["mu"]))

    params_performance = pd.DataFrame(columns=["fold", "model_type", "params", "score"])

    for params in param_combinations:
        if model_type == "bm25":
            if (
                isinstance(params, tuple)
                and len(params) == 1
                and isinstance(params[0], int)
            ):
                k1 = params[0]
                b = None
            elif (
                isinstance(params, tuple)
                and len(params) == 2
                and isinstance(params[0], float)
                and isinstance(params[1], float)
            ):
                k1, b = params
            else:
                raise TypeError(
                    "params should be a tuple of either (int,) or (float, float)"
                )

            model.set_bm25_parameters(k1, b)
            params_str = f"k1: {k1}, b: {b}"
        elif model_type == "lm":
            mu = params[0]
            model.set_qld_parameters(mu)
            params_str = f"mu: {mu}"
        qids = [str(qid) for qid in train_qids]
        run = create_run(model, train_queries, qids)
        measures = evaluate_run(run, qrels, metric=tuning_measure)
        params_str = str(params)

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


def run_cross_validation(
    queries_file: str,
    qrels_file: str,
    index_path: str,
    kfolds: int,
    model_type: str = "bm25",
    tuning_measure: str = "ndcg_cut_10",
) -> Dict[str, Union[str, Dict[str, float], pd.DataFrame, float]]:
    queries, qrels = load_queries_and_qrels(queries_file, qrels_file)
    qrels = pd.read_csv(qrels_file, sep=" ", names=["qid", "Q0", "docid", "rel"])
    seed = 42
    num_folds = kfolds

    rng = np.random.default_rng(seed)
    unique_qids = queries["qid"].unique()
    choice = rng.choice(range(num_folds), size=len(unique_qids))
    groups = {c: [] for c in range(num_folds)}
    for c, qid in zip(choice, unique_qids):
        groups[c].append(qid)

    results_df = pd.DataFrame(columns=["fold", "model_type", "params", "score"])
    model = Model(index_path)

    K = 10
    metric = f"ndcg_cut_{K}"

    metrics = get_metric(get_all_metrics=True)
    response_times = []

    for i in tqdm(range(num_folds), desc="Folds", total=num_folds):
        validation_set = groups[i]
        training_set = set(unique_qids).difference(set(validation_set))
        training_qrels = qrels.loc[qrels["qid"].isin(training_set)]
        train_queries = [
            queries.loc[queries["qid"] == qid]["query"].values[0]
            for qid in training_set
        ]
        train_qids = list(training_set)
        train_qids_str = [str(qid) for qid in train_qids]
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

        best_config = results_df.groupby("params")["score"].mean().idxmax()
        results_df["best_config"] = best_config
        if model_type == "bm25":
            best_config = tuple(map(float, best_config[1:-1].split(", ")))
        elif model_type == "lm":
            best_config = int(float(best_config.strip("()").split(",")[0]))

        set_model_config(model, model_type, best_config)

        start_time = time.time()

        run = create_run(model, train_queries, train_qids_str)
        response_time = (time.time() - start_time) / len(train_qids)
        response_times.append(response_time)

        measures = evaluate_run(run, training_qrels, metric=list(metrics.keys()))
        for metric in metrics:
            metrics[metric] += measures[metric]

    for metric in metrics:
        metrics[metric] /= num_folds
    mean_response_time = sum(response_times) / num_folds

    return {
        "best_config": best_config,
        "metrics": metrics,
        "mean_response_time": mean_response_time,
        "results_df": results_df,
    }
