import numpy as np
import pandas as pd
import itertools
from evaluation.models import Model
from .evaluate import evaluate_run
from .utils import load_queries_and_qrels
import time


def create_run(model, queries, qids):
    qids = [str(item) for item in qids]
    batch_search_output = model.search(queries, qids)
    run = {}
    for qid, search_results in batch_search_output.items():
        run[qid] = {result.docid: result.score for result in search_results}
    return run


def tune_parameters(
    model,
    train_queries,
    train_qids,
    qrels,
    fold,
    model_type="bm25",
    tuning_measure="ndcg_cut_10",
):
    if model_type == "bm25":
        tuning_params = {"k1": [0.9, 1.0, 1.1, 1.2], "b": [0.6, 0.7, 0.8, 0.9]}
        param_combinations = list(
            itertools.product(tuning_params["k1"], tuning_params["b"])
        )
    elif model_type == "lm":
        tuning_params = {"mu": [1000, 1200, 1400]}
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
    queries_file,
    qrels_file,
    index_path,
    kfolds,
    model_type="bm25",
    tuning_measure="ndcg_cut_10",
):
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

    for i in range(num_folds):
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
    print(best_config)
    return {
        "best_config": best_config,
        "metrics": metrics,
        "mean_response_time": mean_response_time,
        "results_df": results_df,
    }
