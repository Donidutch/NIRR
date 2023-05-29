# evaluation/cross_validation/cross_validation.py

from typing import Dict, Union
import numpy as np
import pandas as pd
import pytrec_eval

from evaluation.cross_validation.kfold import generate_folds, get_training_data
from evaluation.model_evaluation import ModelEvaluator
from evaluation.utils.metrics import get_metric
from retrieval.ranking_models.models import Model
from retrieval.model_tuning import ModelTuner


def run_cross_validation(
    queries_file: str,
    qrels_file: str,
    index_path: str,
    kfolds: int,
    model_type: str = "bm25",
    tuning_measure: str = "ndcg_cut_10",
) -> Dict[str, Union[str, Dict[str, float], pd.DataFrame, float]]:
    queries = pd.read_csv(queries_file, dtype={"qid": str})
    qrels = pd.read_csv(qrels_file, dtype={"qid": str, "docno": str, "label": int})
    qrels.rename(
        columns={"label": "rel", "iteration": "Q0", "docno": "docid"}, inplace=True
    )
    new_cols = ["qid", "Q0", "docid", "rel"]
    qrels = qrels.reindex(columns=new_cols)
    unique_qids = queries["qid"].unique()

    groups = generate_folds(unique_qids, kfolds)

    results_df = pd.DataFrame(columns=["fold", "model", "params", "score"])
    model = Model(index_path, model_type)

    metric = tuning_measure
    metrics = get_metric(get_all_metrics=True)

    model_evaluator = ModelEvaluator(model)
    tuner = ModelTuner(model)

    response_times = []
    best_config = None
    mean_scores = []  # To hold mean scores for each fold

    for i in range(kfolds):
        print("Fold", i + 1)
        train_queries, train_qids, training_qrels, train_qids_str = get_training_data(
            groups, i, queries, qrels, unique_qids
        )

        fold_params_performance, best_config = tuner.tune_and_set_parameters(
            model,
            train_queries,
            train_qids,
            training_qrels,
            i + 1,
            model_type,
            metric,
        )
        # Prepare validation data
        validation_set = list(groups[i])
        validation_qrels = qrels.loc[qrels["qid"].isin(validation_set)]
        validation_queries = [
            queries.loc[queries["qid"] == qid]["query"].values[0]
            for qid in validation_set
        ]
        validation_qids_str = [str(qid) for qid in validation_set]
        results_df = pd.concat(
            [results_df, fold_params_performance],
            ignore_index=True,
        )
        response_time, measures = model_evaluator.run_and_evaluate_model(
            validation_queries, validation_qids_str, validation_qrels, metrics
        )
        # response_time, measures = model_evaluator.run_and_evaluate_model(
        #     train_queries, train_qids_str, training_qrels, metrics
        # )
        response_times.append(response_time)
        mean_scores.append(measures[tuning_measure])
        # print(mean_scores)
        # fold_mean_score = pytrec_eval.compute_aggregated_measure(
        #     tuning_measure,
        #     [query_measures[tuning_measure] for query_measures in measures.values()],
        # )
        # mean_scores.append(fold_mean_score)

        metrics = ModelEvaluator.update_metrics(metrics, measures)
        metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    # Compute the mean score over all folds
    mean_score_over_folds = sum(mean_scores) / len(mean_scores)
    # print("scores", mean_score_over_folds)
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    for metric in metrics:
        metrics[metric] /= kfolds

    results_df.set_index("params", inplace=True)

    # print(metrics, metrics_df)
    # metrics = metrics_df.mean().to_dict()

    mean_response_time = np.mean(response_times)
    return {
        "best_config": best_config,
        "metrics": metrics,
        "mean_response_time": mean_response_time,
        "results_df": results_df,
    }
