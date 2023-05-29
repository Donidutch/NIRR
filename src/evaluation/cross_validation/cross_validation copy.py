# evaluation/cross_validation/cross_validation.py
from typing import Dict, Union
import pandas as pd

from evaluation.cross_validation.kfold import generate_folds
from evaluation.model_evaluation import ModelEvaluator
from evaluation.utils.metrics import get_metric
from retrieval.ranking_models.models import Model

# cross_validation.py
from evaluation.model_evaluation import ModelEvaluator
from retrieval.model_tuning import ModelTuner


# class CrossValidator:
#     def __init__(self, model, kfolds, model_type="bm25", tuning_measure="ndcg_cut_10"):
#         self.model_evaluator = ModelEvaluator(model)
#         self.kfolds = kfolds
#         self.model_type = model_type
#         self.tuning_measure = tuning_measure
#         self.metrics = get_metric(get_all_metrics=True)
#         self.results_df = pd.DataFrame(
#             columns=["fold", "model_type", "params", "score"]
#         )

#     def run_cross_validation(
#         self, queries_file: str, qrels_file: str, index_path: str
#     ) -> Dict[str, Union[str, Dict[str, float], pd.DataFrame, float]]:
#         # Load your files and initialize variables
#         # ...
#         unique_qids = pd.Series(queries["qid"].unique())
#         groups = generate_folds(unique_qids, self.kfolds)

#         # Call run_model_on_folds
#         metrics, response_times, results_df, best_config = self.run_model_on_folds(
#             groups, queries, qrels, unique_qids
#         )

#         for metric in metrics:
#             metrics[metric] /= self.kfolds
#         mean_response_time = sum(response_times) / self.kfolds

#         return {
#             "best_config": best_config,
#             "metrics": metrics,
#             "mean_response_time": mean_response_time,
#             "results_df": results_df,
#         }

#     def run_model_on_folds(
#         self,
#         groups: Dict[int, List[str]],
#         queries: pd.DataFrame,
#         qrels: pd.DataFrame,
#         unique_qids: pd.Series,
#     ) -> Tuple[Dict[str, float], List[float], pd.DataFrame, str]:
#         response_times = []
#         best_config = None
#         tuner = ModelTuner(
#             self.model_evaluator.model
#         )  # Initialize ModelTuner with the model

#         for i in tqdm(range(self.kfolds), desc="Folds", total=self.kfolds):
#             # Get training data
#             # ...
#             fold_params_performance, best_config = tuner.tune_and_set_parameters(
#                 train_queries,
#                 train_qids,
#                 training_qrels,
#                 i + 1,
#                 self.model_type,
#                 self.tuning_measure,
#                 self.results_df,
#             )
#             self.results_df = pd.concat(
#                 [self.results_df, fold_params_performance], ignore_index=True
#             )

#             response_time, measures = self.model_evaluator.run_and_evaluate_model(
#                 train_queries, train_qids_str, training_qrels, self.metrics
#             )
#             response_times.append(response_time)
#             self.metrics = self.model_evaluator.update_metrics(self.metrics, measures)

#         return self.metrics, response_times, self.results_df, str(best_config)


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

    model_evaluator = ModelEvaluator(model)

    (
        metrics,
        response_times,
        results_df,
        best_config,
    ) = model_evaluator.run_model_on_folds(
        kfolds,
        groups,
        queries,
        qrels,
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


# cross_validation.py
from typing import Dict, Union
import pandas as pd

from evaluation.model_evaluation import ModelEvaluator
from retrieval.ranking_models.models import Model
from retrieval.model_tuning import ModelTuner
from evaluation.cross_validation.kfold import generate_folds, get_training_data
from evaluation.utils.metrics import get_metric
from tqdm import tqdm


def run_cross_validation(
    queries_file: str,
    qrels_file: str,
    index_path: str,
    kfolds: int,
    model_type: str = "bm25",
    tuning_measure: str = "ndcg_cut_10",
) -> Dict[str, Union[str, Dict[str, float], pd.DataFrame, float]]:
    # Same as before...
    # ...

    model_evaluator = ModelEvaluator(model)
    model_tuner = ModelTuner(model)

    for i in tqdm(range(kfolds), desc="Folds", total=kfolds):
        (
            train_queries,
            train_qids,
            training_qrels,
            train_qids_str,
        ) = get_training_data(groups, i, queries, qrels, unique_qids)

        fold_params_performance, best_config = model_tuner.tune_and_set_parameters(
            train_queries,
            train_qids,
            training_qrels,
            i + 1,
            model_type,
            tuning_measure,
            results_df,
        )

        results_df = pd.concat([results_df, fold_params_performance], ignore_index=True)

        response_time, measures = model_evaluator.run_and_evaluate_model(
            train_queries, train_qids_str, training_qrels, metrics
        )

        response_times.append(response_time)
        for metric, measure in measures.items():
            metrics[metric] += measure

    for metric in metrics:
        metrics[metric] /= kfolds

    mean_response_time = sum(response_times) / kfolds

    return {
        "best_config": best_config,
        "metrics": metrics,
        "mean_response_time": mean_response_time,
        "results_df": results_df,
    }
