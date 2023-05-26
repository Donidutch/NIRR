import time
from typing import List, Dict, Tuple, Any
import pandas as pd
from tqdm import tqdm
from evaluation.evaluate import evaluate_run
from evaluation.models import Model
from evaluation.query_expansion import (
    pseudo_relevance_feedback,
    expand_query_word2vec,
    expand_query_bert,
)
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
import importlib


def create_run(
    model: Model,
    queries: List[str],
    qids: List[str],
    index_path: str,
    corpus_path: str,
    expansion_method=None,
) -> Dict[str, Dict[str, int]]:
    expanded_queries = []
    if expansion_method == "word2vec":
        expansion_model = KeyedVectors.load_word2vec_format(
            "data/embedding/GoogleNews-vectors-negative300.bin", binary=True
        )

    elif expansion_method == "bert":
        expansion_model = SentenceTransformer("all-MiniLM-L6-v2")

    for query in queries:
        if expansion_method == "pseudo_relevance_feedback":
            expanded_query = pseudo_relevance_feedback(
                index_path, query, num_docs=10, num_terms=5
            )
        elif expansion_method == "word2vec":
            expanded_query = expand_query_word2vec(expansion_model, query, num_terms=5)
        elif expansion_method == "bert":
            expanded_query = expand_query_bert(
                expansion_model, query, num_terms=5, corpus=corpus_path
            )
        else:
            expanded_query = query
        expanded_queries.append(expanded_query)

    batch_search_output = model.search(expanded_queries, qids)
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
    index_path: str,
    corpus_path: str,
    expansion_method=None,
) -> Tuple[float, Dict[str, int]]:
    start_time = time.time()

    run = create_run(
        model,
        train_queries,
        train_qids_str,
        index_path,
        corpus_path,
        expansion_method=expansion_method,
    )
    response_time = (time.time() - start_time) / len(train_qids_str)

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
    index_path: str,
    corpus_path: str,
    expansion_method: str = None,
) -> Tuple[Dict[str, float], List[float], pd.DataFrame, str]:
    """
    Runs the specified model on the given data using k-fold cross-validation and returns
    evaluation metrics, response times, updated results DataFrame, and best configuration as a string.

    Args:
        ... (omitted for brevity)
        index_path (str): Path to the index file.
        corpus_path (str): Path to the corpus file.
        expansion_method (str): Query expansion method. Default is None.

    Returns:
        Tuple[Dict[str, float], List[float], pd.DataFrame, str]: Tuple containing the
        updated metrics, response times, updated results DataFrame, and best configuration as a string.
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
            index_path,  # Pass index_path
            corpus_path,  # Pass corpus_path
            model_type,
            metric,
            results_df,
        )

        results_df = pd.concat([results_df, fold_params_performance], ignore_index=True)

        model_evaluation = importlib.import_module("evaluation.model_evaluation")
        run_and_evaluate_model = model_evaluation.run_and_evaluate_model
        update_metrics = model_evaluation.update_metrics
        response_time, measures = run_and_evaluate_model(
            model,
            train_queries,
            train_qids_str,
            training_qrels,
            metrics,
            index_path,  # Pass index_path
            corpus_path,  # Pass corpus_path
            expansion_method,
        )

        response_times.append(response_time)
        metrics = update_metrics(metrics, measures)

    return metrics, response_times, results_df, str(best_config)


def update_metrics(metrics, measures):
    for metric in metrics:
        metrics[metric] += measures[metric]
    return metrics
