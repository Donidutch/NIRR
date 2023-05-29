# evaluation/model_evaluation.py

import time
<<<<<<< HEAD
from typing import List, Dict, Tuple, Any
=======
from typing import Dict, List, Tuple

>>>>>>> 57dfe61 (Add configuration and build files, refactor cross-validation code)
import pandas as pd
from pyserini import collection, index
from pyserini.search import LuceneSearcher

from evaluation.evaluate import evaluate_run
<<<<<<< HEAD
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
=======
from retrieval.ranking_models.models import Model

# from retrieval.model_tuning import ModelTuner
from pyserini.collection import Collection
from pyserini.index.lucene import Generator, LuceneIndexer, IndexReader
from pyserini.index import Lucene

def create_index_from_trec_collection(
    collection_path="data/lab/trec/", index_directory="data/tt/"
):
    collections = Collection('TrecCollection', collection_path)
    generator = Generator('DefaultLuceneDocumentGenerator')
    args = [
        "-input",
        collection_path,
        "-index",
        index_directory,
        "-collection",
        "TrecCollection",
        "-threads",
        str(4),
        "-keepStopwords",
        "-stemmer",
        "none",
    ]
    index_writer = LuceneIndexer(index_directory, args=args, threads=4)
    print(index_writer.args)

    for fs in collections:
        for doc in fs:
            # Convert the contents to a JSON string before adding it
            parsed = generator.create_document(doc)
            doc_id = parsed.get("id")  # FIELD_ID
            contents = parsed.get("contents")  # FIELD_BODY
            # print(doc_id)
            # Create a dictionary with the required fields
            doc_dict = {"id": doc_id, "contents": contents}
            index_writer.add_doc_dict(doc_dict)
    index_writer.close()
    print(IndexReader(index_directory).stats())



class ModelEvaluator:
    def __init__(self, model: Model):
        self.model = model

    def create_qrels_dict(self, qrels):
        qrels_dict = {}
        for _, r in qrels.iterrows():
            qid, _, docno, label = r
            if qid not in qrels_dict:
                qrels_dict[qid] = {}
            qrels_dict[qid][docno] = int(label)

    def create_run(
        self, queries: List[str], qids: List[str]
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
        # print(qids)
        # batch_search_output = self.model.search(queries, qids)
        # run = {}
        # for qid, search_results in batch_search_output.items():
        #     for result in search_results:
        #         print(qid, result.score)

        # test()
        # bm25 = self.model.searcher
        # print(queries)
        # bm25_run = []
        # for _, row in queries.iterrows():
        #     print(row)
        #     qid, query = row
        #     print(query)
        #     res_df = bm25.search(query)
        #     print(res_df)
        #     res_df = pd.DataFrame(res_df)
        #     print(res_df)
        #     for _, res_row in res_df.iterrows():
        #         _, docid, docno, rank, score, query = res_row
        #         row_str = f"{qid} 0 {docno} {rank} {score} tfidf"
        #         bm25_run.append(row_str)
        # with open("outputs/bm25.run", "w") as f:
        #     for l in bm25_run:
        #         f.write(l + "\n")
        # return bm25_run

        # for qid, search_results in batch_search_output.items():
        #     run[qid] = {result.docid: result.score for result in search_results}  # type: ignore  # noqa: E501
        # return run

    def run_and_evaluate_model(
        self,
        train_queries: List[str],
        train_qids_str: List[str],
        training_qrels: pd.DataFrame,
        metrics: Dict[str, float],
    ) -> Tuple[float, Dict[str, int]]:
        start_time = time.time()

        run = self.create_run(train_queries, train_qids_str)
        response_time = (time.time() - start_time) / len(train_qids_str)
        measures = evaluate_run(run, training_qrels, set(metrics.keys()))
        return response_time, measures

    # def run_model_on_folds(
    #     self,
    #     num_folds: int,
    #     groups: Dict[int, List[str]],
    #     queries: pd.DataFrame,
    #     qrels: pd.DataFrame,
    #     model_type: str,
    #     metric: str,
    #     results_df: pd.DataFrame,
    #     metrics: Dict[str, float],
    #     unique_qids: pd.Series,
    # ) -> Tuple[Dict[str, float], List[float], pd.DataFrame, str]:
    #     """
    #     Runs the specified model on the given data using k-fold cross-validation
    #     and returns evaluation metrics,
    #     response times, updated results DataFrame, and best configuration as a string.

    #     Args:
    #         num_folds (int): Number of folds for cross-validation.
    #         groups (Dict[int, List[str]]): Dictionary with fold number as
    #             key and list of query IDs as value.
    #         queries (pd.DataFrame): DataFrame containing query information.
    #         qrels (pd.DataFrame): DataFrame containing relevance information.
    #         model (Any): Model to be used for evaluation.
    #         model_type (str): Type of the model (e.g., "bm25", "lm").
    #         metric (str): Evaluation metric to use for tuning.
    #         results_df (pd.DataFrame): DataFrame to store evaluation results.
    #         metrics (Dict[str, float]): Dict to store aggregated evaluation metrics.
    #         unique_qids (pd.Series): Series containing unique query IDs.

    #     Returns:
    #         Tuple[Dict[str, float], List[float], pd.DataFrame, str]: Tuple containing
    #         the updated metrics,response times, updated results DataFrame, and best
    #         configuration as a string.
    #     """

    #     response_times = []
    #     best_config = None
    #     tuner = ModelTuner(self.model)

    #     for i in tqdm(range(num_folds), desc="Folds", total=num_folds):
    #         (
    #             train_queries,
    #             train_qids,
    #             training_qrels,
    #             train_qids_str,
    #         ) = get_training_data(groups, i, queries, qrels, unique_qids)

    #         fold_params_performance, best_config = tuner.tune_and_set_parameters(self.model,
    #             train_queries,
    #             train_qids,
    #             training_qrels,
    #             i + 1,
    #             model_type,
    #             metric,
    #             results_df,
    #         )

    #         results_df = pd.concat(
    #             [results_df, fold_params_performance], ignore_index=True
    #         )

    #         response_time, measures = self.run_and_evaluate_model(
    #             train_queries, train_qids_str, training_qrels, metrics
    #         )

    #         response_times.append(response_time)
    #         metrics = self.update_metrics(metrics, measures)

    #     return metrics, response_times, results_df, str(best_config)

    @staticmethod
    def update_metrics(metrics, measures):
        for metric in metrics:
            metrics[metric] += measures[metric]
        return metrics


create_index_from_trec_collection()
>>>>>>> 57dfe61 (Add configuration and build files, refactor cross-validation code)
