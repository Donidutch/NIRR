import logging
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytrec_eval
from pyserini.search import LuceneSearcher
from sklearn.model_selection import KFold
from tqdm import tqdm

import utils

# logging.basicConfig(level=logging.ERROR)
metrics = {
    "ndcg",
    "ndcg_cut_5",
    "ndcg_cut_10",
    "ndcg_cut_20",
    "recip_rank",
    "P_5",
    "P_10",
    "P_20",
    "recall_5",
    "recall_10",
    "recall_20",
}


def evaluate_run(
    run: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]
) -> Dict[str, float]:
    """
    Evaluate a run against a set of qrels using standard TREC evaluation measures.

    Args:
        run (Dict[str, Dict[str, float]]): A dict with search results for each query.
        qrels (Dict[str, Dict[str, int]]): A dict containing qrels for each query.

    Returns:
        A dictionary containing the evaluation measures.
    """
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, pytrec_eval.supported_measures)
    results = evaluator.evaluate(run)

    measures = {
        measure: np.mean(
            [query_measures.get(measure, 0) for query_measures in results.values()]
        )
        for measure in metrics
    }
    return measures  # type: ignore


class Model(ABC):
    """
    Abstract base class for a retrieval model.
    All retrieval models should inherit from this class.
    """

    def __init__(self, index_path):
        """
        Constructor for Model class.

        Args:
            index_path (str): Path to the index directory.
        """
        self.searcher = LuceneSearcher(index_path)

    @abstractmethod
    def tune_parameters(self, train_topics, qrels, tuning_measure):
        """
        Abstract method to tune the model's hyperparameters.

        Args:
            train_topics (pd.DataFrame): Dataframe containing the training topics.
            qrels (Dict): Dictionary containing the query relevance judgments.
            tuning_measure (str): Evaluation metric to use for tuning the hyperparams.

        Returns:
            Tuple of best evaluation measure and best hyperparams found during tuning.
        """
        pass

    @abstractmethod
    def set_parameters(self, params):
        """
        Abstract method to set the model's hyperparameters.

        Args:
            params (Dict): Dictionary containing the hyperparameters to set.
        """
        pass

    @abstractmethod
    def search(self, query):
        """
        Abstract method to search the index for a given query.

        Args:
            query (str): Query string.

        Returns:
            Search results.
        """
        pass

    def get_search_time(self):
        """
        Abstract method to get the time taken to perform the last search.
        """
        pass


class LMModel(Model):
    """
    Retrieval model based on the Language Model.
    """

    def __init__(self, index_path):
        """
        Initializes an LMModel object.

        Args:
            index_path (str): Path to the index directory.
        """
        super().__init__(index_path)
        self.mu = 1000
        self.search_time = 0
        self.k = 50
        self.search_times = []

    def tune_parameters(self, train_topics, qrels, tuning_measure="ndcg_cut_10"):
        """
        Tune the LM hyperparameter mu using the specified tuning_measure.

        Args:
            train_topics (pd.DataFrame): DataFrame containing the training topics.
            qrels (Dict): Dictionary containing the query relevance judgments.
            tuning_measure (str): Evaluation metric to use for tuning the hyperparams.

        Returns:
            Tuple of the best evaluation measure and the best
            hyperparams found during tuning.
        """
        tuning_params = {
            "mu": [500, 1000, 1500, 2000],
        }
        best_measure = 0
        best_params = {"mu": 1000}
        try:
            for mu in tuning_params["mu"]:
                self.set_parameters({"mu": mu})
                run = utils.create_run_file(train_topics, self)
                measures = evaluate_run(run, qrels)
                if measures[tuning_measure] > best_measure:
                    best_measure = measures[tuning_measure]
                    best_params = {"mu": mu}
        except Exception as e:
            logging.error(f"Error while tuning parameters for LM: {str(e)}")
        return best_measure, best_params

    def set_parameters(self, params):
        """
        Sets the LM hyperparameter mu.

        Args:
            params (Dict): Dictionary containing the hyperparameters to set.
        """
        self.mu = params["mu"]
        self.searcher.set_qld(self.mu)

    def search(self, query):
        """
        Searches the index for a given query using the LM retrieval model.

        Args:
            query (str): Query string.

        Returns:
            Search results.
        """
        start_time = time.time()
        search_results = self.searcher.search(query, self.k)
        end_time = time.time()
        self.search_time = end_time - start_time
        self.search_times.append(self.search_time)
        return search_results

    def get_mean_search_time(self):
        """
        Computes the mean search time for the LM model.

        Returns:
            The mean search time for the LM model.
        """
        return np.mean(self.search_times)

    def get_search_time(self):
        """
        Returns the search time for the last query.

        Returns:
            The search time for the last query.
        """
        return self.search_time


class BM25Model(Model):
    """
    Retrieval model based on the BM25 algorithm.
    """

    def __init__(self, index_path):
        """
        Constructor for BM25Model class.

        Args:
            index_path (str): Path to the index directory.
        """
        super().__init__(index_path)
        self.k1 = 1.2
        self.b = 0.75
        self.search_time = 0
        self.search_times = []
        self.k = 50

    def tune_parameters(self, train_topics, qrels, tuning_measure="ndcg_cut_10"):
        """
        Tune the BM25 hyperparameters k1 and b using the specified tuning_measure.

        Args:
            train_topics (pd.DataFrame): Dataframe containing the training topics.
            qrels (Dict): Dictionary containing the query relevance judgments.
            tuning_measure (str): Evaluation metric to use for tuning the hyperpars.

        Returns:
            Tuple of best evaluation measure and best hyperparams found during tuning.
        """
        tuning_params = {
            "k1": [1.0, 1.2, 1.5, 1.7, 2.0],
            "b": [0.65, 0.75, 0.85, 0.95],
        }
        best_measure = 0
        best_params = {"k1": 1.0, "b": 0.65}
        try:
            for k1 in tuning_params["k1"]:
                for b in tuning_params["b"]:
                    self.set_parameters({"k1": k1, "b": b})
                    run = utils.create_run_file(train_topics, self)
                    measures = evaluate_run(run, qrels)
                    if measures[tuning_measure] > best_measure:
                        best_measure = measures[tuning_measure]
                        best_params = {"k1": k1, "b": b}
        except Exception as e:
            logging.error(f"Error while tuning parameters for BM25: {str(e)}")
        return best_measure, best_params

    def set_parameters(self, params):
        """
        Set the BM25 hyperparameters k1 and b.

        Args:
            params (Dict): Dictionary containing the hyperparameters to set.
        """
        self.k1 = params["k1"]
        self.b = params["b"]
        self.searcher.set_bm25(self.k1, self.b)

    def search(self, query):
        """
        Search the index for a given query using the BM25 retrieval model.

        Args:
            query (str): Query string.

        Returns:
            Search results.
        """
        start_time = time.time()
        search = self.searcher.search(query, self.k)
        end_time = time.time()
        self.search_time = end_time - start_time
        self.search_times.append(self.search_time)
        return search

    def get_mean_search_time(self):
        """
        Calculate the mean search time across all queries.

        Returns:
            Mean search time.
        """
        return np.mean(self.search_times)

    def get_search_time(self):
        """
        Get the search time for the most recent query.

        Returns:
            Search time.
        """
        return self.search_time


# logging.basicConfig(filename="cross_validation.log", level=logging.INFO)


def run_cross_validation(
    index_variants: List[Dict[str, Any]],
    topic_file: str,
    qrels_file: str,
    tuning_measure: str = "ndcg_cut_10",
    k: int = 5,
) -> None:
    """
    Perform cross-validation on the specified index variants and models using
    the provided topics and qrels.

    Args:
       index_variants: List[Dict[str, str]]
            A list of dictionaries containing information about each index variant.
            Each dictionary should contain the following keys:
                - name (str): The name of the index variant.
                - path (str): The path to the index directory.
        topic_file (str): Path to the file containing the topics to use for cross-val.
        qrels_file (str): Path to the file containing the relevance judgments.
        tuning_measure (str): The metric to use for hyperparameter tuning.
        k (int): The number of folds to use for cross-validation.

    """

    models = {"BM25": BM25Model, "LM": LMModel}
    topics, qrels, qrels_df = utils.load_topics_and_qrels(topic_file, qrels_file)
    topics = topics.iloc[:2000]
    kf = KFold(n_splits=k)

    for index_variant in index_variants:
        print(f"Running cross validation for index: {index_variant['name']}")

        results_df = pd.DataFrame(
            columns=[
                "Fold",
                "Model",
                "Parameters",
                "ndcg",
                "ndcg_cut_5",
                "ndcg_cut_10",
                "ndcg_cut_20",
                "recip_rank",
                "P_5",
                "P_10",
                "P_20",
                "recall_5",
                "recall_10",
                "recall_20",
                "Time",
            ]
        )

        with ThreadPoolExecutor() as executor:
            futures = []
            i = 0
            for i, (train_index, test_index) in tqdm(
                enumerate(kf.split(topics)), total=k
            ):
                train_topics = topics.iloc[train_index]
                test_topics = topics.iloc[test_index]

                for model_name, model_class in tqdm(models.items(), total=len(models)):
                    model = model_class(index_variant["path"])
                    future = executor.submit(
                        model.tune_parameters, train_topics, qrels, tuning_measure
                    )
                    futures.append((model, future, model_name, test_topics))

            for model, future, model_name, test_topics in tqdm(
                futures, total=len(futures), desc="Evaluating models", unit="model"
            ):
                try:
                    best_measure, best_params = future.result()
                    if best_params is None:
                        logging.error(
                            f"No best param found for model {model_name} on fold {i+1}"
                        )
                        continue

                    model.set_parameters(best_params)
                    run = utils.create_run_file(test_topics, model)
                    measures = evaluate_run(run, qrels)

                    measures["Time"] = model.get_search_time()

                    new_row = {
                        "Fold": i + 1,
                        "Model": model_name,
                        "Parameters": best_params,
                        **measures,
                    }
                    new_row_df = pd.DataFrame([new_row])
                    results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                except Exception as e:
                    logging.error(
                        f"Error while training {model_name} on fold {i+1}: {str(e)}"
                    )

        print(f"Index Variant: {index_variant['name']}")
        results = []
        for model_name in models.keys():
            mean_response_time = results_df.loc[
                (results_df["Model"] == model_name) & (results_df["Fold"] == i + 1)
            ]["Time"].mean()
            row = [index_variant["name"], model_name]
            for measure in metrics:
                if measure in results_df.columns:
                    if measure == "MRR":
                        measure_to_fetch = "recip_rank"
                    else:
                        measure_to_fetch = measure

                    score = results_df.loc[
                        (results_df["Model"] == model_name)
                        & (results_df["Fold"] == i + 1)
                    ][measure_to_fetch].mean()
                    row.append(score)
            row.append(mean_response_time)
            results.append(row)

        df = pd.DataFrame(
            results,
            columns=[
                "Index Variant",
                "Ranking Model",
                "NDCG",
                "NDCG@5",
                "NDCG@10",
                "NDCG@20",
                "Precision@5",
                "Precision@10",
                "Precision@20",
                "Recall@5",
                "Recall@10",
                "Recall@20",
                "MRR",
                "Mean Time",
            ],
        )

        df.to_csv("./output/results.csv", index=False, float_format="%.4f")

        results_df.to_csv(
            f"./output/{index_variant['name']}_cross_validation_results.csv",
            index=False,
            float_format="%.4f",
        )


def main():
    if not os.path.exists("output"):
        os.mkdir("output")

    topic_file = "proc_data/train_tsv/subset_queries.doctrain.tsv"
    qrels_file = "proc_data/train_tsv/subset_msmarco-doctrain-qrels.tsv"
    index_path = "./index/"

    kfolds = 5

    index_variants = [
        "full_index",
        "stopwords_removed",
        "stemming",
        "stopwords_removed_stemming",
    ]

    index_dict = []

    for index_variant in index_variants:
        variant_dict = {}
        variant_dict["name"] = index_variant
        variant_dict["path"] = index_path + index_variant + "/"
        index_dict.append(variant_dict)
    run_cross_validation(
        index_dict, topic_file, qrels_file, tuning_measure="ndcg_cut_10", k=kfolds
    )


if __name__ == "__main__":
    main()
