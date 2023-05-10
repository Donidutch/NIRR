import logging
import time
from abc import ABC, abstractmethod

import numpy as np
from pyserini.search import LuceneSearcher
import utils


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
            "mu": [500],
            # "mu": [500, 1000, 1500, 2000],
        }
        best_measure = 0
        best_params = {"mu": 1000}
        try:
            for mu in tuning_params["mu"]:
                self.set_parameters({"mu": mu})
                run = utils.create_run_file(train_topics, self)
                measures = utils.evaluate_run(run, qrels)  # type: ignore
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
            "k1": [1.0, 1.7],
            "b": [0.75, 0.95],
        }

        # tuning_params = {
        #     "k1": [1.2,  1.7],
        #     "b":  [0.75, 0.95],
        #     # "k1": [1.0, 1.2, 1.5, 1.7, 2.0],
        #     # "b": [0.65, 0.75, 0.85, 0.95],
        # }
        best_measure = 0
        best_params = {"k1": 1.0, "b": 0.65}

        for k1 in tuning_params["k1"]:
            for b in tuning_params["b"]:
                try:
                    self.set_parameters({"k1": k1, "b": b})
                except Exception as e:
                    logging.error(f"Error while setting parameters for BM25: {str(e)}")
                    continue

                try:
                    run = utils.create_run_file(train_topics, self)
                except Exception as e:
                    logging.error(f"Error while creating run file for BM25: {str(e)}")
                    continue

                try:
                    measures = utils.evaluate_run(run, qrels)
                    if measures[tuning_measure] > best_measure:
                        best_measure = measures[tuning_measure]
                        best_params = {"k1": k1, "b": b}
                except Exception as e:
                    logging.error(f"Error while evaluating run for BM25: {str(e)}")
                    continue
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
