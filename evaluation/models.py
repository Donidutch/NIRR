import os
from pyserini.search.lucene import LuceneSearcher
from typing import List, Dict


class Model:
    def __init__(self, index_path: str, model_type: str = "bm25", k_hits: int = 10):
        """
        Initialize the Model class.

        Args:
            index_path (str): The path to the Lucene index.
            model_type (str, optional): The ranking model to use.
                Either "bm25" or "qld".
                Defaults to "bm25".
            k_hits (int, optional): The number of hits to retrieve per query.
                Defaults to 10.
        """
        self.searcher = LuceneSearcher(index_path)
        self.model_type = model_type
        self.k = k_hits
        self.k1 = 0.9
        self.b = 0.6
        self.mu = 1000

        if self.model_type == "bm25":
            self.searcher.set_bm25(self.k1, self.b)
        elif self.model_type == "lm":
            self.searcher.set_qld(self.mu)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def search(
        self, queries: List[str], qids: List[str]
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Perform batch search using the specified queries and query IDs.

        Args:
            queries (List[str]): The list of queries to search.
            qids (List[str]): The corresponding list of query IDs.

        Returns:
            Dict[str, List[Dict[str, float]]]: A dictionary mapping each query ID to a
                list of search results, where each result is represented
                as a dictionary with 'docid' and 'score' keys.
        """
        return self.searcher.batch_search(
            queries, qids, k=self.k, threads=os.cpu_count()  # type: ignore
        )

    def set_bm25_parameters(self, k1: float, b: float) -> None:
        """
        Set the parameters for the BM25 ranking model.

        Args:
            k1 (float): The k1 parameter value.
            b (float): The b parameter value.
        """
        self.searcher.set_bm25(k1, b)

    def set_qld_parameters(self, mu: int) -> None:
        """
        Set the parameters for the Query Likelihood (QLD) ranking model.

        Args:
            mu (int): The mu parameter value.
        """
        self.searcher.set_qld(mu)
