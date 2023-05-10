import collections
import os
import time
from typing import IO, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyterrier as pt
import pytrec_eval
from pyserini.index import IndexReader
from pyserini.search import LuceneSearcher, SimpleSearcher


def get_metric():
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
        "MRR",
    }
    return metrics


def index_dataset(path_to_dataset: str, index_path: str) -> None:
    """Indexes a dataset using the TRECCollectionIndexer.

    Args:
        path_to_dataset (str): The path to the dataset.
        index_path (str): The path where the index will be stored.

    Returns:
        None.
    """
    from pyterrier.index import TRECCollectionIndexer

    if not os.path.exists(index_path):
        indexer = TRECCollectionIndexer(index_path, verbose=True)
        indexer.index(path_to_dataset)


def measure_query_time(searcher: LuceneSearcher, query: str) -> float:
    """
    Measure the time taken to execute a query on a given searcher.

    Args:
        searcher (LuceneSearcher): The searcher to use for the query.
        query (str): The query string to execute.

    Returns:
        float: The time taken to execute the query in seconds.
    """
    start_time = time.time()
    searcher.search(query)
    end_time = time.time()
    return end_time - start_time


def get_size(path: str) -> int:
    """
    Recursively computes the total size of a file or directory.

    Args:
        path (str): Path to the file or directory.

    Returns:
        int: Total size in bytes.
    """
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_size(entry.path)
    return total


def measure_index_stats(index_path: str, queries: List[str]) -> Dict[str, Any]:
    """
    Measure statistics for an index, including number of documents, unique terms,
    total terms, and average query time.

    Args:
        index_path (str): The path to the index to measure.
        queries (List[str]): A list of queries to use for measuring the avg query time.

    Returns:
        Dict[str, Any]: A dict with keys for the measured statistics and their values.
    """
    searcher = LuceneSearcher(index_path)
    reader = IndexReader(index_path)

    # Collect index statistics
    num_docs = reader.stats()["documents"]
    num_terms = reader.stats()["unique_terms"]
    total_terms = reader.stats()["total_terms"]

    # Measure search time
    avg_search_time = sum(
        [measure_query_time(searcher, query) for query in queries]
    ) / len(queries)

    index_size = get_size(index_path)

    return {
        "num_docs": num_docs,
        "num_unique_terms": num_terms,
        "total_terms": total_terms,
        "index_size": index_size,
        "avg_search_time": avg_search_time,
    }


def load_dataset_and_index(dataset_file: str, index_path: str) -> Any:
    from pyterrier.index import TRECCollectionIndexer

    """
    Loads the index from the provided file path or index the dataset and load the index.

    Args:
        dataset_file (str): Path to the dataset.
        index_path (str): Path to save the index.

    Returns:
        pyterrier.index.IterDictIndexer : The loaded index.
    """
    if not os.path.exists(index_path):
        indexer = TRECCollectionIndexer(index_path, verbose=True)
        indexer.index(dataset_file)
    indexref = pt.IndexRef.of(index_path)  # type: ignore
    index = pt.IndexFactory.of(indexref)  # type: ignore
    return index


def load_queries(qrels_file: str, query_file: str) -> pd.DataFrame:
    """
    Loads the queries.

    Args:
        qrels_file (str): Path to the qrels file.
        query_file (str): Path to the query file.

    Returns:
        pd.DataFrame : DataFrame containing the queries.
    """
    queries_df = pt.io.read_qrels(qrels_file)  # type: ignore
    topics_df = pt.io.read_topics(query_file, format="singleline")  # type: ignore
    queries_df["query_len"] = topics_df["query"].map(len)
    return queries_df


def measure_average_query_time(searcher: LuceneSearcher, queries: List[str]) -> float:
    """
    Measures the average time taken to perform a search.

    Args:
        searcher (LuceneSearcher): The searcher instance to perform the search.
        queries (List[str]): The list of search queries.

    Returns:
        float : The average time taken to perform the search.
    """
    total_time = 0
    for query in queries:
        total_time += measure_query_time(searcher, query)
    return total_time / len(queries)


def measure_index_statistics(
    queries: List[str], index_variants: List[Dict]
) -> pd.DataFrame:
    results = []
    for variant in index_variants:
        index_path = variant["index_path"]

        # Initialize a searcher and index reader
        searcher = SimpleSearcher(index_path)
        reader = IndexReader(index_path)

        # Collect index statistics
        num_docs = reader.stats()["documents"]
        num_terms = reader.stats()["unique_terms"]
        total_terms = reader.stats()["total_terms"]

        # Measure index size and search time
        index_size = get_size(index_path)
        avg_search_time = measure_average_query_time(searcher, queries)

        results.append(
            {
                "name": variant["name"],
                "num_docs": num_docs,
                "num_unique_terms": num_terms,
                "total_terms": total_terms,
                "index_size": index_size,
                "avg_search_time": avg_search_time,
            }
        )

    # Create pandas DataFrame and save as CSV file
    df = pd.DataFrame(results)
    df.to_csv("index_statistics.csv", index=False)

    return df


def load_topics_and_qrels(
    topic_file: str, qrels_file: str
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]], pd.DataFrame]:
    """
    Loads the topics and qrels from the provided files.

    Args:
        topic_file (str): Path to the topic file.
        qrels_file (str): Path to the qrels file.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Dict[str, int]], pd.DataFrame] : A tuple
        containing three elements: the topics, the qrels in dictionary format,
        and the qrels in DataFrame format.
    """

    qrels = pd.read_csv(qrels_file, sep="\t", names=["qid", "q0", "docid", "rel"])
    topics = pd.read_csv(topic_file, sep="\t", names=["qid", "query"])
    qrel = {}
    with open(qrels_file, "r") as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            if qid not in qrel:
                qrel[qid] = {}
            qrel[qid][docid] = int(rel)
    return topics, qrel, qrels


def write_trec(
    writer: IO[str], qid: str, hits: List[Any], msmarco: bool = True
) -> None:
    """
    Writes the hits to the provided file-like object in TREC format.

    Args:
        writer (IO[str]): File-like object to write to.
        qid (str): Query ID.
        hits (List[Hit]): List of search results.
        msmarco (bool, optional): TREC file is in MSMARCO format or not. Default True.

    Returns:
        None
    """
    for i, hit in enumerate(hits):
        if msmarco:
            writer.write(f"{qid}\t{hit.docid}\t{i+1}\n")
        else:
            writer.write(f"{qid} Q0 {hit.docid} {i+1} {hit.score} pyserini\n")


def load_documents(data_file: str) -> Dict[str, str]:
    """
    Load documents from a file.

    Args:
        data_file (str): Path to the file containing documents, one per line.

    Returns:
        Dict[str, str]: Dictionary containing document IDs and texts.
    """
    with open(data_file, "r", encoding="utf-8") as f:
        documents = {}
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                doc_id, doc_text = parts
                documents[doc_id] = doc_text
    return documents


def parse_run(lines: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Parses a run file and returns a dictionary representation.

    Args:
        lines (List[str]): List of lines in the run file.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing the run.
    """
    run = collections.defaultdict(dict)
    for line in lines:
        query_id, _, object_id, ranking, score = line.strip().split()

        assert object_id not in run[query_id]
        run[query_id][object_id] = float(score)

    return run


def create_run_file(topics: pd.DataFrame, model: Any) -> Dict[str, Dict[str, float]]:
    """
    Generates a run file for the given topics and model.

    Args:
        topics (pd.DataFrame): DataFrame containing the topics (query IDs and texts).
        model (Any): Model object with a `search` method that returns a list
        of hits for a query.

    Returns:
        Dict[str, Dict[str, float]] : A nested dict containing
        the run file (topic IDs and their hits).
    """
    run_lines = []
    for topic_id, topic_text in topics.values:
        hits = model.search(topic_text)

        for i, hit in enumerate(hits):
            rank = i + 1
            doc_id = hit.docid
            score = hit.score
            line = f"{topic_id} Q0 {doc_id} {rank} {score:.6f}\n"
            run_lines.append(line)

    return parse_run(run_lines)


def evaluate_run(
    run: Dict[str, Dict[str, int]], qrels: Dict[str, Dict[str, int]]
) -> Dict[str, float]:
    """
    Evaluate a run using the specified relevance judgments.

    Args:
        run (Dict[str, Dict[str, int]]): Dictionary containing the document
        rankings for each query.
        qrels (Dict[str, Dict[str, int]]): Dictionary containing the relevance
        judgments for each query.

    Returns:
        Dictionary containing the evaluation measures.
    """
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, pytrec_eval.supported_measures)
    results = evaluator.evaluate(run)
    metric = get_metric()
    measures = {
        measure: np.mean(
            [query_measures.get(measure, 0) for query_measures in results.values()]
        )
        for measure in metric
    }
    return measures


def print_all_contents(index_path: str) -> None:
    """
    Prints the contents of all documents in the given index.

    Args:
        index_path (str): Path to the index.

    Returns:
        None
    """
    index_reader = IndexReader(index_path)
    max_docid = index_reader.stats()["documents"]

    for docid in range(max_docid):
        lucene_docid = index_reader.convert_internal_docid_to_collection_docid(docid)

        content = index_reader.doc_raw(lucene_docid)
        print(f"DocID: {lucene_docid}, Content: {content}")
        # document = index_reader.doc(lucene_docid)
        # print(f"DocID: {document.docid()}, Content: {document.raw()}")
