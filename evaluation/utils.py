from typing import Dict, List, Tuple, Any

import pandas as pd


def create_run_file(
    model: Any, queries: List[str], qids: List[str], run_name: str
) -> None:
    """
    Create a run file from the batch search results.

    Args:
        model (Any): The search model.
        queries (List[str]): List of query strings.
        qids (List[str]): List of query IDs.
        run_name (str): Name of the run file.

    Returns:
        None
    """
    batch_search_output = model.search(queries, qids)
    run = []
    for qid, search_results in batch_search_output.items():
        for result in search_results:
            row_str = f"{qid} 0 {result.docno} {result.rank} {result.score} {run_name}"
            run.append(row_str)
    with open(f"outputs/{run_name}.run", "w") as f:
        for l in run:  # noqa: E741
            f.write(l + "\n")


def create_run(
    model: Any, queries: List[str], qids: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Create a run dictionary from the batch search results.

    Args:
        model (Any): The search model.
        queries (List[str]): List of query strings.
        qids (List[str]): List of query IDs.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing the run results.
            The keys are query IDs, and the values are dictionaries where
            the keys are document IDs and the values are scores.
    """
    batch_search_output = model.search(queries, qids)

    return {
        qid: {result.docid: result.score for result in search_results}
        for qid, search_results in batch_search_output.items()
    }


def load_queries_and_qrels(
    queries_file: str, qrels_file: str
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]], pd.DataFrame]:
    """
    Loads the queries and qrels from the provided files.

    Args:
        queries_file (str): Path to the queries file.
        qrels_file (str): Path to the qrels file.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Dict[str, int]], pd.DataFrame] : A tuple
        containing three elements: the queries, the qrels in dictionary format,
        and the qrels in DataFrame format.
    """

    queries = pd.read_csv(queries_file, sep=" ", names=["qid", "query"])
    qrel_dict = {}
    qrel_list = []

    with open(qrels_file, "r") as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            if qid not in qrel_dict:
                qrel_dict[qid] = {}
            qrel_dict[qid][docid] = int(rel)

            # Add qrel to list
            qrel_list.append((qid, docid, (rel)))

    # Convert list to DataFrame
    # qrel_df = pd.DataFrame(qrel_list, columns=["qid", "docid", "rel"])

    return queries, qrel_dict  # type: ignore


def parse_run(lines: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Parse the lines of a run file and return the run as a dictionary.

    Args:
        lines (List[str]): The lines of the run file.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary representing the run,
            where each query ID is mapped to a dictionary
            containing document IDs as keys and corresponding scores as values.

    Raises:
        ValueError: If a duplicated document ID is found for a query.
    """
    run = {}

    for line in lines:
        query_id, _, doc_id, _, score, _ = line.strip().split()
        if query_id not in run:
            run[query_id] = {doc_id: float(score)}
        elif doc_id in run[query_id]:
            raise ValueError(f"Duplicated document ID {doc_id} for query {query_id}")
        else:
            run[query_id][doc_id] = float(score)

    return run
