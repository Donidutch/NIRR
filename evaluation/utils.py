from typing import Dict, Tuple

import pandas as pd


def create_run_file(model, queries, qids, run_name):
    batch_search_output = model.search(queries, qids)
    run = []
    for qid, search_results in batch_search_output.items():
        for result in search_results:
            row_str = f"{qid} 0 {result.docno} {result.rank} {result.score} {run_name}"
            run.append(row_str)
    with open(f"outputs/{run_name}.run", "w") as f:
        for l in run:  # noqa: E741
            f.write(l + "\n")


def create_run(model, queries, qids):
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


def parse_run(lines):
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
