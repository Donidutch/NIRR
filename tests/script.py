import pytrec_eval
import numpy as np
from typing import Dict
from pyserini.search import LuceneSearcher

queries_file = "data/proc_data/train_sample/sample_queries.tsv"
qrels_file = "data/proc_data/train_sample/sample_qrels.tsv"


def create_dummy_qrels() -> Dict[str, Dict[str, int]]:
    return {
        "1": {"doc1": 1, "doc2": 0, "doc3": 1},
        "2": {"doc1": 0, "doc2": 1, "doc3": 0},
    }


def create_dummy_run() -> Dict[str, Dict[str, float]]:
    return {
        "1": {"doc1": 1.2, "doc2": 0.8, "doc3": 1.5},
        "2": {"doc1": 0.1, "doc2": 1.0, "doc3": 0.5},
    }


def runtest(qrels_file, run_file):
    qrels = pytrec_eval.parse_qrel(qrels_file)
    run = pytrec_eval.parse_run(run_file)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map"})
    results = evaluator.evaluate(run)
    print(results)


def create_run_file(model, queries, qids, run_name):
    batch_search_output = model.search(queries, qids)
    run = []
    for qid, search_results in batch_search_output.items():
        for result in search_results:
            row_str = f"{qid} 0 {result.docno} {result.rank} {result.score} {run_name}"
            run.append(row_str)
    with open(f"outputs/{run_name}.run", "w") as f:
        for l in run:
            f.write(l + "\n")


def create_run_file2(queries, qids, run_name):
    searcher = LuceneSearcher("pyserini/indexes/full_index/")
    BM25 = searcher.set_bm25(0.9, 0.4)
    batch_search_output = BM25.search(queries, qids)


def evaluate_run(
    run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    metric: set = {"map", "ndcg"},
) -> Dict[str, float]:
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metric)
    results = evaluator.evaluate(run)

    measures = {
        measure: np.mean(
            [query_measures.get(measure, 0) for query_measures in results.values()]
        )
        for measure in metric
    }
    return measures


def main():
    dummy_qrels = create_dummy_qrels()
    dummy_run = create_dummy_run()

    print("Dummy Qrels:")
    for qid, qrels_docs in dummy_qrels.items():
        print(f"{qid}: {qrels_docs}")

    print("\nDummy Run:")
    for qid, run_docs in dummy_run.items():
        print(f"{qid}: {run_docs}")

    evaluation_results = evaluate_run(dummy_run, dummy_qrels)
    print("\nEvaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
