import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from evaluation.cross_validation import run_cross_validation
from tqdm import tqdm


def create_summary(all_results: List[Tuple], models: List[str]) -> pd.DataFrame:
    """
    Create a summary DataFrame from the list of results.

    Args:
        all_results (List[Tuple]): List of result tuples.
        models (List[str]): List of model names.

    Returns:
        pd.DataFrame: Summary DataFrame containing the results.
    """
    return pd.DataFrame(
        all_results,
        columns=[
            "Index",
            "Model",
            "Best Configuration",
            "NDCG",
            "MRR",
            "P@5",
            "P@10",
            "P@20",
            "Recall@5",
            "Recall@10",
            "Recall@20",
            "Mean Response Time",
        ],
    )


def rank_eval_main(
    topic_file: str,
    qrels_file: str,
    index_path: str,
    kfolds: Optional[int] = None,
    tuning_measure: Optional[str] = "ndcg_cut_10",
    index_variants: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Main function for running the rank evaluation.

    Args:
        topic_file (str): Path to the topic file.
        qrels_file (str): Path to the qrels file.
        index_path (str): Path to the index directory.
        kfolds (Optional[int]): Number of folds for cross-validation. If None, no
            cross-validation will be performed.
        tuning_measure (Optional[str]): Tuning measure for selecting
            the best configuration.
        index_variants (Optional[List[str]]): List of index variants to evaluate.
            If None, all variants will be evaluated.

    Returns:
        pd.DataFrame: Summary DataFrame containing the evaluation results.
    """
    if not os.path.exists("output"):
        os.mkdir("output")

    if index_variants is None:
        index_variants = [
            "full_index",
            # "stopwords_removed",
            # "stemming",
            # "stopwords_removed_stemming",
        ]

    index_dict = []

    for index_variant in index_variants:
        variant_dict = {
            "name": index_variant,
            "path": index_path + index_variant + "/",
        }
        index_dict.append(variant_dict)

    all_results = []

    for index_variant in tqdm(index_dict, desc="Index Variants", total=len(index_dict)):
        headline = "{0}".format(index_variant["name"])
        print("\n")
        print("#" * 10, headline, "#" * 20)

        models = ["bm25", "lm"]
        for model_type in models:
            print("Model: {0}".format(model_type))
            if kfolds is None:
                from evaluation.models import Model
                from evaluation.utils import create_run
                from evaluation.evaluate import evaluate_run

                # from evaluation.utils import load_queries_and_qrels
                searcher = Model(index_variant["path"], model_type=model_type)
                # queries, qrels = load_queries_and_qrels(queries_file, qrels_file)
                queries = pd.read_csv(topic_file, sep=" ", names=["qid", "query"])
                qrels_df = pd.read_csv(
                    qrels_file, sep=" ", names=["qid", "Q0", "docid", "rel"]
                )
                # queries = queries_df["query"].tolist()
                # # qids = queries_df["qid"].tolist()
                # queries = [str(query) for query in queries]s
                qids = queries["qid"]
                qids = [str(qid) for qid in qids]
                run = create_run(searcher, queries, qids)
                metrics = evaluate_run(run, qrels_df, metric=tuning_measure)
                print(metrics)
                best_config = "Not applicable"
                mean_response_time = None
            else:
                result = run_cross_validation(
                    topic_file,
                    qrels_file,
                    index_variant["path"],
                    kfolds,
                    model_type=model_type,
                    tuning_measure=tuning_measure,
                )
                best_config = result["best_config"]
                metrics = result["metrics"]
                mean_response_time = result["mean_response_time"]

            print("result:", metrics)
            all_results.append(
                (
                    index_variant["name"],
                    model_type,
                    best_config,
                    metrics.get("ndcg_cut_10", np.nan),
                    metrics.get("recip_rank", np.nan),
                    metrics.get("P_5", np.nan),
                    metrics.get("P_10", np.nan),
                    metrics.get("P_20", np.nan),
                    metrics.get("recall_5", np.nan),
                    metrics.get("recall_10", np.nan),
                    metrics.get("recall_20", np.nan),
                    mean_response_time,
                )
            )

    summary_df = create_summary(all_results, ["lm", "bm25"])
    summary_df.to_csv("output/results.csv", index=False, float_format="%.3f")
