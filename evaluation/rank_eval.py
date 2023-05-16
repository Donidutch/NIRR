import os
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
    kfolds: Optional[int],
    tuning_measure: Optional[str] = "ndcg_cut_10",
) -> pd.DataFrame:
    """
    Main function for running the rank evaluation.

    Args:
        topic_file (str): Path to the topic file.
        qrels_file (str): Path to the qrels file.
        index_path (str): Path to the index directory.
        kfolds (Optional[int]): Number of folds for cross-validation.
        tuning_measure (Optional[str]): Tuning measure for selecting
            the best configuration.

    Returns:
        pd.DataFrame: Summary DataFrame containing the evaluation results.
    """
    if not os.path.exists("output"):
        os.mkdir("output")

    index_variants = [
        "full_index",
        "stopwords_removed",
        "stemming",
        "stopwords_removed_stemming",
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
            result = run_cross_validation(
                topic_file,
                qrels_file,
                index_variant["path"],
                kfolds,
                model_type=model_type,
                tuning_measure=tuning_measure,
            )
            print("result:", result)
            best_config = result["best_config"]
            metrics = result["metrics"]
            mean_response_time = result["mean_response_time"]

            all_results.extend(
                [
                    (
                        index_variant["name"],
                        model_type,
                        best_config,
                        metrics["ndcg"],
                        metrics["recip_rank"],
                        metrics["P_5"],
                        metrics["P_10"],
                        metrics["P_20"],
                        metrics["recall_5"],
                        metrics["recall_10"],
                        metrics["recall_20"],
                        mean_response_time,
                    )
                ]
            )
    summary_df = create_summary(all_results, ["lm", "bm25"])
    summary_df.to_csv("output/results.csv", index=False, float_format="%.3f")
