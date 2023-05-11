import os
import pandas as pd
from typing import List, Optional, Tuple
from . import kfold as kf


def create_summary(all_results: List[Tuple], models: List[str]) -> pd.DataFrame:
    """
    Creates a summary of the ranking evaluation results.

    Args:
        all_results: A list of tuples containing the evaluation results for each index
        variant and ranking model.
        models: A list of ranking models used in the evaluation.

    Returns:
        A DataFrame containing the summarized results.
    """
    df = pd.DataFrame(
        all_results,
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
    df.to_csv("./output/results.csv", index=False, float_format="%.3f")
    return df


def rank_eval_main(
    topic_file: str,
    qrels_file: str,
    index_path: str,
    kfolds: Optional[int],
    tuning_measure: Optional[str] = "ndcg_cut_10",
) -> pd.DataFrame:
    """
    Main function for running the ranking evaluation.

    Args:
        topic_file: Path to the topic file.
        qrels_file: Path to the qrels file.
        index_path: Path to the index directory.
        kfolds: Number of folds for cross-validation.
        tuning_measure: The measure used for tuning the ranking models.

    Returns:
        A DataFrame containing the summarized results.
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
        variant_dict = {}
        variant_dict["name"] = index_variant
        variant_dict["path"] = index_path + index_variant + "/"
        index_dict.append(variant_dict)

    all_results, models = kf.run_cross_validation(
        index_dict, topic_file, qrels_file, tuning_measure=tuning_measure, k=kfolds  # type: ignore
    )

    summary_df = create_summary(all_results, models)  # type: ignore
    summary_df.to_csv("./output/results.csv", index=False, float_format="%.3f")
    return summary_df
