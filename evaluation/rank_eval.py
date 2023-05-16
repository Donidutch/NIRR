import os
import pandas as pd
from typing import List, Optional, Tuple

from evaluation.cross_validation import run_cross_validation
from tqdm import tqdm


def create_summary(all_results: List[Tuple], models: List[str]) -> pd.DataFrame:
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
) -> pd.DataFrame:  # type: ignore
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
    return summary_df


if __name__ == "__main__":
    if not os.path.exists("output"):
        os.mkdir("output")
    queries_file = "data/proc_data/train_sample/sample_queries.tsv"
    qrels_file = "data/proc_data/train_sample/sample_qrels.tsv"
    kfolds = 2
    output_folder = "pyserini/indexes/"

    summary = rank_eval_main(
        queries_file, qrels_file, output_folder, kfolds, tuning_measure="ndcg_cut_10"
    )
    summary.to_csv("output/results.csv", index=False, float_format="%.3f")
