import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from models import BM25Model, LMModel

import utils


# logging.basicConfig(level=logging.ERROR)

logging.basicConfig(filename="cross_validation.log", level=logging.INFO)


def run_cross_validation(
    index_variants: List[Dict[str, Any]],
    topic_file: str,
    qrels_file: str,
    tuning_measure: str = "ndcg_cut_10",
    k: int = 2,
) -> None:
    models = {"BM25": BM25Model, "LM": LMModel}
    topics, qrels, qrels_df = utils.load_topics_and_qrels(topic_file, qrels_file)
    topics = topics.iloc[:3000]
    kf = KFold(n_splits=k)
    all_results = []
    for index_variant in index_variants:
        print(f"Running cross validation for index: {index_variant['name']}")
        results_df = pd.DataFrame(
            columns=[
                "Fold",
                "Model",
                "Parameters",
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
                "Time",
            ]
        )

        with ThreadPoolExecutor() as executor:
            futures = []
            i = 0
            for i, (train_index, test_index) in tqdm(
                enumerate(kf.split(topics)), total=k
            ):
                train_topics = topics.iloc[train_index]
                test_topics = topics.iloc[test_index]

                for model_name, model_class in tqdm(models.items(), total=len(models)):
                    model = model_class(index_variant["path"])
                    future = executor.submit(
                        model.tune_parameters, train_topics, qrels, tuning_measure
                    )
                    futures.append((model, future, model_name, test_topics))

            for model, future, model_name, test_topics in tqdm(
                futures, total=len(futures), desc="Evaluating models", unit="model"
            ):
                try:
                    best_measure, best_params = future.result()
                    if best_params is None:
                        logging.error(
                            f"No best param found for model {model_name} on fold {i+1}"
                        )
                        continue

                    model.set_parameters(best_params)
                    run = utils.create_run_file(test_topics, model)
                    measures = utils.evaluate_run(run, qrels)  # type: ignore

                    measures["Time"] = model.get_search_time()

                    new_row = {
                        "Fold": i + 1,
                        "Model": model_name,
                        "Parameters": best_params,
                        **measures,
                    }
                    new_row_df = pd.DataFrame([new_row])
                    results_df = pd.concat([results_df, new_row_df], ignore_index=True)
                except Exception as e:
                    logging.error(
                        f"Error while training {model_name} on fold {i+1}: {str(e)}"
                    )

        results_df.to_csv(
            f"./output/{index_variant['name']}_cross_validation_results.csv",
            index=False,
            float_format="%.4f",
        )

        print(f"Index Variant: {index_variant['name']}")
        for model_name in models.keys():
            mean_response_time = results_df.loc[results_df["Model"] == model_name][
                "Time"
            ].mean()
            row = [index_variant["name"], model_name]
            for measure, column_name in zip(
                [
                    "ndcg",
                    "ndcg_cut_5",
                    "ndcg_cut_10",
                    "ndcg_cut_20",
                    "P_5",
                    "P_10",
                    "P_20",
                    "recall_5",
                    "recall_10",
                    "recall_20",
                    "recip_rank",
                ],
                [
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
                ],
            ):
                if measure in results_df.columns:
                    measure_to_fetch = measure
                    score = results_df.loc[results_df["Model"] == model_name][
                        measure_to_fetch
                    ].mean()
                    row.append(score)
            row.append(mean_response_time)
            all_results.append(row)

        df = pd.DataFrame(
            all_results,
            columns=[
                "Index Variant",
                "Ranking Model",
                "NDCG",
                "NDCG@5",
                "NDCG@10",
                "NDCG@20",
                "MRR",
                "Precision@5",
                "Precision@10",
                "Precision@20",
                "Recall@5",
                "Recall@10",
                "Recall@20",
                "Mean Time",
            ],
        )

        df.to_csv("./output/results.csv", index=False, float_format="%.3f")


def main():
    if not os.path.exists("output"):
        os.mkdir("output")

    topic_file = "proc_data/train_tsv/subset_queries.doctrain.tsv"
    qrels_file = "proc_data/train_tsv/subset_msmarco-doctrain-qrels.tsv"
    index_path = "./index/"

    kfolds = 2

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
    run_cross_validation(
        index_dict, topic_file, qrels_file, tuning_measure="ndcg_cut_10", k=kfolds
    )


if __name__ == "__main__":
    main()
