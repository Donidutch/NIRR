import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any

import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from .models import BM25Model, LMModel
from evaluation.utils import evaluate_run
import evaluation.utils as utils


def run_single_fold(
    train_topics,
    test_topics,
    qrels,
    tuning_measure,
    index_variant,
    model_class,
    model_name,
    fold_index,
):
    try:
        model = model_class(index_variant["path"])
        best_measure, best_params = model.tune_parameters(
            train_topics, qrels, tuning_measure
        )

        if best_params is None:
            logging.error(
                f"No best param found for model {model_name} on fold {fold_index + 1}"
            )
            return None

        model.set_parameters(best_params)
        run = utils.create_run_file(test_topics, model)
        measures = evaluate_run(run, qrels)
        measures["Time"] = model.get_search_time()

        return {
            "Fold": fold_index + 1,
            "Model": model_name,
            "Parameters": best_params,
            **measures,
        }
    except Exception as e:
        logging.error(
            f"Error while training {model_name} on fold {fold_index + 1}: {str(e)}"
        )
        return None


def process_single_model(futures, results_df):
    for model, future, model_name, test_topics, fold_index in tqdm(
        futures, total=len(futures), desc="Evaluating models", unit="model"
    ):
        result = future.result()
        if result is not None:
            new_row_df = pd.DataFrame([result])
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)
    return results_df


def create_summary(all_results, models):
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


def run_cross_validation(
    index_variants: List[Dict[str, Any]],
    topic_file: str,
    qrels_file: str,
    tuning_measure: str = "ndcg_cut_10",
    k: int = 5,
) -> pd.DataFrame:
    models = {"BM25": BM25Model, "LM": LMModel}
    topics, qrels, qrels_df = utils.load_topics_and_qrels(topic_file, qrels_file)
    topics = topics.iloc[:5000]
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

            for fold_index, (train_index, test_index) in tqdm(
                enumerate(kf.split(topics)), total=k
            ):
                train_topics = topics.iloc[train_index]
                test_topics = topics.iloc[test_index]

                for model_name, model_class in models.items():
                    future = executor.submit(
                        run_single_fold,
                        train_topics,
                        test_topics,
                        qrels,
                        tuning_measure,
                        index_variant,
                        model_class,
                        model_name,
                        fold_index,
                    )
                    futures.append(
                        (model_class, future, model_name, test_topics, fold_index)
                    )

            results_df = process_single_model(futures, results_df)

        results_df.to_csv(
            f"./output/{index_variant['name']}_cross_validation_results.csv",
            index=False,
            float_format="%.3f",
        )

        summary = results_df.groupby("Model").mean()
        summary = summary.drop(columns=["Fold"])
        summary = summary.reset_index()
        summary["Index Variant"] = index_variant["name"]
        all_results.extend(summary.to_dict(orient="records"))

    summary_df = create_summary(all_results, models)
    return summary_df
