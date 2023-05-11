import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm


from evaluation.models import BM25Model, LMModel

from . import utils


def run_single_fold(
    train_topics: pd.DataFrame,
    test_topics: pd.DataFrame,
    qrels: Dict[str, Dict[str, int]],
    tuning_measure: str,
    index_variant: Dict[str, Any],
    model_class: Any,
    model_name: str,
    fold_index: int,
) -> Dict[str, Any]:
    """
    Train and evaluate a single model on one fold of the cross-validation.

    Args:
        train_topics: The training topic data for the current fold.
        test_topics: The testing topic data for the current fold.
        qrels: The ground truth relevance judgments.
        tuning_measure: The measure used for tuning the ranking models.
        index_variant: A dictionary containing information about the index variant.
        model_class: The class of the ranking model.
        model_name: The name of the ranking model.
        fold_index: The index of the current fold.

    Returns:
        A dictionary containing the results of the evaluation.
    """
    try:
        model = model_class(index_variant["path"])
        best_measure, best_params = model.tune_parameters(
            train_topics, qrels, tuning_measure
        )

        if best_params is None:
            logging.error(
                f"No best param found for model {model_name} on fold {fold_index + 1}"
            )
            return None  # type: ignore

        model.set_parameters(best_params)
        run = utils.create_run_file(test_topics, model)
        measures = utils.evaluate_run(run, qrels)  # type: ignore
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
        return None  # type: ignore


def process_single_model(
    futures: List[Tuple[Any, Future, str, pd.DataFrame, int]], results_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Process the results of the evaluation for each ranking model and fold.

    Args:
        futures: A list of tuples containing the ranking model, the future object,
                 the model name, the test topics, and the fold index.
        results_df: The DataFrame containing the evaluation results.

    Returns:
        A DataFrame with the updated evaluation results.
    """
    for model, future, model_name, test_topics, fold_index in tqdm(
        futures, total=len(futures), desc="Evaluating models", unit="model"
    ):
        result = future.result()
        if result is not None:
            new_row_df = pd.DataFrame([result])
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)
    return results_df


def run_cross_validation(
    index_variants: List[Dict[str, Any]],
    topic_file: str,
    qrels_file: str,
    tuning_measure: str = "ndcg_cut_10",
    k: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run cross-validation for different ranking models on multiple index variants.

    Args:
        index_variants: A list of dictionaries containing information about index variants.
        topic_file: The path to the topic file.
        qrels_file: The path to the qrels file.
        tuning_measure: The measure used for tuning the ranking models (default is "ndcg_cut_10").
        k: The number of folds for cross-validation (default is 5).

    Returns:
        A tuple containing a list of dictionaries with the results of the cross-validation,
        and a dictionary with the ranking models.
    """
    models = {"BM25": BM25Model, "LM": LMModel}
    topics, qrels, qrels_df = utils.load_topics_and_qrels(topic_file, qrels_file)
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
        summary = results_df
        summary = summary.drop(columns=["Fold"])
        summary = summary.reset_index()
        summary["Index Variant"] = index_variant["name"]
        all_results.extend(summary.to_dict(orient="records"))

    return all_results, models
