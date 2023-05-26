import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from evaluation.cross_validation import run_cross_validation
from evaluation.query_expansion import (
    expand_query_bert,
    expand_query_word2vec,
    pseudo_relevance_feedback,
)


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


def read_and_preprocess_files(
    topic_file: str, qrels_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test = True
    if test:
        queries = pd.read_csv(topic_file, sep=",")
        qrels_df = pd.read_csv(qrels_file, sep=",")

        qrels_df.rename(
            columns={"label": "rel", "iteration": "Q0", "docno": "docid"}, inplace=True
        )
        new_cols = ["qid", "Q0", "docid", "rel"]
        qrels_df = qrels_df.reindex(columns=new_cols)
    else:
        queries = pd.read_csv(topic_file, sep=" ", names=["qid", "query"])
        qrels_df = pd.read_csv(qrels_file, sep=" ", names=["qid", "q0", "docid", "rel"])

    return queries, qrels_df


def get_results(
    index_variant,
    model_type,
    tuning_measure,
    kfolds,
    topic_file,
    qrels_file,
    expansion_method,
    corpus_path,
    index_path,
    num_terms,
    num_docs,
) -> Tuple:
    if kfolds is None:
        from evaluation.evaluate import evaluate_run
        from evaluation.metrics import get_metric
        from evaluation.model_evaluation import create_run
        from evaluation.models import Model
        from gensim.models import KeyedVectors
        from sentence_transformers import SentenceTransformer

        searcher = Model(index_variant["path"], model_type=model_type)
        queries, qrels_df = read_and_preprocess_files(topic_file, qrels_file)
        qids = [str(qid) for qid in queries["qid"]]
        if expansion_method == "pseudo_relevance_feedback":
            expanded_queries = [
                pseudo_relevance_feedback(
                    index_variant["path"], str(query), num_docs, num_terms
                )
                for query in queries["query"]
            ]
        elif expansion_method == "word2vec":
            expansion_model = KeyedVectors.load_word2vec_format(
                "data/embedding/GoogleNews-vectors-negative300.bin", binary=True
            )

            expanded_queries = [
                expand_query_word2vec(expansion_model, str(query), num_terms)
                for query in queries["query"]
            ]
        elif expansion_method == "bert":
            expansion_model = SentenceTransformer("all-MiniLM-L6-v2")

            expanded_queries = [
                expand_query_bert(
                    expansion_model, str(query), num_terms, corpus=corpus_path
                )
                for query in queries["query"]
            ]
        else:
            expanded_queries = [str(query) for query in queries["query"]]

        run = create_run(searcher, expanded_queries, qids, index_path, corpus_path)
        metrics = evaluate_run(run, qrels_df, metric=get_metric(get_all_metrics=True))
        best_config = "Not applicable"
        mean_response_time = None

    else:
        print("Running Cross Validation")
        print("expansion_method", expansion_method)
        print(index_variant["path"])
        result = run_cross_validation(
            topic_file,
            qrels_file,
            index_variant["path"],
            corpus_path,
            kfolds,
            model_type=model_type,
            tuning_measure=tuning_measure,
            expansion_method=expansion_method,
        )
        best_config = result["best_config"]
        metrics = result["metrics"]
        mean_response_time = result["mean_response_time"]

    return best_config, metrics, mean_response_time


def rank_eval_main(
    topic_file: str,
    qrels_file: str,
    index_path: str,
    corpus_path: str,
    expansion_method: Optional[str] = None,
    kfolds: Optional[int] = None,
    tuning_measure: Optional[str] = "ndcg_cut_10",
    index_variants: Optional[List[str]] = None,
) -> None:
    all_results = []
    models = ["lm", "bm25"]
    models = ["bm25"]
    if not os.path.exists("output"):
        os.mkdir("output")

    if index_variants is None:
        index_variants = [
            # "full_index",
            # "stopwords_removed",
            # "stemming",
            "stopwords_removed_stemming",
        ]
    index_dict = [
        {"name": variant, "path": index_path + variant + "/"}
        for variant in index_variants
    ]

    for index_variant in tqdm(index_dict, desc="Index Variants", total=len(index_dict)):
        headline = "{0}".format(index_variant["name"])
        print("\n")
        print("#" * 10, headline, "#" * 20)

        for model_type in models:
            print("Model: {0}".format(model_type))
            best_config, metrics, mean_response_time = get_results(
                index_variant,
                model_type,
                tuning_measure,
                kfolds,
                topic_file,
                qrels_file,
                expansion_method,
                corpus_path,
                index_path,
                num_terms=10,
                num_docs=10,
            )

            if isinstance(metrics, dict):
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

    summary_df = create_summary(all_results, models)
    summary_df.to_csv(
        f"output/{expansion_method}.csv", index=False, float_format="%.3f"
    )
