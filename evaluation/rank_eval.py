import typer
import pandas as pd
import os
from . import kfold as kf

app = typer.Typer()


@app.command()
def main(
    topic_file: str = "proc_data/train_tsv/subset_queries.doctrain.tsv",
    qrels_file: str = "proc_data/train_tsv/subset_msmarco-doctrain-qrels.tsv",
    index_path: str = "index/",
    kfolds: int = 2,
    tuning_measure: str = "ndcg_cut_10",
):
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
        index_dict, topic_file, qrels_file, tuning_measure=tuning_measure, k=kfolds
    )

    summary_df = create_summary(all_results, models)
    summary_df.to_csv("./output/results.csv", index=False, float_format="%.3f")
    return summary_df


if __name__ == "__main__":
    app()
