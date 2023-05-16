from typing import Optional
import typer
import os
from evaluation.rank_eval import rank_eval_main  # Add this import

app = typer.Typer()


@app.command()
def create_sample_queries_cmd(
    sample_qid_path: str,
    qrels_input_path: str,
    queries_input_path: str,
    queries_output_path: str,
    qrels_output_path: str,
) -> None:
    from indexing.subset_data import create_sample_queries

    create_sample_queries(
        sample_qid_path,
        qrels_input_path,
        queries_input_path,
        queries_output_path,
        qrels_output_path,
    )


@app.command()
def build_indexes_cmd(
    path_to_dataset: str,
    output_folder: str,
) -> None:
    from indexing.index import build_all_indexes

    if os.path.exists(output_folder):
        print("Index folder already exists. Skipping indexing.")
        print(output_folder)
        return
    else:
        os.mkdir(output_folder)
        build_times = build_all_indexes(path_to_dataset, output_folder)
        print("Build times:")
        for name, build_time in build_times.items():
            print(f"{name}: {build_time:.2f} seconds")


@app.command()
def run_rank_eval_cmd(
    queries_file: str,
    qrels_file: str,
    index_path: str,
    kfolds: int,
    tuning_measure: str = "ndcg_cut_10",
) -> None:
    rank_eval_main(queries_file, qrels_file,
                   index_path, kfolds, tuning_measure)


@app.command()
def run_all_cmd(
    use_samples: bool,
    queries_input_file: str = "data/train/queries.doctrain.tsv",
    queries_output_file_sample: str = "data/train/queries.doctrain.sample.tsv",
    qrels_input_file: str = "data/train/msmarco-doctrain-qrels.tsv",
    qrels_output_file_sample: str = "data/train/msmarco-doctrain-qrels.sample.tsv",
    path_to_dataset: str = "data/trec/",
    indexes_path: str = "pyserini/indexes/",
    kfolds: int = 2,
    tuning_measure: str = "ndcg_cut_10",
    sample_qid_path: Optional[str] = None,
) -> None:
    if use_samples:
        if not sample_qid_path:
            raise ValueError(
                "When using samples, 'sample_qid_path' cannot be None.")
        create_sample_queries_cmd(
            sample_qid_path,
            qrels_input_file,
            queries_input_file,
            queries_output_file_sample,
            qrels_output_file_sample,
        )
        queries_file = queries_output_file_sample
        qrels_file = qrels_output_file_sample
    else:
        queries_file = queries_input_file
        qrels_file = qrels_input_file

    build_indexes_cmd(path_to_dataset, indexes_path)
    run_rank_eval_cmd(queries_file, qrels_file,
                      indexes_path, kfolds, tuning_measure)


if __name__ == "__main__":
    app()
