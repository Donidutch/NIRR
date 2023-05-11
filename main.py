import typer
import os
from indexing.subset_data import create_subsets
from indexing.index import build_all_indexes
from evaluation.rank_eval import rank_eval_main

app = typer.Typer()


@app.command()
def create_subsets_cmd(
    input_trec: str,
    output_trec: str,
    num_docs: int,
    num_topics: int,
    topic_input_file: str,
    topic_output_file: str,
    qrels_input_file: str,
    qrels_output_file: str,
) -> None:
    """
    Command to create subsets of the input data.

    Args:
        input_trec: Path to the input TREC file.
        output_trec: Path to the output TREC file.
        num_docs: Number of documents to include in the subset.
        num_topics: Number of topics to include in the subset.
        topic_input_file: Path to the input topic file.
        topic_output_file: Path to the output topic file.
        qrels_input_file: Path to the input qrels file.
        qrels_output_file: Path to the output qrels file.
    """
    create_subsets(
        input_trec,
        output_trec,
        num_docs,
        num_topics,
        topic_input_file,
        topic_output_file,
        qrels_input_file,
        qrels_output_file,
    )


@app.command()
def build_indexes_cmd(
    path_to_dataset: str,
    output_folder: str,
) -> None:
    """
    Command to build all indexes.

    Args:
        path_to_dataset: Path to the dataset directory.
        output_folder: Path to the output folder.
    """
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    build_times = build_all_indexes(path_to_dataset, output_folder)
    print("Build times:")
    for name, build_time in build_times.items():
        print(f"{name}: {build_time:.2f} seconds")


@app.command()
def run_rank_eval_cmd(
    topic_file: str,
    qrels_file: str,
    index_path: str,
    kfolds: int,
    tuning_measure: str = "ndcg_cut_10",
) -> None:
    """
    Command to run the ranking evaluation.

    Args:
        topic_file: Path to the topic file.
        qrels_file: Path to the qrels file.
        index_path: Path to the index directory.
        kfolds: Number of folds for cross-validation.
        tuning_measure: The measure used for tuning the ranking models.
    """
    rank_eval_main(topic_file, qrels_file, index_path, kfolds, tuning_measure)  # type: ignore


@app.command()
def run_all_cmd(
    num_docs: int,
    num_topics: int,
    input_trec: str,
    output_trec: str,
    topic_input_file: str,
    topic_output_file: str,
    qrels_input_file: str,
    qrels_output_file: str,
    path_to_dataset: str,
    output_folder: str,
    topic_file: str,
    qrels_file: str,
    index_path: str,
    kfolds: int,
    tuning_measure: str = "ndcg_cut_10",
) -> None:
    """
    Command to run all commands (create subsets, build indexes,
    and run ranking evaluation) sequentially.

    Args:
        num_docs: Number of documents to include in the subset.
        num_topics: Number of topics to include in the subset.
        input_trec: Path to the input TREC file.
        output_trec: Path to the output TREC file.
        topic_input_file: Path to the input topic file.
        topic_output_file: Path to the output topic file.
        qrels_input_file: Path to the input qrels file.
        qrels_output_file: Path to the output qrels file.
        path_to_dataset: Path to the dataset directory.
        output_folder: Path to the output folder.
        topic_file: Path to the topic file.
        qrels_file: Path to the qrels file.
        index_path: Path to the index directory.
        kfolds: Number of folds for cross-validation (default is 2).
        tuning_measure: The measure used for tuning the ranking models
        (default is "ndcg_cut_10").
    """
    create_subsets_cmd(
        input_trec,
        output_trec,
        num_docs,
        num_topics,
        topic_input_file,
        topic_output_file,
        qrels_input_file,
        qrels_output_file,
    )
    build_indexes_cmd(path_to_dataset, output_folder)

    run_rank_eval_cmd(topic_file, qrels_file, index_path, kfolds, tuning_measure)


if __name__ == "__main__":
    app()
