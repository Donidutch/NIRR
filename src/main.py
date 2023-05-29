from typing import Optional
import typer
import os

app = typer.Typer()
ExpansionMethods = ["pseudo_relevance_feedback", "word2vec", "bert"]

# ExpansionMethods = ["pseudo_relevance_feedback"]


@app.command()
def create_sample_queries_cmd(
    sample_qid_path: str,
    qrels_input_path: str,
    queries_input_path: str,
    queries_output_path: str,
    qrels_output_path: str,
) -> None:
    from preprocessing.subset_data import create_sample_queries

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
    corpus_path: str,
    kfolds: int,
    tuning_measure: str = "ndcg_cut_10",
    expansion_method: Optional[str] = None,
) -> None:
    from evaluation.rank_eval import rank_eval_main

    rank_eval_main(
        queries_file,
        qrels_file,
        index_path,
        corpus_path,
        expansion_method,
        kfolds,
        tuning_measure,
    )


@app.command()
def run_rank_eval_single_index_cmd(
    queries_file: str,
    qrels_file: str,
    index_path: str,
    tuning_measure: str = "ndcg_cut_10",
) -> None:
    from evaluation.rank_eval import rank_eval_main

    rank_eval_main(queries_file, qrels_file, index_path, None, tuning_measure)


@app.command()
def run_all_cmd(
    use_samples: bool,
    queries_input_file: str = "data/train/queries.doctrain.tsv",
    queries_output_file_sample: str = "data/train/queries.doctrain.sample.tsv",
    qrels_input_file: str = "data/train/msmarco-doctrain-qrels.tsv",
    qrels_output_file_sample: str = "data/train/msmarco-doctrain-qrels.sample.tsv",
    corpus_path: str = "data/msmarco-docs.tsv",
    path_to_dataset: str = "data/trec/",
    indexes_path: str = "data/indexes/pyserini/indexes/",
    kfolds: int = 2,
    tuning_measure: str = "ndcg_cut_10",
    sample_qid_path: Optional[str] = None,
) -> None:
    if use_samples:
        if not sample_qid_path:
            raise ValueError("When using samples, 'sample_qid_path' cannot be None.")
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
    for expansion_method in ExpansionMethods:
        run_rank_eval_cmd(
            queries_file,
            qrels_file,
            indexes_path,
            corpus_path,
            kfolds,
            tuning_measure,
            expansion_method,
        )


@app.command()
def run_test_all_cmd(
    use_samples: bool,
    queries_input_file: str = "data/lab/lab_topics.csv",
    queries_output_file_sample: str = "data/lab/lab_topics.sample.csv",
    qrels_input_file: str = "data/lab/lab_qrels.csv",
    qrels_output_file_sample: str = "data/lab/lab_qrels.sample.csv",
    corpus_path: str = "data/lab/",
    path_to_dataset: str = "data/lab/trec/",
    indexes_path: str = "indexes/pyserini/test_index/",
    kfolds: int = 2,
    tuning_measure: str = "ndcg_cut_10",
    sample_qid_path: Optional[str] = None,
) -> None:
    if use_samples:
        if not sample_qid_path:
            raise ValueError("When using samples, 'sample_qid_path' cannot be None.")
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
    for expansion_method in ExpansionMethods:
        run_rank_eval_cmd(
            queries_file,
            qrels_file,
            indexes_path,
            corpus_path,
            kfolds,
            tuning_measure,
            expansion_method,
        )


@app.command()
def create_pyterrie_index(
    path_to_dataset: str,
    output_folder: str,
) -> None:
    from indexing.index_pyterrier import create_indices

    if os.path.exists(output_folder):
        print("Index folder already exists. Skipping indexing.")
        print(output_folder)
        return
    else:
        # os.mkdir(output_folder)
        create_indices(path_to_dataset, output_folder)


if __name__ == "__main__":
    app()
