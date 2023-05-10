from indexing.subset_data import create_subsets
from indexing.index import build
from evaluation.rank_eval import main as rank_eval_main


def main():
    # Define your parameters
    num_docs = 10000
    num_topics = 10000
    input_file = "data/trec/fulldocs-new.trec"
    output_file = "data/proc_data/train_trec/subset_msmarco.trec"
    topic_input_file = "data/train/queries.doctrain.tsv"
    topic_output_file = "data/proc_data/train_tsv/subset_queries.doctrain.tsv"
    qrels_input_file = "data/train/msmarco-doctrain-qrels.tsv"
    qrels_output_file = "data/proc_data/train_tsv/subset_msmarco-doctrain-qrels.tsv"

    # Call create_subsets function
    create_subsets(
        num_docs,
        num_topics,
        input_file,
        output_file,
        topic_input_file,
        topic_output_file,
        qrels_input_file,
        qrels_output_file,
    )

    # Define your parameters for build function
    path_to_dataset = "data/proc_data/train_trec/"
    output_folder = "index/"

    # Call build function
    build(path_to_dataset, output_folder)
    rank_eval_main(
        topic_file=topic_output_file,
        qrels_file=qrels_output_file,
        index_path=output_folder,
        kfolds=2,
        tuning_measure="ndcg_cut_10",
    )


if __name__ == "__main__":
    main()
