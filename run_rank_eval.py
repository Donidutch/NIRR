import os
from main import app

if __name__ == "__main__":
    if not os.path.exists("output"):
        os.mkdir("output")
    queries_file = "data/proc_data/train_sample/sample_queries.tsv"
    qrels_file = "data/proc_data/train_sample/sample_qrels.tsv"
    kfolds = 2
    output_folder = "pyserini/indexes/"

    app.dispatch(
        [
            "run_rank_eval_cmd",
            "--queries_file",
            queries_file,
            "--qrels_file",
            qrels_file,
            "--index_path",
            output_folder,
            "--kfolds",
            str(kfolds),
            "--tuning_measure",
            "ndcg_cut_10",
        ]
    )
