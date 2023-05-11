import os
import subprocess

if not os.path.exists("pyserini/test_index/"):
    os.mkdir("pyserini/test_index/")

NUM_DOCS = 5000
NUM_TOPICS = 5000

INPUT_TREC = "data/trec/fulldocs-new.trec"
OUTPUT_TREC = "data/proc_data/train_trec/subset_msmarco.trec"
TOPIC_INPUT_FILE = "data/train/queries.doctrain.tsv"
TOPIC_OUTPUT_FILE = "data/proc_data/train_tsv/subset_queries.doctrain.tsv"
QRELS_INPUT_FILE = "data/train/msmarco-doctrain-qrels.tsv"
QRELS_OUTPUT_FILE = "data/proc_data/train_tsv/subset_msmarco-doctrain-qrels.tsv"

PATH_TO_DATASET = "data/proc_data/train_trec/"
OUTPUT_FOLDER = "pyserini/test_index/"

TOPIC_FILE = "data/proc_data/train_tsv/subset_queries.doctrain.tsv"
QRELS_FILE = "data/proc_data/train_tsv/subset_msmarco-doctrain-qrels.tsv"
INDEX_PATH = "pyserini/test_index/"

KFOLDS = 2
TUNING_MEASURE = "ndcg_cut_10"


def test_create_subsets():
    subprocess.run(
        [
            "python",
            "main.py",
            "create-subsets-cmd",
            INPUT_TREC,
            OUTPUT_TREC,
            str(NUM_DOCS),
            str(NUM_TOPICS),
            TOPIC_INPUT_FILE,
            TOPIC_OUTPUT_FILE,
            QRELS_INPUT_FILE,
            QRELS_OUTPUT_FILE,
        ]
    )


def test_build_indexes():
    subprocess.run(
        [
            "python",
            "main.py",
            "build-indexes-cmd",
            PATH_TO_DATASET,
            OUTPUT_FOLDER,
        ]
    )


def test_run_rank_eval():
    subprocess.run(
        [
            "python",
            "main.py",
            "run-rank-eval-cmd",
            TOPIC_FILE,
            QRELS_FILE,
            INDEX_PATH,
            "--kfolds",
            str(KFOLDS),
            "--tuning-measure",
            str(TUNING_MEASURE),
        ]
    )


def test_run_all():
    subprocess.run(
        [
            "python",
            "main.py",
            "run-all-cmd",
            str(NUM_DOCS),
            str(NUM_TOPICS),
            INPUT_TREC,
            OUTPUT_TREC,
            TOPIC_INPUT_FILE,
            TOPIC_OUTPUT_FILE,
            QRELS_INPUT_FILE,
            QRELS_OUTPUT_FILE,
            PATH_TO_DATASET,
            OUTPUT_FOLDER,
            TOPIC_FILE,
            QRELS_FILE,
            INDEX_PATH,
            str(KFOLDS),
            "--tuning-measure",
            str(TUNING_MEASURE),
        ]
    )


if __name__ == "__main__":
    # test_create_subsets()
    # test_build_indexes()
    # test_run_rank_eval()
    test_run_all()
