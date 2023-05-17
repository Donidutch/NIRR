#!/bin/zsh

mkdir -p pyserini/test_index
mkdir -p data/proc_data/train_tsv
mkdir -p data/proc_data/train_sample

INPUT_TREC="data/trec/fulldocs-new.trec"
INDEX_PATH="pyserini/indexes/"

QUERIES_INPUT_FILE="data/train/queries.doctrain.tsv"
QRELS_INPUT_FILE="data/train/msmarco-doctrain-qrels.tsv"

QUERIES_OUTPUT_FILE_SAMPLE="data/proc_data/train_sample/sample_queries.tsv"
QRELS_OUTPUT_FILE_SAMPLE="data/proc_data/train_sample/sample_qrels.tsv"
PATH_TO_SAMPLE_QID="data/train/msmarco-doctrain-queries-sample-qid.tsv"

INDEX_OUTPUT_FOLDER="pyserini/indexes/"
INDEX_OUTPUT_FOLDER_SUBSET="pyserini/test_index"

PATH_TO_DATASET="data/trec/"

KFOLDS=2
TUNING_MEASURE="ndcg_cut_10"

# Function to run all tests
run_all() {
    local use_samples=$1

    python main.py run_all_cmd $use_samples \
    "$QUERIES_INPUT_FILE" "$QUERIES_OUTPUT_FILE_SAMPLE" "$QRELS_INPUT_FILE" "$QRELS_OUTPUT_FILE_SAMPLE" \
    "$PATH_TO_DATASET" "$INDEX_OUTPUT_FOLDER" "$QUERIES_INPUT_FILE" "$QRELS_INPUT_FILE" "$INDEX_PATH" \
    "$KFOLDS" "$TUNING_MEASURE" "$PATH_TO_SAMPLE_QID"
}

# Check that an argument was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 [full|sample]"
    echo "   full: run tests with the full dataset"
    echo "   sample: run tests with a sample of queries"
    exit 1
fi

# Then, based on the provided argument, call the appropriate test function
case "$1" in
    full)
        # run_all false
        ;;
    sample)
        run_all true
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: $0 [full|sample]"
        exit 1
        ;;
esac
