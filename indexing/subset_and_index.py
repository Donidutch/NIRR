import os

os.system(
    "python indexing/subset_data.py create-overlapping-subsets --num-docs 10000 --num-topics 10000 --input-file data/trec/fulldocs-new.trec --output-file proc_data/train_trec/subset_msmarco.trec --topic-input-file train/queries.doctrain.tsv --topic-output-file proc_data/train_tsv/subset_queries.doctrain.tsv --qrels-input-file train/msmarco-doctrain-qrels.tsv --qrels-output-file proc_data/train_tsv/subset_msmarco-doctrain-qrels.tsv"
)

# Run index_opt.py command
os.system("python indexing/index.py proc_data/train_trec/ --output-folder index/")
