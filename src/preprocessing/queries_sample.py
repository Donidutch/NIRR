import pandas as pd
import sys

sys.path.append(".")
sample_qid = pd.read_csv("data/train/msmarco-doctrain-queries-sample-qid.tsv")
qrels_df = pd.read_csv(
    "data/train/msmarco-doctrain-qrels.tsv",
    sep=" ",
    names=["qid", "q0", "docid", "rel"],
)
queries_df = pd.read_csv(
    "data/train/queries.doctrain.tsv", sep="\t", names=["qid", "query"]
)


queries_subset_df = queries_df[queries_df["qid"].isin(sample_qid.values[:, 0])]


qrels_subset_df = qrels_df[qrels_df["qid"].isin(sample_qid.values[:, 0])]

queries_subset_df.to_csv(
    "data/proc_data/train_sample/sample_queries.tsv", sep=" ", index=False
)
qrels_subset_df.to_csv(
    "data/proc_data/train_sample/sample_qrels.tsv", sep=" ", index=False
)
