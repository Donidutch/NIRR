import pandas as pd
import re


def remove_quotes(queries):
    cleaned_queries = []
    for query in queries:
        if isinstance(query, str):
            cleaned_queries.append(query.replace('"', ''))
        else:
            cleaned_queries.append(query)
    return cleaned_queries


def create_sample_queries(
    sample_qid_path: str,
    qrels_input_path: str,
    queries_input_path: str,
    queries_output_path: str,
    qrels_output_path: str,
) -> None:
    sample_qid = pd.read_csv(sample_qid_path)
    qrels_df = pd.read_csv(
        qrels_input_path,
        sep=" ",
        header=None,
        names=["qid", "q0", "docid", "rel"],
    )

    queries_df = pd.read_csv(
        queries_input_path, sep="\t", header=None, names=["qid", "query"]
    )

    queries_subset_df = queries_df[queries_df["qid"].isin(sample_qid.values[:, 0])]
    qrels_subset_df = qrels_df[qrels_df["qid"].isin(sample_qid.values[:, 0])]
   
    queries_subset_df.to_csv(queries_output_path, sep=" ", index=False, header=False)
    qrels_subset_df.to_csv(qrels_output_path, sep=" ", index=False, header=False)
  

def create_subsets(
    input_trec: str,
    output_trec: str,
    num_docs: int,
    num_topics: int,
    topic_input_file: str,
    topic_output_file: str,
    qrels_input_file: str,
    qrels_output_file: str,
):
    # Create the MS MARCO subset
    input_file = input_trec
    output_file = output_trec
    trec_num_docs = 20000
    doc_count = 0
    subset_lines = []
    with open(input_file, "r") as f_in:
        for line in f_in:
            subset_lines.append(line)
            if line.startswith("</DOC>"):
                doc_count += 1
                if doc_count >= trec_num_docs:
                    break

    with open(output_file, "w") as f_out:
        f_out.writelines(subset_lines)

    # Subset train data
    # Subset topics
    topics = []
    with open(topic_input_file, "r") as f:
        for line in f:
            qid, query = line.strip().split("\t", 1)  # Split on tab character
            query = re.sub(r"\W+", " ", query)  # Remove any special characters
            query = re.sub(r"\_+", " ", query)
            query = re.sub(r"\,+", " ", query)
            query = re.sub(r"\-+", " ", query)
            topics.append({"qid": int(qid), "query": query.strip()})

    topics = pd.DataFrame(topics)

    # Read qrels file
    qrels = pd.read_csv(
        qrels_input_file,
        sep=" ",
        names=["qid", "Q0", "docid", "rel"],
        header=None,
        dtype={"qid": int},
    )

    # Filter qrels to keep only rows with qids present in topics
    qrels = qrels[qrels["qid"].isin(topics["qid"])]

    topics_subset_qids = qrels["qid"].unique()[:num_topics]
    topics_subset = topics[topics["qid"].isin(topics_subset_qids)]

    # Write topics_subset to a file directly
    with open(topic_output_file, "w") as f:
        for index, row in topics_subset.iterrows():
            f.write(f"{row['qid']} {row['query']}\n")

    # Subset qrels
    qrel_subset = qrels[qrels["qid"].isin(topics_subset_qids)]
    qrel_subset.to_csv(qrels_output_file, sep=" ", index=False, header=False)

    return output_file, topic_output_file, qrels_output_file
