import csv
import re
from typing import Optional, Tuple

import pandas as pd
import typer

app = typer.Typer()


@app.command()
def subset_marco(
    num_docs: int = typer.Argument(
        10000, help="Number of documents to include in the subset"
    ),
    input_file: str = typer.Argument(
        "data/fulldocs-new.trec", help="Path to the input TREC file"
    ),
    output_file: Optional[str] = typer.Argument(
        None, help="Path to the output TREC file"
    ),
) -> str:
    """
    Subsets a TREC-formatted file by selecting the first `num_docs` documents.

    :param num_docs: Number of documents to include in the subset.
    :param input_file: Path to the input TREC file.
    :param output_file: Path to the output TREC file. If not specified, will be set to
        `proc_data/trec/subset_msmarco.trec`.
    :return: Path to the output TREC file.
    """
    if output_file is None:
        output_file = "proc_data/trec/subset_msmarco.trec"

    doc_count = 0
    subset_lines = []

    with open(input_file, "r") as f_in:
        for line in f_in:
            subset_lines.append(line)
            if line.startswith("</DOC>"):
                doc_count += 1
                if doc_count >= num_docs:
                    break

    with open(output_file, "w") as f_out:
        f_out.writelines(subset_lines)

    return output_file


@app.command()
def marco_csv(
    input_file: str = typer.Argument(..., help="Path to input file in TREC format"),
    output_file: str = typer.Argument(..., help="Path to output CSV file"),
) -> pd.DataFrame:
    """Converts an input TREC file to a CSV file with columns doc_id, url, and content.

    Args:
        input_file (str): Path to input file in TREC format.
        output_file (str): Path to output CSV file.

    Returns:
        pd.DataFrame: Pandas DataFrame with columns doc_id, url, and content.
    """
    with open(input_file, "r") as f_in, open(output_file, "w", newline="") as f_out:
        csv_writer = csv.writer(f_out)
        csv_writer.writerow(["doc_id", "url", "content"])
        doc_id = None
        content = None
        url = None
        capture_content = False
        for line in f_in:
            if line.startswith("<DOCNO>"):
                doc_id = re.sub("<DOCNO>|</DOCNO>", "", line).strip()
            elif line.startswith("<TEXT>"):
                content = []
                capture_content = True
            elif line.startswith("</TEXT>"):
                capture_content = False
                if doc_id and content and url:
                    # Write the extracted document to the CSV file
                    csv_writer.writerow([doc_id, url, "\n".join(content)])
                    doc_id = None
                    content = None
                    url = None
            elif capture_content:
                if not url:
                    url = line.strip()
                else:
                    content.append(line.strip())
    return pd.read_csv(output_file)


@app.command()
def subset_topics(num_topics: int = 100):
    """
    Subsets the topics from the MSMARCO document ranking training set.

    Args:
        num_topics (int): The number of topics to subset.

    Returns:
        str: The filepath to the output subset file.
    """
    input_file = "proc_data/train/queries.doctrain.tsv"
    output_file = "proc_data/train/subset_queries.doctrain.tsv"

    topics = pd.read_csv(input_file, sep="\t", names=["qid", "query"])
    topics_subset = topics.head(num_topics)
    topics_subset.to_csv(output_file, sep="\t", index=False)

    return output_file


@app.command()
def subset_qrels(topic_subset_file: str):
    """
    Subsets the relevance judgments from the MSMARCO document ranking training set.

    Args:
        topic_subset_file (str): The filepath to the topics subset file.

    Returns:
        str: The filepath to the output subset file.
    """
    input_file = "proc_data/train/msmarco-doctrain-qrels.tsv"
    output_file = "proc_data/train/subset_msmarco-doctrain-qrels.tsv"

    topics_subset = pd.read_csv(topic_subset_file, sep="\t", names=["qid", "query"])
    qrel_subset = pd.DataFrame()

    with open(input_file, "r") as f_in:
        qrels = pd.read_csv(f_in, sep="\t", names=["qid", "Q0", "docid", "rel"])

    qrel_subset = pd.DataFrame(columns=["qid", "Q0", "docid", "rel"])
    for qid in topics_subset["qid"]:
        qrel_subset = pd.concat(
            [qrel_subset, qrels[qrels["qid"] == qid]], ignore_index=True
        )

    qrel_subset.to_csv(output_file, sep="\t", index=False)

    return output_file


@app.command()
def subset_train_data(
    num_topics: int = typer.Option(1000, help="Number of topics to subset")
) -> Tuple[str, str]:
    """
    Subsets the training data by selecting the first `num_topics` topics
    and corresponding qrels.

    :param num_topics: Number of topics to include in the subset. Default is 1000.
    :return: A tuple containing the paths to the output topic and qrels files.
    """

    topic_input_file = "proc_data/train/queries.doctrain.tsv"
    topic_output_file = "proc_data/train/subset_queries.doctrain.tsv"
    qrels_input_file = "proc_data/train/msmarco-doctrain-qrels.tsv"
    qrels_output_file = "proc_data/train/subset_msmarco-doctrain-qrels.tsv"

    # Subset topics
    topics = pd.read_csv(
        topic_input_file,
        sep="\t",
        names=["qid", "query"],
        header=None,
        dtype={"qid": int},
    )
    topics_subset = topics.head(num_topics)
    topics_subset.to_csv(topic_output_file, sep="\t", index=False, header=False)

    # Subset qrels
    qrel_subset = pd.DataFrame(columns=["qid", "Q0", "docid", "rel"])
    qrels = pd.read_csv(
        qrels_input_file,
        sep=" ",
        names=["qid", "Q0", "docid", "rel"],
        header=None,
        dtype={"qid": int},
    )

    for qid in topics_subset["qid"]:
        qrel_subset = pd.concat(
            [qrel_subset, qrels[qrels["qid"] == qid]], ignore_index=True
        )

    qrel_subset.to_csv(qrels_output_file, sep="\t", index=False, header=False)

    return topic_output_file, qrels_output_file


@app.command()
def create_overlapping_subsets(
    num_docs: int = typer.Option(10000, help="Number of documents in the subset"),
    num_topics: int = typer.Option(10000, help="Number of topics in the subset"),
    input_file: str = typer.Argument(
        "data/fulldocs-new.trec", help="Path to input file"
    ),
    output_file: str = typer.Argument(
        "proc_data/trec/subset_msmarco.trec", help="Path to output file"
    ),
    topic_input_file: str = typer.Option(
        "proc_data/train/queries.doctrain.tsv", help="Path to topic input file"
    ),
    topic_output_file: str = typer.Option(
        "proc_data/train/subset_queries.doctrain.tsv",
        help="Path to topic output file",
    ),
    qrels_input_file: str = typer.Option(
        "proc_data/train/msmarco-doctrain-qrels.tsv",
        help="Path to qrels input file",
    ),
    qrels_output_file: str = typer.Option(
        "proc_data/train/subset_msmarco-doctrain-qrels.tsv",
        help="Path to qrels output file",
    ),
) -> None:
    """
    Create overlapping subsets of TREC-formatted data for MS MARCO.

    This function creates subsets of the MS MARCO dataset by selecting the first
    `num_docs` documents from the input TREC file and selecting the first `num_topics`
    topics from the input topic file. It then creates new topic
    and qrels files containing only the selected topics and corresponding qrels.
    The output TREC file and new topic/qrels files are written to disk.


    Args:
        num_docs (int):
            Number of documents to include in the TREC subset (default: 10000).
        num_topics (int):
            Number of topics to include in the topic/qrels subset (default: 10000).
        input_file (str):
            Path to the input TREC file (default: 'data/fulldocs-new.trec').
        output_file (str):
            Path to the output TREC file (default: 'proc_data/trec/subset_msmarco.trec')
        topic_input_file (str):
            Path to the input topic file (default: 'train/queries.doctrain.tsv').
        topic_output_file (str):
            Path to the output topic fil (default: 'proc_data/train/subset_queries.tsv')
        qrels_input_file (str):
            Path to the input qrels file (default: 'train/msmarco_doctrain-qrels.tsv').
        qrels_output_file (str):
            Path to the output qrels file (default: 'proc_data/train/subset_qrels.tsv').

    Returns:
        None
    """

    # Create the MS MARCO subset
    doc_count = 0
    subset_lines = []

    with open(input_file, "r") as f_in:
        for line in f_in:
            subset_lines.append(line)
            if line.startswith("</DOC>"):
                doc_count += 1
                if doc_count >= num_docs:
                    break

    with open(output_file, "w") as f_out:
        f_out.writelines(subset_lines)

    topics = pd.read_csv(
        topic_input_file,
        sep="\t",
        names=["qid", "query"],
        header=None,
        dtype={"qid": int},
    )
    topics_subset = topics.head(num_topics)
    topics_subset.to_csv(topic_output_file, sep="\t", index=False, header=False)

    qrel_subset = pd.DataFrame(columns=["qid", "Q0", "docid", "rel"])
    qrels = pd.read_csv(
        qrels_input_file,
        sep=" ",
        names=["qid", "Q0", "docid", "rel"],
        header=None,
        dtype={"qid": int},
    )

    for qid in topics_subset["qid"]:
        qrel_subset = pd.concat(
            [qrel_subset, qrels[qrels["qid"] == qid]], ignore_index=True
        )

    qrel_subset.to_csv(qrels_output_file, sep="\t", index=False, header=False)

    typer.echo(
        f"Created subsets: {output_file}, {topic_output_file}, {qrels_output_file}"
    )


if __name__ == "__main__":
    app()
