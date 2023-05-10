from typing import Tuple

import pandas as pd
import typer

app = typer.Typer()


@app.command()
def subset_marco(
    num_docs: int, input_file: str, output_file: str = "data/trec/subset_msmarco.trec"
) -> str:
    """
    Subsets a TREC-formatted file by selecting the first `num_docs` documents.
    """
    if output_file is None:
        output_file = "data/proc_data/trec/subset_msmarco.trec"

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for _ in range(num_docs):
            f_out.write(next(f_in))

    return output_file


@app.command()
def create_subsets(
    num_docs: int,
    num_topics: int,
    input_file: str,
    output_file: str,
    topic_input_file: str,
    topic_output_file: str,
    qrels_input_file: str,
    qrels_output_file: str,
) -> Tuple[str, str, str]:
    """
    Create overlapping subsets of TREC-formatted data for MS MARCO.
    """
    subset_marco(num_docs, input_file, output_file)

    topics = pd.read_csv(
        topic_input_file, sep="\t", names=["qid", "query"], header=None
    )
    topics.head(num_topics).to_csv(
        topic_output_file, sep="\t", index=False, header=False
    )

    qrels = pd.read_csv(
        qrels_input_file, sep=" ", names=["qid", "Q0", "docid", "rel"], header=None
    )
    qrels[qrels["qid"].isin(topics["qid"])].to_csv(
        qrels_output_file, sep="\t", index=False, header=False
    )

    return output_file, topic_output_file, qrels_output_file


if __name__ == "__main__":
    app()
