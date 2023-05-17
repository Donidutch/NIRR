# NIR Project

The NIR (Neural Information Retrieval) project is a machine learning project that aims to improve information retrieval systems using the [MS MARCO (MicroSoft Machine Reading Comprehension)](https://microsoft.github.io/msmarco/) dataset. This dataset is a large-scale collection of documents that provides fields such as URL, title, and body for each document.

In addition to these documents, the dataset also includes queries (also referred to as topics) and relevance assessments. These assessments indicate which documents are relevant to each query, providing a valuable resource for training information retrieval models.

The dataset includes 367,013 queries along with their corresponding relevance assessments. We use these for training our models, aiming to improve the efficiency and accuracy of information retrieval.

## Dataset

The dataset can be found on Kaggle and can be accessed [here](https://www.kaggle.com/code/nicklasrs/fork-of-fork-of-project/edit).

## Installation and Setup

To set up and run the project, follow these steps:

1. Clone the repository to your local machine.
2. Install the necessary dependencies by running `poetry install` in the project root directory.
3. Run `poetry shell` to activate the virtual environment.
4. Download the dataset and place it in the appropriate directory.
5. Run the project using the command `python main.py`.

## Dependencies

This project relies on several Python libraries, including:

- `spacy`: For natural language processing tasks.
- `pyserini`: For interfacing with the Anserini information retrieval toolkit.
- `typer`: For building command-line interfaces.
- `python-terrier`: For interfacing with the Anserini information retrieval toolkit.
- `python3.8.18`

## Usage 

- `python main.py run-rank-eval-single-index-cmd data/proc_data/train_sample/sample_queries.tsv data/proc_data/train_sample/sample_qrels.tsv pyserini/indexes/full_index`

-`python main.py run-rank-eval-cmd data/proc_data/train_sample/sample_queries.tsv data/proc_data/train_sample/sample_qrels.tsv pyserini/indexes/ 5 `