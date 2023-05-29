import os
import time
from typing import Any, Dict
import pandas as pd
from pyserini.index.lucene import Generator, LuceneIndexer, IndexReader
from pyserini.search import LuceneSearcher
from pyserini.collection import Collection
from pyserini.setup import configure_classpath
from jnius import autoclass
# import logging

# logging.getLogger('pyserini').setLevel(logging.WARNING)
# import warnings

# from tqdm import tqdm

# Suppress unnecessary FutureWarning
# warnings.filterwarnings('ignore', category=FutureWarning)


# configure_classpath("/home/nicklas/Desktop/NIRproject/src/.venv/lib/python3.8/site-packages/pyserini/resources/jars/")
# Define collection directory
collection_dir = "/data/lab/trec/"
JIndexCollection = autoclass('io.anserini.index.IndexCollection')


# JIndexCollection.main(index_dir)
# Create an instance of JLuceneIndexer
# LuceneIndexer = autoclass('io.anserini.index.IndexCollection')
# indexer = LuceneIndexer(index_dir)

# # Define your documents
# # Assuming the documents are in a list where each document is a dictionary with 'id' and 'contents' as keys
# # Prepare some data
documents = [
    {"id": "1", "contents": "This is the first document."},
    {"id": "2", "contents": "This is the second document."},
    {"id": "3", "contents": "And this is the third one."},
]
pd.DataFrame(documents, columns=['id', 'contents']).to_json('data/pyserini/docs.jsonl', orient='records', lines=True)
# # Add the documents to the index
# for doc in documents:
#     doc_str = json.dumps(doc)  # Convert the dictionary to a JSON string
#     indexer.addRawDocument(doc_str)

# # Close the index to write the changes to disk
# indexer.close()

args = ["-collection", "CleanTrecCollection", "-input", "data/lab/trec/", "-index", "my_index"]
indexCollectionInstance = JIndexCollection.main(args)
JIndexCollection.LOG.info("Starting indexer...")

args = ["-collection", "CleanTrecCollection", "-input", "data/lab/trec/", "-index", "my_index"]
indexCollectionInstance = JIndexCollection.main(args)





def verify_document_in_index(index_directory, doc_id):
    reader = IndexReader(index_directory)
    print("stats:", reader.stats())
    searcher = LuceneSearcher(index_directory)
    # Perform the search
    hits = searcher.search(doc_id)
    for i in range(len(hits)):
        # Fetch the document id
        if hits[i].docid == doc_id:
            print("Document with id {} is in the index".format(doc_id))
            print("Document content:", reader.doc(hits[i].docid).raw())
            return
    print("Document with id {} is not in the index".format(doc_id))


def create_index_batch(
    index_writer, collection_path, collection_class="TrecCollection", batch_size=1000
):
    collections = Collection(collection_class, collection_path)
    generator = Generator("DefaultLuceneDocumentGenerator")
    batch = []

    for fs in collections:
        for doc in fs:
            parsed = generator.create_document(doc)
            doc_id = parsed.get("id")  # FIELD_ID
            contents = parsed.get("contents")  # FIELD_BODY
            doc_dict = {"id": doc_id, "contents": contents}
            batch.append(doc_dict)

            if len(batch) == batch_size:
                k = index_writer.add_batch_dict(batch)
                batch = []

    if batch:
        k = index_writer.add_batch_dict(batch)

    index_writer.close()


def create_index(index_writer, collection_path, collection_class="TrecCollection"):
    collections = Collection(collection_class, collection_path)
    generator = Generator("DefaultLuceneDocumentGenerator")

    for fs in collections:
        for doc in fs:
            parsed = generator.create_document(doc)
            doc_id = parsed.get("id")  # FIELD_ID
            contents = parsed.get("contents")  # FIELD_BODY
            doc_dict = {"id": doc_id, "contents": contents}
            index_writer.add_doc_dict(doc_dict)
    index_writer.close()


def build_index(
    variant: Dict[str, Any], path_to_dataset: str, output_folder: str
) -> float:
    """
    Build the index for a specific variant.

    Args:
        variant (Dict[str, Any]): The configuration for the index variant.
        path_to_dataset (str): The path to the dataset.
        output_folder (str): The output folder to store the index.

    Returns:
        float: The time taken to build the index.

    """
    variant["index_path"] = output_folder + variant["index_path"]
    stemmer_arg = "none" if not variant["stemming"] else "porter"
    stopword_arg = (
        ["-keepStopwords"]
        if not variant["stopwords"]
        else ["-stopwords", "stopword.txt"]
    )
    num_threads = 1  # os.cpu_count()

    args = [
        "-input",
        path_to_dataset,
        "-index",
        variant["index_path"],
        "-verbose",
        "-collection",
        "CleanTrecCollection",
        "-generator",
        "DefaultLuceneDocumentGenerator",
        "-threads",
        str(num_threads),
        "-stemmer",
        stemmer_arg,
    ]
    # args.extend(stopword_arg)
    # print(hi)
    # LuceneIndexer(
    #     index_dir=variant["index_path"], args=args
    # )
    # index_writer = LuceneIndexer(
    #     index_dir=variant["index_path"], args=args
    # )

    # print(f"Indexing {variant['name']}...")
    # start_time = time.time()
    create_index(
        LuceneIndexer(index_dir=variant["index_path"], args=args), path_to_dataset
    )
    # end_time = time.time()
    # build_time = end_time - start_time
    # print(f"Finished indexing {variant['name']} in {build_time:.2f} seconds\n")
    build_time = 0
    return build_time


def build_all_indexes(path_to_dataset: str, output_folder: str) -> Dict[str, float]:
    """
    Build all the indexes for the given dataset.

    Args:
        path_to_dataset (str): The path to the dataset.
        output_folder (str): The output folder to store the indexes.

    Returns:
        Dict[str, float]: A dictionary containing the build times for each index variant.

    """
    build_times = {}
    for variant in index_variants:  # noqa: F821
        # print(variant, path_to_dataset)
        build_time = build_index(variant, path_to_dataset, output_folder)
        build_times[variant["name"]] = build_time
    build_times_df = pd.DataFrame(
        list(build_times.items()), columns=["Index_Variant", "Build_Time"]
    )

    build_times_df.to_csv("build_times.csv", index=False)
    return build_times


if __name__ == "__main__":
    dataset_path = "data/lab/trec/"
    index_output_folder = "data/ind/"
    build_times = build_all_indexes(dataset_path, index_output_folder)
