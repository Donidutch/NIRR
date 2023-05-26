import os
import time
from typing import Any, Dict

import pandas as pd

index_variants = [
    {
        "name": "full_index",
        "index_path": "full_index/",
        "stopwords": False,
        "stemming": False,
    },
    {
        "name": "stopwords_removed",
        "index_path": "stopwords_removed/",
        "stopwords": True,
        "stemming": False,
    },
    {
        "name": "stemming",
        "index_path": "stemming/",
        "stopwords": False,
        "stemming": True,
    },
    {
        "name": "stopwords_removed_stemming",
        "index_path": "stopwords_removed_stemming/",
        "stopwords": True,
        "stemming": True,
    },
]


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
    keep_stopwords_arg = "--keepStopwords" if not variant["stopwords"] else ""
    stopwords_file_arg = "" if not variant["stopwords"] else "--stopwords stopword.txt"
    num_threads = 1
    command = f""" python -m pyserini.index.lucene \
        --collection CleanTrecCollection \
        --input {path_to_dataset} \
        --index {variant["index_path"]} \
        --generator DefaultLuceneDocumentGenerator \
        --threads {num_threads} \
        --stemmer {stemmer_arg} \
        {keep_stopwords_arg} \
        {stopwords_file_arg} \
        --storeContents \
    """

    print(f"Indexing {variant['name']}...")
    start_time = time.time()
    os.system(command)
    end_time = time.time()
    build_time = end_time - start_time
    print(f"Finished indexing {variant['name']} in {build_time:.2f} seconds\n")
    return build_time


def build_all_indexes(path_to_dataset: str, output_folder: str) -> Dict[str, float]:
    """
    Build all the indexes for the given dataset.

    Args:
        path_to_dataset (str): The path to the dataset.
        output_folder (str): The output folder to store the indexes.

    Returns:
        Dict[str, float]: A dictionary containing the build times for each index variant

    """
    build_times = {}
    for variant in index_variants:
        build_time = build_index(variant, path_to_dataset, output_folder)
        build_times[variant["name"]] = build_time

    build_times_df = pd.DataFrame(
        list(build_times.items()), columns=["Index_Variant", "Build_Time"]
    )

    build_times_df.to_csv("build_times.csv", index=False)
    return build_times
