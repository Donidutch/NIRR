import os
import time
from typing import Any, Dict

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
    variant["index_path"] = output_folder + variant["index_path"]
    stemmer_arg = "none" if not variant["stemming"] else "porter"
    keep_stopwords_arg = "--keepStopwords" if not variant["stopwords"] else ""
    stopwords_file_arg = "" if not variant["stopwords"] else "--stopwords stopword.txt"
    num_threads = 8  # os.cpu_count()
    command = f"""python -m pyserini.index.lucene \
        --collection CleanTrecCollection \
        --input {path_to_dataset} \
        --index {variant["index_path"]} \
        --generator DefaultLuceneDocumentGenerator \
        --threads {num_threads} \
        --stemmer {stemmer_arg} \
        {keep_stopwords_arg} \
        {stopwords_file_arg} \
        --storeRaw \
    """

    print(f"Indexing {variant['name']}...")
    start_time = time.time()
    os.system(command)
    end_time = time.time()
    build_time = end_time - start_time
    print(f"Finished indexing {variant['name']} in {build_time:.2f} seconds\n")
    return build_time


def build_all_indexes(
    path_to_dataset: str, output_folder: str
) -> Dict[str, float]:
    build_times = {}
    for variant in index_variants:
        build_time = build_index(variant, path_to_dataset, output_folder)
        build_times[variant["name"]] = build_time
    return build_times


if __name__ == "__main__":
    path_to_dataset = "proc_data/train_trec/"
    output_folder = "./index/"
    build_times = build_all_indexes(path_to_dataset, output_folder)
    print("Build times:")
    for name, build_time in build_times.items():
        print(f"{name}: {build_time:.2f} seconds")
