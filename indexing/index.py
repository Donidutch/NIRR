import os
import time
from typing import Dict, List

path_to_dataset = "proc_data/train_trec/"
output_folder = "./index/"

index_variants = [
    {
        "name": "full_index",
        "index_path": output_folder + "full_index/",
        "stopwords": False,
        "stemming": False,
    },
    {
        "name": "stopwords_removed",
        "index_path": output_folder + "stopwords_removed/",
        "stopwords": True,
        "stemming": False,
    },
    {
        "name": "stemming",
        "index_path": output_folder + "stemming/",
        "stopwords": False,
        "stemming": True,
    },
    {
        "name": "stopwords_removed_stemming",
        "index_path": output_folder + "stopwords_removed_stemming/",
        "stopwords": True,
        "stemming": True,
    },
]


def build_index(variant: Dict[str, any]) -> None:  # type: ignore
    stemmer_arg = "none" if not variant["stemming"] else "porter"
    keep_stopwords_arg = "--keepStopwords" if not variant["stopwords"] else ""
    stopwords_file_arg = "" if not variant["stopwords"] else "--stopwords stopword.txt"
    num_threads = os.cpu_count()
    command = f"""python -m pyserini.index.lucene \
        --collection CleanTrecCollection \
        --input {path_to_dataset} \
        --index {variant["index_path"]} \
        --generator DefaultLuceneDocumentGenerator \
        --threads {num_threads} \
        --stemmer {stemmer_arg} \
        {keep_stopwords_arg} \
        {stopwords_file_arg} \
        --storeContents
    """

    print(f"Indexing {variant['name']}...")
    start_time = time.time()
    os.system(command)
    end_time = time.time()
    build_time = end_time - start_time
    print(f"Finished indexing {variant['name']} in {build_time:.2f} seconds\n")
    return build_time  # type: ignore


def build_all_indexes(index_variants: List[Dict[str, any]]) -> Dict[str, float]:  # type: ignore
    build_times = {}
    for variant in index_variants:
        build_time = build_index(variant)
        build_times[variant["name"]] = build_time
    return build_times


if __name__ == "__main__":
    build_times = build_all_indexes(index_variants)
    print("Build times:")
    for name, build_time in build_times.items():
        print(f"{name}: {build_time:.2f} seconds")
