import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any
from tqdm import tqdm
import typer

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-20-openjdk"
app = typer.Typer()


def build_index(variant: Dict[str, Any], path_to_dataset: str) -> float:
    """
    Build a Pyserini index variant for the given TREC dataset.

    Args:
        variant (Dict[str, Any]): A dictionary containing the name, index path, and
            boolean flags for using stopwords and stemming.

    Returns:
        The time it took to build the index variant.
    """
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
    """

    print(f"Indexing {variant['name']}...")
    start_time = time.time()
    os.system(command)
    end_time = time.time()
    build_time = end_time - start_time
    print(f"Finished indexing {variant['name']} in {build_time:.2f} seconds\n")
    return build_time


def build_all_indexes(
    index_variants: List[Dict[str, Any]], path_to_dataset: str
) -> Dict[str, float]:
    """
    Build all Pyserini index variants for the given TREC dataset.

    Args:
        index_variants (List[Dict[str, Any]]): A list of dictionaries, each
        containing the name, index path, and boolean flags for using
        stopwords and stemming.

    Returns:
        A dictionary containing the build times for all index variants.
    """
    build_times = {}
    with ThreadPoolExecutor() as executor:
        future_to_variant = {
            executor.submit(build_index, variant, path_to_dataset): variant["name"]
            for variant in index_variants
        }
        for future in tqdm(
            future_to_variant, total=len(future_to_variant), desc="Indexing"
        ):
            name = future_to_variant[future]
            build_time = future.result()
            build_times[name] = build_time
    return build_times


@app.command()
def main(
    path_to_dataset: str = typer.Argument(...), output_folder: str = "./indexes/"
) -> None:
    """
    Build four variants of the Pyserini index for the given TREC dataset.

    Args:
        path_to_dataset (str): The path to the TREC dataset directory.
        output_folder (str): The path to the output dir for storing the built index.

    Returns:
        None.
    """
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
    build_times = build_all_indexes(index_variants, path_to_dataset)
    # Print build times
    print("Build times:")
    for name, build_time in build_times.items():
        print(f"{name}: {build_time:.2f} seconds")


if __name__ == "__main__":
    app()
