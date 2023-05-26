import os
import pandas as pd
import pyterrier as pt
import time


def create_indices(corpus_dir, index_dir):
    # Make sure to initialize Terrier
    if not pt.started():
        pt.init()

    # Index configurations
    index_configurations = [
        {"index_type": "full_index", "stemmer": None, "stopwords": None},
        {
            "index_type": "no_stopwords",
            "stemmer": None,
            "stopwords": pt.TerrierStopwords.terrier,
        },
        {
            "index_type": "stemming",
            "stemmer": pt.TerrierStemmer.porter,
            "stopwords": None,
        },
        {
            "index_type": "no_stopwords_stemming",
            "stemmer": pt.TerrierStemmer.porter,
            "stopwords": pt.TerrierStopwords.terrier,
        },
    ]

    index_dir = "./pyterrier/indexes/"

    build_times = {}

    for config in index_configurations:
        index_directory = os.path.join(index_dir, config["index_type"])
        os.makedirs(index_directory, exist_ok=True)

        indexer = pt.TRECCollectionIndexer(
            index_directory,
            blocks=True,  # Enable block indexing
            stemmer=config["stemmer"],
            stopwords=config["stopwords"],
            verbose=True,
        )

        # Set additional properties
        indexer.setProperty("invertedfile.lexiconscanner", "pointers")
        indexer.setProperty("ignore.low.idf.terms", "false")

        start_time = time.time()  # start timing
        indexref = indexer.index(corpus_dir)
        end_time = time.time()  # end timing

        build_time = end_time - start_time

        print(f"Indexing of {config['index_type']} completed in {build_time} seconds.")

        build_times[config["index_type"]] = build_time

    build_times_df = pd.DataFrame(
        list(build_times.items()), columns=["Index_Variant", "Build_Time"]
    )

    build_times_df.to_csv("pyterrier_build_times.csv", index=False)

    print("All indexing tasks completed.")

    return build_times
