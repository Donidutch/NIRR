from typing import Dict, List
import numpy as np
import pandas as pd


def generate_folds(
    unique_qids: pd.Series, kfolds: int, seed: int = 42
) -> Dict[int, List[str]]:
    rng = np.random.default_rng(seed)
    choice = rng.choice(range(kfolds), size=len(unique_qids))
    groups = {c: [] for c in range(kfolds)}
    for c, qid in zip(choice, unique_qids):
        groups[c].append(qid)
    # print(groups)
    return groups


def get_training_data(groups, fold, queries, qrels, unique_qids):
    # validation_set = groups[fold]
    # print(unique_qids)
    # validation_set = list(groups[fold])
    # validation_set = list(groups[fold])
    validation_set = [qid for qid in groups[fold]]
    # validation_qrels = qrels.loc[qrels['qid'].isin(validation_set)]

    training_set = set(unique_qids).difference(set(validation_set))
    print(training_set, "training set")
    print(validation_set, "validation set")
    if isinstance(qrels, dict):
        qrels = pd.read_csv(
            "data/proc_data/train_sample/sample_qrels.tsv",
            sep=" ",
            names=["qid", "Q0", "docid", "rel"],
        )

    training_qrels = qrels.loc[qrels["qid"].isin(training_set)]
    train_queries = [
        queries.loc[queries["qid"] == qid]["query"].values[0] for qid in training_set
    ]
    # train_queries = [
    #     queries.loc[queries["qid"] == qid]["query"].values[0] for qid in validation_set
    # ]
    train_queries = queries.loc[queries["qid"].isin(validation_set)]
    # train_queries = queries.loc[queries["qid"].isin(training_set)]

    train_qids = list(training_set)
    train_qids_str = [str(qid) for qid in train_qids]

    return train_queries, train_qids, training_qrels, train_qids_str
