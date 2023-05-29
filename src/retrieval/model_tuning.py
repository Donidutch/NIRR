# retrieval/model_tuning.py

import ast
import itertools
import json
from typing import List, Tuple, Union, Dict
import pandas as pd
from evaluation.evaluate import evaluate_run
from evaluation.model_evaluation import ModelEvaluator
from retrieval.ranking_models.models import Model


class ModelTuner:
    def __init__(self, model: Model):
        self.model = model

    def set_model_config(self, model_type: str, best_config):
        if model_type == "bm25":
            if isinstance(best_config, tuple) and len(best_config) == 2:
                k1, b = best_config
            else:
                raise ValueError("Invalid value for best_config")
            self.model.set_bm25_parameters(k1, b)
        elif model_type == "lm":
            if isinstance(best_config, int):
                mu = best_config
            else:
                raise ValueError("Invalid value for best_config")
            self.model.set_qld_parameters(mu)

    def tune_parameters(
        self,
        model,
        train_queries: List[str],
        train_qids: List[int],
        qrels: Union[pd.DataFrame, Dict[str, Dict[str, int]]],
        fold: int,
        model_type: str = "bm25",
        tuning_measure: str = "ndcg_cut_10",
        config_file: str = "evaluation/gridsearch_params.json",
    ) -> pd.DataFrame:
        """
        Tune the parameters of the model using the given training queries, query IDs,
        relevance judgments, and fold.
        """

        with open(config_file) as f:
            tuning_params = json.load(f)[model_type]
        # print("model", model.model_type, model_type, model)
        evaluator = ModelEvaluator(model)
        param_combinations = list(itertools.product(*tuning_params.values()))

        params_performance = pd.DataFrame(columns=["fold", "model", "params", "score"])
        qids = [str(qid) for qid in train_qids]

        for params in param_combinations:
            print("running with params:", params)

            model.set_parameters(model_type, dict(zip(tuning_params.keys(), params)))
            run = evaluator.create_run(train_queries, qids)
            print(run)
            # print(model.k1, model.b)
            # print(model.set_bm25_parameters(model.k1, model.b))
            measures = evaluate_run(run, qrels, metric=tuning_measure)
            params_str = str(dict(zip(tuning_params.keys(), params)))
            print(measures, "meas")
            new_row = pd.DataFrame(
                {
                    "fold": [fold],
                    "model": [model_type],
                    "params": [params_str],
                    "score": [measures[tuning_measure]],
                }
            )

            params_performance = pd.concat(
                [params_performance, new_row], ignore_index=True
            )

        return params_performance

    def tune_and_set_parameters(
        self,
        model,
        train_queries: List[str],
        train_qids: List[int],
        training_qrels: pd.DataFrame,
        fold: int,
        model_type: str,
        metric: str,
    ) -> Tuple[pd.DataFrame, str]:
        fold_params_performance = self.tune_parameters(
            model,
            train_queries,
            train_qids,
            training_qrels,
            fold=fold,
            model_type=model_type,
            tuning_measure=metric,
        )
        # print("fpp", fold_params_performance)
        best_config = fold_params_performance.groupby("params")["score"].mean().idxmax()

        # Convert the string back into a dictionary
        # print(best_config, model)
        if isinstance(best_config, str):
            best_config_dict = ast.literal_eval(best_config)
        else:
            best_config_dict = ast.literal_eval(str(best_config))
        # print("best_config_dict", best_config_dict)

        # Now you can access the values as you would in a regular dictionary

        if model_type == "bm25":
            best_config = (best_config_dict["k1"], best_config_dict["b"])

        elif model_type == "lm":
            best_config = best_config_dict["mu"]

        self.set_model_config(model_type, best_config)

        return fold_params_performance, str(best_config)
