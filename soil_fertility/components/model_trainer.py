from soil_fertility.components.models.decision_tree import DecisionTree
from soil_fertility.components.models.random_forest import RandomForest
from soil_fertility.logger import logging
from dataclasses import dataclass
from pathlib import Path
from soil_fertility.utils import retreive_base_path, save_object
from soil_fertility.components.model_utils import (
    evaluate_model,
    evaluate_model_gridseach,
)
from soil_fertility.components.models.KNN import KNN
from typing import Dict
import pandas as pd
from sklearn.metrics import accuracy_score


@dataclass
class ModelConfig:
    base_path: str = retreive_base_path()
    trained_model_path: str = Path(base_path, "artifacts/models/model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_config = ModelConfig()

    def init_training(
        self, train_data, test_data, target_column, mode: str = "default"
    ):
        try:
            logging.info("Splitting train and test data")
            X_train = train_data.drop(columns=[target_column])
            y_train = train_data[target_column]
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]

            logging.info("Training model")

            models = {
                "decision_tree": DecisionTree(),
                "random_forest": RandomForest(),
            }

            # change to yaml
            params = {
                "decision_tree": {"max_depth": [10, 20, 30, 40, 50]},
                "random_forest": {
                    "max_depth": [10, 20, 30, 40, 50],
                    "n_trees": [10, 20, 30, 40, 50],
                },
            }
            if mode == "default":
                model_report: dict = evaluate_model(
                    X_train, y_train, X_test, y_test, models, params=params
                )
            elif mode == "gridsearch":
                model_report: dict = evaluate_model_gridseach(
                    X_train, y_train, X_test, y_test, models, params=params
                )

            else:
                logging.error("Mode not supported , please use default or grid_search")
                raise Exception(
                    "Mode not supported , please use default or grid_search"
                )

            logging.info("Training completed")

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Exception(
                    "Models accuracy is less than 0.6, no best model founjd"
                )

            logging.info("Best model found ")

            save_object(file_path=self.model_config.trained_model_path, obj=best_model)

            return model_report

        except Exception as e:
            logging.error(f"Exception occured {e}")
            raise e
