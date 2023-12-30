from soil_fertility.logger import logging
from typing import Dict
import pandas as pd
from sklearn.model_selection import GridSearchCV
from soil_fertility.components.utils.metrics import (
    precision_recall_f1,
    specificity,
    micro_average,
    macro_average,
    accuracy
)
import numpy as np


def evaluate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: Dict[str, object],
) -> Dict[str, object]:
    try:
        report = {"metrics": {}}  
        classes = np.unique(y_test)

        for model_name, model in models.items():
            model.fit(X_train.to_numpy(), y_train.to_numpy())
            y_test_pred = model.predict(X_test.to_numpy())

            metrics_per_class = {}
            for class_label in classes:
                prec, rec, f1 = precision_recall_f1(y_test, y_test_pred, class_label)
                spec = specificity(y_test, y_test_pred, class_label)
                metrics_per_class[str(class_label)] = {
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1,
                    "specificity": spec,
                }
                accuracy_score = accuracy(y_test, y_test_pred)

            micro_precision, micro_recall, micro_f1 = micro_average(
                y_test, y_test_pred, classes
            )

            macro_precision, macro_recall, macro_f1 = macro_average(
                y_test, y_test_pred, classes
            )

            report[model_name] = model
            report["metrics"][model_name] = {
                "accuracy": accuracy_score,
                "micro_average": {
                    "precision": micro_precision,
                    "recall": micro_recall,
                    "f1_score": micro_f1,
                },
                "macro_average": {
                    "precision": macro_precision,
                    "recall": macro_recall,
                    "f1_score": macro_f1,
                },
                "metrics_per_class": metrics_per_class,
            }

        return report
    except Exception as e:
        logging.error(f"Exception occurred: {e}")
        raise


def evaluate_model_gridsearch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: Dict[str, object],
    params: Dict[str, list[int]],
) -> Dict[str, float]:
    try:
        report = {"metrics": {}}
        classes = np.unique(y_test)

        for model_name, model in models.items():
            grid_search = GridSearchCV(
                model, params[model_name], cv=5, scoring="accuracy"
            )
            grid_search.fit(X_train.to_numpy(), y_train.to_numpy())
            y_test_pred = grid_search.predict(X_test.to_numpy())

            metrics_per_class = {}
            for class_label in classes:
                prec, rec, f1 = precision_recall_f1(y_test, y_test_pred, class_label)
                spec = specificity(y_test, y_test_pred, class_label)
                metrics_per_class[str(class_label)] = {
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1,
                    "specificity": spec,
                }
                accuracy_score = accuracy(y_test, y_test_pred)

            micro_precision, micro_recall, micro_f1 = micro_average(
                y_test, y_test_pred, classes
            )

            macro_precision, macro_recall, macro_f1 = macro_average(
                y_test, y_test_pred, classes
            )

            report[model_name + "_gridsearch"] = grid_search
            report["metrics"][model_name] = {
                "accuracy": accuracy_score,
                "micro_average": {
                    "precision": micro_precision,
                    "recall": micro_recall,
                    "f1_score": micro_f1,
                },
                "macro_average": {
                    "precision": macro_precision,
                    "recall": macro_recall,
                    "f1_score": macro_f1,
                },
                "metrics_per_class": metrics_per_class,
            }

        return report
    except Exception as e:
        logging.error(f"Exception occurred {e}")
        raise e
