from soil_fertility.logger import logging
from typing import Dict
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from soil_fertility.components.metrics import (
    calculate_metrics_per_class,
    specificite_per_class,
    accuracy,
    recall,
    precision,
    f1_score,
    specificite,
    confusion_matrix,
)


def evaluate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: Dict[str, object],
) -> Dict[str, object]:
    try:
        report = {"metrics": {}}  # Initialize the report with a metrics dictionary
        classes = list(set(y_test))

        for model_name, model in models.items():
            # Fit the model
            model.fit(X_train.to_numpy(), y_train.to_numpy())
            y_test_pred = model.predict(X_test.to_numpy())

            # Calculate scores
            metrics_per_class = calculate_metrics_per_class(
                y_test, y_test_pred, classes
            )
            specificity_per_class_scores = specificite_per_class(
                y_test, y_test_pred, classes
            )
            test_accuracy = accuracy(y_test, y_test_pred)
            rec = recall(y_test, y_test_pred)
            prec = precision(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)
            spec = specificite(y_test, y_test_pred)
            conf_mat = confusion_matrix(y_test, y_test_pred)

            # Store the model and its metrics
            report[model_name] = model
            report["metrics"][model_name] = {
                "accuracy": test_accuracy,
                "confusion_matrix": conf_mat,
                "recall": rec,
                "precision": prec,
                "f1_score": f1,
                "specificite": spec,
                "metrics_per_class": metrics_per_class,
                "specificity_per_class": specificity_per_class_scores,
            }
        return report
    except Exception as e:
        logging.error(f"Exception occurred: {e}")
        raise


def evaluate_model_gridseach(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: Dict[str, object],
    params: Dict[str, list[int]],
) -> Dict[str, float]:
    try:
        report = {}
        classes = list(set(y_test))
        report = {"metrics": {}}  # Initialize the report with a metrics dictionary
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            para = params[list(models.keys())[i]]
            grid_search = GridSearchCV(model, para, cv=5, scoring="accuracy")
            grid_search.fit(X_train, y_train)

            y_test_pred = grid_search.predict(X_test)

            # Calculate scores
            metrics_per_class = calculate_metrics_per_class(
                y_test, y_test_pred, classes
            )
            specificity_per_class_scores = specificite_per_class(
                y_test, y_test_pred, classes
            )
            test_accuracy = accuracy(y_test, y_test_pred)
            rec = recall(y_test, y_test_pred)
            prec = precision(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)
            spec = specificite(y_test, y_test_pred)
            conf_mat = confusion_matrix(y_test, y_test_pred)

            # Store the model and its metrics
            report[str(model_name) + "_gridsearch"] = grid_search
            report["metrics"][str(model_name) + "_gridsearch"] = {
                "accuracy": test_accuracy,
                "confusion_matrix": conf_mat,
                "recall": rec,
                "precision": prec,
                "f1_score": f1,
                "specificite": spec,
                "metrics_per_class": metrics_per_class,
                "specificity_per_class": specificity_per_class_scores,
            }
        return report

    except Exception as e:
        logging.error(f"Exception occured {e}")
        raise e
