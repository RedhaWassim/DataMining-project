from soil_fertility.logger import logging
from typing import Dict
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def evaluate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: Dict[str, object],
) -> Dict[str, float]:
    try:
        report = {}
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_score

        return report

    except Exception as e:
        logging.error(f"Exception occured {e}")
        raise e


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
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            grid_search = GridSearchCV(model, para, cv=5, scoring="accuracy")
            grid_search.fit(X_train, y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_score

        return report

    except Exception as e:
        logging.error(f"Exception occured {e}")
        raise e
