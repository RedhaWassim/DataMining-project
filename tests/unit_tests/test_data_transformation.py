from soil_fertility.components.data_transformation.data_transformation import (
    DataTransformation,
    GeneralProcessing,
)
import pandas as pd
from sklearn.compose import ColumnTransformer


def test_data_transformation():
    train_path = r"artifacts/raw/1/train.csv"
    test_path = r"artifacts/raw/1/test.csv"
    tester = DataTransformation()

    results = tester.transform(train_path, test_path)
    assert results[0] == 1
    assert type(results[1]) == ColumnTransformer
    assert (
        results[2]
        == "/home/redha/Documents/projects/NLP/datamining project/Soil-Fertility/artifacts/intermediate/1/train.csv"
    )
    assert (
        results[3]
        == "/home/redha/Documents/projects/NLP/datamining project/Soil-Fertility/artifacts/intermediate/1/test.csv"
    )


def test_general_processing():
    tester = GeneralProcessing()
    data = pd.read_csv(
        "/home/redha/Documents/projects/NLP/datamining project/Soil-Fertility/data/Dataset1.csv"
    )
    results = tester.transform(data)
    print(results.shape)

    assert len(results.shape) == 2
    assert results.shape[0] == 622
    assert results.shape[1] == 14
