from soil_fertility.components.data_ingestion import DataIngestion


def test_data_ingestion():
    tester = DataIngestion()
    path = "/home/redha/Documents/projects/NLP/datamining project/Soil-Fertility/data/Dataset1.csv"
    results = tester.init_ingestion(path)
    assert results["part"] == 1
    assert (
        results["train_data_path"]
        == "/home/redha/Documents/projects/NLP/datamining project/Soil-Fertility/artifacts/raw/1/train.csv"
    )
    assert (
        results["test_data_path"]
        == "/home/redha/Documents/projects/NLP/datamining project/Soil-Fertility/artifacts/raw/1/test.csv"
    )
