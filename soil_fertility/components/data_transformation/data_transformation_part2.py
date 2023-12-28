from pydantic import BaseModel
from soil_fertility.components.path_config import PathConfig
from typing import Literal, List, Optional
from sklearn.pipeline import Pipeline
from soil_fertility.components.data_transformation.transformations import (
    DropMissingValues,
    DropDuplicates,
    CustomImputer,
    MinMaxTransformation,
    ZScoreTransformation,
    EqualWidthDescritizer,
    EqualFreqDescritizer,
    DateTimeTransformer,
)
from soil_fertility.logger import logging
import pandas as pd
import os


class DataTransformationConfig(PathConfig):
    def __init__(self):
        super().__init__()
        self.path_type: Literal["raw", "intermediate", "processed"] = "intermediate"
        self.part = 2

        self.update_path()


class DataTransformationTwo(BaseModel):
    transformation_config: DataTransformationConfig = DataTransformationConfig()

    def generate_transformer_second_data(self) -> Pipeline:
        try:
            preprocessor = Pipeline(
                [
                    ("datetime_processing", DateTimeTransformer()),
                ]
            )

            return preprocessor

        except Exception as e:
            logging.error(f"Exception occured {e}")
            raise e

    def transform(
        self,
        data_path: str,
    ):
        try:
            logging.info("reading train and test data")
            df = pd.read_csv(data_path)

            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)

            os.makedirs(
                os.path.dirname(self.transformation_config.raw_data_path), exist_ok=True
            )

            logging.info("data transformation started")

            logging.info("generating preprocessor object")

            preprocessors = self.generate_transformer_second_data()

            logging.info("transforming data")
            processed_data = preprocessors.fit_transform(df)

            logging.info("data transformation completed")

            logging.info("saving transformed data")

            processed_data.to_csv(self.transformation_config.raw_data_path, index=False)

            logging.info("data saved")

            values = (
                self.transformation_config.part,
                preprocessors,
                self.transformation_config.raw_data_path,
            )

            self.transformation_config.part += 1
            self.transformation_config.update_path()

            logging.info("data transformation completed")

            return values
        except Exception as e:
            logging.error(f"Exception occured {e}")
            raise e
