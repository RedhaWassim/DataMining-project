from pydantic import BaseModel
from soil_fertility.components.path_config import PathConfig
from typing import Literal, List, Optional
from sklearn.pipeline import Pipeline
from soil_fertility.components.transformations import (
    DropMissingValues,
    DropDuplicates,
    CustomImputer,
    MinMaxTransformation,
    ZScoreTransformation,
    EqualWidthDescritizer,
    EqualFreqDescritizer,
)
from sklearn.compose import ColumnTransformer
from soil_fertility.logger import logging
import pandas as pd
import os


class DataTransformationConfig(PathConfig):
    def __init__(self):
        super().__init__()
        self.path_type: Literal["raw", "intermediate", "processed"] = "intermediate"
        self.part = 3
        self.update_path()


class DataTransformationThree(BaseModel):
    transformation_config : DataTransformationConfig = DataTransformationConfig()

    def generate_transformer_third_data(
        self, numerical_features: List[str], strategie: str = "frequency", k : int = 5
    ) -> None:
        try:
            if strategie == "frequency":
                preprocessor = Pipeline(
                    [
                        ("equal_freq_descritizer", EqualFreqDescritizer(k,columns=numerical_features)),
                    ]
                )

                return preprocessor

            elif strategie == "width":
                preprocessor = Pipeline(
                    [
                        ("equal_width_descritizer", EqualWidthDescritizer(k,columns=numerical_features)),
                    ]
                )

                return preprocessor
            else:
                logging.error("strategie must be minmax or zscore")
                raise ValueError("strategie must be minmax or zscore")
        except Exception as e:
            logging.error(f"Exception occured {e}")
            raise e
            

    def transform(
        self,
        train_path: str,
        test_path: str,
        target: Optional[str] = None,
        k : int = 5,
        numerical_features : List[str] = ["Temperature"],
        strategie : Literal["frequency", "width"] = "frequency",
    ):
        try:
            logging.info("reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            os.makedirs(
                os.path.dirname(self.transformation_config.raw_data_path), exist_ok=True
            )

            logging.info("data transformation started")

            logging.info("generating preprocessor object")

            preprocessors = self.generate_transformer_third_data(
                numerical_features=numerical_features, strategie=strategie, k=k
            )

            logging.info("transforming train data")
            processed_train = preprocessors.fit_transform(train_df)
            processed_test = preprocessors.fit_transform(test_df)

            logging.info("data transformation completed")

            logging.info("saving transformed data")

            processed_train.to_csv(
                self.transformation_config.train_data_path, index=False
            )
            processed_test.to_csv(
                self.transformation_config.test_data_path, index=False
            )

            logging.info("data saved")

            values = ( 
                self.transformation_config.part,
                preprocessors,
                self.transformation_config.train_data_path,
                self.transformation_config.test_data_path,
            )

            self.transformation_config.part += 1
            self.transformation_config.update_path()

            logging.info("data transformation completed")

            return values
        except Exception as e:
            logging.error(f"Exception occured {e}")
            raise e
        

