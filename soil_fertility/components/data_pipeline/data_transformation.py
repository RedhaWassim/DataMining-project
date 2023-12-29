from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from soil_fertility.components.utils.path_config import PathConfig
from typing import Literal, List, Optional

from sklearn.compose import ColumnTransformer
from soil_fertility.logger import logging
import pandas as pd
import os
from soil_fertility.utils import save_object

from soil_fertility.components.utils.transformations import (
    DropMissingValues,
    DropDuplicates,
    MinMaxTransformation,
    ZScoreTransformation,
    EqualWidthDescritizer,
    EqualFreqDescritizer,
    DateTimeTransformer,
)

class GeneralProcessing(BaseModel):
    def generate_transformer(self):
        all_features_pipeline = Pipeline(
            [
                ("drop_missing_values", DropMissingValues()),
                ("drop_duplicates", DropDuplicates()),
            ]
        )
        return all_features_pipeline

    def transform(self, data):
        try:
            logging.info("general processing started")

            all_features_pipeline = self.generate_transformer()
            processed_data = all_features_pipeline.fit_transform(data)

            logging.info("general processing completed")

            save_object(
                r"/home/redha/Documents/projects/NLP/datamining project/Soil-Fertility/artifacts/preprocessors/all_features_pipeline.pkl",
                all_features_pipeline,
            )

            return processed_data

        except Exception as e:
            logging.error(f"Exception occured {e}")
            raise e


class DataTransformationConfig_one(PathConfig):
    def __init__(self):
        super().__init__()
        self.path_type: Literal["raw", "intermediate", "processed"] = "intermediate"
        self.update_path()

class DataTransformationConfig_two(PathConfig):
    def __init__(self):
        super().__init__()
        self.path_type: Literal["raw", "intermediate", "processed"] = "intermediate"
        self.part = 2

        self.update_path()

class DataTransformationConfig_three(PathConfig):
    def __init__(self):
        super().__init__()
        self.path_type: Literal["raw", "intermediate", "processed"] = "intermediate"
        self.part = 3
        self.update_path()




class DataTransformation(BaseModel):
    transformation_config: DataTransformationConfig_one = DataTransformationConfig_one()

    def generate_transformer_first_data(
        self,
        numerical_features: List[str],
        strategie: str = "minmax",
        target: str = None,
    ) -> None:
        try:
            if strategie == "minmax":
                numerical_pipeline = Pipeline(
                    [
                        ("imputer", MinMaxTransformation()),
                    ]
                )
                preprocessor = ColumnTransformer(
                    [
                        ("numerical", numerical_pipeline, numerical_features),
                    ],
                    remainder="passthrough",
                )
                preprocessor.set_output(transform="pandas")

                return preprocessor

            elif strategie == "zscore":
                numerical_pipeline = Pipeline(
                    [
                        ("imputer", ZScoreTransformation(strategy="median")),
                    ]
                )
                preprocessor = ColumnTransformer(
                    [
                        ("numerical", numerical_pipeline, numerical_features),
                    ],
                    remainder="passthrough",
                )
                preprocessor.set_output(transform="pandas")

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
        target: str = None,
        numerical_features: Optional[List[str]] = None,
        strategie: Literal["minmax", "zscore"] = "minmax",
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

            if numerical_features is None:
                numerical_features = train_df.select_dtypes(
                    include=["int64", "float64"]
                ).columns

            if target is not None:
                numerical_features = numerical_features.drop(target)

            preprocessors = self.generate_transformer_first_data(
                numerical_features=numerical_features,
                strategie=strategie,
                target=target,
            )

            logging.info("transforming train data")
            print("train_coolumns= ", train_df.columns)
            print("test_coolumns= ", test_df.columns)

            processed_train = preprocessors.fit_transform(train_df)
            processed_test = preprocessors.transform(test_df)


            processed_train.columns = train_df.columns
            processed_test.columns = test_df.columns
            
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
            save_object(
                r"/home/redha/Documents/projects/NLP/datamining project/Soil-Fertility/artifacts/preprocessors/preprocessor_first.pkl",
                preprocessors,
            )

            return values
        except Exception as e:
            logging.error(f"Exception occured {e}")
            raise e




class DataTransformationTwo(BaseModel):
    transformation_config: DataTransformationConfig_two = DataTransformationConfig_two()

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



class DataTransformationThree(BaseModel):
    transformation_config: DataTransformationConfig_three = DataTransformationConfig_three()

    def generate_transformer_third_data(
        self, numerical_features: List[str], strategie: str = "frequency", k: int = 5
    ) -> None:
        try:
            if strategie == "frequency":
                preprocessor = Pipeline(
                    [
                        (
                            "equal_freq_descritizer",
                            EqualFreqDescritizer(k, columns=numerical_features),
                        ),
                    ]
                )

                return preprocessor

            elif strategie == "width":
                preprocessor = Pipeline(
                    [
                        (
                            "equal_width_descritizer",
                            EqualWidthDescritizer(k, columns=numerical_features),
                        ),
                    ]
                )

                return preprocessor
            else:
                logging.error("strategie must be width or frequency")
                raise ValueError("strategie must be width or frequency")
        except Exception as e:
            logging.error(f"Exception occured {e}")
            raise e

    def transform(
        self,
        train_path: str = None,
        test_path: str = None,
        target: Optional[str] = None,
        k: int = 5,
        numerical_features: List[str] = ["Temperature"],
        strategie: Literal["frequency", "width"] = "frequency",
        save: bool = True,
        train: Optional[pd.DataFrame] = None,
        test: Optional[pd.DataFrame] = None,
        return_df: bool = False,
    ):
        try:
            if train is not None and test is not None:
                train_df = train
                test_df = test

            else:
                logging.info("reading train and test data")
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

            df = pd.concat([train_df, test_df], axis=0)

            if save:
                os.makedirs(
                    os.path.dirname(self.transformation_config.raw_data_path),
                    exist_ok=True,
                )

            logging.info("data transformation started")

            logging.info("generating preprocessor object")

            preprocessors = self.generate_transformer_third_data(
                numerical_features=numerical_features, strategie=strategie, k=k
            )

            logging.info("transforming train data")
            processed_df = preprocessors.fit_transform(df)

            logging.info("data transformation completed")

            logging.info("saving transformed data")

            if save:
                processed_df.to_csv(
                    self.transformation_config.raw_data_path, index=False
                )

            logging.info("data saved")

            if return_df:
                return processed_df, preprocessors
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
