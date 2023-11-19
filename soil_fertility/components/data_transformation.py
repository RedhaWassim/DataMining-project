from pydantic import BaseModel
from soil_fertility.components.path_config import PathConfig
from typing import Literal, List
from sklearn.pipeline import Pipeline
from soil_fertility.components.transformations import (
    DropMissingValues,
    DropDuplicates,
    CustomImputer,
    MinMaxTransformation,
    ZScoreTransformation
)
from sklearn.compose import ColumnTransformer

class DataTransformationConfig(PathConfig):
    def __init__(self):
        super().__init__()
        self.path_type: Literal["raw", "intermediate", "processed"] = "intermediate"
        self.update_path()


class DataTransformation(BaseModel):
    transformation_config: DataTransformationConfig = DataTransformationConfig()

    def generate_transformer_first_data(
        self, numerical_features: List[str], all_features: List[str]
    ) -> None:
        try:
            all_features_pipeline = Pipeline(
                [
                    ("drop_missing_values", DropMissingValues()),
                    ("drop_duplicates", DropDuplicates()),
                    ("custom_imputer", CustomImputer()),
                ]
            )

            numerical_pipeline = Pipeline(
                [
                    ("imputer", MinMaxTransformation(strategy="median")),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("all_features", all_features_pipeline, all_features),
                    ("numerical", numerical_pipeline, numerical_features),
                ]
            )
            return preprocessor

        except:
            pass
        
    def generate_transformer_second_data(
        self, numerical_features: List[str], all_features: List[str]
    ) -> None:
        try:
            all_features_pipeline = Pipeline(
                [
                    ("drop_missing_values", DropMissingValues()),
                    ("drop_duplicates", DropDuplicates()),
                    ("custom_imputer", CustomImputer()),
                ]
            )

            numerical_pipeline = Pipeline(
                [
                    ("imputer", MinMaxTransformation(strategy="median")),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("all_features", all_features_pipeline, all_features),
                    ("numerical", numerical_pipeline, numerical_features),
                ]
            )
            return preprocessor

        except:
            pass


    def generate_transformer_third_data(
        self, numerical_features: List[str], all_features: List[str]
    ) -> None:
        try:
            all_features_pipeline = Pipeline(
                [
                    ("drop_missing_values", DropMissingValues()),
                    ("drop_duplicates", DropDuplicates()),
                    ("custom_imputer", CustomImputer()),
                ]
            )

            numerical_pipeline = Pipeline(
                [
                    ("imputer", MinMaxTransformation(strategy="median")),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("all_features", all_features_pipeline, all_features),
                    ("numerical", numerical_pipeline, numerical_features),
                ]
            )
            return preprocessor

        except:
            pass


    def transform(self) -> None:
        pass
