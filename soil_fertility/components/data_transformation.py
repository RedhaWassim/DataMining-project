from pydantic import BaseModel
from soil_fertility.components.path_config import PathConfig
from typing import Literal, List
from sklearn.pipeline import Pipeline


class DataTransformationConfig(PathConfig):
    def __init__(self):
        super().__init__()
        self.path_type: Literal["raw", "intermediate", "processed"] = "intermediate"
        self.update_path()


class DataTransformation(BaseModel):
    transformation_config: DataTransformationConfig = DataTransformationConfig()

    def generate_transformer(
        self, numerical_features: List[str], categorical_features: List[str]
    ) -> None:
        try:
            numerical_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            categorical_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            preprocessor = ColumnTransformer(
                [
                    ("numerical", numerical_pipeline, numerical_features),
                    ("categorical", categorical_pipeline, categorical_features),
                ]
            )
            return preprocessor

        except:
            pass

    def transform(self) -> None:
        pass
