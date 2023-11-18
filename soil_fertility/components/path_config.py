from soil_fertility.utils import retreive_base_path
from pydantic import BaseModel
from typing import Literal
from pathlib import Path


class PathConfig(BaseModel):
    part: int = 1
    base_path: str = retreive_base_path()
    path_type: Literal["raw", "intermediate", "processed"] = "raw"
    raw_data_path: str = str(Path(base_path, f"artifacts/{path_type}/{part}/data.csv"))
    train_data_path: str = str(
        Path(base_path, f"artifacts/{path_type}/{part}/train.csv")
    )
    test_data_path: str = str(Path(base_path, f"artifacts/{path_type}/{part}/test.csv"))

    def update_path(self):
        self.raw_data_path = str(
            Path(self.base_path, f"artifacts/{self.path_type}/{self.part}/data.csv")
        )
        self.train_data_path = str(
            Path(self.base_path, f"artifacts/{self.path_type}/{self.part}/train.csv")
        )
        self.test_data_path = str(
            Path(self.base_path, f"artifacts/{self.path_type}/{self.part}/test.csv")
        )
