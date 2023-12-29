import os
from typing import Dict, Literal, Optional

import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from soil_fertility.logger import logging
from soil_fertility.components.path_config import PathConfig
from soil_fertility.components.data_transformation.data_transformation import (
    GeneralProcessing,
)


class DataIngestion(BaseModel):
    ingestion_config: PathConfig = PathConfig()
    general_processing: GeneralProcessing = GeneralProcessing()

    def init_ingestion(
        self,
        path: str = None,
        option: Literal["csv", "xlsx"] = "csv",
        save: Optional[bool] = True,
        data: Optional[pd.DataFrame] = None,
        return_df: bool = False,
    ) -> Dict[str, str | int]:
        try:
            logging.info("ingestion started")
            if data is None and path is None:
                raise Exception("Either data or path should be provided")
            if data is not None and path is not None:
                raise Exception("Either data or path should be provided")

            if data is not None:
                df = data

            if path is not None:
                if option == "csv":
                    df = pd.read_csv(path)
                elif option == "xlsx":
                    df = pd.read_excel(path)

            logging.info("Raw data read from {path}")
            if save:
                os.makedirs(
                    os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True
                )
                df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("starting general processing")

            new_df = self.general_processing.transform(df)

            logging.info("general processing completed")

            logging.info("splitting data into train and test")

            raw_train, raw_test = train_test_split(
                new_df, test_size=0.2, random_state=42
            )
            if save:
                raw_train.to_csv(self.ingestion_config.train_data_path, index=False)
                raw_test.to_csv(self.ingestion_config.test_data_path, index=False)

            if return_df:
                return raw_train, raw_test
            values = (
                self.ingestion_config.part,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

            self.ingestion_config.part += 1

            self.ingestion_config.update_path()

            logging.info("ingestion completed")

            return values

        except FileNotFoundError as fnf:
            logging.error(f"File not found {fnf}")
            raise fnf
        except pd.errors.EmptyDataError as ede:
            logging.error(f"Empty data found {ede}")
            raise ede
        except Exception as e:
            logging.error(f"Exception occured {e}")
            raise e
