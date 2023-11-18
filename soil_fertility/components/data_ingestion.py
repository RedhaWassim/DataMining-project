from pydantic import BaseModel
import os
from soil_fertility.logger import logging
from soil_fertility.utils import retreive_base_path
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict

class DataIngestionConfig(BaseModel):
    part : int = 1
    base_path : str = retreive_base_path()
    raw_data_path : str = os.path.join(base_path, f'artifacts/raw/{part}/data.csv')
    train_data_path : str = os.path.join(base_path, f'artifacts/raw/{part}/train.csv')
    test_data_path : str = os.path.join(base_path, f'artifacts/raw/{part}/test.csv')

    def update_path(self):
        self.part = self.part + 1
        self.raw_data_path = os.path.join(self.base_path, f'artifacts/raw/{self.part}/data.csv')
        self.train_data_path = os.path.join(self.base_path, f'artifacts/raw/{self.part}/train.csv')
        self.test_data_path = os.path.join(self.base_path, f'artifacts/raw/{self.part}/test.csv')


class DataIngestion(BaseModel):
    ingestion_config : DataIngestionConfig = DataIngestionConfig()
    
    def init_ingestion(self, path : str) -> Dict[str, str|int]:
        try :
            logging.info(f'ingestion started')   
            df=pd.read_csv(path)
            logging.info(f'Raw data read from {path}')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info(f'splitting data into train and test')
            raw_train, raw_test = train_test_split(df, test_size=0.2, random_state=42)
            raw_train.to_csv(self.ingestion_config.train_data_path, index=False)
            raw_test.to_csv(self.ingestion_config.test_data_path, index=False)
            
            self.ingestion_config.update_path()
            
            logging.info(f'ingestion completed')

            return {
                'part': self.ingestion_config.part,
                'train_data_path': self.ingestion_config.train_data_path,
                'test_data_path': self.ingestion_config.test_data_path
            }

        except FileNotFoundError as fnf:
            logging.error(f'File not found {fnf}')
            raise fnf
        except pd.errors.EmptyDataError as ede:
            logging.error(f'Empty data found {ede}')
            raise ede
        except Exception as e:
            logging.error(f'Exception occured {e}')
            raise e

    

if __name__=="__main__":
    tester=DataIngestion()
    tester.init_ingestion('/home/redha/Documents/projects/NLP/datamining project/Soil-Fertility/data/Dataset1.csv')
    tester.init_ingestion('/home/redha/Documents/projects/NLP/datamining project/Soil-Fertility/data/Dataset2.csv')