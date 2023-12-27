import pandas as pd
from soil_fertility.logger import logging   
from soil_fertility.utils import retreive_base_path, load_object
import os

class PredictPipeline:
    def __init__(self):
        base_path=retreive_base_path()
        artifacts_path=os.path.join(base_path,"artifacts")
        self.models_path = os.path.join(artifacts_path, "models")
        self.preprocessors_path = os.path.join(artifacts_path, "preprocessors")
    
    def predict(self, features):
        model_path = os.path.join(self.models_path, "model.pkl")
        preprocessor_path = os.path.join(self.preprocessors_path, "preprocessor.pkl")
        model=load_object(model_path)
        preprocessor=load_object(prep)

class InputData:
    def __init__(
        self,
        N: int,
        P: int,
        K: int,
        pH: int,
        EC: int,
        OC: int,
        S: int,
        Zn: int,
        Fe: int,
        Cu: int,
        Mn: int,
        B: int,
        OM: int,
    ):
        self.N = N
        self.P = P
        self.K = K
        self.pH = pH
        self.EC = EC
        self.OC = OC
        self.S = S
        self.Zn = Zn
        self.Fe = Fe
        self.Cu = Cu
        self.Mn = Mn
        self.B = B
        self.OM = OM

    def get_data_as_df(self):
        try:
            data_dict = {
                "N": [self.N],
                "P": [self.P],
                "K": [self.K],
                "pH": [self.pH],
                "EC": [self.EC],
                "OC": [self.OC],
                "S": [self.S],
                "Zn": [self.Zn],
                "Fe": [self.Fe],
                "Cu": [self.Cu],
                "Mn": [self.Mn],
                "B": [self.B],
                "OM": [self.OM],
            }
            df = pd.DataFrame(data_dict)
            return df
        except Exception as e:
            logging.error(f"Exception occured {e}")
            raise e
