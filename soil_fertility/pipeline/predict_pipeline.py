import pandas as pd
from soil_fertility.logger import logging
from soil_fertility.utils import retreive_base_path, load_object
import os
from soil_fertility.components.model_pipeline.monitoring import DataDriftDetector   


def fix_columns_name(data):
    names = [
        "N",
        "P",
        "K",
        "pH",
        "EC",
        "OC",
        "S",
        "Zn",
        "Fe",
        "Cu",
        "Mn",
        "B",
        "OM",
        "Fertility",
    ]
    data.columns = names
    return data


class PredictPipeline:
    def __init__(self):
        base_path = retreive_base_path()
        artifacts_path = os.path.join(base_path, "artifacts")
        self.models_path = os.path.join(artifacts_path, "models")
        self.preprocessors_path = os.path.join(artifacts_path, "preprocessors")
        self.train_data_path = os.path.join(artifacts_path, "/1/train")

    def predict(self, features, model_name: str, drift: bool = False):
        try:
            model_path = os.path.join(self.models_path, model_name)
            all_processor_path = os.path.join(
                self.preprocessors_path, "all_features_pipeline.pkl"
            )
            preprocessor_path = os.path.join(
                self.preprocessors_path, "preprocessor_first.pkl"
            )

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            all_processor = load_object(all_processor_path)


            fixed_data = all_processor.transform(features)
            scaled_data = preprocessor.transform(fixed_data)
            data = fix_columns_name(scaled_data)

            data.drop(columns=["Fertility"], inplace=True)


            prediction = model.predict(data.to_numpy())[0]

            if drift :
                self.data_drift_check(data)

            return prediction
        except Exception as e:
            logging.error(f"Exception occured {e}")
            raise e

    def data_drift_check(self, data , model_name: str):
        drift_detector = DataDriftDetector(data)

        new_pred_data = load_new_prediction_data()

        reference_data = load_reference_data()


        drift_results = drift_detector.detect_drift(new_pred_data)

        if drift_detector.report_drift(drift_results):
    # Handle drift detection, e.g., retrain model, alert, etc.
            handle_drift()




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
                "Fertility": [0],
            }
            df = pd.DataFrame(data_dict)
            return df
        except Exception as e:
            logging.error(f"Exception occured {e}")
            raise e
