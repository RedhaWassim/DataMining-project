import pandas as pd
from scipy.stats import ks_2samp


class DataDriftDetector:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.reference_data = self.reference_data.drop(columns=["Fertility"])

    def detect_drift(self, new_data: pd.DataFrame):
        drift_results = {}
        for column in self.reference_data.columns:
            statistic, p_value = ks_2samp(self.reference_data[column], new_data[column])
            drift_results[column] = {"statistic": statistic, "p_value": p_value}

        return drift_results

    def report_drift(self, drift_results):
        drift_detected = False
        for column, result in drift_results.items():
            if result["p_value"] < 0.0001:  # Threshold for significance
                print("------------------------------------------------------------")
                print(column)
                drift_detected = True
        return drift_detected, drift_results

    def handle_drift():
        return "Drift detected, model retraining required"
