import pandas as pd
from typing import Dict, List
import numpy as np
from collections import Counter
from pydantic import BaseModel


class CalculateTendencies:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe

    @property
    def calculate_central_tendencies(self) -> Dict[str, Dict[str, float]]:
        """this function calculate the central tendencies : mean , median , mode , min , max of each column in the dataframe
        Args:
            df (pd.Dataframe): pandas dataframe to calculate the central tendencies

        Returns:
            Dict: dictionary of the central tendencies of each column
        """
        tendencies = {}
        new_df = self.dataframe.copy()

        for column in new_df.columns:
            if new_df[column].dtype == "object" or new_df[column].dtype == "string":
                tendencies[column] = {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "median": None,
                    "mode": None,
                    "std": None,
                }
            else:
                if new_df[column].dtype == "datetime64[ns]":
                    new_df[column] = new_df[column].astype(np.int64)

                # calculate mean
                col_len = len(new_df[column])
                mean = sum(new_df[column]) / col_len

                # calculate median
                sorted_column = sorted(new_df[column])
                if col_len % 2 == 0:
                    median1 = sorted_column[col_len // 2]
                    median2 = sorted_column[col_len // 2 - 1]
                    median = (median1 + median2) / 2
                else:
                    median = sorted_column[col_len // 2]

                # calculate mode
                counter = Counter(new_df[column])
                mode = counter.most_common(1)[0][0]

                # calculate min
                minimum = sorted_column[0]

                # calculate max
                maximum = sorted_column[-1]

                # calculate standard deviation
                std = 0
                for data in new_df[column]:
                    std += (data - mean) ** 2
                std = std / col_len
                std = std**0.5

                if self.dataframe[column].dtype == "datetime64[ns]":
                    mean, median, mode, minimum, maximum, std = pd.to_datetime(
                        [mean, median, mode, minimum, maximum, std]
                    )

                tendencies[column] = {
                    "min": minimum,
                    "max": maximum,
                    "mean": mean,
                    "median": median,
                    "mode": mode,
                    "std": std,
                }

        return tendencies

    def quartiles(self, percentiles: list) -> Dict[str, List[float]]:
        """this function calculate the quartiles of each column in the dataframe

        Args:
            df (pd.DataFrame): the dataframe to calculate the quartiles
            percentiles (List): list of the percentiles to calculate

        Returns:
            Dict[str,Dict[str,float]]: dictionary of the quartiles of each column
        """
        new_df = self.dataframe.copy()
        quartiles_dict = {}

        for percentile in percentiles:
            if not (0 <= percentile <= 1):
                raise ValueError("Percentile must be between 0 and 1")

        indexes = [
            (int(percentile * len(new_df)), percentile) for percentile in percentiles
        ]

        for col in new_df.columns:
            col_quartiles = []
            sorted_column = sorted(new_df[col])
            for index, percentile in indexes:
                if percentile == 0:
                    tendencies = self.calculate_central_tendencies[col]
                    col_quartiles.append(tendencies.get("min"))
                elif percentile == 1:
                    col_quartiles.append(tendencies.get("max"))
                else:
                    col_quartiles.append(sorted_column[index])
            quartiles_dict[col] = col_quartiles

        return quartiles_dict
