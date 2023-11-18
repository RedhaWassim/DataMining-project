import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Union, Dict
from soil_fertility.components.utils import CalculateTendencies


class DropMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.none_values = [
            "",
            " ",
            "nan",
            "NaN",
            "Nan",
            "NAN",
            "None",
            "none",
            "NONE",
            "null",
            "Null",
            "NULL",
            "?",
            "NA",
            "na",
            "Na",
            "nA",
            None,
            np.nan,
        ]

    def _drop_rows_with_value(
        self, df: pd.DataFrame, custom_drop_values: List[Union[str, int, float, None]]
    ) -> pd.DataFrame:
        """
        Drops rows in a DataFrame where a specified value is found.

        Parameters:
        - df: pandas DataFrame
        - custom_drop_values: the value to drop from the specified column

        Returns:
        - Modified DataFrame with rows containing the specified value dropped
        """

        new_df = df.copy()
        columns_transformed = []
        empty_values = self.none_values + custom_drop_values

        for col in new_df.columns:
            if pd.api.types.is_numeric_dtype(new_df[col]):
                # Handle numeric columns, including NaN
                new_df = new_df[new_df[col].notna() & ~new_df[col].isin(empty_values)]
            else:
                # Handle non-numeric columns
                new_df = new_df[~new_df[col].astype(str).str.lower().isin(empty_values)]
                columns_transformed.append(col)

        for col in columns_transformed:
            try:
                new_df[col] = new_df[col].astype(float)
            except Exception:
                pass

        return new_df

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self._drop_rows_with_value(X, custom_drop_values=["?"])


class DropDuplicates(BaseEstimator, TransformerMixin):
    def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """drop duplicates rows in the dataset

        Args:
            df (pd.DataFrame): dataframe to drop duplicates from

        Returns:
            pd.DataFrame: dataframe without duplicates
        """
        new_df = df.copy()
        dataframe_rows = []

        for _, row in new_df.iterrows():
            row_tuple = tuple(row)
            dataframe_rows.append(row_tuple)

        dropped_set = set(dataframe_rows)

        unique_rows = [list(row_tuple) for row_tuple in dropped_set]

        unique_df = pd.DataFrame(unique_rows, columns=new_df.columns)
        return unique_df

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop_duplicates()


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, stabelize: bool = False, show: bool = False) -> None:
        self.stabelize = stabelize
        self.show = show

    def _tendencies(self, df: pd.DataFrame) -> CalculateTendencies:
        self.tendencies = CalculateTendencies(df)

    def _find_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops outliers from a DataFrame using the IQR method.

        Args:
            df (pd.Dataframe): the dataset to drop outliers from

        Returns:
            pd.Dataframe: the dataset without outliers
        """
        new_df = df.copy()
        self._tendencies(new_df)

        quartiles_dict = self.tendencies.quartiles(percentiles=[0, 0.25, 0.5, 0.75, 1])

        for col in new_df.columns:
            if (
                df[col].dtype == "datetime64[ns]"
                or df[col].dtype == "object"
                or df[col].dtype == "string"
            ):
                continue
            q1, q3 = quartiles_dict[col][1], quartiles_dict[col][3]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            if self.show:
                print(
                    f"Column {col} : lower bound = {lower_bound} , upper bound = {upper_bound}"
                )
                print(
                    f"Outliers : {new_df[(new_df[col] < lower_bound) | (new_df[col] > upper_bound)][col]}"
                )

            new_df = new_df[(new_df[col] >= lower_bound) & (new_df[col] <= upper_bound)]

        return new_df

    def drop_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """drop outliers until data stabilizes

        Args:
            df (pd.DataFrame): the dataframe to drop outliers from

        Returns:
            pd.DataFrame: the dataframe without outliers
        """
        new_df = df.copy()
        if self.stabelize:
            while True:
                df_without_outliers = self._find_outliers(new_df)
                if len(df_without_outliers) == len(new_df):
                    return df_without_outliers
                new_df = df_without_outliers

        else:
            df_without_outliers = self._find_outliers(new_df)
            return df_without_outliers

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.drop_outliers(X)


class MinMaxTransformation(BaseEstimator, TransformerMixin):
    def _tendencies(self, df: pd.DataFrame) -> CalculateTendencies:
        self.tendencies = CalculateTendencies(df)

    def _MinMaxTransformer(self, df: pd.DataFrame) -> pd.DataFrame:
        """MinMaxTransformer function to normalize the data

        Args:
            df (pd.DataFrame): the dataframe to normalize

        Returns:
            pd.DataFrame: the normalized dataframe
        """
        new_df = df.copy()
        infos = self.central_tendenceies
        for col in new_df.columns:
            max = infos[col].get("max")
            min = infos[col].get("min")
            diff = max - min
            x_diff = new_df[col] - min
            new_val = x_diff / diff
            new_df[col] = new_val
        return new_df

    def fit(self, X, y=None):
        self._tendencies(X)
        self.central_tendenceies = self.tendencies.calculate_central_tendencies
        return self

    def transform(self, X, y=None):
        return self._MinMaxTransformer(X)


class ZScoreTransformation(BaseEstimator, TransformerMixin):
    def _tendencies(self, df: pd.DataFrame) -> CalculateTendencies:
        self.tendencies = CalculateTendencies(df)

    def _ZScoreTransformer(self, df: pd.DataFrame) -> pd.DataFrame:
        """this function normalize the data using the zscore method

        Args:
            df (pd.DataFrame): the dataframe to normalize

        Returns:
            pd.DataFrame: dataframe normalized
        """
        new_df = df.copy()
        infos = self.central_tendenceies
        for col in new_df.columns:
            mean = infos[col].get("mean")
            std = infos[col].get("std")
            new_df[col] = (new_df[col] - mean) / std

        return new_df

    def fit(self, X, y=None):
        self._tendencies(X)
        self.central_tendenceies = self.tendencies.calculate_central_tendencies
        return self

    def transform(self, X, y=None):
        return self._ZScoreTransformer(X)
