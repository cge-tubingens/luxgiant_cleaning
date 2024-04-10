"""
Python module to estimate disease duration
"""

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class DiseaseDuration(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for computing disease duration based on assessment date and onset year.

    Parameters:
    ----------
    outputCol : str
        The name of the output column containing computed disease duration.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'DiseaseDuration':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : DiseaseDuration
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by computing disease duration based on assessment date and onset year.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with assessment date, onset year, and additional columns.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with computed disease duration.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self, outputCol:str) -> None:
        """
        Initialize the DiseaseDuration transformer.

        Parameters:
        -----------
        outputCol : str
            The name of the output column containing computed disease duration.
        """
        super().__init__()
        self.outputCol = outputCol

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):
        """
        Transform the input DataFrame by computing disease duration based on assessment date and onset year.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with assessment date, onset year, and additional columns.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with computed disease duration.
        """

        X_copy = X.copy()
        cols = X_copy.columns
        
        X_copy['comp_disease_dur'] = X_copy.apply(
            lambda row: self.compute_disease_duration(row[cols[0]], row[cols[1]]), axis=1
        )
        X_copy['comp_pd_onset'] = X_copy.apply(
            lambda row: self.compute_disease_duration(row[cols[0]], row[cols[4]]), axis=1
        )

        X_copy[self.outputCol] = X_copy.apply(
            lambda row: self.checking_null_values(row[cols[2]], row['comp_disease_dur']), axis=1
        )

        X_copy[self.outputCol] = X_copy.apply(
            lambda row: self.checking_null_values(row[self.outputCol], row[cols[3]]), axis=1
        )

        X_copy[self.outputCol] = X_copy.apply(
            lambda row: self.checking_null_values(row[self.outputCol], row['comp_pd_onset']), axis=1
        )

        return X_copy.drop(columns=['comp_disease_dur', 'comp_pd_onset'], inplace=False)
    
    @staticmethod
    def checking_null_values(x,y):

        """
        Check for null values and return a non-null value.

        Parameters:
        -----------
        x : float or None
            The first value to check.
        y : float or None
            The second value to check.

        Returns:
        --------
        result : float or None
            A non-null value based on the input values.
        """

        if x is None or np.isnan(x) and y is not None:
            return y
        else:
            return x
        
    @staticmethod
    def compute_disease_duration(date_assess: pd.Timestamp, year_onset: float):

        """
        Compute disease duration based on assessment date and onset year.

        Parameters:
        -----------
        date_assess : pd.Timestamp or str
            The assessment date.
        year_onset : float or str
            The onset year.

        Returns:
        --------
        duration : float or np.nan
            Computed disease duration.
        """

        if isinstance(date_assess, str): print(date_assess)

        if not pd.isna(date_assess) and not pd.isna(year_onset) and not isinstance(year_onset, str):
            duration = date_assess.year - float(year_onset)
            if 0 <= duration and duration <=31: return duration
            else: return np.nan
        else:
            return np.nan

class AgeOnset(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for computing age of onset based on birth year and onset year.

    Parameters:
    ----------
    outputCol : str, default='age_of_onset'
        The name of the output column containing computed age of onset.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'AgeOnset':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : AgeOnset
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by computing age of onset based on birth year and onset year.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with birth year and onset year columns.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with computed age of onset.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self, outputCol='age_of_onset') -> None:
        """
        Initialize the AgeOnset transformer.

        Parameters:
        -----------
        outputCol : str, default='age_of_onset'
            The name of the output column containing computed age of onset.
        """
        super().__init__()
        self.outputCol = outputCol

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):

        """
        Transform the input DataFrame by computing age of onset based on birth year and onset year.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with birth year and onset year columns.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with computed age of onset.
        """

        X_copy = X.copy()
        cols = X_copy.columns

        X_copy[self.outputCol] = X_copy[cols[0]].astype(float) - X_copy[cols[1]]

        return X_copy
