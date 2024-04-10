"""
Python module to process MOCA scores
"""

import re

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class CleanerMOCA(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for cleaning data in the MOCA (Montreal Cognitive Assessment) column.

    This transformer applies cleaning operations to the MOCA column, such as removing special characters and extracting relevant information.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'CleanerMOCA':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : CleanerMOCA
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by applying cleaning operations to the MOCA column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the MOCA column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with cleaned MOCA column.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):
        """
        Transform the input DataFrame by applying cleaning operations to the MOCA column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the MOCA column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with cleaned MOCA column.
        """

        X_copy = X.copy()
        col = X.columns[0]

        X_copy[col] = X_copy[col].apply(
            lambda x: x.split('/')[0] if '/' in x else re.sub(r'[^a-zA-Z0-9]', '', x)
        )

        return X_copy

class FormatMOCA(BaseEstimator, TransformerMixin):
    """
    A scikit-learn custom transformer for formatting data in the MOCA (Montreal Cognitive Assessment) column.

    This transformer converts string representations to numerical values in the MOCA column.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'FormatMOCA':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : FormatMOCA
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by converting string representations to numerical values in the MOCA column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the MOCA column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with formatted MOCA column.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):

        """
        Transform the input DataFrame by converting string representations to numerical values in the MOCA column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the MOCA column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with formatted MOCA column.
        """

        X_copy = X.copy()
        col = X.columns[0]

        X_copy[col] = X_copy[col].apply(
            lambda x: self.convert_string_to_nan(x)
        )

        return X_copy
    
    @staticmethod
    def convert_string_to_nan(input_string:str):

        """
        Convert a string representation to a numerical value.

        Parameters:
        -----------
        x : str
            The input string to be converted.

        Returns:
        --------
        value : float or np.nan
            The converted numerical value or np.nan if the conversion fails.
        """

        if input_string=='':
            return np.nan
        
        try:
            return float(input_string)
        except ValueError:
            return np.nan

class ExtremesMOCA(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for handling extreme values in the MOCA (Montreal Cognitive Assessment) column.

    This transformer replaces values in the MOCA column that are outside the range [0, 30] with np.nan.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'ExtremesMOCA':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : ExtremesMOCA
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by replacing values outside the range [0, 30] in the MOCA column with np.nan.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the MOCA column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with extreme values in the MOCA column replaced by np.nan.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):

        """
        Transform the input DataFrame by replacing values outside the range [0, 30] in the MOCA column with np.nan.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the MOCA column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with extreme values in the MOCA column replaced by np.nan.
        """

        X_copy = X.copy()
        col = X.columns[0]

        X_copy[col] = X_copy[col].apply(
            lambda x: self.upper_and_lower(x)
        )

        return X_copy
    
    @staticmethod
    def upper_and_lower(x:float):
        """
        Replace values outside the range [0, 30] with np.nan.

        Parameters:
        -----------
        x : float
            The input value to be checked.

        Returns:
        --------
        value : float or np.nan
            The original value if it is within the range [0, 30], otherwise np.nan.
        """

        if 0<= x and x <= 30:
            return x
        else: return np.nan
