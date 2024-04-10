"""
Python module to correct age
"""

import re

import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator

class AgeCorrector(TransformerMixin, BaseEstimator):

    """
    A scikit-learn custom transformer for correcting age values in a DataFrame column.
    
    This transformer extracts numerical values from the specified column and handles empty or non-numeric values.
    
    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'AgeCorrector':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        self : AgeCorrector
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by correcting age values in the specified column.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with age values in the specified column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with corrected age values.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self) -> None:
        super().__init__()
    
    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None) -> pd.DataFrame:

        """
        Transform the input DataFrame by correcting age values in the specified column.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with age values in the specified column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with corrected age values.
        """

        X_copy = X.copy()
        col = X_copy.columns[0]

        X_copy[col] = X_copy[col].apply(lambda x: self.extract_numbers(x))
        X_copy[col] = X_copy[col].apply(lambda x: None if len(x)==0 else x)

        return X_copy

    @staticmethod
    def extract_numbers(text:str):

        """
        Extract numerical values from a given text.
        
        Parameters:
        -----------
        text : str
            Input text.
            
        Returns:
        --------
        number : str
            Extracted numerical values from the input text.
        """

        number = re.sub(r'[^0-9]', '', text)
        return number
