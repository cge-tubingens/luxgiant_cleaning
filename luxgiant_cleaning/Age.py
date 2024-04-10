"""
Python module to correct age
"""

import re

import pandas as pd
import numpy as np

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

class BasicAgeImputer(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for imputing missing or invalid age values.

    Parameters:
    -----------
    lower_age : int, default=18
        The lower age limit. Ages below this limit will be imputed.
    upper_age : int, default=90
        The upper age limit. Ages above this limit will be imputed.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'BasicAgeImputer':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : BasicAgeImputer
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by imputing missing or invalid age values.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with an age column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with imputed age values.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self, lower_age:int=18, upper_age:int=90) -> None:
        """
        Initialize the BasicAgeImputer transformer.

        Parameters:
        -----------
        lower_age : int, default=18
            The lower age limit. Ages below this limit will be imputed.
        upper_age : int, default=90
            The upper age limit. Ages above this limit will be imputed.
        """
        super().__init__()
        self.lower_age = lower_age
        self.upper_age = upper_age

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:
        """
        Transform the input DataFrame by imputing missing or invalid age values.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with an age column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with imputed age values.
        """

        X_copy = X.copy()
        cols = X_copy.columns

        X_copy[cols[0]] = X_copy[cols[0]].astype(float)

        idx_null = X_copy[X_copy[cols[0]].isnull()].reset_index()['index'].to_list()
        idx_youn = X_copy[X_copy[cols[0]]<self.lower_age].reset_index()['index'].to_list()
        idx_old  = X_copy[X_copy[cols[0]]>self.upper_age].reset_index()['index'].to_list()

        idx_lsts = [idx_null, idx_youn, idx_old]

        for lst in idx_lsts:
            for idx in lst:

                diff = (X_copy.iloc[idx, 2].year - X_copy.iloc[idx, 1].year)
                X_copy.iloc[idx, 0] = diff

        X_copy[cols[0]] = X_copy[cols[0]].apply(lambda x: np.nan if x<18 else x)

        return X_copy
 