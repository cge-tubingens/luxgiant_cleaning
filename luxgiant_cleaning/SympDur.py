"""
Python module to format symptoms duration
"""

import re

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from typing import Callable

class SymptomDuration(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for processing symptom duration values in a DataFrame column.
    
    This transformer applies various string transformations to clean and convert symptom duration values into a standardized format.
    
    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'SymptomDuration':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        self : SymptomDuration
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by processing symptom duration values in the specified column.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with symptom duration values in the specified column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with processed symptom duration values.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        """
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        self : SymptomDuration
            Returns the instance itself.
        """
        return self
    
    def transform(self, X:pd.DataFrame, y=None):
        """
        Transform the input DataFrame by processing symptom duration values in the specified column.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with symptom duration values in the specified column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with processed symptom duration values.
        """

        X_copy = X.copy()
        col = X_copy.columns[0]
        X_copy[col] = X_copy[col].apply(
            lambda x: self.string_transformations(
                input_string = x, 
                fun_replace  = self.replace_non_alphanumeric_with_space, 
                fun_add_space= self.add_space_between_numbers_and_letters
            )
        )

        return X_copy
    
    @staticmethod
    def replace_non_alphanumeric_with_space(input_string:str):
        """
        Replace non-alphanumeric characters with spaces in a given string.
        
        Parameters:
        -----------
        input_string : str
            Input string.
            
        Returns:
        --------
        cleaned_string : str
            String with non-alphanumeric characters replaced by spaces.
        """
        # Use a regular expression to replace non-alphanumeric characters with spaces
        cleaned_string = re.sub(r'[^a-zA-Z0-9.]', ' ', input_string)
        return cleaned_string
    
    @staticmethod
    def add_space_between_numbers_and_letters(input_string:str):
        """
        Add spaces between numbers and letters in a given string.
        
        Parameters:
        -----------
        input_string : str
            Input string.
            
        Returns:
        --------
        result : str
            String with added spaces between numbers and letters.
        """
        # Use a regular expression to find substrings of numbers and letters and add a space
        pattern = r'(\d+)([a-zA-Z]+)|([a-zA-Z]+)(\d+)'
        result = re.sub(pattern, r'\1 \2\3 \4', input_string)
        return result

    @staticmethod
    def string_transformations(input_string:str, fun_replace:Callable[[str], str], fun_add_space:Callable[[str], str]):

        """
        Perform various string transformations on a given string to process symptom duration values.
        
        Parameters:
        -----------
        input_string : str
            Input string representing a symptom duration.
        fun_replace : Callable[[str], str]
            Function to replace non-alphanumeric characters with spaces.
        fun_add_space : Callable[[str], str]
            Function to add spaces between numbers and letters.
            
        Returns:
        --------
        duration : float
            Processed symptom duration value in years.
        """

        def is_castable_as_float(input_string):
            try:
                float(input_string)
                return True
            except ValueError:
                return False
        
        def compute_years(input_string:str):

            if "y" in input_string:
                lst = [txt for txt in input_string.split(' ') if is_castable_as_float(txt) or "y"]
                try:
                    return float(lst[0])
                except ValueError:
                    return 0
            else:
                return 0
                
        def compute_months(input_string:str):

            if "m" in input_string:
                lst = [txt for txt in input_string.split(' ') if is_castable_as_float(txt) or "m"]
                try:
                    return float(lst[-2])
                except ValueError:
                    return 0
            else:
                return 0

        
        if input_string is None or input_string=='': return None
        
        string = input_string.lower()
        # replace non alphanumeric characters by space
        string = fun_replace(string)
        # split numbers and letters substrings
        string = fun_add_space(string)

        if not any(char.isdigit() for char in string):
            return None
        
        if is_castable_as_float(string):
            return float(string)
        
        years = compute_years(string)
        month = compute_months(string)
        if month is None: print(input_string)
        return years + month/12

class SymptomDurationFixer(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for fixing symptom duration values in a DataFrame.
    
    This transformer computes the time difference between symptom start and assessment year and handles values beyond an upper bound.
    
    Parameters:
    ----------
    upper_bound : float
        The upper bound for valid symptom duration values.
    keep_assess_date : bool, default=True
        Whether to keep the assessment date column in the output DataFrame.
    
    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'SymptomDurationFixer':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        self : SymptomDurationFixer
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame or pd.Series:
        Transform the input DataFrame by fixing symptom duration values and handling values beyond the upper bound.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with symptom duration and assessment date columns.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame or pd.Series
            A new DataFrame or Series with fixed symptom duration values.
    """

    def __init__(self, upper_bound, keep_assess_date=True) -> None:
        """
        Initialize the SymptomDurationFixer transformer.
        
        Parameters:
        -----------
        upper_bound : float
            The upper bound for valid symptom duration values.
        keep_assess_date : bool, default=True
            Whether to keep the assessment date column in the output DataFrame.
        """
        super().__init__()
        self.keep_assess_date = keep_assess_date
        self.upper_bound = upper_bound

    def get_feature_names_out():
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):

        """
        Transform the input DataFrame by fixing symptom duration values and handling values beyond the upper bound.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with symptom duration and assessment date columns.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame or pd.Series
            A new DataFrame or Series with fixed symptom duration values.
        """

        X_copy = X.copy()
        cols = X_copy.columns

        X_copy[cols[0]] = X_copy.apply(
            lambda row: self.compute_time(row[cols[0]], row[cols[1]].year), axis=1
        )

        X_copy[cols[0]] = X_copy[cols[0]].apply(
            lambda x: np.nan if x>self.upper_bound else x
        )

        if self.keep_assess_date:
            return X_copy
        else: return X_copy[cols[0]]
    
    @staticmethod
    def compute_time(symptom_start:float, assess_year:float)->float:
        """
        Compute the time difference between symptom start and assessment year.
        
        Parameters:
        -----------
        symptom_start : float
            Symptom start value.
        assess_year : float
            Assessment year value.
            
        Returns:
        --------
        duration : float
            Time difference between symptom start and assessment year.
        """

        if symptom_start>100:
            return assess_year-symptom_start
        else: return symptom_start

class SymptomOnsetFixer(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for fixing symptom onset values in a DataFrame.
    
    This transformer handles values below a specified lower bound by replacing them with NaN.
    
    Parameters:
    ----------
    lower_bound : int
        The lower bound for valid symptom onset values.
    
    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'SymptomOnsetFixer':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        self : SymptomOnsetFixer
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by fixing symptom onset values and handling values below the lower bound.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with symptom onset column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with fixed symptom onset values.
    """

    def __init__(self, lower_bound: int) -> None:
        """
        Initialize the SymptomOnsetFixer transformer.
        
        Parameters:
        -----------
        lower_bound : int
            The lower bound for valid symptom onset values.
        """
        super().__init__()
        self.lower_bound = lower_bound

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        """
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        self : SymptomOnsetFixer
            Returns the instance itself.
        """
        return self
    
    def transform(self, X:pd.DataFrame, y=None):

        """
        Transform the input DataFrame by fixing symptom onset values and handling values below the lower bound.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with symptom onset column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with fixed symptom onset values.
        """

        X_copy = X.copy()
        cols = X_copy.columns

        X_copy[cols[0]] = X_copy[cols[0]].apply(
            lambda x: np.nan if x<self.lower_bound else x
        )

        return X_copy
