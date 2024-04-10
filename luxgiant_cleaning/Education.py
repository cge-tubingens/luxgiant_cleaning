"""
Python module to process education times.
"""

import re

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class EducationStandandizer(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for standardizing education levels in a DataFrame.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'EducationStandandizer':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : EducationStandandizer
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by standardizing education levels.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with education level column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with standardized education levels.

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
        Transform the input DataFrame by standardizing education levels.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with education level column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with standardized education levels.
        """

        X_copy = X.copy()
        col = X_copy.columns

        X_copy[col[0]] = X_copy[col[0]].apply(
            lambda x: x.lower().replace(' ', '')
        )

        return X_copy

class ExtractEducation(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for extracting education information from strings.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'ExtractEducation':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : ExtractEducation
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by extracting education information from strings.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with education information column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with extracted education information.

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
        Transform the input DataFrame by extracting education information from strings.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with education information column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with extracted education information.
        """

        X_copy = X.copy()
        col = X_copy.columns[0]

        X_copy[col] = X_copy[col].apply(
            lambda x: self.extract_number_from_years(x)
        )

        X_copy[col] = X_copy[col].apply(
            lambda x: self.extract_number_from_th(x)
        )

        return X_copy
    
    @staticmethod
    def extract_number_from_years(input_string:str):
        """
        Extract a number from a string containing information about years.

        Parameters:
        -----------
        input_string : str
            The input string.

        Returns:
        --------
        result : int or str
            Extracted number or the original string if no match is found.
        """
        # Check if the string contains 'y' and ends with 'rs'
        if 'y' in input_string and input_string.endswith('rs'):
            # Use regular expression to find and extract the number
            match = re.search(r'\d+', input_string)
            if match:
                return int(match.group())
        # If conditions are not met, return the original string
        return input_string
    
    @staticmethod
    def extract_number_from_th(input_string:str):
        """
        Extract a number from a string starting with numbers followed by "th".

        Parameters:
        -----------
        input_string : str or int
            The input string.

        Returns:
        --------
        result : int or str
            Extracted number or the original string if no match is found.
        """
        # Use regular expression to check if the string starts with numbers followed by "th"
        if isinstance(input_string, int):
            return input_string
        
        match = re.match(r'^(\d+)th', input_string)
        if match:
            return int(match.group(1))  # Extract and return the numbers as an integer
        else:
            return input_string

class EducationSubstition(TransformerMixin, BaseEstimator):

    """
    A scikit-learn custom transformer for substituting values in an education column.

    Parameters:
    ----------
    cats_list : list
        List of values to be substituted with a specified number of years.
    num_years : float
        The number of years to substitute for values in `cats_list`.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'EducationSubstition':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : EducationSubstition
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by substituting values in the education column with a specified number of years.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with an education column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with substituted education values.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self, cats_list:list, num_years:float) -> None:
        """
        Initialize the EducationSubstition transformer.

        Parameters:
        -----------
        cats_list : list
            List of values to be substituted with a specified number of years.
        num_years : float
            The number of years to substitute for values in `cats_list`.
        """
        super().__init__()
        self.cats_list = cats_list
        self.num_years = num_years

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):
        """
        Transform the input DataFrame by substituting values in the education column with a specified number of years.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with an education column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with substituted education values.
        """

        X_copy = X.copy()
        col = X_copy.columns[0]

        X_copy[col] = X_copy[col].apply(
            lambda x: self.num_years if x in self.cats_list else x
        )

        return X_copy

class EducationMissing(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for handling missing values in an education column.

    Parameters:
    ----------
    null_holder : int, float, str, or None, default=np.nan
        The value to be used for representing missing values in the education column.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'EducationMissing':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : EducationMissing
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by handling missing values in the education column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with an education column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with missing values handled in the education column.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self, null_holder=np.nan) -> None:
        """
        Initialize the EducationMissing transformer.

        Parameters:
        -----------
        null_holder : int, float, str, or None, default=np.nan
            The value to be used for representing missing values in the education column.
        """
        super().__init__()
        self.null_holder = null_holder

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):
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
        self : EducationMissing
            Returns the instance itself.
        """

        X_copy = X.copy()
        col = X_copy.columns[0]

        X_copy[col] = X_copy[col]. apply(
            lambda x: self.null_holder if not str(x).isdigit() else float(x)
        )

        return X_copy
    
class EducationExtreme(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for handling extreme values in an education column.

    Parameters:
    ----------
    max_allowed_val : int or float, default=50
        The maximum allowed value in the education column. Values greater than this will be set to NaN.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'EducationExtreme':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : EducationExtreme
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by handling extreme values in the education column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with an education column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with extreme values handled in the education column.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self, max_allowed_val=50) -> None:
        """
        Initialize the EducationExtreme transformer.

        Parameters:
        -----------
        max_allowed_val : int or float, default=50
            The maximum allowed value in the education column. Values greater than this will be set to NaN.
        """
        super().__init__()
        self.max_allowed_val = max_allowed_val

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):
        """
        Transform the input DataFrame by handling extreme values in the education column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with an education column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with extreme values handled in the education column.
        """

        X_copy = X.copy()
        col = X_copy.columns[0]

        X_copy[col] = X_copy[col]. apply(
            lambda x: np.nan if not np.isnan(x) and x>self.max_allowed_val else x
        )

        return X_copy
