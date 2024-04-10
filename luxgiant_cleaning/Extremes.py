"""
Python module to deal with extreme values
"""

import re

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class ExtremesYears(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for handling extreme values in year columns.

    This transformer replaces values in the specified year columns that are outside the range [earlier_year, later_year] with np.nan.

    Parameters:
    -----------
    earlier_year : int, optional (default=1920)
        The lower bound for valid years.
    later_year : int, optional (default=2024)
        The upper bound for valid years.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'ExtremesYears':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : ExtremesYears
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by replacing values outside the range [earlier_year, later_year] in the specified columns with np.nan.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the specified year columns.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with extreme values in the specified year columns replaced by np.nan.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self, earlier_year=1920, later_year=2024) -> None:
        """
        Initialize the ExtremesYears transformer.

        Parameters:
        -----------
        earlier_year : int, optional (default=1920)
            The lower bound for valid years.
        later_year : int, optional (default=2024)
            The upper bound for valid years.
        """
        super().__init__()
        self.earlier_year = earlier_year
        self.later_year = later_year

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):
        """
        Transform the input DataFrame by replacing values outside the range [earlier_year, later_year] in the specified year columns with np.nan.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the specified year columns.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with extreme values in the specified year columns replaced by np.nan.
        """

        X_copy = X.copy()
        cols = X.columns

        for col in cols:
            X_copy[col] = X_copy[col].apply(
                lambda x: x if self.earlier_year < x and x< self.later_year else np.nan
            )

        return X_copy
    
class ExtremeValues(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for handling extreme values in numeric columns.

    This transformer replaces values in the specified numeric column that are outside the range [lower_bound, upper_bound] with np.nan.

    Parameters:
    -----------
    lower_bound : int or float, optional (default=0)
        The lower bound for valid values.
    upper_bound : int or float, optional (default=30)
        The upper bound for valid values.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'ExtremeValues':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : ExtremeValues
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by replacing values outside the range [lower_bound, upper_bound] in the specified column with np.nan.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the specified numeric column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with extreme values in the specified numeric column replaced by np.nan.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self, lower_bound=0, upper_bound=30) -> None:
        """
        Initialize the ExtremeValues transformer.

        Parameters:
        -----------
        lower_bound : int or float, optional (default=0)
            The lower bound for valid values.
        upper_bound : int or float, optional (default=30)
            The upper bound for valid values.
        """
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):

        """
        Transform the input DataFrame by replacing values outside the range [lower_bound, upper_bound] in the specified numeric column with np.nan.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the specified numeric column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with extreme values in the specified numeric column replaced by np.nan.
        """

        X_copy = X.copy()
        col = X.columns[0]

        X_copy[col] = X_copy[col].apply(
            lambda x: x if self.lower_bound < x and x< self.upper_bound else np.nan
        )

        return X_copy

class DateFixer(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for fixing date columns in a DataFrame.

    This transformer converts columns containing date-like values to datetime objects using the pandas to_datetime function.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'DateFixer':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : DateFixer
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by converting date-like values in all columns to datetime objects.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with columns containing date-like values.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with columns containing date-like values converted to datetime objects.

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
        Transform the input DataFrame by converting date-like values in all columns to datetime objects.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with columns containing date-like values.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with columns containing date-like values converted to datetime objects.
        """

        X_copy=X.copy()
        cols = X_copy.columns

        for col in cols:
            X_copy[col] = pd.to_datetime(X_copy[col], errors='coerce')

        return X_copy

class FormatBDI(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for formatting BDI (Beck Depression Inventory) scores in a DataFrame.

    This transformer performs the following operations on a specified column:
    1. Removes any non-alphanumeric characters.
    2. Splits the string at '/' and retains the first part (if '/' is present).
    3. Converts the result to a float, handling empty strings and non-numeric values by converting them to NaN.
    4. Sets values outside the range [0, 63] to NaN.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'FormatBDI':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : FormatBDI
            Returns the instance itself.

    transform(self, X: pd.DataFrame, Y=None) -> pd.DataFrame:
        Transform the input DataFrame by applying the specified formatting operations to the designated column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the column to be formatted.
        Y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with the specified column formatted according to the defined operations.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, Y=None):

        """
        Transform the input DataFrame by applying the specified formatting operations to the designated column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the column to be formatted.
        Y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with the specified column formatted according to the defined operations.
        """

        X_copy = X.copy()
        col = X_copy.columns[0]

        X_copy[col] = X_copy[col].apply(
            lambda x: x.split('/')[0] if '/' in x else re.sub(r'[^a-zA-Z0-9]', '', x)
        )

        X_copy[col] = X_copy[col].apply(
            lambda x: self.convert_string_to_nan(x)
        )

        X_copy[col] = X_copy[col].apply(
            lambda x: x if 0 <= x and x<= 63 else np.nan
        )

        return X_copy
    
    @staticmethod
    def convert_string_to_nan(x:str):

        """
        Convert a string to NaN if it is empty or cannot be converted to a float.

        Parameters:
        -----------
        x : str
            Input string.

        Returns:
        --------
        result : float or NaN
            The converted float value or NaN if the input cannot be converted.
        """

        if x=='':
            return np.nan
        
        try:
            return float(x)
        except ValueError:
            return np.nan

class FormatString2Int(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for formatting string values to integers in a DataFrame.

    This transformer performs the following operations on a specified column:
    1. Splits the string at a specified delimiter and retains the first part (if the delimiter is present).
    2. Removes any non-alphanumeric characters.
    3. Converts the result to a float, handling empty strings and non-numeric values by converting them to NaN.
    4. Sets values outside the specified range to NaN.

    Parameters:
    -----------
    splitter : str
        The delimiter used to split the string.
    lower_bound : int
        The lower bound for valid values.
    upper_bound : int
        The upper bound for valid values.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'FormatString2Int':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : FormatString2Int
            Returns the instance itself.

    transform(self, X: pd.DataFrame, Y=None) -> pd.DataFrame:
        Transform the input DataFrame by applying the specified formatting operations to the designated column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the column to be formatted.
        Y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with the specified column formatted according to the defined operations.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self, splitter:str, lower_bound:int, upper_bound:int) -> None:
        """
        Initialize the FormatString2Int transformer.

        Parameters:
        -----------
        splitter : str
            The delimiter used to split the string.
        lower_bound : int
            The lower bound for valid values.
        upper_bound : int
            The upper bound for valid values.
        """
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.splitter = splitter

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, Y=None):

        """
        Transform the input DataFrame by applying the specified formatting operations to the designated column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the column to be formatted.
        Y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with the specified column formatted according to the defined operations.
        """

        X_copy = X.copy()
        col = X_copy.columns[0]

        if X_copy[col].dtype == 'str':
           X_copy[col] = X_copy[col].apply(
            lambda x: x.split(self.splitter)[0] if self.splitter in x else re.sub(r'[^a-zA-Z0-9]', '', x)
            )

        X_copy[col] = X_copy[col].apply(
            lambda x: self.convert_string_to_nan(x)
        )

        X_copy[col] = X_copy[col].apply(
            lambda x: x if self.lower_bound <= x and x<= self.upper_bound else np.nan
        )

        return X_copy
    
    @staticmethod
    def convert_string_to_nan(input_string:str):
        """
        Convert a string to NaN if it is empty or cannot be converted to a float.

        Parameters:
        -----------
        x : str
            Input string.

        Returns:
        --------
        result : float or NaN
            The converted float value or NaN if the input cannot be converted.
        """

        if input_string=='':
            return np.nan
        
        try:
            return float(input_string)
        except ValueError:
            return np.nan

class UpperValueCapper(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for capping numeric values in a DataFrame to an upper limit.

    This transformer caps the values in a specified column to a maximum specified value.

    Parameters:
    -----------
    cap : float
        The upper limit to which values will be capped.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'UpperValueCapper':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : UpperValueCapper
            Returns the instance itself.

    transform(self, X: pd.DataFrame, Y=None) -> pd.DataFrame:
        Transform the input DataFrame by capping the values in the specified column to the upper limit.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the column to be capped.
        Y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with values in the specified column capped to the upper limit.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self, cap:float) -> None:
        """
        Initialize the UpperValueCapper transformer.

        Parameters:
        -----------
        cap : float
            The upper limit to which values will be capped.
        """
        super().__init__()
        self.cap = cap

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):
        """
        Transform the input DataFrame by capping the values in the specified column to the upper limit.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with the column to be capped.
        Y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with values in the specified column capped to the upper limit.
        """

        X_copy = X.copy()
        col = X_copy.columns

        X_copy[col] = X_copy[col].apply(
            lambda x: self.cap if x>self.cap else x
        )

        return X_copy
