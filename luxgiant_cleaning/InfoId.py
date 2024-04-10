"""
Python module to extract information from the sample ID.
"""

import re

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class centreExtractor(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for extracting the centre ID from Sample IDs.
    
    This transformer extracts the centre ID (characters at positions 1 and 2) from Sample IDs that match a specific pattern.
    
    Parameters:
    ----------
    outputCol : str, default='centre_id'
        The name of the output column where the centre ID will be stored.
    
    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'centreExtractor':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        self : centreExtractor
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by extracting the centre ID from Sample IDs that match the specified pattern.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with Sample IDs in the first column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with an additional column containing the extracted centre IDs.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self, outputCol='centre_id') -> None:
        super().__init__()
        self.outputCol = outputCol

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None) -> pd.DataFrame:

        """
        Transform the input DataFrame by extracting the centre ID from Sample IDs that match the specified pattern.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with Sample IDs in the first column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with an additional column containing the extracted centre IDs.
        """ 

        X_copy = X.copy()
        col = X_copy.columns[0]

        X_copy[self.outputCol] = X_copy[col].apply(lambda x: x[1:3] if self.check_string_pattern(x) else None)

        return X_copy
    
    @staticmethod
    def check_string_pattern(input_string):

        """
        Check if the input Sample ID matches a specific pattern.
        
        Parameters:
        -----------
        input_string : str
            Input Sample ID.
            
        Returns:
        --------
        is_match : bool
            True if the Sample ID matches the pattern, False otherwise.
        """

        # Define a regular expression pattern to match the criteria
        pattern = r"^[PC][ABCD][ABCDEF]\d{4}$"

        # Use re.match to check if the input_string matches the pattern
        if re.match(pattern, input_string):
            return True
        else:
            return False

class ControlStatus(TransformerMixin, BaseEstimator):
        
        """
        A scikit-learn custom transformer for determining control status based on Sample IDs.
    
        This transformer checks if the first character of the Sample ID matches a specific pattern to identify control samples.
    
        Parameters:
        ----------
        outputCol : str, default='is_control'
            The name of the output column where the control status will be stored.
    
        Methods:
        --------
        fit(self, X: pd.DataFrame, y=None) -> 'ControlStatus':
            Fit the transformer. This method does nothing and is included for scikit-learn compatibility.
        
            Parameters:
            -----------
            X : pd.DataFrame
                Input data.
            y : None
                Ignored. This parameter exists only for compatibility.
            
            Returns:
            --------
            self : ControlStatus
                Returns the instance itself.

        transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
            Transform the input DataFrame by determining control status based on the Sample ID pattern.
        
            Parameters:
            -----------
            X : pd.DataFrame
                Input data with Sample IDs in the first column.
            y : None
                Ignored. This parameter exists only for compatibility.
            
            Returns:
            --------
            X_copy : pd.DataFrame
                A new DataFrame with an additional column indicating control status.

        get_feature_names_out(self):
            Pass method for scikit-learn compatibility. Does nothing.
        """

        def __init__(self, outputCol='is_control') -> None:
            super().__init__()
            self.outputCol = outputCol
        
        def get_feature_names_out(self):
            pass

        def fit(self, X:pd.DataFrame, y=None):
            return self
        
        def transform(self, X:pd.DataFrame, y=None):

            """
            Transform the input DataFrame by determining control status based on the Sample ID pattern.

            Parameters:
            -----------
            X : pd.DataFrame
                Input data with Sample IDs in the first column.
            y : None
                Ignored. This parameter exists only for compatibility.

            Returns:
            --------
            X_copy : pd.DataFrame
                A new DataFrame with an additional column indicating control status.
            """

            X_copy = X.copy()
            col = X_copy.columns[0]

            X_copy[self.outputCol] = X_copy[col].apply(lambda x: x[0] if self.check_string_pattern(x) else None)
            X_copy[self.outputCol] = X_copy[self.outputCol].apply(lambda x: 1 if x=='P' else 0)

            return X_copy
        
        @staticmethod
        def check_string_pattern(input_string):

            """
            Check if the input Sample ID matches a specific pattern.

            Parameters:
            -----------
            input_string : str
                Input Sample ID.

            Returns:
            --------
            is_match : bool
                True if the Sample ID matches the pattern, False otherwise.
            """

            # Define a regular expression pattern to match the criteria
            pattern = r"^[PC][ABCD][ABCDEF]\d{4}$"

            # Use re.match to check if the input_string matches the pattern
            if re.match(pattern, input_string):
                return True
            else:
                return False
