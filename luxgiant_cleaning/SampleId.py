"""
Python module with classes to clean and uniformize sample IDs.
"""

import re

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class idKeepAlphanumeric(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for processing sample IDs.
    
    This transformer removes any non-alphanumeric characters from the sample ID.
    
    Parameters:
    ----------
    None
    
    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'idKeepAlphanumeric':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        self : idKeepAlphanumeric
            Returns the instance itself.

    transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Transform the input DataFrame by removing non-alphanumeric characters from the specified column.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with sample IDs in the first column.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with the sample IDs processed (non-alphanumeric characters removed).

    keep_alphanumeric(input_id: str) -> str:
        Static method to remove non-alphanumeric characters from a given string and convert it to uppercase.
        
        Parameters:
        -----------
        input_id : str
            Input sample ID.
            
        Returns:
        --------
        alphanumeric_string : str
            Processed sample ID with non-alphanumeric characters removed and converted to uppercase.
    """

    def __init__(self)->None:
        super().__init__()

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def get_feature_names_out(self):
        pass

    def transform(self, X:pd.DataFrame):

        """
        Transform the input DataFrame by removing non-alphanumeric characters from the specified column.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with sample IDs in the first column.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with the sample IDs processed (non-alphanumeric characters removed).
        """

        X_copy = X.copy()
        col = X_copy.columns[0]

        X_copy[col] = X_copy[col].apply(self.keep_alphanumeric)

        return X_copy

    @staticmethod
    def keep_alphanumeric(input_id):
        """
        Static method to remove non-alphanumeric characters from a given string and convert it to uppercase.
        
        Parameters:
        -----------
        input_id : str
            Input sample ID.
            
        Returns:
        --------
        alphanumeric_string : str
            Processed sample ID with non-alphanumeric characters removed and converted to uppercase.
        """
        alphanumeric_string = re.sub(r'[^a-zA-Z0-9]', '', input_id)
        return alphanumeric_string.upper()

class idMiddleNull(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for processing Sample IDs.
    
    This transformer identifies and removes extra '0' characters mistakenly added in the middle of some Sample IDs.
    
    Parameters:
    ----------
    None
    
    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'idMiddleNull':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        self : idMiddleNull
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by removing extra '0' characters from the middle of the specified column.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with Sample IDs in the first column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with the Sample IDs processed (extra '0' characters removed).

    process_alphanumeric_string(input_string: str) -> str:
        Static method to identify and remove extra '0' characters from the middle of a given Sample ID.
        
        Parameters:
        -----------
        input_string : str
            Input Sample ID.
            
        Returns:
        --------
        processed_string : str
            Processed Sample ID with extra '0' characters removed.
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def get_feature_names_out(self):
        pass

    def transform(self, X:pd.DataFrame, y=None) -> pd.DataFrame:

        """
        Transform the input DataFrame by removing extra '0' characters from the middle of the specified column.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with Sample IDs in the first column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with the Sample IDs processed (extra '0' characters removed).
        """

        X_copy = X.copy()  # Create a copy of the input DataFrame
        col = X_copy.columns[0]

        X_copy[col] = X_copy[col].apply(lambda x: self.process_alphanumeric_string(x))
        
        return X_copy
    
    @staticmethod
    def process_alphanumeric_string(input_string:str):

        """
        Static method to identify and remove extra '0' characters from the middle of a given Sample ID.
        
        Parameters:
        -----------
        input_string : str
            Input Sample ID.
            
        Returns:
        --------
        processed_string : str
            Processed Sample ID with extra '0' characters removed.
        """
    # Use regular expression to check if the last five characters are numbers
        if re.match(r'^\D*(\d{5})$', input_string[-5:]) and len(input_string)==8:
            # Check if the first character is '0' and remove it
            code = input_string[:-5]
            number = input_string[-5:]
            if number[0] == '0' or number[0]=='1':
                number = number[1:]
            return code+number
        return input_string

class idAddZero(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for processing Sample IDs.
    
    This transformer adds an extra '0' to Sample IDs that consist of three letters followed by three numeric values.
    
    Parameters:
    ----------
    None
    
    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'idAddZero':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        self : idAddZero
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by adding an extra '0' to Sample IDs that meet the specified format.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with Sample IDs in the first column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with the Sample IDs processed (extra '0' added).

    add_zero(text: str) -> str:
        Static method to add an extra '0' to Sample IDs that consist of three letters followed by three numeric values.
        
        Parameters:
        -----------
        text : str
            Input Sample ID.
            
        Returns:
        --------
        processed_text : str
            Processed Sample ID with an extra '0' added if it meets the specified format.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:

        """
        Transform the input DataFrame by adding an extra '0' to Sample IDs that meet the specified format.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with Sample IDs in the first column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with the Sample IDs processed (extra '0' added).
        """

        X_copy = X.copy()
        col = X_copy.columns[0]

        X_copy[col] = X_copy[col].apply(lambda x: self.add_zero(x))

        return X_copy

    @staticmethod
    def add_zero(text:str):

        """
        Static method to add an extra '0' to Sample IDs that consist of three letters followed by three numeric values.
        
        Parameters:
        -----------
        text : str
            Input Sample ID.
            
        Returns:
        --------
        processed_text : str
            Processed Sample ID with an extra '0' added if it meets the specified format.
        """

        match = re.match(r'^[A-Za-z]{3}\d{3}$', text)

        if match:
            # The string matches the format, extract the letters and numbers
            letters = text[:3]
            numbers = text[3:]
            return letters + '0' + numbers
        else:
            return text

class idZeroSubstitute(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for processing Sample IDs.
    
    This transformer substitutes 'O' values with '0' values in Sample IDs.
    
    Parameters:
    ----------
    None
    
    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'idZeroSubstitute':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        self : idZeroSubstitute
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by substituting 'O' values with '0' values in Sample IDs.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with Sample IDs in the first column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with the Sample IDs processed (substituted 'O' with '0').

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
        Transform the input DataFrame by substituting 'O' values with '0' values in Sample IDs.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with Sample IDs in the first column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with the Sample IDs processed (substituted 'O' with '0').
        """

        X_copy = X.copy()
        col = X_copy.columns[0]

        X_copy[col] = X_copy[col].apply(lambda x: x.replace('O', '0'))
        return X_copy

class idFormatMarker(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for processing Sample IDs.
    
    This transformer checks if the format of Sample IDs is correct.
    
    Parameters:
    ----------
    outputCol : str, default='has_ID_prob'
        The name of the output column that indicates whether the Sample ID has the correct format.
    
    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'idFormatMarker':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        self : idFormatMarker
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by adding a column indicating whether the Sample ID has the correct format.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with Sample IDs in the first column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with an additional column indicating whether the Sample ID has the correct format.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """
    
    def __init__(self, outputCol='has_ID_prob') -> None:
        """
        Initialize the idFormatMarker transformer.
        
        Parameters:
        -----------
        outputCol : str, default='has_ID_prob'
            The name of the output column that indicates whether the Sample ID has the correct format.
        """
        super().__init__()
        self.outputCol = outputCol

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None) -> pd.DataFrame:

        """
        Transform the input DataFrame by adding a column indicating whether the Sample ID has the correct format.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input data with Sample IDs in the first column.
        y : None
            Ignored. This parameter exists only for compatibility.
            
        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with an additional column indicating whether the Sample ID has the correct format.
        """

        X_copy = X.copy()
        col = X_copy.columns[0]

        X_copy[self.outputCol] = X_copy[col].apply(lambda x: self.check_string_pattern(x))

        return X_copy
    
    @staticmethod
    def check_string_pattern(input_string):

        """
        Check if the input Sample ID has the correct format.
        
        Parameters:
        -----------
        input_string : str
            Input Sample ID.
            
        Returns:
        --------
        has_correct_format : bool
            True if the Sample ID has the correct format, False otherwise.
        """

        # Define a regular expression pattern to match the criteria
        pattern = r"^[PC][ABCD][ABCDEF]\d{4}$"

        # Use re.match to check if the input_string matches the pattern
        if re.match(pattern, input_string):
            return True
        else:
            return False

class idLength(BaseEstimator, TransformerMixin):

    def __init__(self, outputCol:str='id_length') -> None:

        """
        Initialize the idLength transformer.

        Parameters:
        - outputCol (str): Name of the output column containing the computed lengths. Default is 'id_length'.
        """

        super().__init__()
        self.outputCol = outputCol

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):

        """
        Transform method to compute the length of identifiers in a DataFrame column.

        This method calculates the length of identifiers in the specified DataFrame column
        and returns a new DataFrame with the length values in the output column.

        Parameters:
        -----------
        X (pd.DataFrame): Input DataFrame.
        y : None

        Returns:
        --------
        - pd.DataFrame: Transformed DataFrame with the computed lengths in the output column.
        """

        X_copy = X.copy()
        col = X_copy.columns[0]

        X_copy[col] = X_copy[col].apply(lambda x: len(x))

        return X_copy
