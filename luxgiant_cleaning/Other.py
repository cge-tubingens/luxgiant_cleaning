"""
Python module with stand-alone classes
"""

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class AssessmentDate(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for imputing missing values in the assessment date column.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'AssessmentDate':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : AssessmentDate
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by imputing missing values in the assessment date column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with an assessment date column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with imputed assessment date values.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None)->pd.DataFrame:
        """
        Transform the input DataFrame by imputing missing values in the assessment date column.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with an assessment date column.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with imputed assessment date values.
        """

        X_copy = X.copy()
        cols = X_copy.columns

        X_copy = X_copy.sort_values(by=cols[0]).reset_index()
        X_copy.columns = ['index_orig'] + [col for col in X_copy.columns if col!='index']

        lst_idx = X_copy[X_copy['date_of_assessment'].isnull()].reset_index()['index'].to_list()
        
        endpoints = self.endpoints_finder(lst_idx)

        for point in endpoints.keys():
            first_date = X_copy.iloc[endpoints[point][0], 2]
            scond_date = X_copy.iloc[endpoints[point][1], 2]

            imp_date = self.find_mid_date(first_date, scond_date)
            X_copy.iloc[point, 2] = imp_date

        return X_copy.sort_values(by='index_orig').reset_index(drop=True).drop(columns='index_orig', inplace=False)
    
    @staticmethod
    def endpoints_finder(idx_lst:list)->list:
        """
        Find endpoints for each missing segment in the assessment date column.

        Parameters:
        -----------
        idx_lst : list
            List of indices where assessment dates are missing.

        Returns:
        --------
        endpoints_dict : dict
            A dictionary where keys are indices with missing assessment dates, and values are lists of
            adjacent indices indicating the endpoints of each missing segment.
        """

        endpoints_dict = {}
    
        for idx in idx_lst:

            prev = (idx-1 in idx_lst)
            next = (idx+1 in idx_lst)

            if not prev and not next:
                endpoints_dict[idx] = [idx-1, idx+1]
            if not prev and next:
                endpoints_dict[idx] = [idx-1]
                idx_carrier = idx
            if prev and next: continue
            if prev and not next:
                endpoints_dict[idx_carrier].append(idx+1)

        return endpoints_dict
    
    @staticmethod
    def find_mid_date(start_date, end_date):
        """
        Find the midpoint date between two given dates.

        Parameters:
        -----------
        start_date : pd.Timestamp
            The starting date.
        end_date : pd.Timestamp
            The ending date.

        Returns:
        --------
        imputed : pd.Timestamp
            The midpoint date between `start_date` and `end_date`.
        """

        diff = start_date - end_date
        imputed = start_date + pd.tseries.offsets.DateOffset(days=diff.days)

        return imputed
   
class HeightWeight(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for handling extreme values in height and weight columns.

    Parameters:
    -----------
    height_extremes : list, default=[150, 190]
        The valid range for height values.
    weight_extremes : list, default=[40, 100]
        The valid range for weight values.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'HeightWeight':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : HeightWeight
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by handling extreme values in height and weight columns.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with height and weight columns.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with adjusted height and weight values.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self, height_extremes=[150, 190], weight_extremes=[40, 100]) -> None:
        """
        Initialize the HeightWeight transformer.

        Parameters:
        -----------
        height_extremes : list, default=[150, 190]
            The valid range for height values.
        weight_extremes : list, default=[40, 100]
            The valid range for weight values.
        """
        super().__init__()
        self.height_extremes = height_extremes
        self.weight_extremes = weight_extremes

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):
        """
        Transform the input DataFrame by handling extreme values in height and weight columns.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with height and weight columns.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with adjusted height and weight values.
        """

        X_copy = X.copy()
        cols = X_copy.columns

        mask_h_in_w = (X_copy[cols[1]]<self.height_extremes[1]) & (X_copy[cols[1]]>self.height_extremes[1])
        mask_w_in_h = (X_copy[cols[0]]<self.weight_extremes[1]) & (X_copy[cols[0]]>self.weight_extremes[1])

        idx_lst = X_copy[mask_h_in_w & mask_w_in_h].reset_index()['index'].to_list()

        for idx in idx_lst:
            X_copy.iloc[idx, 0], X_copy.iloc[idx, 1] = X_copy.iloc[idx, 1], X_copy.iloc[idx, 0]

        mask_extreme_h = (X_copy[cols[0]]<self.height_extremes[0]) | (X_copy[cols[0]]>self.height_extremes[1])
        mask_extreme_w = (X_copy[cols[1]]<self.weight_extremes[0]) | (X_copy[cols[1]]>self.weight_extremes[1])

        idx_lst_h = X_copy[mask_extreme_h].reset_index()['index'].to_list()
        idx_lst_w = X_copy[mask_extreme_w].reset_index()['index'].to_list()

        for idx in idx_lst_h:
            X_copy.iloc[idx, 0] = np.nan

        for idx in idx_lst_w:
            X_copy.iloc[idx, 1] = np.nan

        return X_copy

class BMICalculator(BaseEstimator, TransformerMixin):

    """
    A scikit-learn custom transformer for calculating BMI (Body Mass Index) based on height and weight columns.

    Parameters:
    -----------
    outputCol : str, default='bmi'
        The name of the column where BMI values will be stored.

    Methods:
    --------
    fit(self, X: pd.DataFrame, y=None) -> 'BMICalculator':
        Fit the transformer. This method does nothing and is included for scikit-learn compatibility.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        self : BMICalculator
            Returns the instance itself.

    transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        Transform the input DataFrame by calculating BMI based on height and weight columns.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with height and weight columns.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with added BMI column.

    get_feature_names_out(self):
        Pass method for scikit-learn compatibility. Does nothing.
    """

    def __init__(self, outputCol='bmi') -> None:
        super().__init__()
        self.outputCol = outputCol

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):

        """
        Transform the input DataFrame by calculating BMI based on height and weight columns.

        Parameters:
        -----------
        X : pd.DataFrame
            Input data with height and weight columns.
        y : None
            Ignored. This parameter exists only for compatibility.

        Returns:
        --------
        X_copy : pd.DataFrame
            A new DataFrame with added BMI column.
        """

        X_copy = X.copy()
        cols = X_copy.columns

        mask = (X_copy[cols[0]].isnull() | X_copy[cols[1]].isnull())

        X_copy[self.outputCol] = X_copy.apply(
            lambda row: (10**4)*(row[cols[1]])/(row[cols[0]]*row[cols[0]]), axis=1
        )

        for k in range(len(mask)):
            if mask[k]:
                X_copy.iloc[k,2] = np.nan

        return X_copy
