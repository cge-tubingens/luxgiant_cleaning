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

class Move2Other(BaseEstimator, TransformerMixin):

    def __init__(self, feature_name:str, sep:str='___') -> None:
        super().__init__()
        self.sep = sep
        self.feature_name = feature_name

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):

        X_copy = X.copy()
        cols = X_copy.columns

        result = X_copy.iloc[:, :-1].sum(axis=1)
        for k in range(len(result)):
            if result[k] == 0 and X_copy.iloc[k,-1]==0:
                X_copy.iloc[k,-1]=1
        
        return X_copy

class ClassifyOnset(BaseEstimator, TransformerMixin):

    def __init__(self, outputCol:str='onset_type') -> None:
        super().__init__()
        self.outputCol = outputCol

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):

        X_copy = X.copy()
        col = X_copy.columns

        X_copy[self.outputCol] = X_copy[col[0]].apply(
            lambda x: self.type_onset(x)
        )

        return X_copy
    
    @staticmethod
    def type_onset(x):

        if np.isnan(x): return None 

        if x <= 18: return 0
        elif x <= 50: return 1
        else:
            return 2
        
class ClassifyEducation(BaseEstimator, TransformerMixin):

    def __init__(self, outputCol:str='education_level') -> None:
        super().__init__()
        self.outputCol = outputCol

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):

        X_copy = X.copy()
        col = X_copy.columns

        X_copy[self.outputCol] = X_copy[col[0]].apply(
            lambda x: self.ed_level(x)
        )

        return X_copy
    
    @staticmethod
    def ed_level(x):

        if np.isnan(x): return None 

        if x == 0: return 0
        elif x <= 7: return 1
        elif x <= 12: return 2
        else:
            return 3

class RecodeGeography(TransformerMixin, BaseEstimator):

    def __init__(self, outputCol:str='zone_of_origin') -> None:
        super().__init__()
        self.outputCol = outputCol

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):

        X_copy = X.copy()
        col = X_copy.columns

        

        X_copy[self.outputCol] = X_copy[col[0]].apply(
            lambda x: self.recoding(x)
        )

        return X_copy
    
    @staticmethod
    def recoding(x):

        recode_dict = {
            1:5, 2:3, 3:3, 4:3, 5:2, 6:5, 7:4, 8:1, 9:1, 10:1, 11:3, 12:5, 13:5, 14:2, 15:4, 16:3, 
            17:3, 18:3, 19:3, 20:3, 21:1, 22:1, 23:3, 24:5, 25:5, 26:3, 27:2, 28:2, 29:3, 30:5, 
            31:1, 32:4, 33:4, 34:1, 35:5, 36:5,
        }

        if np.isnan(x):return None
        else:
            return recode_dict[x]

class FromUPDRStoMDS(TransformerMixin, BaseEstimator):

    def __init__(self, outputStr:str='estim_MDS') -> None:
        super().__init__()
        self.outputStr = outputStr

    def get_feature_names_out(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):

        X_copy = X.copy()
        cols = X_copy.columns

        X_copy['updrs_part_iv'] = X_copy[cols[4]] + X_copy[cols[5]] + X_copy[cols[6]]

        X_copy[self.outputStr + '_part_i'] = X_copy.apply(
            lambda row: self.compute_mds1_from_updrs1(hoehn_yahr=row[0], updrs_1=row[1]), axis=1
        )

        X_copy[self.outputStr + '_part_ii'] = X_copy.apply(
            lambda row: self.compute_mds2_from_updrs2(hoehn_yahr=row[0], updrs_2=row[2]), axis=1
        )
        X_copy[self.outputStr + '_part_iii'] = X_copy.apply(
            lambda row: self.compute_mds3_from_updrs3(hoehn_yahr=row[0], updrs_3=row[3]), axis=1
        )
        X_copy[self.outputStr + '_part_iv'] = X_copy.apply(
            lambda row: self.compute_mds4_from_updrs4(hoehn_yahr=row[0], updrs_4=row['updrs_part_iv']), axis=1
        )
        X_copy[self.outputStr + 'total'] = X_copy[self.outputStr + '_part_i']\
             + X_copy[self.outputStr + '_part_ii'] + X_copy[self.outputStr + '_part_iii']\
             + X_copy[self.outputStr + '_part_iv']

        return X_copy.drop(columns='updrs_part_iv', inplace=False)
    
    @staticmethod
    def compute_mds1_from_updrs1(hoehn_yahr:str, updrs_1:str):

        if np.isnan(hoehn_yahr): return np.nan

        elif hoehn_yahr == 0: return 0
        elif 1 <= hoehn_yahr and hoehn_yahr <=2.5: return np.round(updrs_1*2.5 + 4.7,0)
        elif hoehn_yahr == 3: return np.round(updrs_1*2. + 7.7,0)
        else:
            return np.round(updrs_1*1.6 + 10.8,0)

    @staticmethod
    def compute_mds2_from_updrs2(hoehn_yahr:str, updrs_2:str):

        if np.isnan(hoehn_yahr): return np.nan

        elif hoehn_yahr == 0: return 0
        elif 1 <= hoehn_yahr and hoehn_yahr <=2.5: return np.round(updrs_2*1.1 + 0.2,0)
        elif hoehn_yahr == 3: return np.round(updrs_2 + 1.5,0)
        else:
            return np.round(updrs_2 + 4.7,0)
        
    @staticmethod
    def compute_mds3_from_updrs3(hoehn_yahr:str, updrs_3:str):

        if np.isnan(hoehn_yahr): return np.nan

        elif hoehn_yahr == 0: return 0
        elif 1 <= hoehn_yahr and hoehn_yahr <=2.5: return np.round(updrs_3*1.2 + 2.3,0)
        elif hoehn_yahr == 3: return np.round(updrs_3*1.2 + 1.0,0)
        else:
            return np.round(updrs_3*1.1 + 7.5,0)
        
    @staticmethod
    def compute_mds4_from_updrs4(hoehn_yahr:str, updrs_4:str):

        if np.isnan(hoehn_yahr): return np.nan

        elif hoehn_yahr == 0: return 0
        elif 1 <= hoehn_yahr and hoehn_yahr <=2.5: return np.round(updrs_4*1.0 - 0.3,0)
        elif hoehn_yahr == 3: return np.round(updrs_4*1.0 - 0.3,0)
        else:
            return np.round(updrs_4*1.1 + 0.8,0)

