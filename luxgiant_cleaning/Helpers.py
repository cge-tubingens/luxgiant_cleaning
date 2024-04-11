"""
Python module with stand-alone funtions to perform well defined tasks
"""

import argparse

import pandas as pd

def recover_columns_names(columns:list)->list:

    """
    Recover the original column names from a list of transformed column names.

    Parameters:
    -----------
    columns : list
        List of transformed column names.

    Returns:
    --------
    old_columns : list
        List of original column names.
    """

    old_columns = []

    for col in columns:
        splitted = col.split('__')
        if len(splitted)==2:
            old_columns.append(splitted[1])
        else:
            old_columns.append(splitted[1] + '__' + splitted[2])
    return old_columns

def detect_datetime_cols(X:pd.DataFrame)->dict:

    """
    Detect columns with datetime values in a pandas DataFrame.

    Parameters:
    -----------
    X : pd.DataFrame
        Input DataFrame.

    Returns:
    --------
    datetime_cols : dict
        Dictionary mapping column names to the type of datetime values ('tc' for timestamp, 'td' for timedelta).
    """

    cols = X.columns
    datetime_lst = {}

    for col in cols:

        if not isinstance(X[col], pd.DataFrame):
            if X[col].dtype == 'datetime64' or X[col].dtype == 'datetime64[ns]':
                datetime_lst[col] = 'tc'

    return datetime_lst

def select_value_labels(X:pd.DataFrame, original_labels:dict)->dict:

    """
    Select labels corresponding to columns in the DataFrame from the original_labels dictionary.

    Parameters:
    -----------
    X : pd.DataFrame
        Input DataFrame.
    original_labels : dict
        Dictionary mapping formatted column names to labels.

    Returns:
    --------
    new_labels : dict
        Dictionary mapping original column names to labels.
    """

    cols = X.columns
    new_labels = {}
    for col in cols:
         
         format_col = col + '_'
         if format_col in original_labels.keys():
             new_labels[col] = original_labels[format_col]

    return new_labels

def rearrange_columns(X:pd.DataFrame, original_cols:list)->pd.DataFrame:

    """
    Rearrange the columns of the DataFrame based on the order specified in the original_cols list.

    Parameters:
    -----------
    X : pd.DataFrame
        Input DataFrame.
    original_cols : list
        List specifying the desired order of columns.

    Returns:
    --------
    pd.DataFrame
    """

    new_cols = []
    old_cols = []

    for col in original_cols:
        if col in X.columns: old_cols.append(col)

    for col in X.columns:
        if col not in original_cols: new_cols.append(col)

    return new_cols + old_cols

def arg_parser()->dict:

    # define parser
    parser = argparse.ArgumentParser(description='Adresses to input STATA file and output folder')

    # parameters of quality control
    parser.add_argument('--input-file', type=str, nargs='?', default=None, const=None, help='Full path to the STATA file with REDCap raw data.')

    # path to data and names of files
    parser.add_argument('--output-folder', type=str, nargs='?', default=None, const=None, help='Full path to the to folder where cleaned file will be saved.')

    # parse args and turn into dict
    args = parser.parse_args()

    return args
