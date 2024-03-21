import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def data_cleanup(raw_data):
    """Takes the raw data and removes/replaces unwanted values

    Args:
        raw_data (pandas DataFrame): The Raw Data sent to the model

    Returns:
        pandas DataFrame: A dataframe with columns x12 and x63 cleaned
    """    
    raw_data['x12'] = raw_data['x12'].str.replace('$','').str.replace(',','').str.replace(')','').str.replace('(','-').astype(float)
    raw_data['x63'] = raw_data['x63'].str.replace('%','').astype(float)

    return raw_data

def imputation_and_standardization(cleaned_data, non_imputed_columns):
    """Replaces missing values in the data via imputation, the standardizes the data. This does not apply to the 
    or columns x5, x31, x81, or x82

    Args:
        cleaned_data (pandas DataFrame): A dataframe containing the previously cleaned data
        non_imputed_columns (list): A list containing the columns not to be imputed and standardized

    Returns:
        pandas DataFrame: a DataFrame with columns now 
    """    
    cleaned_data  = cleaned_data.drop(columns=non_imputed_columns)
    empty_features = cleaned_data.loc[:, ~cleaned_data.any()]
    non_empty_features = cleaned_data.loc[:, cleaned_data.any()]
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    clean_imputed = pd.DataFrame(imputer.fit_transform(cleaned_data), columns=non_empty_features.columns)
    clean_imputed = pd.concat([clean_imputed, empty_features], axis=1, sort=False)
    std_scaler = StandardScaler()
    clean_imputed_std = pd.DataFrame(std_scaler.fit_transform(clean_imputed), columns=clean_imputed.columns)
    return clean_imputed_std

def create_dummies(one_hot_data, one_hot_columns):
    """Performs one hot enoding on the provided data, for the provided columns

    Args:
        one_hot_data (pandas DataFrame): A pandas dataframe containing the data that we would like to add one-hot encoded
        columns to
        one_hot_columns (list): A list of columns that we would like to one-hot encode

    Returns:
        pandas DataFrame: a pandas dataframe containing the original data as well as the new one-hot encoded columns
    """    
    prefix_dict = {key : key for key in one_hot_columns}
    one_hot_data = pd.get_dummies(one_hot_data, columns=one_hot_columns, drop_first=True,\
                                  prefix=prefix_dict, prefix_sep='_', dummy_na=True)
    return one_hot_data

def fill_missing_data(dummies_data, col_list):
    """Fills in the missing columns that we are expecting if they are not present

    Args:
        dummies_data (pandas DataFrame): A dataframe containing the one-hot encoded and imputed data.
        col_list (list): The final list of columns that the model is expeceting our pre-processed data to have
    Returns:
        pandas DataFrame: A dataframe containing the missing columns which have been set to 0 and filtered down to only the columns
        that our model is expecting
    """    
    for feat in col_list:
        if feat not in dummies_data.columns:
            dummies_data[feat] = 0
    return dummies_data[col_list]


def feature_preprocessing(data, one_hot_columns=['x5', 'x31', 'x81', 'x82'], non_imputed_columns=['x5', 'x31', 'x81', 'x82']):
    """A function to apply all of the feature engineering steps outlined in the original model

    Args:
        data (pandas DataFrame): The raw data that we would like preproccessed
        one_hot_columns (list, optional):The columns to be one_hot encoded. Defaults to ['x5', 'x31', 'x81', 'x82'].
        non_imputed_columns (list, optional): The columns that are not to be imputed. Defaults to ['x5', 'x31', 'x81', 'x82'].

    Returns:
       Pandas DataFrame: A DataFrame containing the cleaned, standardized and one-hot encoded data with missing values filled
       via imputation filtered to the selected features from model training
    """    
    selected_features = ['x5_saturday',
    'x81_July',
    'x81_December',
    'x31_japan',
    'x81_October',
    'x5_sunday',
    'x31_asia',
    'x81_February',
    'x91',
    'x81_May',
    'x5_monday',
    'x81_September',
    'x81_March',
    'x53',
    'x81_November',
    'x44',
    'x81_June',
    'x12',
    'x5_tuesday',
    'x81_August',
    'x81_January',
    'x62',
    'x31_germany',
    'x58',
    'x56']
    data=data.copy(deep=True)
    cleaned_data = data_cleanup(data)
    imputed_data = imputation_and_standardization(cleaned_data, non_imputed_columns)
    imputed_data = pd.concat([data[['x5', 'x31', 'x81', 'x82']], imputed_data], axis=1, sort=False)
    dummies = create_dummies(imputed_data, one_hot_columns)
    preprocessed_data = fill_missing_data(dummies, selected_features)
    return preprocessed_data