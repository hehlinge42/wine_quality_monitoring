# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger("main_logger")

from sklearn.model_selection import train_test_split
import utils as u
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler

# TODO: complete this script, or create sub_scripts such as preprocessing_smote etc..

# Description of this script:
# main function calling the sub preprocessing function for the dataset selected
# subfunctions applying preprocessing (ex: one hot encoding, dropping etc..)


def main_preprocessing_from_name(df, conf):
    """
    Main Preprocessing function: it launches the correct function in order to preprocess the selected dataset
    Args:
        df: Dataframe
        conf: Conf file

    Returns: Preprocessed Dataframe

    """

    dict_function_preprocess = {
        "drift": "preprocessing_for_drift_dataset",
        "fraude": "preprocessing_for_fraud_dataset",
        "stroke": "preprocessing_for_stroke_dataset",
        "banking": "preprocessing_for_banking_dataset",
        "diabetic": "preprocessing_for_diabetic_dataset",
        "wine": "preprocessing_for_wine_dataset",
        "marketing": "preprocessing_for_marketing_dataset",
    }

    selected_dataset = conf["selected_dataset"]
    function_preprocess = globals()[dict_function_preprocess[selected_dataset]]
    logger.info(
        "Beginning of preprocessing function: "
        + dict_function_preprocess[selected_dataset]
    )
    df_preprocessed = function_preprocess(df, conf)
    logger.info(
        "End of preprocessing function: " + dict_function_preprocess[selected_dataset]
    )

    return df_preprocessed


def preprocessing_for_wine_dataset(df, conf):

    """
    Preprocessing for the WINE dataset
    Args:
        df: Wine dataset
        conf:  conf file

    Returns: Preprocessed Wine Dataset

    """
    # Steps:
    # Clean the output (as 0 or 1)
    # one hot elevel, car, zipcode
    # drop id

    logger.debug("Cleaning Output")
    # Cleaning Output:
    df["quality"] = np.where(df["quality"] > 5, 1, 0)

    logger.debug("Dropping unique columns")
    # Drop id:

    logger.debug("Selection of X and Y")
    # returning columns for train test split
    y_column = u.get_y_column_from_conf(conf)
    X_columns = [x for x in df.columns if x != y_column]

    logger.debug("Verification of float and na values ")
    # verification:
    for col in df.columns:
        try:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
        except:
            logger.error(col + " cannot be typed as float")
        if df[col].isna().sum() > 0:
            logger.warning("NA présent dans " + col)
    logger.info("preprocessing Wine ok")

    return df, X_columns, y_column


############################## Preprocessing Utils Functions ###########
def one_hot_encoder(df, cols):
    """
    One hot encoder, while dropping the encoded columns
    Args:
        df: dataframe
        cols: cols to encode

    Returns:dataframe with one hot encoded columns

    """
    # transform categorical features in OHE
    df_added = pd.get_dummies(df[cols], prefix=cols)
    df = pd.concat([df.drop(cols, axis=1), df_added], axis=1)
    return df


# This part can be optimized as a selection of function from the conf file (as the preprocessing is)
def basic_split(df, size_of_test, X_columns, y_column, seed=42):
    """
    Split the dataframe in train, test sets
    Args:
        df: Dataframe to Split
        size_of_test: proportion of test dataset
        X_columns: Columns for the variables
        y_column: Column for the output
        seed: Random state/seed

    Returns: Train and test datasets for variables and output

    """
    X_train, X_test, y_train, y_test = train_test_split(
        df[X_columns], df[y_column], test_size=size_of_test, random_state=seed
    )
    return X_train, X_test, y_train, y_test