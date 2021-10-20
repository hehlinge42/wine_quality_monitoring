# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import sys
import json
import altair as alt

from logzero import logger
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "../loading/")
sys.path.insert(0, "../preprocessing/")
sys.path.insert(0, "../modeling/")
sys.path.insert(0, "../evaluation/")
sys.path.insert(0, "../interpretability/")
sys.path.insert(0, "../utils/")

import utils as u
import loading
import evaluation
import preprocessing

import monitoring
import modeling

ROWS_AT_START = 500


def get_recap(db, global_data):
    """Output for streamlit displaying detected feature drift.

    Args:
        db (pd.DataFrame): complete summary of batches previously processed
        global_data (dict): immutable data stored in streamlit cache

    Returns:
        [pd.DataFrame]: p-values related to drifts for each feature and target
    """

    try:
        features_pvalue = [col + "_p_value" for col in global_data["X_columns"]]
        cols = features_pvalue + [global_data["y_column"] + "_p_value"]
        df = db.loc[len(db) - 1, cols].T.to_frame()
        logger.debug(f"df.columns: {df.columns}")
        df.rename({df.columns[-1]: "p-value"}, axis=1, inplace=True)
        features = {
            features_pvalue[i]: global_data["X_columns"][i]
            for i in range(len(features_pvalue))
        }
        df.rename(features, axis=0, inplace=True)
        df.sort_values(by=["p-value"], inplace=True)
    except KeyError:
        df = None
    except:
        logger.exception("")
    return df


def get_advice(adwin):
    """Output for streamlit displaying detected drifts using ADAWIN.

    Args:
        adwin ([dict]): feature-wise detected drifts (e.g. [idx_drift_1, idx_drift_2, ...])

    Returns:
        [str]: advice shown to user.
    """

    if adwin is None:
        advice = "No advice for now."
    else:
        advice = (
            "You may wish to look at the distribution around the following index: <br/>"
        )
        for k, v in adwin.items():
            if len(v) != 0:
                advice += f"- {k} look at indices **{v}**<br/>"
    return advice


def get_current_state(db_batch, global_data):
    """Loads models in production and other shortcuts into a dictionary

    Args:
        db_batch (pd.DataFrame): summary of all batches previously processed
        global_data (dict): immutable data stored in streamlit cache

    Returns:
        [dict]: models and other shortcuts
    """

    current_state = {}
    current_state["nb_lines"] = db_batch.at[len(db_batch) - 1, "nb_lines"]
    current_state["current_model_version"] = db_batch.at[len(db_batch) - 1, "version"]
    current_state["current_model_version_int"] = int(
        current_state["current_model_version"][1:]
    )
    current_state["next_model_version"] = "_" + str(
        current_state["current_model_version_int"] + 1
    )
    current_state["classifier"] = u.load_model(
        global_data["conf"], version=current_state["current_model_version"]
    )
    current_state["training_set"] = None
    current_state["iso_forest"] = u.load_model(
        global_data["conf"],
        name="isolation_forest",
        version=current_state["current_model_version"],
    )
    return current_state


def init_monitoring(global_data, init_nb_rows=ROWS_AT_START):
    """Called when files monitoring_db_batch.csv and monitoring_db_models.csv
    cannot be read.
    - trains first model on 500 first lines and saves it to disk
    - writes first model summary on monitoring_db_models.csv
    - writes first batch summary on monitoring_db_batch.csv

    Args:
        global_data ([dict]): configuration settings
        init_nb_rows ([int], optional): Used rows for initial model training. Defaults to ROWS_AT_START.

    Returns:
        [dict]: database of batch characteristics
        [dict]: database of model characteristics
    """

    nb_outliers, f1_score, db_models = train_model(
        global_data, current_state=None, features_to_keep=None, db_models=None
    )
    past_data = global_data["df_preprocessed"].loc[0 : init_nb_rows - 1, :]
    first_row = {
        "batch": 0,
        "batch_size": init_nb_rows,
        "nb_outliers": nb_outliers / init_nb_rows,
        "f1_score": f1_score,
        "nb_lines": init_nb_rows,
        "version": "_0",
    }
    means_and_std = monitoring.get_means_std(past_data)
    first_row = {**first_row, **means_and_std}
    db_batch = pd.DataFrame(first_row, index=[0])
    u.save_metrics(global_data["conf"], db_batch)
    return db_batch, db_models


def update_models_db(
    global_data,
    version="_0",
    start=0,
    end=ROWS_AT_START - 1,
    cols_to_keep=None,
    db_models=None,
):
    """Registers new model characteristics (e.g. selected features to keep,
    version, ...) in model database.

    Args:
        global_data ([dict]): configuration settings
        version (str, optional): model version. Defaults to "_0".
        start (int, optional): first observation used for training in historical data. Defaults to 0.
        end ([int], optional): last observation used for training in historical data. Defaults to ROWS_AT_START-1.
        cols_to_keep ([list], optional): selected columns to keep in model training (user choice). Defaults to None (i.e. all).
        db_models ([dict], optional): current database of model characteristics. Defaults to None (i.e. empty at launch).

    Returns:
        [dict]: updated database of model characteristics
    """
    models_dict = {"version": version, "start": start, "end": end}
    for col in cols_to_keep:
        models_dict[col] = 1
    cols_to_remove = set(global_data["X_columns"]) - set(cols_to_keep)
    for col in cols_to_remove:
        models_dict[col] = 0

    if db_models is None:
        db_models = pd.DataFrame(models_dict, index=[0])
    else:
        new_row = pd.DataFrame(models_dict, index=[len(db_models)])
        db_models = db_models.append(new_row, ignore_index=True)
    u.save_metrics(global_data["conf"], db_models, metric_type="model")
    return db_models


def train_model(
    global_data,
    current_state=None,
    features_to_keep=None,
    db_models=None,
    data_span=(0, ROWS_AT_START),
):
    """Trains a new model based on a range of historical data and a subset of
    features. Outliers are remove using an IsolationForest.

    Args:
        global_data ([dict]): unmutable dict of data cached by streamlit
        current_state ([dict], optional): Data related to the current model in
            production and the current batch being processes. Defaults to None.
        features_to_keep ([list], optional): List of features to train the model on.
            Defaults to None. In this case, all features are used
        db_models ([pd.DataFrame], optional): Summary of all models trained.
            Defaults to None.
        data_span (tuple, optional): Range of historical data to train the model on.
            Defaults to (0, 500).

    Returns:
        [int]: number of outliers on the training set
        [float]: f1-score on the testing set
        [pd.DataFrame]: db_models appended by new model's summary
    """
    if features_to_keep is None:
        features_to_keep = global_data["X_columns"]
    if current_state is None:
        version = "_0"
    else:
        version = current_state["next_model_version"]

    features_to_keep_y = features_to_keep + [global_data["y_column"]]
    df_train = global_data["df_preprocessed"].loc[data_span[0] : data_span[1] - 1, :]
    df_train = df_train[features_to_keep_y]
    X_train, X_test, y_train, y_test = preprocessing.basic_split(
        df_train, 0.20, features_to_keep, global_data["y_column"]
    )
    u.save_train_data_description(X_train.describe(), global_data["conf"])
    outliers = monitoring.fit_isolation_forest(X_train, global_data["conf"], version)
    classifier, best_params = modeling.main_modeling_from_name(
        X_train, y_train, global_data["conf"]
    )
    logger.info(f"New model trained!")
    u.save_model(classifier, global_data["conf"], version=version)
    dict_metrics = evaluation.main_evaluation(
        classifier, X_test, y_test, global_data["conf"]
    )
    f1_score = round(dict_metrics["f1_score"], 2)
    db_models = update_models_db(
        global_data,
        version=version,
        start=data_span[0],
        end=data_span[1] - 1,
        cols_to_keep=features_to_keep,
        db_models=db_models,
    )
    return len(outliers), f1_score, db_models


def process_next_batch(global_data, db_batch, db_models, current_state, batch_size=50):
    """Loads current model on new batch. Computes model metrics, outliers and drifs found in new batch.

    Args:
        global_data ([dict]): configuration settings
        db_batch ([dict]): database of batch characteristics
        db_models ([dict]): database of model characteristics
        current_state ([dict]): state containing current model (isolation forest for outliers and prediction model)
        batch_size (int, optional): Next batch size (user input). Defaults to 50.

    Returns:
        [dict]: updated database of batch characteristics
    """
    last_model = db_models.tail(1)
    cols_to_remove = [global_data["y_column"]]
    for col in global_data["X_columns"]:
        if last_model[col].values[0] == 0:
            cols_to_remove.append(col)
    new_batch = global_data["df_preprocessed"].loc[
        current_state["nb_lines"] : current_state["nb_lines"] + batch_size - 1, :
    ]
    past_data = u.get_sub_df_preprocessed(global_data["df_preprocessed"], db_models)

    y_past_data = past_data[global_data["y_column"]]
    X_past_data = past_data.drop(cols_to_remove, axis=1)
    y_new_batch = new_batch[global_data["y_column"]]
    X_new_batch = new_batch.drop(cols_to_remove, axis=1)

    # Use Isolation forest fitting on current model training data
    # to find (if any) outliers in new batch
    outliers = monitoring.get_outliers(
        X_new_batch, current_state["iso_forest"], global_data["conf"]
    )

    # Detect feature drift (variation in distribution of new batch vs training data)
    drifts = monitoring.detect_features_drift(
        past_data, new_batch, global_data["column_mapping"]
    )

    # Get user input indices of first and last training observation (for ADWIN)
    start = db_models.tail(1)["start"].values[0]
    end = db_batch.tail(1)["nb_lines"].values[0]

    drifts_less_threshold = [drift[0] for drift in drifts if drift[1] < 0.05]

    # Apply ADWIN to features where distribution is statistically significant (pvalue < 5%)
    adwin_results = monitoring.check_adwin_concept_drift(
        global_data["df_preprocessed"].loc[start:end, :], drifts_less_threshold
    )
    u.save_adwin(adwin_results, global_data["conf"])

    # Evaluate model on new batch
    dict_metrics_new_batch = evaluation.main_evaluation(
        current_state["classifier"], X_new_batch, y_new_batch, global_data["conf"]
    )
    f1_score = round(dict_metrics_new_batch["f1_score"], 2)

    # Write to batch characteristics database
    new_row = {
        "batch": len(db_batch),
        "batch_size": batch_size,
        "nb_outliers": len(outliers) / batch_size,
        "f1_score": f1_score,
        "nb_lines": current_state["nb_lines"] + batch_size,
        "version": current_state["current_model_version"],
    }
    for col, p_value in drifts:
        new_row[col + "_p_value"] = p_value
    means_and_std = monitoring.get_means_std(new_batch)
    new_row = {**new_row, **means_and_std}
    new_row = pd.DataFrame(new_row, index=[0])
    db_batch = db_batch.append(new_row, ignore_index=True)
    u.save_metrics(global_data["conf"], db_batch)
    return db_batch


def rewrite_batch_db(db_batch, db_models, nb_outliers, f1_score, global_data):
    """Overwrite previous model batch database.

    Args:
        db_batch (pd.DataFrame): Data summary of all previous batches processed
        db_models (pd.DataFrame): Data summary of all previous models trained
        nb_outliers (int): Number of outliers detected in new training dataset
        f1_score (float): f1-score on testing set
        global_data (dict): configuration settings

    Returns:
        [dict]: database of batch characteristics
    """
    updated_row = db_batch.tail(1)
    batch_size = (db_models.tail(1)["end"] - db_models.tail(1)["start"] + 1).values[0]
    version = db_models.tail(1)["version"].values[0]
    updated_row["f1_score"] = f1_score
    updated_row["batch_size"] = batch_size
    updated_row["nb_outliers"] = nb_outliers / batch_size
    updated_row["version"] = version

    for col in updated_row.columns:
        if "p_value" in col:
            updated_row[col] = None

    new_db_batch = pd.DataFrame(updated_row)
    try:
        new_db_batch.reset_index(inplace=True)
    except:
        pass
    u.save_metrics(global_data["conf"], new_db_batch, metric_type="batch")
    return new_db_batch
