# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import sys

from logzero import logger

sys.path.insert(0, "loading/")
sys.path.insert(0, "preprocessing/")
sys.path.insert(0, "modeling/")
sys.path.insert(0, "evaluation/")
sys.path.insert(0, "interpretability/")
sys.path.insert(0, "utils/")
sys.path.insert(0, "monitoring/")

import utils as u
import loading
import json
import evaluation
import preprocessing
import monitoring
import modeling

import warnings

warnings.filterwarnings("ignore")

BATCH_SIZE = 50


@st.cache
def cached_load(path_conf):
    global_data = {}
    with open(path_conf, "r") as json_fd:
        global_data["conf"] = json.load(json_fd)
    global_data["df_preprocessed"] = loading.load_preprocessed_csv_from_name(
        global_data["conf"]
    )
    global_data["y_column"] = u.get_y_column_from_conf(global_data["conf"])
    global_data["X_columns"] = [
        x
        for x in global_data["df_preprocessed"].columns
        if x != global_data["y_column"]
    ]
    global_data["column_mapping"] = u.get_column_mapping(
        global_data["conf"], global_data["df_preprocessed"]
    )
    return global_data


def init_monitoring_batch(global_data):
    past_data = global_data["df_preprocessed"].loc[0:499, :]
    X_train, X_test, y_train, y_test = preprocessing.basic_split(
        past_data,
        0.25,
        global_data["X_columns"],
        global_data["y_column"],
    )
    init_nb_rows = len(past_data)
    nb_outliers = monitoring.fit_isolation_forest(X_train, global_data["conf"])
    classifier = u.load_model(global_data["conf"], version="_0")
    dict_metrics = evaluation.main_evaluation(
        classifier, X_test, y_test, global_data["conf"]
    )
    f1_score = round(dict_metrics["f1_score"], 2)
    first_row = {
        "batch": 0,
        "batch_size": init_nb_rows,
        "nb_outliers": nb_outliers / init_nb_rows,
        "f1_score": [f1_score],
        "nb_lines": init_nb_rows,
        "version": "_0",
    }
    means_and_std = monitoring.get_means_std(past_data)
    first_row = {**first_row, **means_and_std}
    db = pd.DataFrame(first_row)
    u.save_metrics(global_data["conf"], db)
    return db


def update_models_db(global_data, version="_0", start=0, end=499, cols_to_keep=None, db_models=None):
    models_dict = {
        "version": version,
        "start": start,
        "end": end
    }
    for col in cols_to_keep:
        models_dict[col] = 1
    cols_to_remove = set(global_data["X_columns"]) - set(cols_to_keep)
    for col in cols_to_remove:
        models_dict[col] = 0

    if db_models is None:
        db_models = pd.DataFrame(models_dict)
    else:
        new_row = pd.DataFrame(models_dict)
        db_models = db_models.append(new_row, ignore_index=True)
    u.save_metrics(global_data["conf"], db_models, type="model")
    return db_models


def retrain_model(global_data, current_state, features_to_keep=None, db_models=None):
    if features_to_keep is None:
        features_to_keep = global_data["X_columns"]
    features_to_keep_y = features_to_keep + [global_data["y_column"]]
    df_train = global_data["df_preprocessed"].loc[0 : current_state["nb_lines"] - 1, :]
    df_train = df_train[features_to_keep_y]
    X_train, X_test, y_train, y_test = preprocessing.basic_split(
        df_train, 0.20, features_to_keep, global_data["y_column"]
    )
    classifier, best_params = modeling.main_modeling_from_name(
        X_train, y_train, global_data["conf"]
    )
    u.save_model(
        classifier, global_data["conf"], version=current_state["next_model_version"]
    )
    dict_metrics = evaluation.main_evaluation(
        classifier, X_test, y_test, global_data["conf"]
    )
    f1_score = round(dict_metrics["f1_score"], 2)
    update_models_db(global_data, version="_0", start=0, end=499, cols_to_keep=None, db_models=None)

def process_next_batch(global_data, db, current_state):
    new_batch = global_data["df_preprocessed"].loc[
        current_state["nb_lines"] : current_state["nb_lines"] + BATCH_SIZE - 1, :
    ]
    past_data = global_data["df_preprocessed"].loc[0 : current_state["nb_lines"] - 1, :]

    classifier = u.load_model(
        global_data["conf"], version=current_state["current_model_version"]
    )
    y_past_data = past_data[global_data["y_column"]]
    X_past_data = past_data.drop(global_data["y_column"], axis=1)
    y_new_batch = new_batch[global_data["y_column"]]
    X_new_batch = new_batch.drop(global_data["y_column"], axis=1)

    nb_outliers = monitoring.get_nb_outliers(X_new_batch, current_state["iso_forest"])

    dict_metrics_past_data = evaluation.main_evaluation(
        classifier, X_past_data, y_past_data, global_data["conf"]
    )
    dict_metrics_new_batch = evaluation.main_evaluation(
        classifier, X_new_batch, y_new_batch, global_data["conf"]
    )
    new_batch["prediction"] = dict_metrics_new_batch["prediction"]
    past_data["prediction"] = dict_metrics_past_data["prediction"]

    # Custom Preprocessing to check drifts
    drifts = monitoring.detect_features_drift(
        past_data, new_batch, global_data["column_mapping"]
    )
    f1_score = round(dict_metrics_new_batch["f1_score"], 2)
    new_row = {
        "batch": len(db),
        "batch_size": BATCH_SIZE,
        "nb_outliers": nb_outliers / BATCH_SIZE,
        "f1_score": [f1_score],
        "nb_lines": current_state["nb_lines"] + BATCH_SIZE,
        "version": current_state["current_model_version"],
    }
    for col, p_value in drifts:
        new_row[col + "_p_value"] = p_value

    means_and_std = monitoring.get_means_std(new_batch)
    new_row = {**new_row, **means_and_std}

    new_row = pd.DataFrame(new_row)
    db = db.append(new_row, ignore_index=True)
    u.save_metrics(global_data["conf"], db)
    return db

global_data = cached_load("../params/conf/conf.json")

try:
    db_batch = u.load_metrics(global_data["conf"], type="batch")
except:
    db_batch = init_monitoring_batch(global_data)

current_state = monitoring.get_current_state(db_batch, global_data)
try:
    db_models = u.load_metrics(global_data["conf"], type="model")
except:
    db_models = update_models_db(global_data)

with st.sidebar:
    st.header("Load next batch")
    load_next_batch = st.button("Load")
    if load_next_batch:
        try:
            db = process_next_batch(global_data, db_batch, current_state)
        except:
            logger.exception("")
            st.write("You've reached the end of the dataset")

    drifts_df = monitoring.get_recap(db_batch, global_data)
    if drifts_df is not None:
        st.header("Recap")
        st.write("p-values for detecting drifts in current batch")
        st.dataframe(drifts_df)

    st.header("Advice")
    st.write(monitoring.get_advice(db_batch))

    st.header("Retrain parameters")
    features_to_keep = []
    for feature in global_data["X_columns"]:
        feature_checkbox = st.checkbox(feature, value=True)
        if feature_checkbox:
            features_to_keep.append(feature)

    retrain_model_button = st.button("Retrain model")
    if retrain_model_button:
        retrain_model(global_data, db, current_state, features_to_keep)

st.title("Monitoring wine quality prediction model")
st.header(
    f"Model in production: version {str(current_state['current_model_version_int'])}"
)

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        f1_score_widget = st.metric(f"F1-score", str(db.at[len(db) - 1, "f1_score"]))
    with col2:
        f1_score_widget2 = st.metric(
            f"Share of outliers in last batch", monitoring.get_share_outliers(db)
        )
    f1_score_chart = st.line_chart(db[["f1_score", "nb_outliers"]])

try:
    option = st.selectbox("Select a feature ", global_data["X_columns"])
    st.metric(f"p-value of col {option}", str(db.at[len(db) - 1, option + "_p_value"]))
    st.line_chart(db[option + "_mean"])

except:
    logger.exception("")
    st.write("Expecting first batch")
