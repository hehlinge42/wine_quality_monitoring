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


def process_next_batch(global_data, db):
    nb_lines = db.at[len(db) - 1, "nb_lines"]
    new_batch = global_data["df_preprocessed"].loc[
        nb_lines : nb_lines + BATCH_SIZE - 1, :
    ]
    past_data = global_data["df_preprocessed"].loc[0 : nb_lines - 1, :]

    classifier = u.load_model(global_data["conf"], version="_0")
    y_past_data = past_data[global_data["y_column"]]
    X_past_data = past_data.drop(global_data["y_column"], axis=1)
    y_new_batch = new_batch[global_data["y_column"]]
    X_new_batch = new_batch.drop(global_data["y_column"], axis=1)
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
    sorted_drifts = sorted(drifts, key=lambda tup: tup[1])[:5]
    logger.debug(f"sorted_drifts: {sorted_drifts}")

    f1_score = round(dict_metrics_new_batch["f1_score"], 2)

    new_row = {
        "batch": len(db),
        "f1_score": [f1_score],
        "nb_lines": nb_lines + BATCH_SIZE,
        "version": "_0",
    }
    means_and_std = monitoring.get_means_std(new_batch)
    new_row = {**new_row, **means_and_std}
    for i, drift in enumerate(sorted_drifts):
        col, p_value = drift
        new_row["col" + str(i + 1)] = col
        new_row["pvalue" + str(i + 1)] = p_value

    new_row = pd.DataFrame(new_row)
    db = db.append(new_row, ignore_index=True)
    u.save_metrics(global_data["conf"], db)
    return db


global_data = cached_load("../params/conf/conf.json")

try:
    db = u.load_metrics(global_data["conf"])
except:
    logger.debug("Load at start")
    X_train, X_test, y_train, y_test = preprocessing.basic_split(
        global_data["df_preprocessed"],
        0.25,
        global_data["X_columns"],
        global_data["y_column"],
    )
    classifier = u.load_model(global_data["conf"], version="_0")
    dict_metrics = evaluation.main_evaluation(
        classifier, X_test, y_test, global_data["conf"]
    )
    f1_score = round(dict_metrics["f1_score"], 2)
    first_row = {"batch": 0, "f1_score": [f1_score], "nb_lines": 500, "version": "_0"}
    past_data = global_data["df_preprocessed"].loc[0:499, :]
    means_and_std = monitoring.get_means_std(past_data)
    first_row = {**first_row, **means_and_std}
    logger.debug(f"\n{first_row}")

    db = pd.DataFrame(first_row)
    u.save_metrics(global_data["conf"], db)

with st.sidebar:
    st.header("Load next batch")
    load_next_batch = st.button("Load")
    if load_next_batch:
        try:
            db = process_next_batch(global_data, db)
        except:
            st.write("You've reached the end of the dataset")

    for feature in global_data["X_columns"]:
        st.checkbox(feature, value=True)

st.title("Monitoring wine quality prediction model")
f1_score_widget = st.metric(f"F1-score", str(db.at[len(db) - 1, "f1_score"]))
f1_score_chart = st.line_chart(db[["f1_score"]])
try:
    for i in range(1, 6):
        col = db.at[len(db) - 1, "col" + str(i)]
        pvalue = db.at[len(db) - 1, "pvalue" + str(i)]
        st.metric(f"p-value of col {col}", str(pvalue))
        st.line_chart(db[col + "_mean"])
except:
    st.write("Expecting first batch")