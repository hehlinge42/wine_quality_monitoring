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
warnings.filterwarnings('ignore')

from monitoring.model_monitor import ModelMonitor

BATCH_SIZE = 50

@st.cache
def cached_load(path_conf):
    global_data = {}
    with open(path_conf, "r") as json_fd:
        global_data["conf"] = json.load(json_fd)
    global_data["df_preprocessed"] = loading.load_preprocessed_csv_from_name(global_data["conf"])
    global_data["y_column"] = u.get_y_column_from_conf(global_data["conf"])
    global_data["X_columns"] = [x for x in global_data["df_preprocessed"].columns if x != global_data["y_column"]]
    return global_data

def process_next_batch(global_data, db):
    nb_lines = db.at[len(db) - 1, "nb_lines"]
    new_batch = global_data["df_preprocessed"].loc[
                nb_lines: nb_lines + BATCH_SIZE - 1, :
                ]
    check_
    classifier = u.load_model(global_data["conf"], version="_0")
    y_test = new_batch[global_data["y_column"]]
    X_test = new_batch.drop(global_data["y_column"], axis=1)
    dict_metrics = evaluation.main_evaluation(
        classifier, X_test, y_test, global_data["conf"]
    )
    f1_score = round(dict_metrics["f1_score"], 2)
    new_row = pd.DataFrame({"batch": len(db), "f1_score": [f1_score], "nb_lines": nb_lines + BATCH_SIZE, "version": "_0"})
    db = db.append(new_row, ignore_index=True)
    u.save_metrics(global_data["conf"], db)
    return db


global_data = cached_load("../params/conf/conf.json")

try:
    db = u.load_metrics(global_data["conf"])
except:
    logger.debug("Load at start")
    X_train, X_test, y_train, y_test = preprocessing.basic_split(
        global_data["df_preprocessed"], 0.25, global_data["X_columns"], global_data["y_column"]
    )
    classifier = u.load_model(global_data["conf"], version="_0")
    dict_metrics = evaluation.main_evaluation(
        classifier, X_test, y_test, global_data["conf"]
    )
    f1_score = round(dict_metrics["f1_score"], 2)
    db = pd.DataFrame({"batch": 0, "f1_score": [f1_score], "nb_lines": 500, "version": "_0"})
    u.save_metrics(global_data["conf"], db)

with st.sidebar:
    st.header("Load next batch")
    load_next_batch = st.button("Load")
    if load_next_batch:
        db = process_next_batch(global_data, db)

st.title("Monitoring wine quality prediction model")
f1_score_widget = st.metric(f"F1-score", str(db.at[len(db) - 1, "f1_score"]))
f1_score_chart = st.line_chart(db[["f1_score"]])
db
