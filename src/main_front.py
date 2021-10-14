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

import utils as u
import loading
import json
import evaluation
import preprocessing
import warnings
warnings.filterwarnings('ignore')

class ModelMonitor():
    def __init__(self, conf):
        logger.debug("In ModelMonitor constructor")
        self.conf = conf
        self.df_preprocessed = loading.load_preprocessed_csv_from_name(self.conf)
        self.y_column = u.get_y_column_from_conf(self.conf)
        self.X_columns = [x for x in self.df_preprocessed.columns if x != self.y_column]
        X_train, X_test, y_train, y_test = preprocessing.basic_split(
            self.df_preprocessed, 0.25, self.X_columns, self.y_column
        )
        self.classifier = u.load_model(self.conf)
        self.dict_metrics = evaluation.main_evaluation(
            self.classifier, X_test, y_test, self.conf
        )
        f1_score = round(self.dict_metrics["f1_score"], 2)
        self.f1_per_batch = pd.DataFrame({"batch": [0], "f1_score": [f1_score]})
        self.f1_per_batch.set_index("batch", inplace=True)
        self.nb_lines = 500
        self.nb_clicks = 0

    def load_next_batch(self):
        new_batch = self.df_preprocessed.loc[self.nb_lines:self.nb_lines + BATCH_SIZE - 1, :]
        self.nb_lines += BATCH_SIZE
        self.nb_clicks += 1
        y_test = new_batch[self.y_column]
        X_test = new_batch.drop(self.y_column, axis=1)
        self.dict_metrics = evaluation.main_evaluation(
            self.classifier, X_test, y_test, self.conf
        )
        new_row = {'f1_score': self.dict_metrics["f1_score"]}
        self.f1_per_batch = self.f1_per_batch.append(new_row, ignore_index=True)
        logger.debug(f"\n{self.f1_per_batch}")
        logger.debug(f"monitor: \n{str(self)}")
        # monitor.f1_per_batch
        # f1_score_widget = st.metric(f"F1-score", str(self.f1_per_batch.at[len(self.f1_per_batch), "f1_score"]))
        # f1_score_chart = st.line_chart(self.f1_per_batch)

    # def __str__(self):
    #     return f"nb_lines: {str(self.nb_lines)}, nb_clicks: {str(self.nb_clicks)}"

path_conf = "../params/conf/conf.json"
BATCH_SIZE = 50
with open(path_conf, "r") as json_fd:
    conf = json.load(json_fd)

monitor = ModelMonitor(conf)
st.title("Monitoring wine quality prediction model")
f1_score_widget = st.metric(f"F1-score", str(monitor.f1_per_batch.at[len(monitor.f1_per_batch) - 1, "f1_score"]))
f1_score_chart = st.line_chart(monitor.f1_per_batch)
monitor.f1_per_batch

with st.sidebar:
    st.header("Load next batch")
    load_next_batch = st.button("Load")
    if load_next_batch:
        monitor.load_next_batch()
