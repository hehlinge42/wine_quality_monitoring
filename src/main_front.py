# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import sys
import json
import altair as alt

from logzero import logger
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "loading/")
sys.path.insert(0, "preprocessing/")
sys.path.insert(0, "modeling/")
sys.path.insert(0, "evaluation/")
sys.path.insert(0, "interpretability/")
sys.path.insert(0, "utils/")
sys.path.insert(0, "monitoring/")

import utils as u
import loading
import evaluation
import preprocessing
import monitoring
import modeling
import st_utils


@st.cache
def cached_load(path_conf):
    """Caching config file settings

    Args:
        path_conf ([str]): filepath of the global configuration file related
        to the `str` folder

    Returns:
        [dict]: configuration settings
    """

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


# Start of Streamlit App
global_data = cached_load("../params/conf/conf.json")

try:
    # Load historical batch characteristics and model performance
    db_batch = u.load_metrics(global_data["conf"], metric_type="batch")
    db_models = u.load_metrics(global_data["conf"], metric_type="model")
except FileNotFoundError:
    db_batch, db_models = st_utils.init_monitoring(global_data)  # Train first model
    adwin = None
except Exception:
    logger.exception("")

# Load current outliers and statistics on training data
outliers = u.load_outliers(global_data["conf"])
train_data_description = u.load_train_data_description(global_data["conf"])
current_state = st_utils.get_current_state(db_batch, global_data)

with st.sidebar:
    st.header("Load next batch")
    batch_size = st.number_input(
        "Observations to load", min_value=0, max_value=100, value=50, step=10
    )
    load_next_batch = st.button("Load")
    if load_next_batch:
        try:
            # Evaluate model on new batch and looks for drifts
            db_batch = st_utils.process_next_batch(
                global_data, db_batch, db_models, current_state, batch_size
            )
        except:
            logger.exception("")
            st.write("You've reached the end of the dataset")

    # Display drifts in feature distribution (batch vs train data)
    drifts_df = st_utils.get_recap(db_batch, global_data)
    if drifts_df is not None:
        st.header("Recap")
        st.write("p-values for detecting drifts in current batch")
        st.dataframe(drifts_df)

    # Outputs (if any) concept drift detected by Adaptive Windowing (feature-wise)
    st.header("ADWIN Advice")
    adwin = u.load_adwin(global_data["conf"])
    st.markdown(st_utils.get_advice(adwin), unsafe_allow_html=True)

    # User can choose to retrain model with subset of features
    st.header("Retrain parameters")
    features_to_keep = []
    for feature in global_data["X_columns"]:
        feature_checkbox = st.checkbox(feature, value=True)
        if feature_checkbox:
            features_to_keep.append(feature)

    # User can choose to retrain model with subset of observations
    max_data_span = current_state["nb_lines"].item()
    data_span_retrain = st.slider(
        "Data span for retraining",
        min_value=0,
        max_value=max_data_span,
        value=(max_data_span - 500, max_data_span),
        step=1,
    )

    # Retrain model based on above user inputs
    retrain_model_button = st.button("Retrain model")
    if retrain_model_button:
        logger.critical(f"db_models {db_models}")
        nb_outliers, f1_score, db_models = st_utils.train_model(
            global_data, current_state, features_to_keep, db_models, data_span_retrain
        )
        # Write historical batch db for new model (overwriting past db)
        db_batch = st_utils.rewrite_batch_db(
            db_batch, db_models, nb_outliers, f1_score, global_data
        )

st.title(f"Monitoring {global_data['conf']['selected_dataset']} quality predictions")
st.header(f"Model in production: version {db_models.tail(1)['version'].values[0][1:]}")

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        f1_score_widget = st.metric(
            f"F1-score", str(db_batch.at[len(db_batch) - 1, "f1_score"])
        )
    with col2:
        f1_score_widget2 = st.metric(
            f"Share of outliers in last batch", monitoring.get_share_outliers(db_batch)
        )
    with col3:
        # User input to show description of outliers or model performance on different batches
        display_outliers = st.checkbox("Show outliers", value=False)
    if display_outliers:
        st.dataframe(outliers)
        st.dataframe(train_data_description)
    else:
        f1_score_chart = st.line_chart(db_batch[["f1_score", "nb_outliers"]])

try:
    with st.container():
        # Feature-wise distribution and p_value of train / current batch data
        option = st.selectbox(
            "Select a feature ", global_data["X_columns"] + [global_data["y_column"]]
        )
        st.metric(
            f"p-value of col {option}",
            str(db_batch.at[len(db_batch) - 1, option + "_p_value"]),
        )
        col1, col2 = st.columns(2)

        with col1:
            train_data = u.get_sub_df_preprocessed(
                global_data["df_preprocessed"], db_models
            )
            train_chart = alt.Chart(train_data).mark_bar().encode(x=option, y="count()")
            train_data_hist = st.altair_chart(train_chart)
        with col2:
            batch_data = u.get_sub_df_preprocessed_batch(
                global_data["df_preprocessed"], db_batch
            )
            batch_chart = alt.Chart(batch_data).mark_bar().encode(x=option, y="count()")
            batch_data_hist = st.altair_chart(batch_chart)
        st.line_chart(db_batch[option + "_mean"])
        st.line_chart(db_batch[option + "_std"])

except KeyError:
    pass
except:
    logger.exception("")
    st.write("Expecting first batch")
