# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import sys
import altair as alt

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
ROWS_AT_START = 500


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


def init_monitoring(global_data, init_nb_rows=ROWS_AT_START):
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
    data_span=(0, 500),
):
    logger.critical(f"In train_model")
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


def process_next_batch(global_data, db, db_models, current_state, batch_size=50):
    last_model = db_models.tail(1)
    cols_to_remove = [global_data["y_column"]]
    for col in global_data["X_columns"]:
        if last_model[col].values[0] == 0:
            cols_to_remove.append(col)
    new_batch = global_data["df_preprocessed"].loc[
        current_state["nb_lines"] : current_state["nb_lines"] + batch_size - 1, :
    ]
    past_data = u.get_sub_df_preprocessed(global_data["df_preprocessed"], db_models)

    classifier = u.load_model(
        global_data["conf"], version=current_state["current_model_version"]
    )
    y_past_data = past_data[global_data["y_column"]]
    X_past_data = past_data.drop(cols_to_remove, axis=1)
    y_new_batch = new_batch[global_data["y_column"]]
    X_new_batch = new_batch.drop(cols_to_remove, axis=1)

    outliers = monitoring.get_outliers(
        X_new_batch, current_state["iso_forest"], global_data["conf"]
    )

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

    start = db_models.tail(1)["start"].values[0]
    end = db_batch.tail(1)["nb_lines"].values[0]

    drifts_less_threshold = [drift[0] for drift in drifts if drift[1] < 0.05]
    adwin_results = monitoring.check_adwin_concept_drift(
        global_data["df_preprocessed"].loc[start:end, :], drifts_less_threshold
    )

    u.save_adwin(adwin_results, global_data["conf"])

    f1_score = round(dict_metrics_new_batch["f1_score"], 2)
    new_row = {
        "batch": len(db),
        "batch_size": batch_size,
        "nb_outliers": len(outliers) / batch_size,
        "f1_score": [f1_score],
        "nb_lines": current_state["nb_lines"] + batch_size,
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


def rewrite_batch_db(db_batch, db_models, nb_outliers, f1_score, current_state):
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


# Start of Streamlit App
global_data = cached_load("../params/conf/conf.json")

try:
    db_batch = u.load_metrics(global_data["conf"], metric_type="batch")
    db_models = u.load_metrics(global_data["conf"], metric_type="model")
except FileNotFoundError:
    db_batch, db_models = init_monitoring(global_data)
    adwin = None
except Exception:
    logger.exception("")

outliers = u.load_outliers(global_data["conf"])
train_data_description = u.load_train_data_description(global_data["conf"])
current_state = monitoring.get_current_state(db_batch, global_data)

with st.sidebar:
    st.header("Load next batch")
    batch_size = st.number_input(
        "Observations to load", min_value=0, max_value=100, value=50, step=10
    )
    load_next_batch = st.button("Load")
    if load_next_batch:
        try:
            db_batch = process_next_batch(
                global_data, db_batch, db_models, current_state, batch_size
            )
        except:
            logger.exception("")
            st.write("You've reached the end of the dataset")

    drifts_df = monitoring.get_recap(db_batch, global_data)
    if drifts_df is not None:
        st.header("Recap")
        st.write("p-values for detecting drifts in current batch")
        st.dataframe(drifts_df)

    st.header("ADWIN Advice")
    adwin = u.load_adwin(global_data["conf"])
    logger.debug(f"adwin = {adwin}")
    st.markdown(monitoring.get_advice(adwin), unsafe_allow_html=True)

    st.header("Retrain parameters")
    features_to_keep = []
    for feature in global_data["X_columns"]:
        feature_checkbox = st.checkbox(feature, value=True)
        if feature_checkbox:
            features_to_keep.append(feature)

    max_data_span = current_state["nb_lines"].item()
    data_span_retrain = st.slider(
        "Data span for retraining",
        min_value=0,
        max_value=max_data_span,
        value=(max_data_span - 500, max_data_span),
        step=1,
    )

    retrain_model_button = st.button("Retrain model")
    if retrain_model_button:
        logger.critical(f"db_models {db_models}")
        nb_outliers, f1_score, db_models = train_model(
            global_data, current_state, features_to_keep, db_models, data_span_retrain
        )
        db_batch = rewrite_batch_db(
            db_batch, db_models, nb_outliers, f1_score, current_state
        )

st.title("Monitoring wine quality predictions")
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
        display_outliers = st.checkbox("Show outliers", value=False)
    if display_outliers:
        st.dataframe(outliers)
        st.dataframe(train_data_description)
    else:
        f1_score_chart = st.line_chart(db_batch[["f1_score", "nb_outliers"]])

try:
    with st.container():
        option = st.selectbox("Select a feature ", global_data["X_columns"])
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
