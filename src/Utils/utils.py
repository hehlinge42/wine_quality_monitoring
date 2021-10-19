# -*- coding: utf-8 -*-
import logging
import pickle
import pandas as pd
import json

from logzero import logger


def my_get_logger(path_log, log_level, my_name=""):
    """
    Instanciation du logger et paramétrisation
    :param path_log: chemin du fichier de log
    :param log_level: Niveau du log
    :return: Fichier de log
    """

    log_level_dict = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }

    LOG_LEVEL = log_level_dict[log_level]

    if my_name != "":
        logger = logging.getLogger(my_name)
        logger.setLevel(LOG_LEVEL)
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(LOG_LEVEL)

    # create a file handler
    handler = logging.FileHandler(path_log)
    handler.setLevel(LOG_LEVEL)

    # create a logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(funcName)s - %(levelname)-8s: %(message)s"
    )
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


def save_model(clf, conf, name="", version=""):
    if len(name) == 0:
        name = conf["selected_dataset"] + "_" + conf["selected_model"]
    filename = (
        conf["paths"]["Outputs_path"]
        + conf["paths"]["folder_models"]
        + name
        + version
        + ".sav"
    )
    pickle.dump(clf, open(filename, "wb"))
    logger.info("Modele sauvergarde: " + filename)
    return "OK"


def load_model(conf, name=None, version=""):
    if name is None:
        name = conf["selected_dataset"] + "_" + conf["selected_model"]
    filename = (
        conf["paths"]["Outputs_path"]
        + conf["paths"]["folder_models"]
        + name
        + version
        + ".sav"
    )
    logger.debug(f"loading model at path: {filename}")
    with open(filename, "rb") as fd:
        clf = pickle.load(fd)
    logger.info("Model loaded: " + filename)
    return clf


def load_metrics(conf, metric_type="batch"):
    if metric_type == "batch":
        filepath = (
            conf["paths"]["Outputs_path"]
            + conf["paths"]["folder_metrics"]
            + conf["monitoring_db_path_batch"]
        )
    elif metric_type == "model":
        filepath = (
            conf["paths"]["Outputs_path"]
            + conf["paths"]["folder_metrics"]
            + conf["monitoring_db_path_model"]
        )
    else:
        raise NotImplementedError("Only batch and model csv are saved.")
    db = pd.read_csv(filepath)
    return db


def save_metrics(conf, db, metric_type="batch"):
    if metric_type == "batch":
        filepath = (
            conf["paths"]["Outputs_path"]
            + conf["paths"]["folder_metrics"]
            + conf["monitoring_db_path_batch"]
        )
    elif metric_type == "model":
        filepath = (
            conf["paths"]["Outputs_path"]
            + conf["paths"]["folder_metrics"]
            + conf["monitoring_db_path_model"]
        )
    else:
        raise NotImplementedError("Only batch and model csv are saved.")
    db.to_csv(filepath, index=False)


def save_outliers(outliers, conf):
    filepath = (
        conf["paths"]["Outputs_path"]
        + conf["paths"]["folder_metrics"]
        + conf["outliers_filename"]
    )
    outliers.to_csv(filepath, index_label="Time")


def load_outliers(conf):
    filepath = (
        conf["paths"]["Outputs_path"]
        + conf["paths"]["folder_metrics"]
        + conf["outliers_filename"]
    )
    outliers = pd.read_csv(filepath)
    outliers.set_index("Time", inplace=True)
    return outliers


def save_train_data_description(description, conf):
    filepath = (
        conf["paths"]["Outputs_path"]
        + conf["paths"]["folder_metrics"]
        + conf["description_filename"]
    )
    description.to_csv(filepath, index_label="Statistics")


def load_train_data_description(conf):
    filepath = (
        conf["paths"]["Outputs_path"]
        + conf["paths"]["folder_metrics"]
        + conf["description_filename"]
    )
    try:
        description = pd.read_csv(filepath)
        description.set_index("Statistics", inplace=True)
    except FileNotFoundError:
        description = None
    return description


def save_adwin(adwin, conf):
    filepath = (
        conf["paths"]["Outputs_path"]
        + conf["paths"]["folder_metrics"]
        + conf["adwin_filename"]
    )
    with open(filepath, "w") as fd:
        json.dump(adwin, fd)


def load_adwin(conf):
    filepath = (
        conf["paths"]["Outputs_path"]
        + conf["paths"]["folder_metrics"]
        + conf["adwin_filename"]
    )
    try:
        with open(filepath, "r") as fd:
            adwin = json.load(fd)
    except:
        adwin = None
    return adwin


def get_y_column_from_conf(conf):
    return conf["dict_info_files"][conf["selected_dataset"]]["y_name"]


def get_column_mapping(conf, df):
    y_column = get_y_column_from_conf(conf)
    X_columns = [x for x in df.columns if x != y_column]
    column_mapping = {
        "target": y_column,
        "prediction": "prediction",
        "numerical_features": X_columns,
    }
    return column_mapping


def get_sub_df_preprocessed(df_preprocessed, db_models):
    start_data = db_models.tail(1)["start"].values[0]
    end_data = db_models.tail(1)["end"].values[0]
    sub_df = df_preprocessed.loc[start_data:end_data, :]
    return sub_df


def get_sub_df_preprocessed_batch(df_preprocessed, db_batch):
    start_data = (
        db_batch.tail(1)["nb_lines"].values[0]
        - db_batch.tail(1)["batch_size"].values[0]
    )
    end_data = db_batch.tail(1)["nb_lines"].values[0]
    sub_df = df_preprocessed.loc[start_data:end_data, :]
    return sub_df
