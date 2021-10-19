# -*- coding: utf-8 -*-
import logging
import pickle
import pandas as pd

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


def save_metrics(conf, db, type="batch"):
    if type == "batch":
        filepath = (
            conf["paths"]["Outputs_path"]
            + conf["paths"]["folder_metrics"]
            + conf["monitoring_db_path_batch"]
        )
    else:
        filepath = (
            conf["paths"]["Outputs_path"]
            + conf["paths"]["folder_metrics"]
            + conf["monitoring_db_path_model"]
        )
    db.to_csv(filepath, index=False)


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
