from skmultiflow.drift_detection.adwin import ADWIN
from evidently.model_profile import Profile
from evidently.profile_sections import (
    DataDriftProfileSection,
    CatTargetDriftProfileSection,
)
from sklearn.ensemble import IsolationForest

import json
import sys
import pandas as pd

sys.path.insert(0, "utils/")

import utils as u
from logzero import logger


def check_adwin_concept_drift(data, features):

    adwin_results = {}
    for feature in features:
        feature_idx_changes = check_featurewise_concept_drift(data, feature)
        adwin_results[feature] = feature_idx_changes
    return adwin_results


def check_featurewise_concept_drift(data, feature):
    """
    Args:
        past_data:
        new_batch:

    Returns:

    """

    adwin = ADWIN()
    idx_of_changes = []
    for idx, row in enumerate(data[feature]):
        adwin.add_element(row)
        if adwin.detected_change():
            idx_of_changes.append(idx)
    return idx_of_changes


def detect_features_drift(
    past_data,
    new_batch,
    column_mapping,
    confidence=0.95,
    get_pvalues=True,
):
    """
    Returns 1 if Data Drift is detected, else returns 0.
    If get_pvalues is True, returns p-value for each feature.
    The Data Drift detection depends on the confidence level and the threshold.
    For each individual feature Data Drift is detected with the selected confidence (default value is 0.95).
    """

    data_drift_profile = Profile(
        sections=[DataDriftProfileSection, CatTargetDriftProfileSection]
    )
    data_drift_profile.calculate(past_data, new_batch, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)

    drifts = []
    num_features = (
        column_mapping.get("numerical_features")
        if column_mapping.get("numerical_features")
        else []
    )
    for feature in num_features:
        p_value = json_report["data_drift"]["data"]["metrics"][feature]["p_value"]
        if get_pvalues:
            drifts.append((feature, p_value))
        else:
            drifts.append((feature, True if p_value < (1.0 - confidence) else False))

    p_value_target = json_report["cat_target_drift"]["data"]["metrics"]["target_drift"]
    if get_pvalues:
        drifts.append((column_mapping["target"], p_value_target))
    else:
        drifts.append(
            (
                column_mapping["target"],
                True if p_value_target < (1.0 - confidence) else False,
            )
        )

    return drifts


def get_means_std(df_preprocessed):
    """

    Args:
        df_preprocessed:

    Returns:

    """
    means = df_preprocessed.mean(axis=0).to_dict()
    std = df_preprocessed.std(axis=0).to_dict()
    ret = {}
    for k in means.keys():
        ret[k + "_mean"] = means[k]
        ret[k + "_std"] = std[k]
    return ret


def get_current_state(db_batch, global_data):
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


def fit_isolation_forest(X_train, conf, version):
    iso = IsolationForest(contamination=0.05)
    iso.fit(X_train)
    u.save_model(iso, conf, name="isolation_forest", version=version)
    outliers = get_outliers(X_train, iso, conf)
    return outliers


def get_outliers(X, iso_forest, conf):
    y_pred = iso_forest.predict(X)
    outliers_mask = y_pred == -1
    X_outliers = X.loc[outliers_mask, :]
    u.save_outliers(X_outliers, conf)
    return X_outliers


def get_share_outliers(db):
    return str(round(db.at[len(db) - 1, "nb_outliers"] * 100)) + "%"


def get_recap(db, global_data):
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
