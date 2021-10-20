from skmultiflow.drift_detection.adwin import ADWIN
from evidently.model_profile import Profile
from evidently.profile_sections import (
    DataDriftProfileSection,
    CatTargetDriftProfileSection,
)
from sklearn.ensemble import IsolationForest

import sys

sys.path.insert(0, "utils/")
import utils as u

import json
import pandas as pd
from logzero import logger


def check_adwin_concept_drift(data, features):
    """

    Args:
        data ([pd.DataFrame]): historical data
        features ([list]): list of columns to check

    Returns:
        [dict]: feature-wise
    """

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
    """Detects drifts between a past data dataset and a new batch on both
        features and target

    Args:
        past_data (pd.DataFrame): Training dataset of the model in production
            (both features and target)
        new_batch (pd.DataFrame): New batch to be evaluated
        column_mapping (dict): {
            "numerical_features": [list of numerical features],
            "target": target
        }
        confidence (float, optional): Confidence level for the Kolmogorov-Smirnov test.
            Defaults to 0.95.
        get_pvalues (bool, optional): Returns 1 if Data Drift is detected, else returns 0.
            If get_pvalues is True, returns p-value for each feature. Defaults to True.

    Returns:
        [list]: Returns 1 if Data Drift is detected, else returns 0.
            If get_pvalues is True, returns p-value for each feature.
    """

    logger.debug(f"column_mapping: {column_mapping}")
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
    cat_features = (
        column_mapping.get("categorical_features")
        if column_mapping.get("categorical_features")
        else []
    )
    for feature in num_features + cat_features:
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
