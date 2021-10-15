import skmultiflow
from skmultiflow.drift_detection.adwin import ADWIN
from evidently.model_profile import Profile
from evidently.profile_sections import DataDriftProfileSection, CatTargetDriftProfileSection
import json

def check_concept_drift(past_data, new_batch):
    """
    Args:
        past_data:
        new_batch:

    Returns:

    """

    adwin = ADWIN()
    adwin.add_element(past_data)
    adwin.add_element(new_batch)
    detected_change = adwin.detected_change()
    detected_warning = adwin.detected_warning_zone()
    return detected_warning, detected_change


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

    data_drift_profile = Profile(sections=[DataDriftProfileSection, CatTargetDriftProfileSection])
    data_drift_profile.calculate(past_data, new_batch, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)

    drifts = []
    num_features = column_mapping.get('numerical_features') if column_mapping.get('numerical_features') else []
    for feature in num_features:
        p_value = json_report["data_drift"]["data"]["metrics"][feature]["p_value"]
        if get_pvalues:
            drifts.append((feature, p_value))
        else:
            drifts.append((feature, True if p_value < (1.0 - confidence) else False))

    p_value_target = json_report["cat_target_drift"]["data"]["metrics"]["target_drift"]
    if get_pvalues:
        drifts.append(("target", p_value_target))
    else:
        drifts.append(("target", True if p_value_target < (1.0 - confidence) else False))

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