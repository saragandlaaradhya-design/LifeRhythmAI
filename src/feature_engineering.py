"""Leakage-safe feature engineering for next-activity prediction.

A training "sample" is created at the moment an activity ENDS. Every feature
describes only what is already known at that moment (the current activity, the
two before it, the time of day, the gap before it). The label (target) is the
category of the FOLLOWING activity and never appears in the feature table X.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import clean_activities

# If the next activity starts more than this long after the current one ends,
# the log probably has a hole, so we do not treat it as "the next activity".
MAX_GAP_MINUTES = 12 * 60
NONE_TOKEN = "None"  # used when there is no previous activity available

CATEGORICAL_FEATURES = ["current_category", "prev_category", "prev2_category"]
NUMERIC_FEATURES = [
    "end_hour",
    "end_hour_sin",
    "end_hour_cos",
    "end_dow",
    "is_weekend",
    "start_hour",
    "current_duration_min",
    "gap_since_prev_min",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


class InsufficientDataError(Exception):
    """Raised when there is not enough (valid) data for the requested operation."""


def compute_features(df: pd.DataFrame):
    """Return (features, cleaned_activities).

    `features` has one row per cleaned activity (same index) and uses ONLY
    information from that activity and earlier ones.
    """
    d = clean_activities(df)
    if d.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS), d

    cat = d["category"]
    start, end = d["start_dt"], d["end_dt"]

    # minutes between the end of the previous activity and the start of this one
    gap_before = (start - end.shift(1)).dt.total_seconds() / 60.0
    broken1 = gap_before.isna() | (gap_before > MAX_GAP_MINUTES)
    broken2 = broken1 | broken1.shift(1, fill_value=True)

    end_hour = end.dt.hour + end.dt.minute / 60.0
    start_hour = start.dt.hour + start.dt.minute / 60.0
    end_dow = end.dt.dayofweek

    feats = pd.DataFrame(
        {
            "current_category": cat,
            "prev_category": cat.shift(1).where(~broken1, NONE_TOKEN).fillna(NONE_TOKEN),
            "prev2_category": cat.shift(2).where(~broken2, NONE_TOKEN).fillna(NONE_TOKEN),
            "end_hour": end_hour,
            "end_hour_sin": np.sin(2 * np.pi * end_hour / 24.0),
            "end_hour_cos": np.cos(2 * np.pi * end_hour / 24.0),
            "end_dow": end_dow,
            "is_weekend": (end_dow >= 5).astype(int),
            "start_hour": start_hour,
            "current_duration_min": d["duration_min"].astype(float),
            "gap_since_prev_min": gap_before.fillna(0).clip(lower=0, upper=MAX_GAP_MINUTES),
        }
    )
    return feats[FEATURE_COLUMNS], d


def build_training_set(df: pd.DataFrame):
    """Return (X, y, meta) for supervised learning, in chronological order.

    y[i] is the category of the activity that follows sample i.
    meta has: row_index (position in the cleaned data), predict_time (when the
    prediction would be made = end of the current activity).
    """
    feats, d = compute_features(df)
    if d.empty:
        raise InsufficientDataError("There are no valid activity records.")

    next_cat = d["category"].shift(-1)
    gap_next = (d["start_dt"].shift(-1) - d["end_dt"]).dt.total_seconds() / 60.0
    valid = (next_cat.notna() & (gap_next <= MAX_GAP_MINUTES)).values
    if not valid.any():
        raise InsufficientDataError("No consecutive activity pairs are available for training.")

    X = feats[valid].reset_index(drop=True)
    y = next_cat[valid].reset_index(drop=True)
    meta = pd.DataFrame(
        {
            "row_index": d.index[valid].values,
            "predict_time": d.loc[valid, "end_dt"].values,
        }
    )
    return X, y, meta


def build_prediction_row(df: pd.DataFrame, as_of=None):
    """Features describing 'right now' (the end of the latest activity).

    If `as_of` is given, activities that ended after it are ignored.
    Returns (one_row_dataframe, context_dict).
    """
    data = clean_activities(df)
    if as_of is not None:
        data = data[data["end_dt"] <= pd.Timestamp(as_of)]
    feats, d = compute_features(data)
    if d.empty:
        raise InsufficientDataError(
            "No activity history is available. Record some activities (or load the demo data) first."
        )
    last = d.iloc[-1]
    context = {
        "last_category": str(last["category"]),
        "ended_at": last["end_dt"],
        "last_duration_min": int(last["duration_min"]),
    }
    return feats.iloc[[-1]].reset_index(drop=True), context