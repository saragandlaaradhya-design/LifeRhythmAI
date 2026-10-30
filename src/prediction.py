"""Prediction service: load the trained model and predict the next activity."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH
from src.feature_engineering import build_prediction_row

DISCLAIMER = (
    "This is a statistical estimate based on your past routine, not a certainty. "
    "Confidence is the share of Random Forest trees voting for the activity and is not a guaranteed probability."
)


class ModelNotTrainedError(Exception):
    """Raised when no usable trained model file exists."""


def load_model(model_path: Optional[Path] = None) -> dict:
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise ModelNotTrainedError("No trained model found. Train the model first (Model page, or: python -m src.train_model).")
    try:
        artifact = joblib.load(path)
    except Exception as exc:  # corrupt file, version mismatch...
        raise ModelNotTrainedError(f"The saved model could not be loaded ({exc}). Please retrain it.") from exc
    if not isinstance(artifact, dict) or not {"pipeline", "feature_columns"} <= set(artifact):
        raise ModelNotTrainedError("The saved model file has an unexpected format. Please retrain it.")
    return artifact


def _confidence_label(p: float) -> str:
    if p >= 0.6:
        return "higher confidence"
    if p >= 0.4:
        return "moderate confidence"
    return "low confidence"


def predict_next_activity(
    history: pd.DataFrame,
    as_of=None,
    model_path: Optional[Path] = None,
    artifact: Optional[dict] = None,
    top_k: int = 3,
) -> dict:
    """Predict the activity most likely to follow the latest completed one.

    Raises ModelNotTrainedError / InsufficientDataError (from feature_engineering).
    """
    if artifact is None:
        artifact = load_model(model_path)
    row, context = build_prediction_row(history, as_of)
    row = row[artifact["feature_columns"]]

    pipe = artifact["pipeline"]
    proba = pipe.predict_proba(row)[0]
    classes = list(pipe.classes_)
    order = np.argsort(proba)[::-1][: max(1, top_k)]
    top = [{"category": str(classes[i]), "probability": float(proba[i])} for i in order]

    reference = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(datetime.now())
    hours_since = float((reference - context["ended_at"]).total_seconds() / 3600.0)
    warnings = []
    if hours_since > 12:
        warnings.append(
            f"Your last logged activity ended about {hours_since:.0f} hours ago, so this prediction may be outdated."
        )
    if artifact.get("data_source") != "user_records":
        warnings.append("The model was trained on synthetic or mixed data, not only on your own records.")

    return {
        "predicted": top[0]["category"],
        "confidence": top[0]["probability"],
        "confidence_label": _confidence_label(top[0]["probability"]),
        "top_k": top,
        "based_on": context,
        "hours_since_last_activity": hours_since,
        "data_source": artifact.get("data_source"),
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }