"""Train and evaluate the next-activity classifiers.

ML parts:   RandomForestClassifier (primary) vs two simple baselines.
Evaluation: chronological split (train = earlier part, test = later part).
Leakage:    the one-hot encoder is fitted inside the Pipeline on training data only.

Run from the project folder:   python -m src.train_model
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import METRICS_PATH, MIN_RECORDS_FOR_TRAINING, MODEL_PATH
from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    InsufficientDataError,
    build_training_set,
)
from src.preprocessing import clean_activities


class MarkovBaseline:
    """Simple baseline: predict the most common successor of the current category."""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = pd.Series(np.asarray(y))
        self.fallback_ = y.value_counts().idxmax()
        table = pd.crosstab(np.asarray(X["current_category"]), y.values)
        self.table_ = table.idxmax(axis=1).to_dict()
        return self

    def predict(self, X: pd.DataFrame):
        return np.array([self.table_.get(c, self.fallback_) for c in X["current_category"]])


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


def make_random_forest(n_estimators: int = 200, random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=3,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def make_majority_baseline() -> Pipeline:
    return Pipeline([("prep", _preprocessor()), ("model", DummyClassifier(strategy="most_frequent"))])


def chronological_split_index(n_samples: int, test_fraction: float = 0.2) -> int:
    """Index where the test set starts. Samples before it train, samples from it onward test."""
    n_test = max(1, int(round(n_samples * test_fraction)))
    return n_samples - n_test


def _score(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _data_source(cleaned: pd.DataFrame) -> str:
    flags = pd.to_numeric(cleaned["is_demo"], errors="coerce").fillna(0)
    if (flags == 1).all():
        return "synthetic_demo"
    if (flags == 0).all():
        return "user_records"
    return "mixed"


def train_and_evaluate(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    n_estimators: int = 200,
    random_state: int = 42,
    save: bool = True,
    model_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
) -> dict:
    """Train baselines + Random Forest, evaluate on the chronological test set.

    Raises InsufficientDataError if the history is too small. Returns the metrics
    dictionary (also written to JSON when save=True). All numbers are computed
    from real predictions on held-out data.
    """
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5")

    cleaned = clean_activities(df)
    if len(cleaned) < MIN_RECORDS_FOR_TRAINING:
        raise InsufficientDataError(
            f"Only {len(cleaned)} valid records found; at least {MIN_RECORDS_FOR_TRAINING} are needed "
            "for a reliable chronological train/test evaluation. Record more activities or load the demo data."
        )

    X, y, meta = build_training_set(cleaned)
    split = chronological_split_index(len(X), test_fraction)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if y_train.nunique() < 2:
        raise InsufficientDataError("The training period contains only one activity type; cannot learn patterns.")
    if len(X_test) < 10:
        raise InsufficientDataError("Too few test samples after the chronological split.")

    majority = make_majority_baseline().fit(X_train, y_train)
    markov = MarkovBaseline().fit(X_train, y_train)
    forest = make_random_forest(n_estimators, random_state).fit(X_train, y_train)

    pred_rf = forest.predict(X_test)
    models = {
        "majority_baseline": _score(y_test, majority.predict(X_test)),
        "markov_baseline": _score(y_test, markov.predict(X_test)),
        "random_forest": _score(y_test, pred_rf),
    }

    labels = sorted(set(y_test) | set(pred_rf))
    cm = confusion_matrix(y_test, pred_rf, labels=labels)
    report = classification_report(y_test, pred_rf, labels=labels, output_dict=True, zero_division=0)
    per_class = {
        lab: {
            "precision": float(report[lab]["precision"]),
            "recall": float(report[lab]["recall"]),
            "f1": float(report[lab]["f1-score"]),
            "support": int(report[lab]["support"]),
        }
        for lab in labels
    }

    names = forest.named_steps["prep"].get_feature_names_out()
    importances = forest.named_steps["model"].feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    top_features = [
        {"feature": str(names[i]).split("__", 1)[-1], "importance": float(importances[i])} for i in top_idx
    ]

    source = _data_source(cleaned)
    notes = [
        "Metrics come from a chronological split: the model is trained on earlier samples and tested on later ones.",
        "Macro F1 averages F1 over classes appearing in the test labels or predictions, so rare classes count equally.",
        "The saved model is refitted on ALL samples; the metrics above describe the train-only model.",
    ]
    if source != "user_records":
        notes.append("Part or all of the data is synthetic, so these metrics do NOT represent real-world performance.")

    metrics = {
        "data_source": source,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_records": int(len(cleaned)),
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_fraction": float(test_fraction),
        "train_period": {"start": str(meta["predict_time"].iloc[0]), "end": str(meta["predict_time"].iloc[split - 1])},
        "test_period": {"start": str(meta["predict_time"].iloc[split]), "end": str(meta["predict_time"].iloc[-1])},
        "models": models,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "top_features": top_features,
        "test_class_distribution": {str(k): int(v) for k, v in y_test.value_counts().items()},
        "notes": notes,
    }

    if save:
        final = make_random_forest(n_estimators, random_state).fit(X, y)
        artifact = {
            "pipeline": final,
            "feature_columns": FEATURE_COLUMNS,
            "data_source": source,
            "trained_at": metrics["trained_at"],
            "n_training_samples": int(len(X)),
            "sklearn_version": sklearn.__version__,
        }
        mp = Path(model_path) if model_path else MODEL_PATH
        jp = Path(metrics_path) if metrics_path else METRICS_PATH
        mp.parent.mkdir(parents=True, exist_ok=True)
        jp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, mp)
        jp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def load_metrics(metrics_path: Optional[Path] = None) -> Optional[dict]:
    """Return the saved metrics dictionary, or None if no model has been trained."""
    jp = Path(metrics_path) if metrics_path else METRICS_PATH
    if not jp.exists():
        return None
    try:
        return json.loads(jp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def train_from_database(db_path: Optional[Path] = None, **kwargs) -> dict:
    """Train on whatever activities are currently stored in the database."""
    from LifeRhythmAI.src import database as db

    return train_and_evaluate(db.get_activities(db_path=db_path), **kwargs)


def main() -> None:
    from LifeRhythmAI.src import database as db
    from LifeRhythmAI.src.data_generator import generate_demo_data

    df = db.get_activities()
    if df.empty:
        print("Database is empty: training on SYNTHETIC demo data generated in memory (not saved to the database).")
        df = generate_demo_data()
        df["is_demo"] = 1
    try:
        m = train_and_evaluate(df)
    except InsufficientDataError as exc:
        print(f"Cannot train: {exc}")
        sys.exit(1)

    print(f"\nData source: {m['data_source']}  |  records: {m['n_records']}  |  samples: {m['n_samples']}")
    print(f"Train samples: {m['n_train']}  ({m['train_period']['start']} -> {m['train_period']['end']})")
    print(f"Test samples:  {m['n_test']}  ({m['test_period']['start']} -> {m['test_period']['end']})\n")
    print(f"{'Model':<20}{'Accuracy':>10}{'Macro F1':>10}")
    for name, s in m["models"].items():
        print(f"{name:<20}{s['accuracy']:>10.3f}{s['macro_f1']:>10.3f}")
    print("\nTop features:")
    for f in m["top_features"][:5]:
        print(f"  {f['feature']:<35}{f['importance']:.3f}")
    for note in m["notes"]:
        print(f"* {note}")
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()