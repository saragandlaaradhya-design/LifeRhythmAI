"""Central configuration: paths, categories and shared constants."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "liferhythm.db"
MODEL_PATH = MODELS_DIR / "next_activity_model.joblib"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

CATEGORIES = [
    "Sleeping",
    "Studying",
    "Attending classes",
    "Exercise",
    "Eating",
    "Travelling",
    "Leisure",
    "Social activities",
    "Personal tasks",
    "Other",
]

# Storage format for datetimes in SQLite (sortable as text)
DB_DT_FORMAT = "%Y-%m-%d %H:%M:%S"

# Longest single activity we accept (minutes). 16 h allows long sleep.
MAX_DURATION_MIN = 16 * 60

# Minimum number of cleaned records before supervised training is attempted
MIN_RECORDS_FOR_TRAINING = 120

# Suggestion statuses
SUGGESTION_STATUSES = ("pending", "accepted", "dismissed", "completed")