"""Input validation, interval construction, overlap detection and cleaning."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Optional, Union

import pandas as pd

from src.config import CATEGORIES, MAX_DURATION_MIN

ACTIVITY_COLUMNS = ["id", "start_dt", "end_dt", "duration_min", "category", "notes", "is_demo"]


class ValidationError(ValueError):
    """Raised when user input is invalid."""


def parse_date(value: Union[str, date, datetime]) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        parsed = datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValidationError(f"Invalid date '{value}'. Use the format YYYY-MM-DD.")
    if parsed.year < 2000 or parsed.year > 2100:
        raise ValidationError(f"Date '{value}' is outside the supported range (2000-2100).")
    return parsed


def parse_time(value: Union[str, time, datetime]) -> time:
    if isinstance(value, datetime):
        return value.time().replace(second=0, microsecond=0)
    if isinstance(value, time):
        return value.replace(second=0, microsecond=0)
    try:
        return datetime.strptime(str(value).strip(), "%H:%M").time()
    except ValueError:
        raise ValidationError(f"Invalid time '{value}'. Use the 24-hour format HH:MM.")


def validate_category(category: str) -> str:
    if category not in CATEGORIES:
        raise ValidationError(f"Unknown category '{category}'. Choose one of: {', '.join(CATEGORIES)}.")
    return category


def build_interval(
    date_value,
    start_time,
    end_time: Optional[Union[str, time]] = None,
    duration_min: Optional[int] = None,
) -> dict:
    """Return {'start_dt', 'end_dt', 'duration_min'} after validating the inputs.

    Exactly one of end_time or duration_min must be given. If end_time is earlier
    than start_time the activity is assumed to cross midnight (e.g. sleep).
    """
    if (end_time is None) == (duration_min is None):
        raise ValidationError("Provide either an end time or a duration (not both, not neither).")

    start_dt = datetime.combine(parse_date(date_value), parse_time(start_time))

    if duration_min is not None:
        try:
            duration = int(duration_min)
        except (TypeError, ValueError):
            raise ValidationError("Duration must be a whole number of minutes.")
        if duration <= 0:
            raise ValidationError("Duration must be greater than zero.")
        end_dt = start_dt + timedelta(minutes=duration)
    else:
        end_t = parse_time(end_time)
        end_dt = datetime.combine(start_dt.date(), end_t)
        if end_dt == start_dt:
            raise ValidationError("End time equals start time, so the duration would be zero.")
        if end_dt < start_dt:
            end_dt += timedelta(days=1)  # crosses midnight
        duration = int((end_dt - start_dt).total_seconds() // 60)

    if duration > MAX_DURATION_MIN:
        raise ValidationError(
            f"Duration of {duration} min exceeds the maximum of {MAX_DURATION_MIN} min. "
            "Please check the start and end times."
        )
    return {"start_dt": start_dt, "end_dt": end_dt, "duration_min": duration}


def find_overlaps(
    start_dt: datetime,
    end_dt: datetime,
    existing: pd.DataFrame,
    exclude_id: Optional[int] = None,
) -> pd.DataFrame:
    """Return rows of `existing` that overlap [start_dt, end_dt). Touching edges are fine."""
    if existing is None or existing.empty:
        return pd.DataFrame(columns=ACTIVITY_COLUMNS)
    mask = (existing["start_dt"] < end_dt) & (existing["end_dt"] > start_dt)
    if exclude_id is not None and "id" in existing.columns:
        mask &= existing["id"] != exclude_id
    return existing[mask].copy()


def clean_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Clean an activity DataFrame for analysis/ML.

    - converts datetimes, drops unparseable rows
    - drops rows with end <= start or unknown category
    - recomputes duration_min from the timestamps
    - drops exact duplicates (same start, end, category)
    - sorts chronologically
    """
    columns = ["start_dt", "end_dt", "duration_min", "category"]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=ACTIVITY_COLUMNS).astype(
            {"start_dt": "datetime64[ns]", "end_dt": "datetime64[ns]"}
        )
    missing = [c for c in ("start_dt", "end_dt", "category") if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")

    out = df.copy()
    out["start_dt"] = pd.to_datetime(out["start_dt"], errors="coerce")
    out["end_dt"] = pd.to_datetime(out["end_dt"], errors="coerce")
    out = out.dropna(subset=["start_dt", "end_dt", "category"])
    out = out[out["end_dt"] > out["start_dt"]]
    out = out[out["category"].isin(CATEGORIES)]
    out["duration_min"] = ((out["end_dt"] - out["start_dt"]).dt.total_seconds() // 60).astype(int)
    out = out[out["duration_min"] > 0]
    out = out.drop_duplicates(subset=["start_dt", "end_dt", "category"])
    for col, default in (("id", None), ("notes", ""), ("is_demo", 0)):
        if col not in out.columns:
            out[col] = default
    out = out.sort_values("start_dt").reset_index(drop=True)
    return out[ACTIVITY_COLUMNS]