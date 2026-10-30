"""Synthetic demo-data generator.

IMPORTANT: this data is SIMULATED. Model accuracy measured on it says nothing
about real-world performance. The last `drift_weeks` weeks deliberately contain
a routine change (later bedtime, less studying, less exercise) so that the
drift-detection module has something genuine to find.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.config import CATEGORIES


def _clip_int(rng, mean, sd, lo, hi) -> int:
    return int(np.clip(rng.normal(mean, sd), lo, hi))


def _weekday_plan(rng, wake, drifted):
    """(category, target_start_minute, duration_min, probability) in order."""
    study_p = 0.5 if drifted else 0.9
    study_dur = _clip_int(rng, 55 if drifted else 110, 20, 25, 180)
    exercise_p = 0.2 if drifted else 0.6
    return [
        ("Personal tasks", wake, _clip_int(rng, 35, 8, 15, 60), 1.0),
        ("Eating", wake + 40, _clip_int(rng, 25, 5, 12, 40), 1.0),
        ("Travelling", 8 * 60 + 20, _clip_int(rng, 35, 8, 20, 60), 0.95),
        ("Attending classes", 9 * 60, _clip_int(rng, 200, 25, 120, 270), 0.9),
        ("Eating", 13 * 60, _clip_int(rng, 40, 8, 20, 60), 1.0),
        ("Attending classes", 14 * 60, _clip_int(rng, 110, 20, 60, 160), 0.7),
        ("Travelling", 16 * 60 + 30, _clip_int(rng, 35, 8, 20, 60), 0.9),
        ("Exercise", 17 * 60 + 30, _clip_int(rng, 50, 10, 25, 80), exercise_p),
        ("Leisure", 18 * 60 + 30, _clip_int(rng, 75 if not drifted else 105, 20, 30, 150), 0.9),
        ("Eating", 20 * 60, _clip_int(rng, 40, 8, 20, 60), 1.0),
        ("Studying", 20 * 60 + 50, study_dur, study_p),
        ("Social activities", 19 * 60 + 15, _clip_int(rng, 60, 20, 30, 120), 0.15),
        ("Leisure", 22 * 60 + 30, _clip_int(rng, 45, 15, 20, 90), 0.7),
        ("Other", 15 * 60, _clip_int(rng, 30, 10, 15, 60), 0.1),
    ]


def _weekend_plan(rng, wake, drifted):
    exercise_p = 0.25 if drifted else 0.55
    study_p = 0.45 if drifted else 0.75
    return [
        ("Personal tasks", wake, _clip_int(rng, 40, 10, 15, 70), 1.0),
        ("Eating", wake + 50, _clip_int(rng, 35, 8, 15, 50), 1.0),
        ("Exercise", 10 * 60 + 30, _clip_int(rng, 55, 12, 25, 90), exercise_p),
        ("Studying", 12 * 60, _clip_int(rng, 80 if drifted else 105, 20, 30, 150), study_p),
        ("Eating", 14 * 60, _clip_int(rng, 45, 10, 20, 70), 1.0),
        ("Social activities", 15 * 60 + 30, _clip_int(rng, 120, 30, 45, 200), 0.7),
        ("Leisure", 18 * 60, _clip_int(rng, 100, 25, 40, 180), 0.9),
        ("Eating", 20 * 60 + 15, _clip_int(rng, 45, 10, 20, 70), 1.0),
        ("Leisure", 21 * 60 + 15, _clip_int(rng, 70, 20, 30, 120), 0.8),
        ("Travelling", 11 * 60 + 45, _clip_int(rng, 30, 10, 15, 50), 0.2),
        ("Other", 16 * 60, _clip_int(rng, 40, 15, 15, 80), 0.15),
    ]


def generate_demo_data(
    start_date: Optional[date] = None,
    weeks: int = 8,
    drift_weeks: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Create `weeks` of synthetic, non-overlapping activity records.

    Returns columns: start_dt, end_dt, duration_min, category, notes.
    """
    if weeks < 1:
        raise ValueError("weeks must be at least 1")
    drift_weeks = max(0, min(drift_weeks, weeks - 1)) if weeks > 1 else 0

    rng = np.random.default_rng(seed)
    if start_date is None:
        start_date = date.today() - timedelta(days=weeks * 7)
    n_days = weeks * 7
    drift_start_day = n_days - drift_weeks * 7

    rows = []
    wake = 7 * 60 + 15  # minutes after midnight of the current day

    for i in range(n_days):
        day = start_date + timedelta(days=i)
        midnight = datetime.combine(day, datetime.min.time())
        weekend = day.weekday() >= 5
        drifted = i >= drift_start_day

        # Bedtime in minutes after this day's midnight (>=1440 means after midnight)
        if weekend:
            bedtime = _clip_int(rng, 24 * 60 + 5, 40, 22 * 60 + 45, 26 * 60)
        else:
            bedtime = _clip_int(rng, 23 * 60, 25, 22 * 60, 24 * 60 + 40)
        if drifted:
            bedtime += 75
        sleep_dur = _clip_int(rng, 7 * 60 + 10, 40, 5 * 60 + 30, 9 * 60 + 30)

        plan = (_weekend_plan if weekend else _weekday_plan)(rng, wake, drifted)
        plan.sort(key=lambda p: p[1])  # order by target start

        cursor = wake
        for category, target, duration, prob in plan:
            if rng.random() > prob:
                continue
            start = max(cursor + int(rng.integers(0, 10)), target + int(rng.normal(0, 8)))
            if start + duration > bedtime - 20:
                continue
            rows.append((midnight + timedelta(minutes=start),
                         midnight + timedelta(minutes=start + duration),
                         duration, category))
            cursor = start + duration

        rows.append((midnight + timedelta(minutes=bedtime),
                     midnight + timedelta(minutes=bedtime + sleep_dur),
                     sleep_dur, "Sleeping"))
        wake = bedtime + sleep_dur - 1440
        wake = max(wake, 5 * 60)

    df = pd.DataFrame(rows, columns=["start_dt", "end_dt", "duration_min", "category"])
    df["notes"] = "synthetic"
    df = df.sort_values("start_dt").reset_index(drop=True)
    assert set(df["category"]).issubset(set(CATEGORIES))
    return df


if __name__ == "__main__":
    # Quick manual check:  python -m src.data_generator
    data = generate_demo_data()
    print(f"Generated {len(data)} synthetic records, {data['start_dt'].min()} -> {data['end_dt'].max()}")
    print(data["category"].value_counts())