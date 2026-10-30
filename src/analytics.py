"""Descriptive analytics: plain statistics about the logged routine.

This module is NOT machine learning. It only summarises what was recorded
(averages, counts, spreads). Circular statistics are used for clock times so
that 23:50 and 00:10 are treated as 20 minutes apart, not 23 hours and 40 minutes.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.config import CATEGORIES
from src.preprocessing import clean_activities, parse_date

MAIN_SLEEP_MIN_DURATION = 180  # minutes; shorter Sleeping records are treated as naps

METRIC_COLUMNS = ["study_min", "exercise_sessions", "exercise_day", "bedtime_min", "sleep_min", "weekend"]


# ------------------------------------------------------------ clock helpers
def circular_mean_minutes(values) -> float:
    """Mean of clock times given as minutes after midnight (wraps around midnight)."""
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan")
    ang = a * 2 * np.pi / 1440.0
    mean = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    return float((mean % (2 * np.pi)) * 1440.0 / (2 * np.pi))


def circular_std_minutes(values) -> float:
    """Spread of clock times in minutes (circular standard deviation). Needs 2+ values."""
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if a.size < 2:
        return float("nan")
    ang = a * 2 * np.pi / 1440.0
    r = float(np.hypot(np.sin(ang).mean(), np.cos(ang).mean()))
    r = min(max(r, 1e-12), 1.0)
    return float(np.sqrt(-2.0 * np.log(r)) * 1440.0 / (2 * np.pi))


def circular_diff_minutes(a: float, b: float) -> float:
    """Signed difference a - b between two clock times, in the range [-720, 720)."""
    return float((a - b + 720.0) % 1440.0 - 720.0)


def format_clock(minutes) -> str:
    if minutes is None or pd.isna(minutes):
        return "n/a"
    m = int(round(float(minutes))) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def regularity_label(spread_min) -> str:
    if spread_min is None or pd.isna(spread_min):
        return "not enough data"
    if spread_min < 30:
        return "very regular"
    if spread_min < 60:
        return "fairly regular"
    return "variable"


def _none_if_nan(x):
    return None if x is None or pd.isna(x) else float(x)


# ------------------------------------------------------------ day assignment
def _with_day(d: pd.DataFrame) -> pd.DataFrame:
    """Add a 'day' column = the calendar date the activity belongs to.

    Normally the start date. A Sleeping record that starts after midnight but
    before 06:00 belongs to the PREVIOUS day (it is that evening's sleep), so a
    late bedtime never creates a phantom "day" that contains only sleep.
    """
    out = d.copy()
    day = out["start_dt"].dt.normalize()
    early_sleep = (out["category"] == "Sleeping") & (out["start_dt"].dt.hour < 6)
    out["day"] = day.where(~early_sleep, day - pd.Timedelta(days=1))
    return out


def daily_metrics(df: pd.DataFrame, timing_categories: Iterable[str] = ()) -> pd.DataFrame:
    """One row per LOGGED day (a day with at least one record).

    Columns: study_min, exercise_sessions, exercise_day (0/1), bedtime_min (clock
    minutes of the main night sleep start), sleep_min, weekend (bool), and for each
    category in timing_categories a column 'start_<category>' with the clock minute
    of its first start that day.

    A day with no records at all is NOT a row: "not logged" is not the same as
    "did not study".
    """
    timing_categories = list(timing_categories)
    timing_cols = [f"start_{c}" for c in timing_categories]
    columns = METRIC_COLUMNS + timing_cols
    d = clean_activities(df)
    if d.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="day"))

    d = _with_day(d)
    index = pd.DatetimeIndex(np.sort(d["day"].unique()), name="day")
    m = pd.DataFrame(index=index)

    study = d.loc[d["category"] == "Studying"].groupby("day")["duration_min"].sum()
    m["study_min"] = study.reindex(index).fillna(0).astype(float)
    ex = d.loc[d["category"] == "Exercise"].groupby("day").size()
    m["exercise_sessions"] = ex.reindex(index).fillna(0).astype(float)
    m["exercise_day"] = (m["exercise_sessions"] > 0).astype(float)

    sl = d.loc[(d["category"] == "Sleeping") & (d["duration_min"] >= MAIN_SLEEP_MIN_DURATION)].copy()
    hour = sl["start_dt"].dt.hour
    sl = sl.loc[(hour >= 18) | (hour < 6)]
    if sl.empty:
        m["bedtime_min"] = np.nan
        m["sleep_min"] = np.nan
    else:
        sl = sl.sort_values("duration_min").groupby("day").tail(1).set_index("day")
        bed = (sl["start_dt"].dt.hour * 60 + sl["start_dt"].dt.minute).astype(float)
        m["bedtime_min"] = bed.reindex(index)
        m["sleep_min"] = sl["duration_min"].astype(float).reindex(index)

    m["weekend"] = index.dayofweek >= 5

    for cat, col in zip(timing_categories, timing_cols):
        sub = d.loc[d["category"] == cat]
        if sub.empty:
            m[col] = np.nan
            continue
        first = sub.groupby("day")["start_dt"].min()
        m[col] = (first.dt.hour * 60 + first.dt.minute).astype(float).reindex(index)

    return m[columns]


# ------------------------------------------------------------ summaries
def average_daily_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Average minutes per logged day for every category (by start date)."""
    d = clean_activities(df)
    if d.empty:
        return pd.DataFrame({"category": CATEGORIES, "avg_min_per_day": 0.0, "share_pct": 0.0})
    d = _with_day(d)
    n_days = d["day"].nunique()
    totals = d.groupby("category")["duration_min"].sum().reindex(CATEGORIES).fillna(0.0)
    avg = totals / n_days
    total = float(avg.sum())
    share = avg / total * 100.0 if total > 0 else avg * 0.0
    out = pd.DataFrame({"category": CATEGORIES, "avg_min_per_day": avg.values, "share_pct": share.values})
    return out.sort_values("avg_min_per_day", ascending=False).reset_index(drop=True)


def weekday_weekend_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Average minutes per logged weekday vs weekend day, for every category."""
    d = clean_activities(df)
    res = pd.DataFrame({"category": CATEGORIES})
    if d.empty:
        res["weekday_avg_min"] = np.nan
        res["weekend_avg_min"] = np.nan
        res["difference_min"] = np.nan
        return res
    d = _with_day(d)
    d["weekend"] = d["day"].dt.dayofweek >= 5
    for flag, col in ((False, "weekday_avg_min"), (True, "weekend_avg_min")):
        sub = d[d["weekend"] == flag]
        n = sub["day"].nunique()
        if n == 0:
            res[col] = np.nan
        else:
            res[col] = (sub.groupby("category")["duration_min"].sum().reindex(CATEGORIES).fillna(0.0) / n).values
    res["difference_min"] = res["weekend_avg_min"] - res["weekday_avg_min"]
    return res


def hourly_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Average minutes per logged day spent in each category during each clock hour (0-23)."""
    occ = pd.DataFrame(0.0, index=range(24), columns=CATEGORIES)
    d = clean_activities(df)
    if d.empty:
        return occ
    n_days = _with_day(d)["day"].nunique()
    starts = [t.to_pydatetime() for t in d["start_dt"]]
    ends = [t.to_pydatetime() for t in d["end_dt"]]
    for s, e, c in zip(starts, ends, d["category"]):
        cur = s
        while cur < e:
            nxt = min(cur.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1), e)
            occ.at[cur.hour, c] += (nxt - cur).total_seconds() / 60.0
            cur = nxt
    return occ / n_days


def typical_activity_by_hour(df: pd.DataFrame, min_minutes: float = 5.0) -> pd.DataFrame:
    """For each hour of the day, the category that occupies most of it on average."""
    occ = hourly_occupancy(df)
    rows = []
    for h in range(24):
        row = occ.loc[h]
        if row.sum() < min_minutes:
            rows.append({"hour": h, "hour_label": f"{h:02d}:00", "typical_category": "No clear data",
                         "avg_min_in_hour": 0.0, "share_of_hour_pct": 0.0})
        else:
            cat = str(row.idxmax())
            rows.append({"hour": h, "hour_label": f"{h:02d}:00", "typical_category": cat,
                         "avg_min_in_hour": float(row.max()), "share_of_hour_pct": float(row.max() / 60.0 * 100.0)})
    return pd.DataFrame(rows)


def summary_stats(df: pd.DataFrame) -> dict:
    """Headline numbers: sleep, study and exercise habits (descriptive statistics only)."""
    m = daily_metrics(df)
    if m.empty:
        return {"logged_days": 0}
    nights = m["bedtime_min"].dropna()
    return {
        "logged_days": int(len(m)),
        "n_nights": int(len(nights)),
        "avg_sleep_min": _none_if_nan(m["sleep_min"].mean()),
        "typical_bedtime": format_clock(circular_mean_minutes(nights)) if len(nights) else None,
        "bedtime_spread_min": _none_if_nan(circular_std_minutes(nights)),
        "avg_study_min_per_day": float(m["study_min"].mean()),
        "study_days_pct": float((m["study_min"] > 0).mean() * 100.0),
        "exercise_days_per_week": float(m["exercise_day"].mean() * 7.0),
    }


def consistency_by_category(df: pd.DataFrame, min_days: int = 5) -> pd.DataFrame:
    """How steady is the daily start time of each category? (lower spread = more consistent)

    Uses the first start of the category each day (night sleep start for Sleeping).
    Sorted from most consistent to most variable.
    """
    cols = ["category", "days_observed", "typical_start", "start_spread_min", "regularity"]
    others = [c for c in CATEGORIES if c != "Sleeping"]
    m = daily_metrics(df, others)
    series = {"Sleeping": m["bedtime_min"]}
    for c in others:
        series[c] = m[f"start_{c}"]
    rows = []
    for cat, s in series.items():
        v = s.dropna().astype(float)
        if len(v) >= min_days:
            spread = circular_std_minutes(v)
            rows.append({"category": cat, "days_observed": int(len(v)),
                         "typical_start": format_clock(circular_mean_minutes(v)),
                         "start_spread_min": spread, "regularity": regularity_label(spread)})
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("start_spread_min").reset_index(drop=True) if not out.empty else out


def weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar week (Monday start): how the routine changed over time."""
    cols = ["week_start", "logged_days", "avg_sleep_min", "typical_bedtime",
            "bedtime_spread_min", "regularity", "study_min_per_day", "exercise_days"]
    m = daily_metrics(df)
    if m.empty:
        return pd.DataFrame(columns=cols)
    week_start = m.index - pd.to_timedelta(m.index.dayofweek, unit="D")
    rows = []
    for ws, g in m.groupby(week_start):
        nights = g["bedtime_min"].dropna()
        spread = circular_std_minutes(nights) if len(nights) >= 3 else float("nan")
        rows.append({
            "week_start": ws.date(),
            "logged_days": int(len(g)),
            "avg_sleep_min": _none_if_nan(g["sleep_min"].mean()),
            "typical_bedtime": format_clock(circular_mean_minutes(nights)) if len(nights) else "n/a",
            "bedtime_spread_min": _none_if_nan(spread),
            "regularity": regularity_label(spread),
            "study_min_per_day": float(g["study_min"].mean()),
            "exercise_days": int(g["exercise_day"].sum()),
        })
    return pd.DataFrame(rows, columns=cols)


def daily_timeline(df: pd.DataFrame, day) -> pd.DataFrame:
    """Chronological list of activities that overlap the given calendar day."""
    d = clean_activities(df)
    start = pd.Timestamp(parse_date(day))
    end = start + pd.Timedelta(days=1)
    sub = d[(d["start_dt"] < end) & (d["end_dt"] > start)].copy()
    if sub.empty:
        return sub
    sub["start_label"] = sub["start_dt"].dt.strftime("%H:%M")
    sub["end_label"] = sub["end_dt"].dt.strftime("%H:%M")
    return sub.sort_values("start_dt").reset_index(drop=True)


# ------------------------------------------------- helpers for recommendations
def bedtime_summary(df: pd.DataFrame, weekend: Optional[bool] = None, last_days: int = 28) -> dict:
    """Recent bedtime statistics, optionally for weekdays or weekends only."""
    empty = {"n": 0, "mean_min": float("nan"), "std_min": float("nan"), "mean_sleep_min": float("nan")}
    m = daily_metrics(df)
    if m.empty:
        return empty
    m = m[m.index > m.index.max() - pd.Timedelta(days=last_days)]
    if weekend is not None:
        m = m[m["weekend"] == weekend]
    nights = m["bedtime_min"].dropna()
    if nights.empty:
        return empty
    return {
        "n": int(len(nights)),
        "mean_min": circular_mean_minutes(nights),
        "std_min": circular_std_minutes(nights),
        "mean_sleep_min": float(m["sleep_min"].mean()) if m["sleep_min"].notna().any() else float("nan"),
    }


def typical_start_minute(df: pd.DataFrame, category: str, weekend: Optional[bool] = None,
                         last_days: int = 42, min_days: int = 3) -> float:
    """Usual clock time (minutes after midnight) at which a category first starts on a day."""
    m = daily_metrics(df, [category])
    if m.empty:
        return float("nan")
    m = m[m.index > m.index.max() - pd.Timedelta(days=last_days)]
    if weekend is not None:
        m = m[m["weekend"] == weekend]
    v = m[f"start_{category}"].dropna()
    return circular_mean_minutes(v) if len(v) >= min_days else float("nan")