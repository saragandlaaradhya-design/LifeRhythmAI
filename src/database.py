"""SQLite persistence for activities, suggestions and settings."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DB_DT_FORMAT, DB_PATH, SUGGESTION_STATUSES
from src.preprocessing import (
    ACTIVITY_COLUMNS,
    ValidationError,
    build_interval,
    parse_date,
    validate_category,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS activities (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    start_dt     TEXT NOT NULL,
    end_dt       TEXT NOT NULL,
    duration_min INTEGER NOT NULL CHECK (duration_min > 0),
    category     TEXT NOT NULL,
    notes        TEXT DEFAULT '',
    is_demo      INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_activities_start ON activities(start_dt);

CREATE TABLE IF NOT EXISTS suggestions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    suggestion_date TEXT NOT NULL,
    start_dt        TEXT NOT NULL,
    end_dt          TEXT NOT NULL,
    category        TEXT NOT NULL,
    title           TEXT NOT NULL,
    reason          TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','accepted','dismissed','completed')),
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class OverlapError(Exception):
    """Raised when a new/updated activity overlaps existing ones."""

    def __init__(self, conflicts: pd.DataFrame):
        self.conflicts = conflicts
        super().__init__(f"The activity overlaps {len(conflicts)} existing record(s).")


class ExistingDataError(Exception):
    """Raised when an operation would mix demo data into the user's real records."""


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Open a connection (creating the DB and tables if needed) and always close it."""
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Create the database file and tables (safe to call repeatedly)."""
    with get_connection(db_path):
        pass


def _fmt(dt: datetime) -> str:
    return dt.strftime(DB_DT_FORMAT)


def _now() -> str:
    return _fmt(datetime.now())


def _rows_to_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame([dict(r) for r in rows], columns=ACTIVITY_COLUMNS)
    if df.empty:
        return df.astype({"start_dt": "datetime64[ns]", "end_dt": "datetime64[ns]"})
    df["start_dt"] = pd.to_datetime(df["start_dt"])
    df["end_dt"] = pd.to_datetime(df["end_dt"])
    return df


def _find_conflicts(conn, start_dt, end_dt, exclude_id=None) -> pd.DataFrame:
    sql = "SELECT id, start_dt, end_dt, duration_min, category, notes, is_demo FROM activities WHERE start_dt < ? AND end_dt > ?"
    params = [_fmt(end_dt), _fmt(start_dt)]
    if exclude_id is not None:
        sql += " AND id != ?"
        params.append(int(exclude_id))
    return _rows_to_frame(conn.execute(sql + " ORDER BY start_dt", params).fetchall())


# ----------------------------------------------------------------- activities
def add_activity(
    date_value,
    start_time,
    category: str,
    end_time=None,
    duration_min: Optional[int] = None,
    notes: str = "",
    allow_overlap: bool = False,
    is_demo: bool = False,
    db_path: Optional[Path] = None,
) -> int:
    """Validate and insert an activity. Returns the new id.

    Raises ValidationError for bad input and OverlapError (with .conflicts)
    if it overlaps existing records and allow_overlap is False.
    """
    validate_category(category)
    interval = build_interval(date_value, start_time, end_time, duration_min)
    with get_connection(db_path) as conn:
        if not allow_overlap:
            conflicts = _find_conflicts(conn, interval["start_dt"], interval["end_dt"])
            if not conflicts.empty:
                raise OverlapError(conflicts)
        cur = conn.execute(
            "INSERT INTO activities (start_dt, end_dt, duration_min, category, notes, is_demo, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                _fmt(interval["start_dt"]),
                _fmt(interval["end_dt"]),
                interval["duration_min"],
                category,
                (notes or "").strip(),
                int(is_demo),
                _now(),
            ),
        )
        return int(cur.lastrowid)


def update_activity(
    activity_id: int,
    date_value,
    start_time,
    category: str,
    end_time=None,
    duration_min: Optional[int] = None,
    notes: str = "",
    allow_overlap: bool = False,
    db_path: Optional[Path] = None,
) -> None:
    validate_category(category)
    interval = build_interval(date_value, start_time, end_time, duration_min)
    with get_connection(db_path) as conn:
        exists = conn.execute("SELECT 1 FROM activities WHERE id = ?", (int(activity_id),)).fetchone()
        if not exists:
            raise ValidationError(f"Activity {activity_id} does not exist.")
        if not allow_overlap:
            conflicts = _find_conflicts(conn, interval["start_dt"], interval["end_dt"], exclude_id=activity_id)
            if not conflicts.empty:
                raise OverlapError(conflicts)
        conn.execute(
            "UPDATE activities SET start_dt=?, end_dt=?, duration_min=?, category=?, notes=? WHERE id=?",
            (
                _fmt(interval["start_dt"]),
                _fmt(interval["end_dt"]),
                interval["duration_min"],
                category,
                (notes or "").strip(),
                int(activity_id),
            ),
        )


def delete_activity(activity_id: int, db_path: Optional[Path] = None) -> bool:
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM activities WHERE id = ?", (int(activity_id),))
        return cur.rowcount > 0


def get_activities(start_date=None, end_date=None, db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return activities (optionally those starting within [start_date, end_date]) in chronological order."""
    sql = "SELECT id, start_dt, end_dt, duration_min, category, notes, is_demo FROM activities"
    clauses, params = [], []
    if start_date is not None:
        clauses.append("start_dt >= ?")
        params.append(_fmt(datetime.combine(parse_date(start_date), datetime.min.time())))
    if end_date is not None:
        nxt = datetime.combine(parse_date(end_date), datetime.min.time()) + timedelta(days=1)
        clauses.append("start_dt < ?")
        params.append(_fmt(nxt))
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY start_dt, id"
    with get_connection(db_path) as conn:
        return _rows_to_frame(conn.execute(sql, params).fetchall())


def count_activities(db_path: Optional[Path] = None, demo: Optional[bool] = None) -> int:
    sql = "SELECT COUNT(*) FROM activities"
    params = []
    if demo is not None:
        sql += " WHERE is_demo = ?"
        params.append(int(demo))
    with get_connection(db_path) as conn:
        return int(conn.execute(sql, params).fetchone()[0])


# ------------------------------------------------------------------ demo data
def insert_demo_data(
    demo_df: pd.DataFrame,
    confirm_with_existing: bool = False,
    db_path: Optional[Path] = None,
) -> int:
    """Insert synthetic records (tagged is_demo=1). Replaces older demo rows.

    If the user already has real records, raises ExistingDataError unless
    confirm_with_existing=True. Real records are never deleted here.
    """
    if count_activities(db_path, demo=False) > 0 and not confirm_with_existing:
        raise ExistingDataError(
            "You already have your own records. Loading demo data would mix them with synthetic data."
        )
    rows = [
        (
            _fmt(r.start_dt.to_pydatetime()),
            _fmt(r.end_dt.to_pydatetime()),
            int(r.duration_min),
            r.category,
            r.notes if isinstance(r.notes, str) else "",
            1,
            _now(),
        )
        for r in demo_df.itertuples(index=False)
    ]
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM activities WHERE is_demo = 1")
        conn.executemany(
            "INSERT INTO activities (start_dt, end_dt, duration_min, category, notes, is_demo, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def clear_demo_data(db_path: Optional[Path] = None) -> int:
    """Delete only synthetic records. Returns the number removed."""
    with get_connection(db_path) as conn:
        return conn.execute("DELETE FROM activities WHERE is_demo = 1").rowcount


def clear_all_data(db_path: Optional[Path] = None) -> int:
    """Delete every activity record (the UI must ask for confirmation first)."""
    with get_connection(db_path) as conn:
        return conn.execute("DELETE FROM activities").rowcount


# ---------------------------------------------------------------- suggestions
def add_suggestion(suggestion_date, start_dt, end_dt, category, title, reason,
                   db_path: Optional[Path] = None) -> int:
    validate_category(category)
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO suggestions (suggestion_date, start_dt, end_dt, category, title, reason, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
            (str(parse_date(suggestion_date)), _fmt(pd.Timestamp(start_dt).to_pydatetime()),
             _fmt(pd.Timestamp(end_dt).to_pydatetime()), category, title, reason, _now()),
        )
        return int(cur.lastrowid)


def get_suggestions(status: Optional[str] = None, db_path: Optional[Path] = None) -> pd.DataFrame:
    cols = ["id", "suggestion_date", "start_dt", "end_dt", "category", "title", "reason", "status", "created_at"]
    sql = f"SELECT {', '.join(cols)} FROM suggestions"
    params = []
    if status is not None:
        sql += " WHERE status = ?"
        params.append(status)
    sql += " ORDER BY start_dt, id"
    with get_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    df = pd.DataFrame([dict(r) for r in rows], columns=cols)
    if not df.empty:
        df["start_dt"] = pd.to_datetime(df["start_dt"])
        df["end_dt"] = pd.to_datetime(df["end_dt"])
    return df


def set_suggestion_status(suggestion_id: int, status: str, db_path: Optional[Path] = None) -> None:
    if status not in SUGGESTION_STATUSES:
        raise ValidationError(f"Status must be one of {SUGGESTION_STATUSES}.")
    with get_connection(db_path) as conn:
        cur = conn.execute("UPDATE suggestions SET status = ? WHERE id = ?", (status, int(suggestion_id)))
        if cur.rowcount == 0:
            raise ValidationError(f"Suggestion {suggestion_id} does not exist.")


def clear_pending_suggestions(db_path: Optional[Path] = None) -> int:
    with get_connection(db_path) as conn:
        return conn.execute("DELETE FROM suggestions WHERE status = 'pending'").rowcount


# ------------------------------------------------------------------- settings
def set_setting(key: str, value: str, db_path: Optional[Path] = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def get_setting(key: str, default: Optional[str] = None, db_path: Optional[Path] = None) -> Optional[str]:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default