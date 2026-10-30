"""Rule-based recommendation engine (deterministic rules, NOT machine learning).

Every suggestion is placed only inside a free time window: it never overlaps a
logged activity, a blocked window, an accepted suggestion, or another new
suggestion. Every suggestion carries a plain-language reason. Suggestions are
built from the user's OWN goals and OWN history, not from a universal ideal routine.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src import database as db
from src.analytics import bedtime_summary, format_clock, typical_start_minute
from src.drift_detection import PERSISTENT, DriftConfig, DriftResult, analyze_drift
from src.preprocessing import ValidationError, clean_activities, parse_date, parse_time

GOALS = ("study", "exercise", "sleep")
MIN_DAYS_FOR_PATTERNS = 7
MIN_SESSION_MIN = 20
MAX_SESSIONS_PER_GOAL = 2
BEDTIME_VARIABILITY_MIN = 45.0
DEFAULT_PREFERRED_START = {"Studying": 16 * 60, "Exercise": 17 * 60 + 30, "Personal tasks": 19 * 60}
RULE_LABELS = {
    "bedtime_shift": "bedtime",
    "study_decrease": "study time",
    "exercise_decrease": "exercise frequency",
    "timing_shift": "activity timing",
    "irregular_bedtime": "bedtime regularity",
}


@dataclass
class Preferences:
    """Goals and constraints entered by the user."""

    goals: List[str] = field(default_factory=list)
    study_goal_min_per_day: int = 90
    study_session_min: int = 45
    exercise_goal_per_week: int = 3
    exercise_session_min: int = 30
    available_start: str = "07:00"
    available_end: str = "22:00"
    blocked_windows: List[List[str]] = field(default_factory=list)   # [["12:00", "13:00"], ...]
    max_suggestions: int = 5

    def validate(self) -> "Preferences":
        bad = [g for g in self.goals if g not in GOALS]
        if bad:
            raise ValidationError(f"Unknown goal(s) {bad}. Choose from {list(GOALS)}.")
        limits = (
            ("study_goal_min_per_day", 15, 600), ("study_session_min", 15, 240),
            ("exercise_goal_per_week", 1, 14), ("exercise_session_min", 10, 180),
            ("max_suggestions", 1, 10),
        )
        for name, lo, hi in limits:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                raise ValidationError(f"{name} must be a whole number between {lo} and {hi}.")
        start, end = parse_time(self.available_start), parse_time(self.available_end)
        if end <= start:
            raise ValidationError("The available-time window must end after it starts.")
        for window in self.blocked_windows:
            if len(window) != 2:
                raise ValidationError("Each blocked window needs a start and an end time.")
            if parse_time(window[1]) <= parse_time(window[0]):
                raise ValidationError(f"Blocked window {window[0]}-{window[1]} must end after it starts.")
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "Preferences":
        try:
            data = json.loads(text)
            allowed = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in allowed}).validate()
        except (ValueError, TypeError, ValidationError, AttributeError):
            return cls()


def save_preferences(prefs: Preferences, db_path=None) -> None:
    prefs.validate()
    db.set_setting("preferences", prefs.to_json(), db_path=db_path)


def load_preferences(db_path=None) -> Preferences:
    text = db.get_setting("preferences", None, db_path=db_path)
    return Preferences() if text is None else Preferences.from_json(text)


# ------------------------------------------------------------ calendar
def _py(x) -> datetime:
    return pd.Timestamp(x).to_pydatetime()


def _ceil5(dt: datetime) -> datetime:
    if dt.second or dt.microsecond:
        dt += timedelta(minutes=1)
    dt = dt.replace(second=0, microsecond=0)
    return dt + timedelta(minutes=(-dt.minute) % 5)


class _Calendar:
    """Busy intervals for one planning day. Used to guarantee no overlaps."""

    def __init__(self):
        self.items: List[Tuple[datetime, datetime]] = []

    def add(self, start, end) -> None:
        s, e = _py(start), _py(end)
        if e > s:
            self.items.append((s, e))

    def conflicts(self, start: datetime, end: datetime) -> bool:
        return any(start < e and end > s for s, e in self.items)

    def free_windows(self, start: datetime, end: datetime, min_minutes: int = 15):
        busy = sorted((max(s, start), min(e, end)) for s, e in self.items if e > start and s < end)
        free, cursor = [], start
        for s, e in busy:
            if s > cursor:
                free.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < end:
            free.append((cursor, end))
        need = timedelta(minutes=min_minutes)
        return [(s, e) for s, e in free if e - s >= need]


def _build_calendar(hist: pd.DataFrame, day, prefs: Preferences, suggestions: Optional[pd.DataFrame]) -> _Calendar:
    cal = _Calendar()
    lo = datetime.combine(day, datetime.min.time()) - timedelta(days=1)
    hi = lo + timedelta(days=4)
    if not hist.empty:
        for s, e in zip(hist["start_dt"], hist["end_dt"]):
            if e > lo and s < hi:
                cal.add(s, e)
    for bs, be in prefs.blocked_windows:
        cal.add(datetime.combine(day, parse_time(bs)), datetime.combine(day, parse_time(be)))
    if suggestions is not None and not suggestions.empty:
        for _, r in suggestions[suggestions["status"] == "accepted"].iterrows():
            if r["end_dt"] > lo and r["start_dt"] < hi:
                cal.add(r["start_dt"], r["end_dt"])
    return cal


def _day_window(day, prefs: Preferences, now=None):
    start = datetime.combine(day, parse_time(prefs.available_start))
    end = datetime.combine(day, parse_time(prefs.available_end))
    if now is not None:
        now = _py(now)
        if now.date() > day:
            return start, start
        if now.date() == day:
            start = max(start, _ceil5(now))
    return start, end


def free_windows(history, target_date, prefs: Optional[Preferences] = None,
                 suggestions: Optional[pd.DataFrame] = None, now=None, min_minutes: int = 15):
    """List of (start, end) free windows on the target day (public helper, also used by the UI)."""
    prefs = (prefs or Preferences()).validate()
    day = parse_date(target_date)
    start, end = _day_window(day, prefs, now)
    if end <= start:
        return []
    cal = _build_calendar(clean_activities(history), day, prefs, suggestions)
    return cal.free_windows(start, end, min_minutes)


def _place(cal: _Calendar, win_start, win_end, pref_min: float, duration: int, day):
    """Best slot of `duration` minutes closest to the preferred clock time, or None."""
    midnight = datetime.combine(day, datetime.min.time())
    pref_dt = midnight + timedelta(minutes=5 * round(float(pref_min) / 5))
    dur = timedelta(minutes=duration)
    best = None
    for ws, we in cal.free_windows(win_start, win_end, min_minutes=duration):
        s = min(max(pref_dt, ws), we - dur)
        dist = abs((s - pref_dt).total_seconds())
        if best is None or dist < best[0]:
            best = (dist, s, s + dur)
    return None if best is None else (best[1], best[2])


def _place_flexible(cal, win_start, win_end, pref_min, duration, day, min_duration=MIN_SESSION_MIN):
    """Try the full length first, then shorter sessions. Returns ((start, end), minutes) or (None, None)."""
    for dur in dict.fromkeys([duration, max(min_duration, duration * 2 // 3), min_duration]):
        if dur > duration:
            continue
        slot = _place(cal, win_start, win_end, pref_min, int(dur), day)
        if slot is not None:
            return slot, int(dur)
    return None, None


def _range(s: datetime, e: datetime) -> str:
    return f"{s:%H:%M}-{e:%H:%M}"


# ------------------------------------------------------------ main generator
def generate_recommendations(history, target_date, prefs: Optional[Preferences] = None,
                             drift: Optional[DriftResult] = None,
                             suggestions: Optional[pd.DataFrame] = None, now=None):
    """Return (recommendations, notes) for one day.

    recommendations: list of dicts with category, title, start_dt, end_dt, reason, source, priority.
    notes: plain-text messages explaining anything that was skipped.
    `suggestions` is the table of earlier suggestions (any status): accepted ones block
    time, and categories dismissed for this date are not suggested again.
    """
    prefs = (prefs or Preferences()).validate()
    day = parse_date(target_date)
    hist = clean_activities(history)
    notes: List[str] = []
    found: List[dict] = []

    if now is not None and _py(now).date() > day:
        return [], ["The chosen date is in the past, so no suggestions were made."]

    win_start, win_end = _day_window(day, prefs, now)
    cal = _build_calendar(hist, day, prefs, suggestions)
    weekend = day.weekday() >= 5
    scope = "weekends" if weekend else "weekdays"
    logged_days = int(hist["start_dt"].dt.normalize().nunique()) if not hist.empty else 0
    have_patterns = logged_days >= MIN_DAYS_FOR_PATTERNS
    if not have_patterns:
        notes.append(f"Only {logged_days} day(s) of history are logged (at least {MIN_DAYS_FOR_PATTERNS} are needed "
                     "to learn your usual times), so suggestions rely on your stated goals and default times.")

    if drift is not None and not drift.sufficient:
        notes.append(f"Routine-change analysis was not used: {drift.message}")
    findings = drift.findings if drift is not None and drift.sufficient else []

    dismissed = set()
    if suggestions is not None and not suggestions.empty:
        dm = suggestions[(suggestions["status"] == "dismissed") & (suggestions["suggestion_date"].astype(str) == str(day))]
        dismissed = set(dm["category"])

    def add(category, title, slot, reason, source, priority):
        cal.add(slot[0], slot[1])
        found.append({"category": category, "title": title, "start_dt": slot[0], "end_dt": slot[1],
                      "reason": reason, "source": source, "priority": priority})

    # ---- 1. bedtime consistency
    bed_findings = [f for f in findings if f.rule in ("bedtime_shift", "irregular_bedtime") and f.scope == scope]
    stats = bedtime_summary(hist, weekend=weekend) if have_patterns else None
    wants_sleep = "sleep" in prefs.goals
    need_bed = False
    if stats is not None and "Sleeping" not in dismissed and stats["n"] >= 4:
        steady = not np.isnan(stats["std_min"]) and stats["std_min"] >= BEDTIME_VARIABILITY_MIN
        need_bed = bool(bed_findings) or (wants_sleep and steady)
    if need_bed:
        shift = next((f for f in bed_findings if f.rule == "bedtime_shift"), None)
        target_min = shift.baseline if shift is not None else stats["mean_min"]
        target_min = int(5 * round(target_min / 5)) % 1440
        sleep_len = stats["mean_sleep_min"]
        sleep_len = 450 if np.isnan(sleep_len) else int(5 * round(min(max(sleep_len, 300), 600) / 5))
        bed_start = datetime.combine(day, datetime.min.time()) + timedelta(minutes=target_min)
        if target_min < 12 * 60:
            bed_start += timedelta(days=1)
        bed_end = bed_start + timedelta(minutes=sleep_len)
        clock = format_clock(target_min)
        if now is not None and bed_start < _py(now):
            pass  # that bedtime has already passed
        elif cal.conflicts(bed_start, bed_end):
            notes.append("A sleep period is already logged or planned for that night, so no bedtime suggestion was made.")
        else:
            if shift is not None:
                reason = (f"On {scope} your typical bedtime moved from about {shift.baseline_text} to about "
                          f"{shift.current_text} ({shift.change_text}). Aiming for your earlier time of about "
                          f"{shift.baseline_text} would restore your previous pattern. Ignore this if your new "
                          "schedule works for you.")
            else:
                reason = (f"Your recent {scope} bedtimes vary by about +/-{stats['std_min']:.0f} min. Choosing one "
                          f"target time ({clock}) is an option if you want a more predictable evening"
                          + (", and you chose a regular-sleep goal." if wants_sleep else "."))
            add("Sleeping", f"Aim for a bedtime near {clock}", (bed_start, bed_end), reason, "drift" if shift else "goal", 1)

    # ---- 2. study sessions
    if "study" in prefs.goals and "Studying" not in dismissed:
        logged = int(hist.loc[(hist["category"] == "Studying") &
                              (hist["start_dt"].dt.normalize() == pd.Timestamp(day)), "duration_min"].sum())
        remaining = prefs.study_goal_min_per_day - logged
        if remaining < MIN_SESSION_MIN:
            notes.append("Your study goal for that day is already met by logged activity.")
        else:
            typical = typical_start_minute(hist, "Studying", weekend) if have_patterns else float("nan")
            pref = DEFAULT_PREFERRED_START["Studying"] if np.isnan(typical) else typical
            drop = next((f for f in findings if f.rule == "study_decrease" and f.scope == scope), None)
            for _ in range(MAX_SESSIONS_PER_GOAL):
                want = min(prefs.study_session_min, remaining)
                if want < MIN_SESSION_MIN:
                    break
                slot, dur = _place_flexible(cal, win_start, win_end, pref, int(want), day)
                if slot is None:
                    notes.append("No free window was long enough for another study session.")
                    break
                parts = [f"Your study goal is {prefs.study_goal_min_per_day} min/day and {logged} min are logged for {day}.",
                         f"This slot ({_range(*slot)}) is free in your schedule"
                         + (f" and close to when you usually study (about {format_clock(typical)})." if not np.isnan(typical) else ".")]
                if drop is not None:
                    parts.append(f"Your logged study time on {scope} is also below your earlier baseline ({drop.change_text}).")
                add("Studying", f"Study session ({dur} min)", slot, " ".join(parts), "goal", 2)
                remaining -= dur

    # ---- 3. exercise session
    if "exercise" in prefs.goals and "Exercise" not in dismissed:
        ex = hist[hist["category"] == "Exercise"]
        monday = day - timedelta(days=day.weekday())
        day_ts, next_ts = pd.Timestamp(day), pd.Timestamp(day + timedelta(days=1))
        week_count = int(((ex["start_dt"] >= pd.Timestamp(monday)) & (ex["start_dt"] < next_ts)).sum())
        today_count = int(((ex["start_dt"] >= day_ts) & (ex["start_dt"] < next_ts)).sum())
        yest_count = int(((ex["start_dt"] >= day_ts - pd.Timedelta(days=1)) & (ex["start_dt"] < day_ts)).sum())
        remaining = prefs.exercise_goal_per_week - week_count
        days_left = 7 - day.weekday()
        if today_count > 0:
            notes.append("You already logged exercise for that day.")
        elif remaining <= 0:
            notes.append("Your weekly exercise goal is already met.")
        elif yest_count > 0 and remaining <= days_left - 1:
            notes.append("You exercised the day before and can still reach your weekly goal, so a rest day was kept.")
        else:
            typical = typical_start_minute(hist, "Exercise", weekend) if have_patterns else float("nan")
            pref = DEFAULT_PREFERRED_START["Exercise"] if np.isnan(typical) else typical
            slot, dur = _place_flexible(cal, win_start, win_end, pref, prefs.exercise_session_min, day, min_duration=15)
            if slot is None:
                notes.append("No free window was long enough for an exercise session.")
            else:
                drop = next((f for f in findings if f.rule == "exercise_decrease" and f.scope == scope), None)
                parts = [f"Your goal is {prefs.exercise_goal_per_week} exercise sessions per week; {week_count} logged so far "
                         f"this week ({remaining} to go, {days_left} day(s) left including this one).",
                         f"This slot ({_range(*slot)}) is free"
                         + (f" and near your usual exercise time (about {format_clock(typical)})." if not np.isnan(typical) else ".")]
                if drop is not None:
                    parts.append(f"Your exercise on {scope} fell from {drop.baseline_text} to {drop.current_text}.")
                add("Exercise", f"Exercise session ({dur} min)", slot, " ".join(parts), "goal", 3)

    # ---- 4. planning block after repeated disruptions
    if findings and "Personal tasks" not in dismissed:
        persistent_rules = sorted({f.rule for f in findings if f.persistence == PERSISTENT})
        n_unusual = len(drift.unusual_days)
        if len(persistent_rules) >= 2 or n_unusual >= 3:
            slot, dur = _place_flexible(cal, win_start, win_end, DEFAULT_PREFERRED_START["Personal tasks"], 15, day, min_duration=15)
            if slot is not None:
                names = ", ".join(RULE_LABELS.get(r, r) for r in persistent_rules) or "several days"
                reason = (f"Several parts of your recent routine changed ({names}; {n_unusual} unusual day(s) logged). "
                          "A short planning block lets you decide whether to adjust your schedule or keep the new pattern.")
                add("Personal tasks", "Plan your week (15 min)", slot, reason, "drift", 4)

    found.sort(key=lambda r: r["priority"])
    found = found[: prefs.max_suggestions]
    found.sort(key=lambda r: r["start_dt"])
    return found, notes


def generate_and_store(target_date, prefs: Optional[Preferences] = None, db_path=None,
                       now=None, drift_config: Optional[DriftConfig] = None):
    """Read history from the database, generate suggestions, and replace the pending ones."""
    history = db.get_activities(db_path=db_path)
    prefs = prefs or load_preferences(db_path)
    existing = db.get_suggestions(db_path=db_path)
    drift = analyze_drift(history, drift_config)
    recs, notes = generate_recommendations(history, target_date, prefs, drift, existing, now)
    db.clear_pending_suggestions(db_path)
    for r in recs:
        r["id"] = db.add_suggestion(parse_date(target_date), r["start_dt"], r["end_dt"],
                                    r["category"], r["title"], r["reason"], db_path=db_path)
    return recs, notes