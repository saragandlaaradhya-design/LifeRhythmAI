from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

import src.analytics

PERSISTENT = "persistent change"
EMERGING = "emerging (later half only)"
EARLIER_ONLY = "earlier half only (may be fading)"
MIXED = "mixed (not clearly lasting)"
NOT_ENOUGH = "too few days to judge persistence"
WHOLE_PERIOD = "whole-period measure"

PERSIST_TEXT = {
    PERSISTENT: "The change appears in both the earlier and the later half of the current period, so it looks lasting rather than a single unusual day.",
    EMERGING: "It shows up only in the later half of the current period, so it may be just starting. Keep watching.",
    EARLIER_ONLY: "It was present earlier in the current period but not recently, so it may be fading.",
    MIXED: "Neither half of the current period shows it clearly, so the overall difference may come from a few unusual days.",
    NOT_ENOUGH: "There are too few days in this group to judge whether it is lasting.",
    WHOLE_PERIOD: "This compares the spread over the whole period, so it cannot be split into single days.",
}

DISCLAIMER = (
    "These flags describe changes in the activities you logged. They are not a diagnosis, "
    "and a change is not automatically a problem. Missing logs can also look like a change."
)

SCOPES = (("weekdays", False), ("weekends", True))
UNUSUAL_COLUMNS = ["date", "scope", "metric", "value", "typical", "difference_min", "note"]
COMPARISON_COLUMNS = ["metric", "scope", "baseline", "current", "change"]


@dataclass
class DriftConfig:
    """All thresholds in one place, so they can be changed from the UI."""

    current_days: int = 14               # length of the recent window
    baseline_days: int = 28              # length of the baseline window just before it
    min_baseline_days: int = 10          # logged days needed in the baseline window
    min_current_days: int = 5            # logged days needed in the current window
    min_scope_days_baseline: int = 4     # logged weekdays (or weekend days) needed in baseline
    min_scope_days_current: int = 3      # ... and in the current window
    bedtime_shift_min: float = 45.0      # flag if typical bedtime moves by this many minutes
    study_drop_pct: float = 25.0         # flag if average study time falls by this percent
    exercise_drop_pct: float = 30.0      # flag if the share of days with exercise falls by this percent
    timing_shift_min: float = 60.0       # flag if an activity's usual start moves by this many minutes
    irregularity_increase_pct: float = 30.0   # flag if bedtime spread grows by this percent ...
    irregularity_min_abs_min: float = 15.0    # ... and by at least this many minutes
    single_day_shift_min: float = 120.0  # a single day this far from typical is listed as unusual
    min_baseline_study_min: float = 20.0       # ignore study rule if baseline study is below this
    min_baseline_exercise_ratio: float = 0.15  # ignore exercise rule if baseline exercise is rarer than this
    timing_categories: tuple = ("Studying", "Exercise", "Attending classes")

    def validate(self) -> "DriftConfig":
        for name in (
            "current_days", "baseline_days", "min_baseline_days", "min_current_days",
            "min_scope_days_baseline", "min_scope_days_current", "bedtime_shift_min",
            "study_drop_pct", "exercise_drop_pct", "timing_shift_min",
            "irregularity_increase_pct", "irregularity_min_abs_min", "single_day_shift_min",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero.")
        for name in ("study_drop_pct", "exercise_drop_pct"):
            if getattr(self, name) > 100:
                raise ValueError(f"{name} cannot be more than 100.")
        if self.min_current_days > self.current_days:
            raise ValueError("min_current_days cannot be larger than current_days.")
        if self.min_baseline_days > self.baseline_days:
            raise ValueError("min_baseline_days cannot be larger than baseline_days.")
        return self


@dataclass
class DriftFinding:
    rule: str            # bedtime_shift | study_decrease | exercise_decrease | timing_shift | irregular_bedtime
    title: str
    scope: str           # weekdays | weekends
    severity: str        # mild | moderate | notable
    persistence: str
    baseline: float      # numeric baseline value (clock minutes for time-based rules)
    current: float
    ratio: float         # observed change divided by its threshold
    baseline_text: str
    current_text: str
    change_text: str
    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftResult:
    sufficient: bool
    message: str
    config: DriftConfig
    baseline_period: Optional[tuple] = None
    current_period: Optional[tuple] = None
    n_baseline_days: int = 0
    n_current_days: int = 0
    findings: List[DriftFinding] = field(default_factory=list)
    unusual_days: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=UNUSUAL_COLUMNS))
    comparison: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=COMPARISON_COLUMNS))
    notes: List[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


# ------------------------------------------------------------ statistics
def _bed_mean(f: pd.DataFrame) -> float:
    v = f["bedtime_min"].dropna()
    return src.analytics.circular_mean_minutes(v) if len(v) >= 2 else float("nan")


def _bed_std(f: pd.DataFrame) -> float:
    v = f["bedtime_min"].dropna()
    return src.analytics.circular_std_minutes(v) if len(v) >= 3 else float("nan")


def _study_mean(f: pd.DataFrame) -> float:
    return float(f["study_min"].mean()) if len(f) >= 2 else float("nan")


def _exercise_ratio(f: pd.DataFrame) -> float:
    return float(f["exercise_day"].mean()) if len(f) >= 2 else float("nan")


def _timing_mean(col: str):
    def fn(f: pd.DataFrame) -> float:
        v = f[col].dropna()
        return src.analytics.circular_mean_minutes(v) if len(v) >= 2 else float("nan")
    return fn


def _persistence(cur_scope: pd.DataFrame, stat_fn, deviates) -> str:
    """Does the deviation show up in BOTH halves of the current period?"""
    half = len(cur_scope) // 2
    if half < 2:
        return NOT_ENOUGH
    first, second = stat_fn(cur_scope.iloc[:half]), stat_fn(cur_scope.iloc[half:])
    if pd.isna(first) or pd.isna(second):
        return NOT_ENOUGH
    a, b = deviates(first), deviates(second)
    if a and b:
        return PERSISTENT
    if b:
        return EMERGING
    if a:
        return EARLIER_ONLY
    return MIXED


def _severity(ratio: float, persistence: str) -> str:
    if persistence == PERSISTENT:
        return "notable" if ratio >= 1.5 else "moderate"
    if persistence == WHOLE_PERIOD:
        return "moderate" if ratio >= 1.5 else "mild"
    return "mild"


def _finding(rule, title, scope, ratio, persistence, baseline, current,
             baseline_text, current_text, change_text, what) -> DriftFinding:
    return DriftFinding(
        rule=rule, title=title, scope=scope, severity=_severity(ratio, persistence),
        persistence=persistence, baseline=float(baseline), current=float(current), ratio=float(ratio),
        baseline_text=baseline_text, current_text=current_text, change_text=change_text,
        explanation=f"{what} {PERSIST_TEXT[persistence]}",
    )


# ------------------------------------------------------------ rules
def _evaluate_scope(base_s: pd.DataFrame, cur_s: pd.DataFrame, scope: str, cfg: DriftConfig) -> List[DriftFinding]:
    """Apply every rule to one scope (weekdays only, or weekends only)."""
    out: List[DriftFinding] = []
    if len(base_s) < cfg.min_scope_days_baseline or len(cur_s) < cfg.min_scope_days_current:
        return out

    # Rule 1: bedtime moved
    if base_s["bedtime_min"].notna().sum() >= cfg.min_scope_days_baseline and \
            cur_s["bedtime_min"].notna().sum() >= cfg.min_scope_days_current:
        b, c = _bed_mean(base_s), _bed_mean(cur_s)
        diff = src.analytics.circular_diff_minutes(c, b)
        if abs(diff) >= cfg.bedtime_shift_min:
            sign = 1 if diff > 0 else -1

            def dev_bed(s, b=b, sign=sign):
                return (not pd.isna(s)) and src.analytics.circular_diff_minutes(s, b) * sign >= cfg.bedtime_shift_min

            pers = _persistence(cur_s, _bed_mean, dev_bed)
            word = "later" if diff > 0 else "earlier"
            what = (f"On {scope}, you typically went to sleep around {src.analytics.format_clock(b)} in the baseline period "
                    f"but around {src.analytics.format_clock(c)} in the current period, which is {abs(diff):.0f} min {word} "
                    f"(threshold: {cfg.bedtime_shift_min:.0f} min).")
            out.append(_finding("bedtime_shift", f"Bedtime moved {word}", scope, abs(diff) / cfg.bedtime_shift_min,
                                pers, b, c, src.analytics.format_clock(b), src.analytics.format_clock(c), f"{abs(diff):.0f} min {word}", what))

    # Rule 2: study time fell
    bm, cm = _study_mean(base_s), _study_mean(cur_s)
    if not pd.isna(bm) and not pd.isna(cm) and bm >= cfg.min_baseline_study_min:
        drop = (bm - cm) / bm * 100.0
        if drop >= cfg.study_drop_pct:
            def dev_study(s, bm=bm):
                return (not pd.isna(s)) and (bm - s) / bm * 100.0 >= cfg.study_drop_pct

            pers = _persistence(cur_s, _study_mean, dev_study)
            what = (f"On {scope}, your logged study time averaged {bm:.0f} min per logged day in the baseline period "
                    f"and {cm:.0f} min in the current period, a drop of {drop:.0f}% (threshold: {cfg.study_drop_pct:.0f}%).")
            out.append(_finding("study_decrease", "Study time decreased", scope, drop / cfg.study_drop_pct,
                                pers, bm, cm, f"{bm:.0f} min/day", f"{cm:.0f} min/day", f"-{drop:.0f}%", what))

    # Rule 3: exercise became less frequent
    br, cr = _exercise_ratio(base_s), _exercise_ratio(cur_s)
    if not pd.isna(br) and not pd.isna(cr) and br >= cfg.min_baseline_exercise_ratio:
        drop = (br - cr) / br * 100.0
        if drop >= cfg.exercise_drop_pct:
            def dev_ex(s, br=br):
                return (not pd.isna(s)) and (br - s) / br * 100.0 >= cfg.exercise_drop_pct

            pers = _persistence(cur_s, _exercise_ratio, dev_ex)
            what = (f"On {scope}, you logged exercise on {br * 100:.0f}% of logged days in the baseline period "
                    f"but on {cr * 100:.0f}% in the current period, a relative drop of {drop:.0f}% "
                    f"(threshold: {cfg.exercise_drop_pct:.0f}%).")
            out.append(_finding("exercise_decrease", "Exercise became less frequent", scope,
                                drop / cfg.exercise_drop_pct, pers, br, cr,
                                f"{br * 100:.0f}% of days", f"{cr * 100:.0f}% of days", f"-{drop:.0f}%", what))

    # Rule 4: major change in the usual start time of an activity
    for cat in cfg.timing_categories:
        col = f"start_{cat}"
        if col not in base_s.columns or col not in cur_s.columns:
            continue
        if base_s[col].notna().sum() < 4 or cur_s[col].notna().sum() < 3:
            continue
        fn = _timing_mean(col)
        b, c = fn(base_s), fn(cur_s)
        if pd.isna(b) or pd.isna(c):
            continue
        diff = src.analytics.circular_diff_minutes(c, b)
        if abs(diff) >= cfg.timing_shift_min:
            sign = 1 if diff > 0 else -1

            def dev_t(s, b=b, sign=sign):
                return (not pd.isna(s)) and src.analytics.circular_diff_minutes(s, b) * sign >= cfg.timing_shift_min

            pers = _persistence(cur_s, fn, dev_t)
            word = "later" if diff > 0 else "earlier"
            what = (f"On {scope}, '{cat}' usually started around {src.analytics.format_clock(b)} in the baseline period "
                    f"and around {src.analytics.format_clock(c)} in the current period ({abs(diff):.0f} min {word}; "
                    f"threshold: {cfg.timing_shift_min:.0f} min).")
            out.append(_finding("timing_shift", f"{cat} now starts {word}", scope, abs(diff) / cfg.timing_shift_min,
                                pers, b, c, src.analytics.format_clock(b), src.analytics.format_clock(c), f"{abs(diff):.0f} min {word}", what))

    # Rule 5: bedtime became less regular
    bs, cs = _bed_std(base_s), _bed_std(cur_s)
    if not pd.isna(bs) and not pd.isna(cs) and base_s["bedtime_min"].notna().sum() >= cfg.min_scope_days_baseline:
        base_eff = max(bs, 5.0)
        if cs >= base_eff * (1 + cfg.irregularity_increase_pct / 100.0) and (cs - bs) >= cfg.irregularity_min_abs_min:
            ratio = ((cs / base_eff) - 1.0) / (cfg.irregularity_increase_pct / 100.0)
            pct = (cs / base_eff - 1.0) * 100.0
            what = (f"On {scope}, your bedtime varied by about +/-{bs:.0f} min in the baseline period and "
                    f"+/-{cs:.0f} min in the current period ({pct:.0f}% more spread; threshold: "
                    f"{cfg.irregularity_increase_pct:.0f}% and at least {cfg.irregularity_min_abs_min:.0f} min more).")
            out.append(_finding("irregular_bedtime", "Bedtime became less regular", scope, ratio, WHOLE_PERIOD,
                                bs, cs, f"+/-{bs:.0f} min", f"+/-{cs:.0f} min", f"+{cs - bs:.0f} min spread", what))
    return out


def _unusual_days(base: pd.DataFrame, cur: pd.DataFrame, cfg: DriftConfig, findings: List[DriftFinding]) -> pd.DataFrame:
    """Single days far from the typical bedtime or sleep length (not necessarily a routine change)."""
    rows = []
    for scope, wk in SCOPES:
        b, c = base[base["weekend"] == wk], cur[cur["weekend"] == wk]
        if b["bedtime_min"].notna().sum() < cfg.min_scope_days_baseline:
            continue
        bm = _bed_mean(b)
        sm = b["sleep_min"].mean()
        persistent_bed = any(f.rule == "bedtime_shift" and f.scope == scope and f.persistence == PERSISTENT
                             for f in findings)
        for day, row in c.iterrows():
            if not pd.isna(row["bedtime_min"]) and not pd.isna(bm):
                dd = src.analytics.circular_diff_minutes(row["bedtime_min"], bm)
                if abs(dd) >= cfg.single_day_shift_min:
                    note = ("Part of the wider bedtime shift reported above." if persistent_bed else
                            "Single unusual day. One day on its own is not treated as a routine change.")
                    rows.append({"date": day.date(), "scope": scope, "metric": "Bedtime",
                                 "value": src.analytics.format_clock(row["bedtime_min"]), "typical": src.analytics.format_clock(bm),
                                 "difference_min": float(dd), "note": note})
            if not pd.isna(row["sleep_min"]) and not pd.isna(sm):
                ds = float(row["sleep_min"] - sm)
                if abs(ds) >= cfg.single_day_shift_min:
                    rows.append({"date": day.date(), "scope": scope, "metric": "Sleep length",
                                 "value": f"{row['sleep_min']:.0f} min", "typical": f"{sm:.0f} min",
                                 "difference_min": ds,
                                 "note": "Single unusual day. One day on its own is not treated as a routine change."})
    return pd.DataFrame(rows, columns=UNUSUAL_COLUMNS)


def _num(x, nd: int = 0) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def _delta(a, b, suffix: str = "") -> str:
    return "n/a" if pd.isna(a) or pd.isna(b) else f"{b - a:+.0f}{suffix}"


def _comparison_rows(b: pd.DataFrame, c: pd.DataFrame, scope: str) -> list:
    rows = []
    if b.empty or c.empty:
        return rows

    def add(metric, bt, ct, change):
        rows.append({"metric": metric, "scope": scope, "baseline": bt, "current": ct, "change": change})

    bb, cc = _bed_mean(b), _bed_mean(c)
    add("Typical bedtime", src.analytics.format_clock(bb), src.analytics.format_clock(cc),
        "n/a" if pd.isna(bb) or pd.isna(cc) else f"{src.analytics.circular_diff_minutes(cc, bb):+.0f} min")
    bs, cs = _bed_std(b), _bed_std(c)
    add("Bedtime spread (+/- min)", _num(bs), _num(cs), _delta(bs, cs, " min"))
    sb, sc = b["sleep_min"].mean(), c["sleep_min"].mean()
    add("Average sleep (min)", _num(sb), _num(sc), _delta(sb, sc, " min"))
    stb, stc = _study_mean(b), _study_mean(c)
    add("Study (min per logged day)", _num(stb), _num(stc), _delta(stb, stc, " min"))
    eb, ec = _exercise_ratio(b), _exercise_ratio(c)
    add("Days with exercise (%)", _num(eb * 100 if not pd.isna(eb) else eb),
        _num(ec * 100 if not pd.isna(ec) else ec),
        "n/a" if pd.isna(eb) or pd.isna(ec) else f"{(ec - eb) * 100:+.0f} points")
    return rows


# ------------------------------------------------------------ main entry point
def _split_windows(m: pd.DataFrame, cfg: DriftConfig):
    last = m.index.max()
    cur_start = last - pd.Timedelta(days=cfg.current_days - 1)
    base_end = cur_start - pd.Timedelta(days=1)
    base_start = base_end - pd.Timedelta(days=cfg.baseline_days - 1)
    cur = m[(m.index >= cur_start) & (m.index <= last)]
    base = m[(m.index >= base_start) & (m.index <= base_end)]
    return base, cur, (base_start.date(), base_end.date()), (cur_start.date(), last.date())


def analyze_drift(df: pd.DataFrame, config: Optional[DriftConfig] = None) -> DriftResult:
    """Compare the latest `current_days` with the `baseline_days` just before them.

    Never raises for missing data: returns a result with sufficient=False and a message.
    """
    cfg = (config or DriftConfig()).validate()
    m = src.analytics.daily_metrics(df, cfg.timing_categories)
    if m.empty:
        return DriftResult(False, "There are no valid activity records yet, so no routine can be compared.", cfg)

    base, cur, bp, cp = _split_windows(m, cfg)
    if len(base) < cfg.min_baseline_days or len(cur) < cfg.min_current_days:
        msg = (f"Not enough logged days to compare routines: found {len(base)} baseline day(s) "
               f"(need {cfg.min_baseline_days}) and {len(cur)} current day(s) (need {cfg.min_current_days}). "
               f"The baseline window is the {cfg.baseline_days} days before the latest {cfg.current_days} days.")
        return DriftResult(False, msg, cfg, bp, cp, len(base), len(cur))

    findings: List[DriftFinding] = []
    comparison: list = []
    notes: List[str] = []
    for scope, wk in SCOPES:
        b, c = base[base["weekend"] == wk], cur[cur["weekend"] == wk]
        if len(b) < cfg.min_scope_days_baseline or len(c) < cfg.min_scope_days_current:
            notes.append(f"{scope.capitalize()}: too few logged days "
                         f"({len(b)} baseline, {len(c)} current) for a fair comparison, so they were skipped.")
            continue
        findings += _evaluate_scope(b, c, scope, cfg)
        comparison += _comparison_rows(b, c, scope)

    order = {"notable": 0, "moderate": 1, "mild": 2}
    findings.sort(key=lambda f: (order[f.severity], -f.ratio))
    unusual = _unusual_days(base, cur, cfg, findings)

    msg = (f"Compared the latest {len(cur)} logged day(s) ({cp[0]} to {cp[1]}) with {len(base)} baseline "
           f"logged day(s) ({bp[0]} to {bp[1]}). {len(findings)} change(s) flagged.")
    return DriftResult(True, msg, cfg, bp, cp, len(base), len(cur), findings, unusual,
                       pd.DataFrame(comparison, columns=COMPARISON_COLUMNS), notes)


if __name__ == "__main__":
    # Quick manual check on SYNTHETIC data:  python -m src.drift_detection
    from LifeRhythmAI.src.data_generator import generate_demo_data

    result = analyze_drift(generate_demo_data())
    print("(synthetic demo data)")
    print(result.message)
    for n in result.notes:
        print("Note:", n)
    for f in result.findings:
        print(f"\n[{f.severity.upper()}] {f.title} ({f.scope}) - {f.persistence}\n  {f.explanation}")
    print("\n" + result.disclaimer)