"""LifeRhythm AI - Streamlit application.

Run from the project folder with:   streamlit run app.py

Which parts are machine learning?
  * Next-activity prediction (Random Forest)  -> ML
  * Everything else (drift detection, recommendations, analytics) -> rules / statistics
"""
from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
import plotly.express as px
import streamlit as st

from src import analytics as an
from src import database as db
from src.config import CATEGORIES, MIN_RECORDS_FOR_TRAINING, MODEL_PATH
from src.data_generator import generate_demo_data
from src.drift_detection import DriftConfig, analyze_drift
from src.feature_engineering import InsufficientDataError
from src.prediction import ModelNotTrainedError, predict_next_activity
from src.preprocessing import ValidationError, parse_time
from src.recommendations import (
    GOALS,
    Preferences,
    free_windows,
    generate_and_store,
    load_preferences,
    save_preferences,
)
from src.train_model import load_metrics, train_from_database

st.set_page_config(page_title="LifeRhythm AI", page_icon="🌙", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    .tag {display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.78rem;
          font-weight:600; margin-right:6px;}
    .tag-ml {background:#e3f2fd; color:#0d47a1;}
    .tag-rule {background:#fff3e0; color:#e65100;}
    .tag-stat {background:#e8f5e9; color:#1b5e20;}
    </style>
    """,
    unsafe_allow_html=True,
)

db.init_db()

st.title("LifeRhythm AI")
st.success("Streamlit is working!")
st.write("Testing dashboard output...")

GOAL_LABELS = {"study": "Study regularly", "exercise": "Exercise regularly", "sleep": "Keep a regular bedtime"}


# ----------------------------------------------------------------- helpers
def tag(kind: str) -> None:
    labels = {
        "ml": ("tag-ml", "Machine learning"),
        "rule": ("tag-rule", "Rule-based (not ML)"),
        "stat": ("tag-stat", "Descriptive statistics (not ML)"),
    }
    css, text = labels[kind]
    st.markdown(f'<span class="tag {css}">{text}</span>', unsafe_allow_html=True)


def empty_state(message: str) -> None:
    st.info(message)


def display_table(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    out = df.copy()
    out["date"] = out["start_dt"].dt.strftime("%Y-%m-%d")
    out["start"] = out["start_dt"].dt.strftime("%H:%M")
    out["end"] = out["end_dt"].dt.strftime("%H:%M")
    out["source"] = out["is_demo"].map({1: "synthetic", 0: "yours"}).fillna("yours")
    st.dataframe(
        out[["id", "date", "start", "end", "duration_min", "category", "notes", "source"]],
        use_container_width=True,
        hide_index=True,
    )


def get_drift_config() -> DriftConfig:
    return st.session_state.get("drift_cfg", DriftConfig())


def parse_blocked(text: str):
    windows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("-")]
        if len(parts) != 2:
            raise ValidationError(f"Blocked window '{line}' must look like 12:00-13:00.")
        parse_time(parts[0])
        parse_time(parts[1])
        windows.append([parts[0], parts[1]])
    return windows


# ------------------------------------------------------------- page: log
def page_log() -> None:
    st.header("Record an activity")
    st.caption("Activities are entered manually. LifeRhythm AI does not detect what you do automatically.")

    with st.form("add_form"):
        c1, c2, c3 = st.columns(3)
        d = c1.date_input("Date", value=date.today())
        start = c2.time_input("Start time", value=time(9, 0), step=300)
        category = c3.selectbox("Category", CATEGORIES)
        mode = st.radio("How do you want to give the length?", ["End time", "Duration (minutes)"], horizontal=True)
        c4, c5 = st.columns(2)
        end = c4.time_input("End time (used when 'End time' is selected)", value=time(10, 0), step=300)
        dur = c5.number_input("Duration in minutes (used when 'Duration' is selected)",
                              min_value=1, max_value=960, value=60, step=5)
        notes = st.text_input("Notes (optional)")
        allow = st.checkbox("Save even if it overlaps another record")
        submitted = st.form_submit_button("Save activity")

    if submitted:
        try:
            if mode == "End time":
                new_id = db.add_activity(d, start, category, end_time=end, notes=notes, allow_overlap=allow)
            else:
                new_id = db.add_activity(d, start, category, duration_min=int(dur), notes=notes, allow_overlap=allow)
            st.success(f"Saved activity #{new_id}. (An end time earlier than the start time means it crosses midnight.)")
        except db.OverlapError as exc:
            st.warning(f"{exc} Nothing was saved. Tick 'Save even if it overlaps' to keep both records.")
            display_table(exc.conflicts)
        except ValidationError as exc:
            st.error(str(exc))

    st.subheader("Most recent records")
    df = db.get_activities()
    if df.empty:
        empty_state("No activities yet. Add one above, or open **Data & demo** to load synthetic demo data.")
    else:
        display_table(df.sort_values("start_dt", ascending=False).head(15))


# --------------------------------------------------------- page: timeline
def page_timeline() -> None:
    st.header("Daily timeline")
    df = db.get_activities()
    if df.empty:
        empty_state("No activities yet. Record one first, or load the demo data.")
        return

    day = st.date_input("Day", value=df["start_dt"].max().date(), key="timeline_day")
    tl = an.daily_timeline(df, day)
    if tl.empty:
        empty_state(f"Nothing was logged on {day}. Pick another day.")
        return

    st.metric("Logged time on this day (minutes)", int(tl["duration_min"].sum()))
    display_table(tl)

    fig = px.timeline(tl, x_start="start_dt", x_end="end_dt", y="category", color="category",
                      hover_data=["duration_min", "notes"])
    fig.update_layout(showlegend=False, height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Overnight activities such as sleep extend past midnight.")

    with st.expander("Edit or delete a record"):
        labels = {int(r.id): f"{r.start_label}-{r.end_label}  {r.category}" for r in tl.itertuples()}
        rid = st.selectbox("Choose a record", list(labels), format_func=lambda i: labels[i])
        row = tl[tl["id"].astype(int) == rid].iloc[0]

        with st.form(f"edit_form_{rid}"):
            e1, e2, e3 = st.columns(3)
            ed = e1.date_input("Date", value=row["start_dt"].date())
            es = e2.time_input("Start time", value=row["start_dt"].time().replace(second=0), step=300)
            ec = e3.selectbox("Category", CATEGORIES, index=CATEGORIES.index(row["category"]))
            ee = st.time_input("End time", value=row["end_dt"].time().replace(second=0), step=300)
            en = st.text_input("Notes", value=row["notes"] or "")
            eallow = st.checkbox("Save even if it overlaps another record")
            save_edit = st.form_submit_button("Save changes")
        if save_edit:
            try:
                db.update_activity(rid, ed, es, ec, end_time=ee, notes=en, allow_overlap=eallow)
                st.success("Record updated. Reload the page view (press R) to see it in the table.")
            except db.OverlapError as exc:
                st.warning(f"{exc} Nothing was changed.")
                display_table(exc.conflicts)
            except ValidationError as exc:
                st.error(str(exc))

        confirm = st.checkbox("I want to delete this record", key=f"del_confirm_{rid}")
        if st.button("Delete record", key=f"del_btn_{rid}"):
            if not confirm:
                st.warning("Tick the confirmation box first.")
            elif db.delete_activity(rid):
                st.success("Record deleted. Press R to refresh the view.")
                st.rerun()


# ---------------------------------------------------------- page: predict
def page_predict() -> None:
    st.header("Predict the next activity")
    tag("ml")
    st.write("A Random Forest model, trained on your history, estimates which activity is most likely to follow "
             "your latest logged one.")

    df = db.get_activities()
    if df.empty:
        empty_state("There is no history to predict from. Record activities or load the demo data.")
        return
    if not MODEL_PATH.exists():
        st.warning("No trained model yet. Open **Model & evaluation** and press **Train model**.")
        return

    custom = st.checkbox("Predict as of a specific date and time (instead of 'now')")
    as_of = None
    if custom:
        last = df["end_dt"].max().to_pydatetime()
        c1, c2 = st.columns(2)
        ad = c1.date_input("Date", value=last.date(), key="asof_date")
        at = c2.time_input("Time", value=last.time().replace(second=0, microsecond=0), key="asof_time")
        as_of = datetime.combine(ad, at)

    if st.button("Predict next activity", type="primary"):
        try:
            res = predict_next_activity(df, as_of=as_of)
        except ModelNotTrainedError as exc:
            st.error(str(exc))
            return
        except InsufficientDataError as exc:
            st.error(str(exc))
            return

        ctx = res["based_on"]
        st.subheader(f"Most likely next activity: {res['predicted']}")
        m1, m2 = st.columns(2)
        m1.metric("Model confidence", f"{res['confidence'] * 100:.0f}%")
        m2.metric("Confidence level", res["confidence_label"])
        st.write(f"Based on your last logged activity: **{ctx['last_category']}** "
                 f"({ctx['last_duration_min']} min, ended {ctx['ended_at']:%Y-%m-%d %H:%M}).")
        st.markdown("**Top candidates**")
        for t in res["top_k"]:
            st.progress(min(1.0, t["probability"]), text=f"{t['category']} - {t['probability'] * 100:.0f}%")
        for w in res["warnings"]:
            st.warning(w)
        st.caption(res["disclaimer"])


# ------------------------------------------------------------ page: drift
def page_drift() -> None:
    st.header("Routine changes (drift detection)")
    tag("rule")
    st.write("Your recent routine is compared with an earlier baseline. Weekdays are compared with weekdays and "
             "weekends with weekends. Every flag explains itself.")

    with st.expander("Adjust thresholds", expanded=False):
        c1, c2 = st.columns(2)
        cur = c1.slider("Current period (days)", 7, 28, 14)
        base = c2.slider("Baseline period (days)", 14, 60, 28)
        bed = c1.slider("Flag bedtime shift of at least (min)", 15, 180, 45, step=5)
        timing = c2.slider("Flag activity start-time shift of at least (min)", 30, 180, 60, step=5)
        study = c1.slider("Flag study-time drop of at least (%)", 10, 80, 25)
        ex = c2.slider("Flag exercise-frequency drop of at least (%)", 10, 80, 30)
        irr = c1.slider("Flag bedtime-spread increase of at least (%)", 10, 100, 30)
    cfg = DriftConfig(current_days=cur, baseline_days=base, bedtime_shift_min=float(bed),
                      timing_shift_min=float(timing), study_drop_pct=float(study),
                      exercise_drop_pct=float(ex), irregularity_increase_pct=float(irr))
    st.session_state["drift_cfg"] = cfg

    df = db.get_activities()
    if df.empty:
        empty_state("No activities yet. Record some or load the demo data.")
        return
    try:
        result = analyze_drift(df, cfg)
    except ValueError as exc:
        st.error(f"Invalid settings: {exc}")
        return

    if not result.sufficient:
        st.info(result.message)
        return

    st.write(result.message)
    c1, c2 = st.columns(2)
    c1.metric("Baseline period", f"{result.baseline_period[0]} to {result.baseline_period[1]}",
              f"{result.n_baseline_days} logged days", delta_color="off")
    c2.metric("Current period", f"{result.current_period[0]} to {result.current_period[1]}",
              f"{result.n_current_days} logged days", delta_color="off")
    for note in result.notes:
        st.caption(f"Note: {note}")

    st.subheader("Flagged changes")
    if not result.findings:
        st.success("No change exceeded your thresholds. This does not prove nothing changed, only that "
                   "nothing crossed the limits you set.")
    for f in result.findings:
        with st.expander(f"{f.severity.upper()}: {f.title} ({f.scope})  -  {f.persistence}", expanded=True):
            a, b, c = st.columns(3)
            a.metric("Baseline", f.baseline_text)
            b.metric("Current", f.current_text)
            c.metric("Change", f.change_text)
            st.write(f.explanation)

    st.subheader("Baseline vs current")
    if result.comparison.empty:
        empty_state("No comparison table is available for this data.")
    else:
        st.dataframe(result.comparison, use_container_width=True, hide_index=True)

    st.subheader("Single unusual days")
    if result.unusual_days.empty:
        st.write("No single day was far from your typical pattern.")
    else:
        st.dataframe(result.unusual_days, use_container_width=True, hide_index=True)
    st.caption(result.disclaimer)


# ---------------------------------------------------- page: recommendations
def page_recommendations() -> None:
    st.header("Personalised recommendations")
    tag("rule")
    st.write("Suggestions are built from your goals, your free time and your own history. They never overlap "
             "logged activities, blocked windows or other suggestions, and each one gives its reason.")

    prefs = load_preferences()
    with st.expander("My goals and constraints", expanded=not prefs.goals):
        with st.form("prefs_form"):
            goals = st.multiselect("Goals", list(GOALS), default=[g for g in prefs.goals if g in GOALS],
                                   format_func=lambda g: GOAL_LABELS[g])
            c1, c2 = st.columns(2)
            study_goal = c1.number_input("Study goal (minutes per day)", 15, 600, int(prefs.study_goal_min_per_day), 15)
            study_sess = c2.number_input("Preferred study session (minutes)", 15, 240, int(prefs.study_session_min), 5)
            ex_goal = c1.number_input("Exercise goal (sessions per week)", 1, 14, int(prefs.exercise_goal_per_week), 1)
            ex_sess = c2.number_input("Preferred exercise session (minutes)", 10, 180, int(prefs.exercise_session_min), 5)
            a1, a2 = st.columns(2)
            av_start = a1.time_input("Available from", value=parse_time(prefs.available_start), step=900)
            av_end = a2.time_input("Available until", value=parse_time(prefs.available_end), step=900)
            blocked_text = st.text_area("Blocked times (one per line, e.g. 12:00-13:00)",
                                        value="\n".join(f"{a}-{b}" for a, b in prefs.blocked_windows))
            max_s = st.number_input("Maximum suggestions per day", 1, 10, int(prefs.max_suggestions), 1)
            saved = st.form_submit_button("Save my preferences")
        if saved:
            try:
                new = Preferences(
                    goals=list(goals), study_goal_min_per_day=int(study_goal), study_session_min=int(study_sess),
                    exercise_goal_per_week=int(ex_goal), exercise_session_min=int(ex_sess),
                    available_start=av_start.strftime("%H:%M"), available_end=av_end.strftime("%H:%M"),
                    blocked_windows=parse_blocked(blocked_text), max_suggestions=int(max_s),
                )
                save_preferences(new)
                prefs = new
                st.success("Preferences saved.")
            except ValidationError as exc:
                st.error(str(exc))

    st.subheader("Get suggestions for a day")
    c1, c2 = st.columns([1, 2])
    target = c1.date_input("Day to plan", value=date.today(), key="rec_day")
    from_now = c2.checkbox("Only suggest times that have not passed yet", value=True)

    history = db.get_activities()
    with st.expander("Show my free time windows for this day"):
        wins = free_windows(history, target, prefs, db.get_suggestions(),
                            now=datetime.now() if from_now else None)
        if not wins:
            st.write("No free windows of at least 15 minutes inside your available hours.")
        for s, e in wins:
            st.write(f"{s:%H:%M} to {e:%H:%M}  ({int((e - s).total_seconds() // 60)} min)")

    if st.button("Generate suggestions", type="primary"):
        try:
            recs, notes = generate_and_store(target, prefs=prefs, now=datetime.now() if from_now else None,
                                             drift_config=get_drift_config())
            st.session_state["rec_notes"] = notes
            if not recs:
                st.info("No suggestions for this day. See the notes below.")
        except (ValidationError, ValueError) as exc:
            st.error(str(exc))

    for n in st.session_state.get("rec_notes", []):
        st.caption(f"Note: {n}")

    st.subheader("My suggestions")
    sug = db.get_suggestions()
    if sug.empty:
        empty_state("No suggestions yet. Set a goal above and press **Generate suggestions**.")
        return
    show = st.radio("Show", ["pending", "accepted", "completed", "dismissed"], horizontal=True)
    sub = sug[sug["status"] == show]
    if sub.empty:
        st.write(f"No {show} suggestions.")
    for r in sub.itertuples():
        with st.container(border=True):
            st.markdown(f"**{r.title}**  \n{r.start_dt:%a %d %b, %H:%M} to {r.end_dt:%H:%M}  ({r.category})")
            st.write(f"*Why:* {r.reason}")
            b1, b2, _ = st.columns([1, 1, 4])
            if r.status == "pending":
                if b1.button("Accept", key=f"acc_{r.id}"):
                    db.set_suggestion_status(int(r.id), "accepted")
                    st.rerun()
                if b2.button("Dismiss", key=f"dis_{r.id}"):
                    db.set_suggestion_status(int(r.id), "dismissed")
                    st.rerun()
            elif r.status == "accepted":
                if b1.button("Mark completed", key=f"done_{r.id}"):
                    db.set_suggestion_status(int(r.id), "completed")
                    st.rerun()
                if b2.button("Dismiss", key=f"dis_{r.id}"):
                    db.set_suggestion_status(int(r.id), "dismissed")
                    st.rerun()
    st.caption("Suggestions are options, not instructions. They make no claims about health or productivity.")


# ---------------------------------------------------------- page: model
def page_model() -> None:
    st.header("Model & evaluation")
    tag("ml")
    n = db.count_activities()
    st.write(f"Records in the database: **{n}**. At least **{MIN_RECORDS_FOR_TRAINING}** valid records are needed.")

    if st.button("Train model on my current records", type="primary"):
        with st.spinner("Training Random Forest and baselines..."):
            try:
                train_from_database()
                st.success("Model trained and saved. The numbers below were calculated from real held-out predictions.")
            except InsufficientDataError as exc:
                st.error(str(exc))
            except Exception as exc:  # keep the app alive on unexpected training errors
                st.error(f"Training failed: {exc}")

    m = load_metrics()
    if m is None:
        empty_state("No evaluation results yet. Train the model first.")
        return

    st.write(f"Data source: **{m['data_source']}**  |  records: {m['n_records']}  |  "
             f"train samples: {m['n_train']}  |  test samples: {m['n_test']}")
    st.caption(f"Train period: {m['train_period']['start']} to {m['train_period']['end']}.  "
               f"Test period: {m['test_period']['start']} to {m['test_period']['end']}.")
    if m["data_source"] != "user_records":
        st.warning("This model used synthetic or mixed data. These scores do NOT represent real-world performance.")

    names = {"majority_baseline": "Baseline 1: always most common activity",
             "markov_baseline": "Baseline 2: most common successor",
             "random_forest": "Random Forest"}
    table = pd.DataFrame(
        [{"Model": names[k], "Accuracy": v["accuracy"], "Macro F1": v["macro_f1"]} for k, v in m["models"].items()]
    )
    st.subheader("Comparison on the chronological test set")
    st.dataframe(table.style.format({"Accuracy": "{:.3f}", "Macro F1": "{:.3f}"}),
                 use_container_width=True, hide_index=True)

    st.subheader("Confusion matrix (Random Forest)")
    fig = px.imshow(m["confusion_matrix"], x=m["labels"], y=m["labels"], text_auto=True,
                    labels=dict(x="Predicted", y="Actual", color="Count"), aspect="auto")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Per-activity results")
        per = pd.DataFrame(m["per_class"]).T.reset_index().rename(columns={"index": "activity"})
        st.dataframe(per, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Most influential features")
        st.dataframe(pd.DataFrame(m["top_features"]), use_container_width=True, hide_index=True)

    for note in m["notes"]:
        st.caption(f"* {note}")


# ---------------------------------------------------------- page: rhythm
def page_rhythm() -> None:
    st.header("My rhythm")
    tag("stat")
    st.write("A summary of what you logged. These are averages and spreads, not predictions.")
    df = db.get_activities()
    if df.empty:
        empty_state("No activities yet. Record some or load the demo data.")
        return

    s = an.summary_stats(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Logged days", s["logged_days"])
    c2.metric("Typical bedtime", s.get("typical_bedtime") or "n/a")
    c3.metric("Average sleep", f"{s['avg_sleep_min'] / 60:.1f} h" if s.get("avg_sleep_min") else "n/a")
    c4.metric("Exercise days per week", f"{s['exercise_days_per_week']:.1f}")
    spread = s.get("bedtime_spread_min")
    if spread is not None:
        st.write(f"Your bedtime varies by about +/-{spread:.0f} min ({an.regularity_label(spread)}).")

    st.subheader("Average daily time by category")
    avg = an.average_daily_minutes(df)
    fig = px.bar(avg[avg["avg_min_per_day"] > 0], x="category", y="avg_min_per_day",
                 labels={"avg_min_per_day": "Minutes per logged day", "category": ""})
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weekday vs weekend")
    st.dataframe(an.weekday_weekend_comparison(df).round(0), use_container_width=True, hide_index=True)

    st.subheader("Typical activity by hour")
    st.dataframe(an.typical_activity_by_hour(df)[["hour_label", "typical_category", "share_of_hour_pct"]]
                 .round(0), use_container_width=True, hide_index=True)
    with st.expander("Hourly heatmap"):
        occ = an.hourly_occupancy(df)
        hm = px.imshow(occ.T, aspect="auto", labels=dict(x="Hour of day", y="Category", color="Min per day"))
        hm.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(hm, use_container_width=True)

    st.subheader("Which activities are most regular?")
    cons = an.consistency_by_category(df)
    if cons.empty:
        st.write("Not enough days logged yet (at least 5 days per activity).")
    else:
        st.dataframe(cons.round(0), use_container_width=True, hide_index=True)

    st.subheader("Week by week")
    wk = an.weekly_summary(df)
    st.dataframe(wk.round(1), use_container_width=True, hide_index=True)
    if len(wk) >= 2:
        w1, w2 = st.columns(2)
        w1.plotly_chart(px.line(wk, x="week_start", y="study_min_per_day", markers=True,
                                labels={"study_min_per_day": "Study min per logged day"}),
                        use_container_width=True)
        w2.plotly_chart(px.line(wk, x="week_start", y="avg_sleep_min", markers=True,
                                labels={"avg_sleep_min": "Average sleep (min)"}),
                        use_container_width=True)


# ------------------------------------------------------------ page: data
def page_data() -> None:
    st.header("Data & demo mode")
    total, demo = db.count_activities(), db.count_activities(demo=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("All records", total)
    c2.metric("Synthetic demo records", demo)
    c3.metric("Your own records", total - demo)

    st.subheader("Load demo data")
    st.warning("The demo data is SYNTHETIC (simulated). It covers 8 weeks and deliberately includes a routine "
               "change in the last 2 weeks. Accuracy measured on it does not represent real-world performance.")
    own = total - demo
    confirm = False
    if own > 0:
        confirm = st.checkbox(f"I understand this will add synthetic records alongside my {own} own record(s). "
                              "My own records will not be deleted.")
    if st.button("Load synthetic demo data"):
        try:
            n = db.insert_demo_data(generate_demo_data(), confirm_with_existing=confirm)
            st.success(f"Loaded {n} synthetic records. Next: train the model on the **Model & evaluation** page.")
        except db.ExistingDataError as exc:
            st.error(f"{exc} Tick the confirmation box above to continue.")

    st.subheader("Clear data")
    if st.button("Clear demo data only"):
        st.success(f"Removed {db.clear_demo_data()} synthetic records. Your own records were kept.")
    wipe = st.checkbox("I understand that deleting ALL records cannot be undone")
    if st.button("Delete ALL records"):
        if wipe:
            st.success(f"Deleted {db.clear_all_data()} records.")
        else:
            st.warning("Tick the confirmation box first.")

    st.subheader("Export")
    df = db.get_activities()
    if df.empty:
        st.write("Nothing to export yet.")
    else:
        st.download_button("Download my records (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name="liferhythm_records.csv", mime="text/csv")


# --------------------------------------------------------------------- main
PAGES = {
    "1. Record activity": page_log,
    "2. Daily timeline": page_timeline,
    "3. Predict next activity": page_predict,
    "4. Routine changes": page_drift,
    "5. Recommendations": page_recommendations,
    "6. Model & evaluation": page_model,
    "7. My rhythm": page_rhythm,
    "8. Data & demo": page_data,
}

st.sidebar.title("🌙 LifeRhythm AI")
st.sidebar.caption("Learns your daily activity cycle from the activities you log.")
choice = st.sidebar.radio("Go to", list(PAGES))
st.sidebar.markdown("---")
st.sidebar.caption("Predictions and flags are estimates based on your logs. They are not medical or "
                   "psychological advice.")

if db.count_activities(demo=True) > 0:
    st.info("Your database contains SYNTHETIC demo records. Results based on them are for demonstration only.")

PAGES[choice]()