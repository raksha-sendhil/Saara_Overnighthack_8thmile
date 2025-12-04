# dashboard.py
#
# Streamlit dashboard that automatically discovers recent sessions from the Flask backend,
# auto-selects the newest finished session, and then polls the backend for live cheating scores
# and features. It also loads the raw event log file from ./logs/<session_id>.json and lets
# the user download it.
#
# Run:
#   streamlit run dashboard.py
#
# Requirements: streamlit, requests, pandas
#
import streamlit as st
import requests
import time
import pandas as pd
import os
import json
from datetime import datetime

# ------------------ Configuration ------------------
BACKEND = "http://localhost:8000"
SESSIONS_URL = f"{BACKEND}/api/sessions"
SCORE_URL_BASE = f"{BACKEND}/api/score"   # append /<session_id>
LOG_DIR = "logs"
DEFAULT_POLL_INTERVAL = 2
# ---------------------------------------------------

st.set_page_config(page_title="Proctor Dashboard", layout="wide")
st.title("Proctor Dashboard — Behavioral Analytics")

# Sidebar controls
st.sidebar.header("Controls")
poll_interval = st.sidebar.number_input("Poll interval (s)", min_value=1, max_value=10, value=DEFAULT_POLL_INTERVAL)
score_threshold = st.sidebar.slider("Alert threshold", 0, 100, 70)
auto_start = st.sidebar.checkbox("Auto-start newest finished session", value=True)
refresh_sessions_btn = st.sidebar.button("Refresh sessions")

# Helpers
def fetch_sessions():
    try:
        r = requests.get(SESSIONS_URL, timeout=2.0)
        if r.status_code == 200:
            return r.json().get("sessions", [])
        return []
    except Exception:
        return []

def fetch_score(session_id):
    try:
        r = requests.get(f"{SCORE_URL_BASE}/{session_id}", timeout=2.0)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

def load_log_from_disk(session_id):
    path = os.path.join(LOG_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def pick_most_recent_session(sessions):
    if not sessions:
        return ""
    # Prefer finished sessions (most recent), else most recent active
    finished = [s for s in sessions if s.get("finished")]
    if finished:
        return finished[0]["session_id"]
    return sessions[0]["session_id"]

# Discover sessions from backend
sessions = fetch_sessions()
session_options = [s["session_id"] for s in sessions]
most_recent = pick_most_recent_session(sessions) if sessions else ""

# Layout columns
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Available sessions (from backend)")
    if sessions:
        sess_table = pd.DataFrame(sessions)
        # add human-readable timestamp if available
        if "last_t" in sess_table.columns:
            sess_table["last_activity_utc"] = sess_table["last_t"].apply(
                lambda x: datetime.utcfromtimestamp(x/1000).strftime("%Y-%m-%d %H:%M:%S") if x and x>0 else ""
            )
        display_cols = ["session_id", "score", "finished", "last_activity_utc"] if "last_activity_utc" in sess_table.columns else ["session_id", "score", "finished"]
        st.dataframe(sess_table[display_cols].rename(columns={"last_activity_utc":"last_activity_utc"}), height=260)
    else:
        st.info("No sessions discovered yet. Ensure backend is running and clients sent events.")

with col_right:
    st.subheader("Monitor")
    # Auto-select most recent if requested
    if auto_start and most_recent:
        # make the most recent option first for convenience
        opts = [most_recent] + [o for o in session_options if o != most_recent]
        selected_session = st.selectbox("Session ID", options=[""] + opts, index=1 if most_recent else 0)
    else:
        selected_session = st.selectbox("Session ID", options=[""] + session_options, index=0)
    start_btn = st.button("Start monitoring")
    stop_btn = st.button("Stop monitoring")

# Allow manual refresh of sessions list
if refresh_sessions_btn:
    sessions = fetch_sessions()
    st.experimental_rerun()

# Prepare placeholders for dynamic content
chart_ph = st.empty()
metrics_ph = st.empty()
features_ph = st.empty()
alerts_ph = st.empty()
log_ph = st.empty()
raw_download_ph = st.empty()
info_ph = st.empty()

# Session state initialization
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "timeline" not in st.session_state:
    st.session_state.timeline = []  # list of {"ts": datetime, "score": int}
if "last_session" not in st.session_state:
    st.session_state.last_session = None

# Start/stop behavior
if start_btn and selected_session:
    st.session_state.monitoring = True
    st.session_state.last_session = selected_session
    st.session_state.timeline = []
if stop_btn:
    st.session_state.monitoring = False

# Auto-start if checked and a most_recent exists and user didn't explicitly stop
if auto_start and most_recent and not st.session_state.monitoring and not start_btn and not stop_btn:
    st.session_state.monitoring = True
    st.session_state.last_session = most_recent
    st.session_state.timeline = []

# Determine which session to monitor
session_to_monitor = st.session_state.last_session if st.session_state.last_session else selected_session

# Monitoring loop (in-place updates, no rerun)
if st.session_state.monitoring and session_to_monitor:
    info_ph.info(f"Monitoring session: {session_to_monitor} — polling every {poll_interval}s")
    try:
        # Poll loop: runs until monitoring flag is turned off or session finishes
        while st.session_state.monitoring:
            data = fetch_score(session_to_monitor)
            now = datetime.utcnow()

            if data is None:
                alerts_ph.error("No response from backend. Is backend running and session id correct?")
                st.session_state.monitoring = False
                break

            score = int(data.get("score", 0))
            feats = data.get("features", {}) or {}
            finished = data.get("finished", False)

            # append timeline
            st.session_state.timeline.append({"ts": now, "score": score})
            if len(st.session_state.timeline) > 500:
                st.session_state.timeline = st.session_state.timeline[-500:]

            # Chart
            with chart_ph.container():
                st.subheader("Live risk score")
                df_chart = pd.DataFrame(st.session_state.timeline)
                if not df_chart.empty:
                    df_chart = df_chart.set_index("ts")
                    st.line_chart(df_chart["score"])
                else:
                    st.write("No score yet")

            # Metrics + features
            with metrics_ph.container():
                st.subheader("Latest")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Score", f"{score}/100")
                    st.write("Finished:", finished)
                    if score >= score_threshold:
                        st.error(f"High risk (≥ {score_threshold})")
                    else:
                        st.success("Score within threshold")
                with col2:
                    st.write("Latest features")
                    if feats:
                        try:
                            f_df = pd.DataFrame(list(feats.items()), columns=["feature", "value"])
                            st.table(f_df)
                        except Exception:
                            st.write(feats)
                    else:
                        st.write("No features available")

            # Raw logs (load from disk)
            raw = load_log_from_disk(session_to_monitor)
            with log_ph.container():
                if raw is not None:
                    st.subheader("Raw event log (tail)")
                    # show tail (last 100 events)
                    try:
                        tail_df = pd.DataFrame(raw[-100:])
                        st.dataframe(tail_df)
                    except Exception:
                        st.write(raw[-100:])
                    raw_download_ph.download_button(
                        label="Download full raw log (JSON)",
                        data=json.dumps(raw, indent=2),
                        file_name=f"{session_to_monitor}_log.json",
                        mime="application/json"
                    )
                else:
                    st.info("No raw log file found on disk yet (waiting for backend writes).")

            # If session finished, stop monitoring after brief pause
            if finished:
                st.success("Session finished on backend. Stopping monitoring.")
                time.sleep(2)
                st.session_state.monitoring = False
                break

            # Sleep until next poll
            time.sleep(poll_interval)
    except Exception as e:
        st.error(f"Monitoring loop error: {e}")
        st.session_state.monitoring = False

else:
    # not monitoring: show informative panels
    if not selected_session:
        st.info("Select a session from the dropdown or wait for sessions to appear.")
    else:
        st.write("Monitoring is stopped. Click 'Start monitoring' to begin.")
        # show last timeline if present
        if st.session_state.timeline:
            chart_ph.subheader("Last recorded timeline")
            df_chart = pd.DataFrame(st.session_state.timeline).set_index("ts")
            chart_ph.line_chart(df_chart["score"])

st.write("---")
st.write("Notes: This dashboard polls the backend at", BACKEND)
