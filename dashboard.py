# dashboard.py
# Simplified Streamlit dashboard (no live score, no polling controls, no timestamps)
#
# Run:
#   streamlit run dashboard.py
#
# Expects backend at http://localhost:8000 exposing:
# - GET /api/sessions  -> {"sessions":[{"session_id","score","finished", ...}, ...]}
# - GET /api/score/<session_id> -> {"session_id","score","features","finished"}
#
import streamlit as st
import requests
import pandas as pd
import json
import os

BACKEND = "http://localhost:8000"
SESSIONS_URL = f"{BACKEND}/api/sessions"
SCORE_URL = f"{BACKEND}/api/score"
LOG_DIR = "logs"

st.set_page_config(page_title="Cheating Analysis", layout="wide")
st.title("Cheating Analysis Dashboard")

# ---------- helpers ----------
def fetch_sessions():
    try:
        r = requests.get(SESSIONS_URL, timeout=2.0)
        if r.status_code == 200:
            return r.json().get("sessions", [])
    except Exception:
        pass
    return []

def fetch_score(sid):
    try:
        r = requests.get(f"{SCORE_URL}/{sid}", timeout=2.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def load_log(session_id):
    p = os.path.join(LOG_DIR, f"{session_id}.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# ---------- UI ----------
sessions = fetch_sessions()
if not sessions:
    st.warning("No sessions found. Start a test to generate sessions.")
    st.stop()

# show simple table: session_id, score, finished (no timestamps)
simple_table = pd.DataFrame([{"session_id": s.get("session_id"),
                              "score": s.get("score", 0),
                              "finished": s.get("finished", False)} for s in sessions])
st.subheader("Sessions")
st.dataframe(simple_table, height=200)

session_ids = simple_table["session_id"].tolist()
selected = st.selectbox("Select session", session_ids)

if not selected:
    st.info("Choose a session to view details.")
    st.stop()

result = fetch_score(selected)
if not result:
    st.error("Unable to fetch session data from backend.")
    st.stop()

# Final score
st.subheader("Final Cheating Score")
st.metric("Score (0–100)", int(result.get("score", 0)))

# Features
st.subheader("Extracted Features")
features = result.get("features") or {}
if features:
    feat_df = pd.DataFrame(list(features.items()), columns=["feature", "value"])
    st.table(feat_df)
else:
    st.write("No features available for this session.")

# Raw log tail + download
st.subheader("Raw Event Log (tail)")
raw = load_log(selected)
if raw:
    tail = raw[-50:] if isinstance(raw, list) and len(raw) > 50 else raw
    try:
        tail_df = pd.DataFrame(tail)
        st.dataframe(tail_df, height=300)
    except Exception:
        st.write(tail)
    st.download_button(
        "Download full log (JSON)",
        data=json.dumps(raw, indent=2),
        file_name=f"{selected}_log.json",
        mime="application/json"
    )
else:
    st.info("Raw log file not present on disk yet.")
