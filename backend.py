# backend.py
#
# Flask backend for behavioral exam prototype.
# - Serves index.html
# - POST /api/events   -> accept batched events { session_id, question_id, events: [...] }
# - GET  /api/score/<session_id> -> latest score + features + finished flag
# - GET  /api/sessions -> list of sessions for dashboard auto-discovery
#
# Requires:
# pip install flask flask-cors numpy pandas scikit-learn joblib
#
import os
import json
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib

# ---------------- config ----------------
HOST = "0.0.0.0"
PORT = 8000
LOG_DIR = "logs"
MODEL_PATH = "model.pkl"
WINDOW_MS = 10_000   # sliding window for features (10s)
MAX_EVENTS_KEEP = 20000  # per-session in-memory cap

# create directories
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------- app init ----------------
app = Flask(__name__, static_folder="static")
CORS(app)

# ---------------- in-memory state ----------------
SESSION_EVENTS = {}    # session_id -> list of event dicts
SESSION_FEATURES = {}  # session_id -> last computed feature dict
SESSION_SCORE = {}     # session_id -> last combined score (0-100)
SESSION_FINISHED = {}  # session_id -> bool

# ---------------- load model if available ----------------
MODEL = None
try:
    if os.path.exists(MODEL_PATH):
        MODEL = joblib.load(MODEL_PATH)
        print(f"[INFO] Loaded model from {MODEL_PATH}")
    else:
        print("[INFO] No model.pkl found — using heuristic scoring.")
except Exception as e:
    print("[WARNING] Failed to load model.pkl:", e)
    MODEL = None

# ---------------- utility helpers ----------------
def save_log(session_id):
    """Persist the full event list for a session to logs/<session_id>.json"""
    try:
        path = os.path.join(LOG_DIR, f"{session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(SESSION_EVENTS.get(session_id, []), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[ERROR] saving log:", e)

def events_to_df(events):
    """Convert list of dict events to pandas DataFrame with safe columns."""
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    # ensure timestamp column exists as numeric ms
    if "t" in df.columns:
        df["t"] = pd.to_numeric(df["t"], errors="coerce")
    else:
        df["t"] = np.nan
    return df

def sliding_window_df(events, now_ms=None, window_ms=WINDOW_MS):
    """Return DataFrame of events within last window_ms (uses event 't' fields)."""
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    df = events_to_df(events)
    if df.empty:
        return df
    # If t has NaNs, treat whole df as available and rely on last events for time
    if df["t"].isnull().all():
        return df
    cutoff = now_ms - window_ms
    return df[df["t"] >= cutoff]

def safe_count(df, evtype):
    if df is None or df.empty:
        return 0
    return int((df["type"] == evtype).sum()) if "type" in df.columns else 0

def compute_features_for_session(session_id):
    """Compute numeric features for last WINDOW_MS ms from session events.
       Returns a dict keyed to the same feature names used for training."""
    events = SESSION_EVENTS.get(session_id, [])
    if not events:
        feats = {
            "paste_count_10s": 0,
            "copy_count_10s": 0,
            "rightclick_count_10s": 0,
            "tab_switch_count_10s": 0,
            "selection_count_10s": 0,
            "mousemove_count_10s": 0,
            "keydown_count_10s": 0,
            "iki_mean_10s": 0.0,
            "iki_std_10s": 0.0,
            "event_rate_10s": 0.0
        }
        SESSION_FEATURES[session_id] = feats
        return feats

    df_all = events_to_df(events)
    now_ms = int(time.time() * 1000)
    df_win = sliding_window_df(events, now_ms=now_ms, window_ms=WINDOW_MS)
    # counts
    paste = safe_count(df_win, "cheat_paste")
    copy = safe_count(df_win, "cheat_copy")
    right = safe_count(df_win, "cheat_rightclick")
    tab = safe_count(df_win, "cheat_tab_switch")
    selection = safe_count(df_win, "selection")
    mouse_moves = int((df_win["type"] == "mousemove").sum()) if "type" in df_win.columns else 0
    keydowns = int((df_win["type"] == "keydown").sum()) if "type" in df_win.columns else 0

    # IKIs: prefer 'iki' events with 'dt' field, otherwise derive from keydown timestamps
    ikis = []
    if "type" in df_win.columns and "dt" in df_win.columns and (df_win["type"] == "iki").any():
        ikis = df_win[df_win["type"] == "iki"]["dt"].dropna().astype(float).tolist()
    else:
        # derive from keydown timestamps (in the window)
        if "type" in df_all.columns and "t" in df_all.columns:
            kdowns = df_all[df_all["type"] == "keydown"].sort_values("t")
            if len(kdowns) >= 2:
                times = kdowns["t"].dropna().values
                if len(times) >= 2:
                    diffs = np.diff(times).astype(float)
                    diffs = diffs[(diffs > 0) & (diffs < 2000)]  # filter extremes
                    ikis = diffs.tolist() if len(diffs) > 0 else []

    iki_mean = float(np.mean(ikis)) if len(ikis) > 0 else 0.0
    iki_std = float(np.std(ikis)) if len(ikis) > 0 else 0.0

    # event_rate = events-per-10s (in window)
    ev_count = len(df_win) if not df_win.empty else 0
    event_rate = float(ev_count) / (WINDOW_MS / 1000.0)  # events per second, but training used per-10s approx; keep consistent
    # For our synthetic training we used event_rate roughly = (mouse+key)/10
    # It's okay; downstream model was trained with the same derived column.

    feats = {
        "paste_count_10s": int(paste),
        "copy_count_10s": int(copy),
        "rightclick_count_10s": int(right),
        "tab_switch_count_10s": int(tab),
        "selection_count_10s": int(selection),
        "mousemove_count_10s": int(mouse_moves),
        "keydown_count_10s": int(keydowns),
        "iki_mean_10s": float(iki_mean),
        "iki_std_10s": float(iki_std),
        "event_rate_10s": float(event_rate)
    }

    SESSION_FEATURES[session_id] = feats
    return feats

def heuristic_score(features):
    """Quick heuristic mapping of features -> 0-100 score, fallback when no model."""
    score = 0.0
    score += features.get("paste_count_10s", 0) * 40.0
    score += features.get("copy_count_10s", 0) * 30.0
    score += features.get("rightclick_count_10s", 0) * 25.0
    score += features.get("tab_switch_count_10s", 0) * 50.0
    score += features.get("selection_count_10s", 0) * 5.0
    # robotic typing
    if 0 < features.get("iki_std_10s", 9999) < 20:
        score += 10.0
    # low mouse + lots of key activity
    if features.get("mousemove_count_10s", 0) < 2 and features.get("keydown_count_10s", 0) > 5:
        score += 8.0
    return int(max(0, min(100, score)))

def model_score(features):
    """If model loaded, use it, else heuristic."""
    if MODEL is None:
        return heuristic_score(features)
    # Model expects feature ordering used in training
    order = [
        "paste_count_10s",
        "copy_count_10s",
        "rightclick_count_10s",
        "tab_switch_count_10s",
        "selection_count_10s",
        "mousemove_count_10s",
        "keydown_count_10s",
        "iki_mean_10s",
        "iki_std_10s",
        "event_rate_10s"
    ]
    x = np.array([features.get(k, 0.0) for k in order], dtype=float).reshape(1, -1)
    try:
        if hasattr(MODEL, "predict_proba"):
            p = MODEL.predict_proba(x)[0, 1]
            return int(max(0, min(100, round(p * 100.0))))
        else:
            # classifier without probas
            cl = MODEL.predict(x)[0]
            return 100 if cl == 1 else 0
    except Exception as e:
        print("[ERROR] model prediction failed:", e)
        return heuristic_score(features)

# ---------------- API endpoints ----------------

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    """
    Serve frontend files:
    - If a static file exists in ./static, serve it.
    - Otherwise serve ./index.html
    """
    # If path corresponds to a static file, serve from static folder
    static_path = os.path.join(app.static_folder or "static", path)
    if path and os.path.exists(static_path):
        return send_from_directory(app.static_folder, path)
    # If index.html exists in project root, serve that
    index_path = os.path.join(".", "index.html")
    if os.path.exists(index_path):
        return send_from_directory(".", "index.html")
    return "index.html not found on server.", 404

@app.route("/api/events", methods=["POST"])
def api_events():
    """
    Accept POSTed event batches. Payload format:
    {
      "session_id": "sess_xxx",
      "question_id": 0,
      "events": [ { "type": "...", "t": 173..., ... }, ... ]
    }
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "invalid json", "detail": str(e)}), 400

    session_id = data.get("session_id")
    events = data.get("events", [])
    if not session_id:
        return jsonify({"error": "missing session_id"}), 400
    if not isinstance(events, list):
        return jsonify({"error": "events must be list"}), 400

    # ensure session exists
    if session_id not in SESSION_EVENTS:
        SESSION_EVENTS[session_id] = []

    # Append events; guard size
    for ev in events:
        # minimal validation: ensure it's a dict; add t if missing
        if not isinstance(ev, dict):
            continue
        if "t" not in ev:
            ev["t"] = int(time.time() * 1000)
        SESSION_EVENTS[session_id].append(ev)
        # detect test_end event to mark finished
        if ev.get("type") == "test_end":
            SESSION_FINISHED[session_id] = True

    # cap list size
    if len(SESSION_EVENTS[session_id]) > MAX_EVENTS_KEEP:
        SESSION_EVENTS[session_id] = SESSION_EVENTS[session_id][-MAX_EVENTS_KEEP:]

    # persist log (overwrite) for dashboard access
    save_log(session_id)

    # compute features and score
    feats = compute_features_for_session(session_id)
    score = model_score(feats)
    SESSION_SCORE[session_id] = int(score)

    return jsonify({"status": "ok"}), 200

@app.route("/api/score/<session_id>", methods=["GET"])
def api_score(session_id):
    """Return current score + features + finished flag for session."""
    feats = SESSION_FEATURES.get(session_id, {})
    score = int(SESSION_SCORE.get(session_id, 0))
    finished = bool(SESSION_FINISHED.get(session_id, False))
    last_t = None
    # find last event timestamp if available
    events = SESSION_EVENTS.get(session_id, [])
    if events:
        try:
            last_t = max([int(ev.get("t", 0)) for ev in events if isinstance(ev, dict)])
        except Exception:
            last_t = None
    return jsonify({
        "session_id": session_id,
        "score": score,
        "features": feats,
        "finished": finished,
        "last_t": last_t
    }), 200

@app.route("/api/sessions", methods=["GET"])
def api_sessions():
    """
    Return list of known sessions with metadata for dashboard auto-detection:
    { "sessions": [ { session_id, score, last_t, finished }, ... ] }
    Sorted by last_t desc.
    """
    out = []
    for sid, events in SESSION_EVENTS.items():
        last_t = 0
        if events:
            try:
                last_t = int(max([ev.get("t", 0) for ev in events if isinstance(ev, dict)]))
            except Exception:
                last_t = 0
        out.append({
            "session_id": sid,
            "score": int(SESSION_SCORE.get(sid, 0)),
            "last_t": last_t,
            "finished": bool(SESSION_FINISHED.get(sid, False))
        })
    out = sorted(out, key=lambda x: x["last_t"], reverse=True)
    return jsonify({"sessions": out}), 200

# ---------------- CLI: small helper to print sessions (optional) ----------------
if __name__ == "__main__":
    print(f"[INFO] Starting backend on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=True, threaded=True)
