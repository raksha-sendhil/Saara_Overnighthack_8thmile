# Behavioral Analytics for Online Exam Integrity

This project is a privacy-friendly cheating-detection system for online examinations.  
The system does not use webcam, microphone, or screen recording.  
It relies only on browser interaction patterns to estimate a real-time risk score.

video link:
https://drive.google.com/file/d/1PKlBJu2witDwU1nAd_wV2XVP_WrJ3aOc/view?usp=sharing

ppt link:
https://drive.google.com/file/d/1PKlBJu2witDwU1nAd_wV2XVP_WrJ3aOc/view?usp=sharing

## Core Features
- Test Platform with disabled copy-paste and right click functionality
- Detects tab switching and window visibility changes
- Captures cursor movement patterns and drag behaviour
- Tracks keystroke timing (inter-key intervals and bursts)
- Computes a cheating-risk score (0–100)
- Displays score and recent events on a proctor dashboard

## System Components
1. Frontend (HTML + JavaScript)  
   - Simple exam page (questions + answer fields)  
   - JS event listeners collect interaction data  
   - Events are batched and sent to backend every few seconds

2. Backend (Python + Flask)  
   - Receives event streams  
   - Extracts behavioral features  
   - Runs a lightweight ML model (Random Forest)  
   - Updates and returns a live risk score

3. Dashboard (Streamlit)  
   - Polls backend for the latest score  
   - Shows score trends and flagged behaviours  
   - Can show a warning if risk crosses a threshold

## Data Flow
Browser events → Flask API (/api/events) → Feature extraction → ML scoring → Streamlit dashboard query (/api/score)

## Tech Stack
- HTML, CSS, JavaScript (event capture)
- Flask (backend API)
- Python (pandas, numpy)
- scikit-learn (risk model)
- Streamlit (proctor dashboard)


