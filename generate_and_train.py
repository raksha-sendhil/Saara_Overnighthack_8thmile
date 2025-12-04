import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import random

# ============================================================
#  FEATURE DEFINITIONS (must match backend.py)
# ============================================================
FEATURE_COLUMNS = [
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

# ============================================================
#  SYNTHETIC DATASET GENERATION
# ============================================================

def generate_sample(is_cheater):
    """
    Create 1 synthetic feature row.
    Patterns:
      - cheaters have more paste/copy/right-click/tab-switch
      - cheaters have lower IKI variance (bot-like)
      - honest users have more natural key & mouse variety
    """
    if is_cheater:
        paste = np.random.poisson(2)
        copy = np.random.poisson(1)
        right = np.random.poisson(1)
        tab = np.random.poisson(2)
        selection = np.random.poisson(3)

        mouse = int(np.random.normal(80, 15))
        keydown = int(np.random.normal(60, 10))

        iki_mean = abs(np.random.normal(110, 25))
        iki_std = abs(np.random.normal(12, 5))  # robotic typing = low variance

        event_rate = mouse + keydown + paste + copy
        event_rate = event_rate / 10.0

    else:
        paste = np.random.poisson(0.1)
        copy = np.random.poisson(0.2)
        right = np.random.poisson(0.1)
        tab = np.random.poisson(0.05)
        selection = np.random.poisson(1)

        mouse = int(np.random.normal(120, 20))
        keydown = int(np.random.normal(90, 15))

        iki_mean = abs(np.random.normal(180, 40))
        iki_std = abs(np.random.normal(40, 10))  # human typing variance is higher

        event_rate = mouse + keydown
        event_rate = event_rate / 10.0

    return [
        paste,
        copy,
        right,
        tab,
        selection,
        mouse,
        keydown,
        iki_mean,
        iki_std,
        event_rate
    ]

# Generate final dataset
N = 2500   # total rows
cheat_fraction = 0.35

rows = []
labels = []

for i in range(N):
    is_cheater = (random.random() < cheat_fraction)
    rows.append(generate_sample(is_cheater))
    labels.append(1 if is_cheater else 0)

df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
df["label"] = labels

df.to_csv("synthetic_dataset.csv", index=False)
print("[INFO] Synthetic dataset written to synthetic_dataset.csv")

# ============================================================
#  TRAINING
# ============================================================

X = df[FEATURE_COLUMNS]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n[INFO] Classification report on synthetic data:")
print(classification_report(y_test, y_pred, digits=4))

# Save the trained model
joblib.dump(model, "model.pkl")
print("[INFO] model.pkl saved successfully.")
