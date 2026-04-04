"""
train_model.py
Trains a Random Forest classifier on symptom-disease data and saves model + encoder.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# ── Load Dataset ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"))

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Diseases: {df['disease'].unique()}")

# ── Features & Labels ────────────────────────────────────────────────────────
SYMPTOM_COLS = [c for c in df.columns if c != "disease"]
X = df[SYMPTOM_COLS].values
y = df["disease"].values

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42
)

# ── Train Model ───────────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Save Artifacts ────────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

with open(os.path.join(BASE_DIR, "symptom_cols.pkl"), "wb") as f:
    pickle.dump(SYMPTOM_COLS, f)

print("\n✅ Saved: model.pkl, label_encoder.pkl, symptom_cols.pkl")
