# ============================================================
# DecodeLabs — Project 2: Data Classification Using AI
# Algorithm : K-Nearest Neighbors (KNN)
# Dataset   : Iris Benchmark (150 samples, 3 classes, 4 features)
# Pipeline  : IPO Framework — Input → Process → Output
# ============================================================

# --- IMPORTS -------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import numpy as np

# ============================================================
# PHASE 1 — INPUT: Load & Understand the Dataset
# ============================================================

print("=" * 60)
print("  DecodeLabs — Project 2: Data Classification Using AI")
print("  Algorithm: K-Nearest Neighbors | Dataset: Iris")
print("=" * 60)

# Load the Iris benchmark dataset (built into scikit-learn)
iris = load_iris()
X = iris.data        # Features: sepal length, sepal width, petal length, petal width
y = iris.target      # Labels: 0=Setosa, 1=Versicolor, 2=Virginica
class_names = iris.target_names

print("\n[ PHASE 1 — INPUT: Dataset Overview ]")
print(f"  Total samples    : {X.shape[0]}")
print(f"  Features         : {X.shape[1]} ({', '.join(iris.feature_names)})")
print(f"  Classes          : {len(class_names)} ({', '.join(class_names)})")
print(f"  Samples per class: {np.bincount(y).tolist()} (perfectly balanced)")

# Peek at first 3 rows
print("\n  First 3 data rows (raw):")
for i in range(3):
    print(f"    Sample {i+1}: {X[i]} → Class: {class_names[y[i]]}")

# ============================================================
# PHASE 1b — SANITIZATION: Feature Scaling (StandardScaler)
# ============================================================
# KNN is distance-based — unscaled features cause bias.
# StandardScaler transforms each feature to mean=0, variance=1.

print("\n[ PHASE 1b — SANITIZATION: Feature Scaling ]")

# Split BEFORE scaling to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,       # 80% train / 20% test
    random_state=42,      # reproducibility seed
    stratify=y            # keep class balance in both splits
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
X_test_scaled  = scaler.transform(X_test)        # apply to test (no leakage)

print(f"  Training set size : {X_train_scaled.shape[0]} samples (80%)")
print(f"  Testing set size  : {X_test_scaled.shape[0]} samples (20%)")
print(f"  Scaling applied   : mean=0, variance=1 per feature")

# ============================================================
# PHASE 2 — PROCESS: Find Optimal K + Train KNN
# ============================================================

print("\n[ PHASE 2 — PROCESS: Finding Optimal K ]")

# Test k values from 1 to 15 and pick the one with best F1 score
best_k    = 1
best_f1   = 0.0
k_results = {}

for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    preds = knn.predict(X_test_scaled)
    f1    = f1_score(preds, y_test, average="weighted")
    k_results[k] = round(f1, 4)
    if f1 > best_f1:
        best_f1 = f1
        best_k  = k

print(f"  K tested          : 1 to 15")
print(f"  Best K found      : K={best_k} (F1={best_f1:.4f})")
print(f"  K scan results    : {k_results}")

# Train final model with optimal K
print(f"\n  Training final KNN model with K={best_k}...")
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)
print("  Model training complete.")

# ============================================================
# PHASE 3 — OUTPUT: Evaluate, Validate & Report
# ============================================================

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)
f1       = f1_score(y_test, predictions, average="weighted")
cm       = confusion_matrix(y_test, predictions)

print("\n[ PHASE 3 — OUTPUT: Model Validation ]")
print(f"\n  Accuracy  : {accuracy * 100:.2f}%")
print(f"  F1 Score  : {f1:.4f} (weighted average)")

# Confusion Matrix
print("\n  Confusion Matrix:")
print(f"  {'':12s} | {'Pred Setosa':>12} | {'Pred Versic':>12} | {'Pred Virgn':>12}")
print("  " + "-" * 52)
for i, row in enumerate(cm):
    print(f"  {'True '+class_names[i]:12s} | {row[0]:>12} | {row[1]:>12} | {row[2]:>12}")

# Per-class classification report
print("\n  Per-Class Classification Report:")
print(classification_report(y_test, predictions, target_names=class_names))

# ============================================================
# BONUS — Live Prediction: Test with Custom Input
# ============================================================

print("[ BONUS — Live Flower Predictor ]")
print("  Enter flower measurements to get a real-time prediction.")
print("  (Type 'skip' to bypass this section)\n")

sample_flowers = [
    [5.1, 3.5, 1.4, 0.2],   # typical Setosa
    [6.0, 2.9, 4.5, 1.5],   # typical Versicolor
    [6.7, 3.1, 5.6, 2.4],   # typical Virginica
]

print("  Auto-predicting 3 reference flowers:")
for flower in sample_flowers:
    scaled  = scaler.transform([flower])
    pred    = model.predict(scaled)[0]
    probs   = model.predict_proba(scaled)[0]
    conf    = max(probs) * 100
    print(f"    Input {flower} → Predicted: {class_names[pred]:12s} (confidence: {conf:.1f}%)")

# Interactive input
user_input = input("\n  Try your own flower? Enter 4 values (e.g. 5.1 3.5 1.4 0.2) or 'skip': ")
if user_input.strip().lower() != "skip":
    try:
        values  = list(map(float, user_input.strip().split()))
        scaled  = scaler.transform([values])
        pred    = model.predict(scaled)[0]
        probs   = model.predict_proba(scaled)[0]
        conf    = max(probs) * 100
        print(f"\n  Your flower → Predicted class: {class_names[pred].upper()} ({conf:.1f}% confidence)")
    except Exception:
        print("  Invalid input — skipped.")

print("\n" + "=" * 60)
print("  Project 2 complete. Model trained, tested, and validated.")
print("=" * 60)