# Project 2 — Data Classification Using AI

A supervised machine learning pipeline built with Python and scikit-learn, classifying Iris flowers using the K-Nearest Neighbors (KNN) algorithm.

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 96.67% |
| F1 Score (weighted) | 0.9666 |
| Optimal K | 1 (auto-tuned) |

## How to run

### 1. Install dependencies
```
pip install scikit-learn numpy
```

### 2. Run the classifier
```
python classifier.py
```

## Pipeline (IPO Framework)
- **Input** — Iris dataset (150 samples, 4 features, 3 classes) + StandardScaler
- **Process** — 80/20 train-test split + automatic K tuning (K=1 to 15) + KNN training
- **Output** — Accuracy, F1 Score, Confusion Matrix, per-class report, live predictor

## Key concepts demonstrated
- Supervised learning vs rule-based AI
- Feature scaling (StandardScaler) to prevent distance bias
- Train/test split with stratification to prevent order bias
- KNN algorithm (proximity principle, majority vote)
- Model validation: accuracy vs F1 score (why accuracy alone can be misleading)
- Confusion matrix: TP, FP, FN, TN
- Hyperparameter tuning: finding the optimal K via elbow method
