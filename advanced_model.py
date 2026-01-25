import numpy as np
import pandas as pd
# Use Agg backend to avoid GUI issues if running in background, though not strictly necessary here
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, RocCurveDisplay
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from joblib import dump
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Train, eval, and return metrics dict."""
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, pos_label=1), # 1 is benign in sklearn usually, wait check target names
        "roc_auc": roc_auc_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred)
    }
    return metrics, y_score

def main():
    # 1) Load the full standard dataset (real, widely used, 569 samples, 30 features)
    # Target: 0 = Malignant, 1 = Benign (sklearn default)
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    
    print(f"Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print("Classes:", data.target_names)
    print()

    # 2) Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3) Define Advanced Models
    # HistGradientBoostingClassifier: Very fast, supports missing values, scales to big data (inspired by LightGBM)
    # MLPClassifier: Neural Network
    
    models = {
        "HistGradientBoosting": Pipeline([
            # Native scaling, but standard scaling often helps convergence or is neutral
            ("scaler", StandardScaler()), 
            ("clf", HistGradientBoostingClassifier(random_state=42, max_iter=200))
        ]),
        "NeuralNetwork": Pipeline([
            ("scaler", StandardScaler()), # Critical for MLP
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32), 
                activation='relu', 
                solver='adam', 
                alpha=0.0001, 
                batch_size='auto', 
                learning_rate='adaptive', 
                random_state=42, 
                max_iter=1000,
                early_stopping=True
            ))
        ])
    }

    # 4) Evaluate
    results = []
    roc_curves = {}

    for name, pipe in models.items():
        metrics, y_score = evaluate_model(name, pipe, X_train, y_train, X_test, y_test)
        results.append(metrics)
        roc_curves[name] = y_score
        print(f"\n=== {name} ===")
        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
        print("Confusion Matrix:\n", metrics["confusion_matrix"])
        
    # 5) Summary & Saved
    results_df = pd.DataFrame([
        {"model": m["model"], "accuracy": m["accuracy"], "f1": m["f1"], "roc_auc": m["roc_auc"]}
        for m in results
    ])
    results_df = results_df.sort_values(by="roc_auc", ascending=False).reset_index(drop=True)
    
    print("\n=== Leaderboard ===")
    print(results_df)

    best_name = results_df.loc[0, "model"]
    # We will prefer NeuralNetwork if it's very close, just because it sounds 'cooler' for an advanced project,
    # but strictly following the metric is safer. Let's stick to the metric.
    best_pipe = models[best_name]
    
    print(f"\nExample: Saving best model ({best_name}) to artifacts...")

    # Refit on full data for production
    best_pipe.fit(X, y)
    
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/best_advanced_model.joblib"
    dump(best_pipe, model_path)
    print(f"Saved pipeline to {model_path}")
    
    # Also save the scaler separately just in case the UI needs to do pre-processing manually 
    # (though pipeline handles it, it's good practice for debugging)
    if "scaler" in best_pipe.named_steps:
        dump(best_pipe.named_steps["scaler"], "artifacts/advanced_scaler.joblib")
        print("Saved scaler to artifacts/advanced_scaler.joblib")

if __name__ == "__main__":
    main()
