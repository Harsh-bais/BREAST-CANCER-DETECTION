
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from joblib import dump, load
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Train, eval, and return metrics dict."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Some models need decision_function, others have predict_proba
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        # fallback (not ideal), treat predictions as scores
        y_score = y_pred

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, target_names=["malignant(0)", "benign(1)"])
    }
    return metrics, y_score

def main():
    # 1) Load data (from CSV provided by user)
    csv_path = "Reduced_Breast_Cancer_Dataset.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path} in the current directory.")
    df = pd.read_csv(csv_path)
    if "diagnosis" not in df.columns:
        raise ValueError("CSV must contain a 'diagnosis' column as the target.")
    # If diagnosis is strings like 'M'/'B', map to 0/1. If already numeric, this is a no-op.
    if df["diagnosis"].dtype == object:
        df["diagnosis"] = df["diagnosis"].map({"M": 0, "B": 1}).astype(int)
    y = df["diagnosis"].astype(int).rename("target")  # 0/1 target
    X = df.drop(columns=["diagnosis"])  # all remaining columns are features

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print("Classes: 0=malignant, 1=benign")
    print()

    # Drop up to 3-4 features with the LOWEST correlation to the target
    max_drop = 4
    target_corr_series = (
        X.assign(target=y).corr()["target"].drop("target").abs().sort_values(ascending=True)
    )
    to_drop_auto = list(target_corr_series.index[:max_drop])
    if to_drop_auto:
        X = X.drop(columns=to_drop_auto)
        print(f"Dropped (lowest |corr with target|): {to_drop_auto}")
        print(f"New feature count: {X.shape[1]}")
        print()

    # 2) Train/Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3) Define classic models with sensible defaults
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", C=1.0))
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, class_weight="balanced"))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),  # scaler harmless; RF doesn't need it
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=None, random_state=42, class_weight="balanced_subsample"
            ))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),  # tree-based; scaler not required
            ("clf", GradientBoostingClassifier(random_state=42))
        ]),
    }

    # 4) Cross-validate (quick sanity check)
    print("Cross-validation (Stratified 5-fold) ROC-AUC on training set:")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, pipe in models.items():
        # Need probability/decision function for ROC-AUC; use predict_proba if available
        scoring = "roc_auc"
        scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring=scoring)
        print(f"  {name:>16}: mean={scores.mean():.4f} ± {scores.std():.4f}")
    print()

    # 5) Fit, evaluate on held-out test set + collect ROC curves
    results = []
    roc_curves = {}

    for name, pipe in models.items():
        metrics, y_score = evaluate_model(name, pipe, X_train, y_train, X_test, y_test)
        results.append(metrics)
        roc_curves[name] = y_score
        print(f"=== {name} ===")
        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"F1-score  : {metrics['f1']:.4f}")
        print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
        print("Confusion matrix:\n", metrics["confusion_matrix"])
        print("Classification report:\n", metrics["classification_report"])
        print()

    # 6) Pick best model by ROC-AUC (primary), then F1 as tiebreaker
    results_df = pd.DataFrame([
        {"model": m["model"], "accuracy": m["accuracy"], "f1": m["f1"], "roc_auc": m["roc_auc"]}
        for m in results
    ])
    
    results_df = results_df.sort_values(by=["roc_auc", "f1"], ascending=False).reset_index(drop=True)

    print("=== Summary (Test Set) ===")
    print(results_df)
    print()

    best_name = results_df.loc[0, "model"]
    best_pipe = models[best_name]
    print(f"Best model by ROC-AUC/F1: {best_name}")

    # 7) Plot ROC curves
    plt.figure(figsize=(8, 6))
    for name, y_score in roc_curves.items():
        RocCurveDisplay.from_predictions(
            y_test, y_score, name=name, plot_chance_level=True
        )
    plt.title("ROC Curves - Breast Cancer (Classic ML)")
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    print("Saved ROC curves to roc_curves.png")

    # 8) Save best model + an explicit scaler for production usage
    # For convenience, also extract fitted scaler if present.
    fitted_pipe = best_pipe.fit(X, y)  # fit on full data for final model
    scaler = None
    if "scaler" in fitted_pipe.named_steps:
        scaler = fitted_pipe.named_steps["scaler"]

    os.makedirs("artifacts", exist_ok=True)
    dump(fitted_pipe, "artifacts/best_model.joblib")
    if scaler is not None:
        dump(scaler, "artifacts/scaler.joblib")

    print("Saved best pipeline to artifacts/best_model.joblib")
    if scaler is not None:
        print("Saved scaler to artifacts/scaler.joblib")

    # 9) Example: load and predict on first 5 test samples
    loaded_pipe = load("artifacts/best_model.joblib")
    sample = X_test.iloc[:5]
    sample_pred = loaded_pipe.predict(sample)
    sample_proba = (loaded_pipe.predict_proba(sample)[:, 1]
                    if hasattr(loaded_pipe, "predict_proba") else None)
    print("\nExample predictions on 5 samples:")
    print("Predicted classes:", sample_pred.tolist())
    if sample_proba is not None:
        print("Predicted benign probabilities:", np.round(sample_proba, 4).tolist())

if __name__ == "__main__":
    main()
