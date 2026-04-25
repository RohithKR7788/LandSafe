"""
train.py
~~~~~~~~
End-to-end training pipeline for LandSafe.

Usage
-----
    python generate_data.py          # creates data/processed/kerala_landslide_dataset.csv
    python train.py                  # trains model, prints metrics, saves to models/
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

from src import (
    load_and_prepare, build_model,
    evaluate, compute_shap, save_model,
)

DATA_PATH    = "data/processed/kerala_landslide_dataset.csv"
REPORTS_PATH = Path("models")


def main():
    print("=" * 60)
    print("  LandSafe — Landslide Prediction Training Pipeline")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    if not Path(DATA_PATH).exists():
        print("\n[!] Dataset not found. Running generate_data.py first...\n")
        import generate_data
        generate_data.generate().to_csv(DATA_PATH, index=False)

    print(f"\n[1/5] Loading dataset from {DATA_PATH}")
    X, y = load_and_prepare(DATA_PATH)
    print(f"      Shape: {X.shape}  |  Positive rate: {y.mean()*100:.1f}%")

    # ── 2. Train / test split ────────────────────────────────────────────────
    print("\n[2/5] Splitting data (80% train / 20% test, stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── 3. Cross-validation ──────────────────────────────────────────────────
    print("\n[3/5] Running 5-fold stratified cross-validation...")
    pipeline = build_model()
    cv_scores = evaluate(pipeline, X_train, y_train)
    print(f"      ROC-AUC : {cv_scores['roc_auc_mean']:.4f} ± {cv_scores['roc_auc_std']:.4f}")
    print(f"      F1      : {cv_scores['f1_mean']:.4f} ± {cv_scores['f1_std']:.4f}")
    print(f"      Accuracy: {cv_scores['acc_mean']:.4f}")

    # ── 4. Final fit + held-out evaluation ───────────────────────────────────
    print("\n[4/5] Training final model on full training set...")
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n      ── Test-set Classification Report ──")
    print(classification_report(y_test, y_pred,
                                 target_names=["No landslide", "Landslide"]))
    print(f"      Test ROC-AUC: {roc_auc:.4f}")

    # ── 5. Save artefacts ─────────────────────────────────────────────────────
    print("\n[5/5] Saving model and plots...")
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    save_model(pipeline)

    # Confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["No landslide", "Landslide"],
        ax=axes[0], colorbar=False,
    )
    axes[0].set_title("Confusion Matrix")

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title(f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / "evaluation.png", dpi=150)
    print(f"      Evaluation plot saved → {REPORTS_PATH}/evaluation.png")

    # Feature importance via XGBoost
    xgb_model = pipeline.named_steps["model"]
    importance = xgb_model.feature_importances_

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    n_features = min(15, len(importance))
    idx = np.argsort(importance)[-n_features:]
    ax2.barh(range(n_features), importance[idx], color="#e74c3c", alpha=0.8)
    ax2.set_yticks(range(n_features))
    ax2.set_yticklabels([f"Feature {i}" for i in idx])
    ax2.set_title("Top Feature Importances (XGBoost)")
    ax2.set_xlabel("Importance score")
    plt.tight_layout()
    plt.savefig(REPORTS_PATH / "feature_importance.png", dpi=150)
    print(f"      Feature importance plot saved → {REPORTS_PATH}/feature_importance.png")

    # Save metrics JSON
    metrics = {**cv_scores, "test_roc_auc": roc_auc}
    with open(REPORTS_PATH / "metrics.json", "w") as f:
        json.dump({k: round(float(v), 4) for k, v in metrics.items()}, f, indent=2)

    print("\n✅ Training complete! Run the dashboard with:")
    print("   streamlit run app.py\n")


if __name__ == "__main__":
    main()
