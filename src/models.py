"""Model training and evaluation.

Provides training pipelines for Random Forest and Linear Regression with CV,
and utilities to extract feature names and importance.
"""
from typing import Tuple, Dict, Any, List
import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def train_and_evaluate(df: pd.DataFrame, feature_cols: List[str], target_col: str, output_dir: str = "outputs") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # Linear regression pipeline (assumes numeric features are scaled in preprocess)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["linear"] = _evaluate(y_test, y_pred_lr)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["random_forest"] = _evaluate(y_test, y_pred_rf)

    # Cross-validation (RF)
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    cv_mae = -cross_val_score(rf, X, y, cv=cv, scoring="neg_mean_absolute_error")
    results["rf_cv_mae_mean"] = float(np.mean(cv_mae))
    results["rf_cv_mae_std"] = float(np.std(cv_mae))

    # Save models
    models_dir = os.path.join(output_dir, "../models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(rf, os.path.join(models_dir, "model_rf.joblib"))
    joblib.dump(lr, os.path.join(models_dir, "model_lr.joblib"))
    logger.info("Saved models to %s", models_dir)

    # Feature importances (RF)
    try:
        importances = rf.feature_importances_
        results["feature_importances"] = importances.tolist()
    except Exception:
        results["feature_importances"] = None

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Return predictions and models for downstream saving
    preds = pd.DataFrame({
        "y_true": y_test,
        "y_pred_rf": y_pred_rf,
        "y_pred_lr": y_pred_lr,
    }, index=y_test.index)

    return {"results": results, "predictions": preds, "rf": rf, "lr": lr}


def _evaluate(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    # Compute RMSE manually to avoid compatibility issues with 'squared' kwarg
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
