"""Data cleaning and preprocessing utilities.

Provides functions to standardize column names, parse dates, clean weight columns,
infer the latest weight per row, and prepare feature matrices and transformers.
"""
from typing import List, Tuple, Optional
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case lower strings."""
    df = df.copy()
    df.columns = ["_".join(c.strip().lower().split()) for c in df.columns.astype(str)]
    return df


def parse_dates(df: pd.DataFrame, date_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    if date_cols is None:
        date_candidates = [c for c in df.columns if "birth" in c or "date" in c]
    else:
        date_candidates = date_cols

    for col in date_candidates:
        try:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
        except Exception:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_weight_value(x: object) -> Optional[float]:
    if pd.isna(x):
        return np.nan
    s = str(x)
    # Remove non-digit and non-dot characters (e.g., 'kg', commas)
    s = re.sub(r"[^0-9.\-]", "", s)
    if s == "":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def infer_weight_columns(df: pd.DataFrame) -> List[str]:
    """Return likely weight columns by heuristic matching.

    Heuristics: column name contains 'weigh' or 'weight', or values contain 'kg'.
    """
    candidates = [c for c in df.columns if "weigh" in c or "weight" in c]
    if candidates:
        return candidates

    # Fallback: check for 'kg' in values
    candidates = []
    sample = df.head(50).astype(str)
    for c in df.columns:
        if sample[c].str.contains("kg", na=False).any():
            candidates.append(c)
    return candidates


def clean_weight_columns(df: pd.DataFrame, weight_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in weight_cols:
        df[c] = df[c].apply(clean_weight_value)
    return df


def select_latest_weight(df: pd.DataFrame, weight_cols: List[str], target_name: str = "final_weight") -> pd.DataFrame:
    """Select the latest non-null weight across ordered weight_cols as the target.

    Assumes weight_cols are ordered chronologically left-to-right as in the CSV.
    """
    df = df.copy()
    df[target_name] = df[weight_cols].bfill(axis=1).iloc[:, 0]
    # Alternatively, pick last non-null: use apply with last valid
    def last_nonnull(row):
        for c in reversed(weight_cols):
            v = row.get(c)
            if pd.notna(v):
                return v
        return np.nan

    df[target_name] = df.apply(last_nonnull, axis=1)
    return df


def feature_engineering(df: pd.DataFrame, weight_cols: List[str], birth_col: str = "birth_date") -> pd.DataFrame:
    df = df.copy()
    # Age at last weight (days)
    if birth_col in df.columns and np.issubdtype(df[birth_col].dtype, np.datetime64):
        # Try to extract a proxy date for last weight from column names if possible
        # Otherwise age will be NA
        df["age_at_last_weight_days"] = pd.NA
        # If weight column names encode dates, user can extend this function
    # Early-weight summary features
    df["early_weight_mean"] = df[weight_cols].iloc[:, :max(1, min(3, len(weight_cols)))].mean(axis=1)
    df["early_weight_std"] = df[weight_cols].iloc[:, :max(1, min(3, len(weight_cols)))].std(axis=1)
    return df


def build_preprocessor(df: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    """Create a ColumnTransformer for numeric and categorical preprocessing."""
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    # Use `sparse_output=False` for compatibility with newer scikit-learn
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor
