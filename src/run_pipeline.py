"""Orchestrate full pipeline: load, preprocess, EDA, train, save outputs.

Usage:
    python src/run_pipeline.py --data /path/to/cattle.csv --out outputs/
"""
import argparse
import logging
import os
import pandas as pd

from data_loader import load_data
from preprocessing import (
    standardize_columns,
    parse_dates,
    infer_weight_columns,
    clean_weight_columns,
    select_latest_weight,
    feature_engineering,
    build_preprocessor,
)
from eda import summary_stats, correlation_heatmap, plot_growth_curves, plot_weight_distributions
from models import train_and_evaluate
from visualization import plot_feature_importance, plot_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(data_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    df = load_data(data_path)

    df = standardize_columns(df)
    df = parse_dates(df)

    weight_cols = infer_weight_columns(df)
    logger.info("Inferred weight columns: %s", weight_cols)
    df = clean_weight_columns(df, weight_cols)

    df = select_latest_weight(df, weight_cols, target_name="final_weight")
    df = feature_engineering(df, weight_cols)

    # Basic EDA
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    stats = summary_stats(df, numeric_cols)
    stats.to_csv(os.path.join(outdir, "summary_stats.csv"))
    correlation_heatmap(df, numeric_cols, outpath=os.path.join(outdir, "figures/corr_heatmap.png"))
    if weight_cols:
        plot_growth_curves(df, weight_cols, id_col=df.columns[0], outpath=os.path.join(outdir, "figures/growth_curves.png"))
        plot_weight_distributions(df, weight_cols, outdir=os.path.join(outdir, "figures"))

    # Prepare features: choose sensible defaults
    target_col = "final_weight"
    candidate_numeric = ["early_weight_mean", "early_weight_std"]
    candidate_categorical = [c for c in ["combination", "owner", "category"] if c in df.columns]

    # Fill simple missing values
    model_df = df[[target_col] + candidate_numeric + candidate_categorical].copy()
    model_df = model_df.dropna(subset=[target_col])
    model_df[candidate_numeric] = model_df[candidate_numeric].fillna(model_df[candidate_numeric].median())
    model_df[candidate_categorical] = model_df[candidate_categorical].fillna("unknown")

    # For modeling convenience, convert categorical columns to string
    for c in candidate_categorical:
        model_df[c] = model_df[c].astype(str)

    # Build preprocessor and expand features into numeric matrix for models
    preprocessor = build_preprocessor(model_df, categorical_cols=candidate_categorical, numeric_cols=candidate_numeric)
    # Fit transform
    X = preprocessor.fit_transform(model_df.drop(columns=[target_col]))

    # Build flattened feature names
    feature_names = []
    # numeric names
    feature_names.extend(candidate_numeric)
    # categorical names from onehot
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cats = ohe.get_feature_names_out(candidate_categorical)
        feature_names.extend(list(cats))
    except Exception:
        # Best-effort fallback
        feature_names.extend(candidate_categorical)

    # Convert X to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names, index=model_df.index)
    y = model_df[target_col]

    # Train and evaluate
    results = train_and_evaluate(pd.concat([X_df, y], axis=1), feature_names, target_col, output_dir=outdir)

    # Save predictions
    preds = results["predictions"].reset_index()
    preds.to_csv(os.path.join(outdir, "predictions.csv"), index=False)
    logger.info("Saved predictions to %s", os.path.join(outdir, "predictions.csv"))

    # Save feature importance figure if available
    if results["results"].get("feature_importances") is not None:
        plot_feature_importance(results["results"]["feature_importances"], feature_names, outpath=os.path.join(outdir, "figures/feature_importance.png"))
    plot_predictions(preds["y_true"], preds["y_pred_rf"], outpath=os.path.join(outdir, "figures/predictions_parity.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cattle ML pipeline")
    parser.add_argument("--data", type=str, help="Path to CSV data file", required=True)
    parser.add_argument("--out", type=str, help="Output directory", default="outputs")
    args = parser.parse_args()
    main(args.data, args.out)
