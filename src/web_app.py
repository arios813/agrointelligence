"""Streamlit web app to run the cattle ML pipeline interactively.

Run:
    streamlit run src/web_app.py

Features:
- Upload or select CSV
- Display summary stats and correlation heatmap
- Train RandomForest and LinearRegression models
- Show metrics, feature importance and prediction downloads
"""
import streamlit as st
import pandas as pd
import os
import io
import logging

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
from breeding_evaluator import CombinationEvaluator

logging.basicConfig(level=logging.INFO)


DEFAULT_CSV = "/Users/andrewrios/Downloads/pukavy_bulls.csv"


st.set_page_config(page_title="Cattle ML demo", layout="wide")
st.title("Cattle Breeding Analytics — Demo Web App")

uploaded = st.file_uploader("Upload a CSV (or leave empty to use detected sample)", type=["csv"]) 
use_default = st.checkbox("Use default sample CSV", value=True)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_default and os.path.exists(DEFAULT_CSV):
    df = load_data(DEFAULT_CSV)
else:
    st.warning("No CSV provided and default not found. Upload a file to continue.")
    st.stop()

# Preprocess
df = standardize_columns(df)
df = parse_dates(df)
weight_cols = infer_weight_columns(df)
df = clean_weight_columns(df, weight_cols)
df = select_latest_weight(df, weight_cols, target_name="final_weight")
df = feature_engineering(df, weight_cols)

st.header("Data preview")
st.dataframe(df.head(10))

st.header("Summary statistics")
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
stats = summary_stats(df, numeric_cols)
st.dataframe(stats)

st.markdown("### Correlation heatmap")
corr_path = "outputs/figures/corr_heatmap_streamlit.png"
correlation_heatmap(df, numeric_cols, outpath=corr_path)
st.image(corr_path, use_column_width=True)

if weight_cols:
    st.markdown("### Growth curve (average)")
    gc_path = "outputs/figures/growth_curves_streamlit.png"
    plot_growth_curves(df, weight_cols, id_col=df.columns[0], outpath=gc_path)
    st.image(gc_path, use_column_width=True)

    st.markdown("### Weight distributions")
    plot_weight_distributions(df, weight_cols, outdir="outputs/figures")
    cols = st.columns(3)
    for i, c in enumerate(weight_cols):
        img_path = os.path.join("outputs/figures", f"dist_{c}.png")
        if os.path.exists(img_path):
            cols[i % 3].image(img_path, caption=c)

st.header("Model training & evaluation")
if st.button("Train models"):
    with st.spinner("Training models..."):
        # Simple feature selection
        target_col = "final_weight"
        candidate_numeric = ["early_weight_mean", "early_weight_std"]
        candidate_categorical = [c for c in ["combination", "owner", "category"] if c in df.columns]

        model_df = df[[target_col] + candidate_numeric + candidate_categorical].copy()
        model_df = model_df.dropna(subset=[target_col])
        model_df[candidate_numeric] = model_df[candidate_numeric].fillna(model_df[candidate_numeric].median())
        model_df[candidate_categorical] = model_df[candidate_categorical].fillna("unknown")
        for c in candidate_categorical:
            model_df[c] = model_df[c].astype(str)

        preprocessor = build_preprocessor(model_df, categorical_cols=candidate_categorical, numeric_cols=candidate_numeric)
        X = preprocessor.fit_transform(model_df.drop(columns=[target_col]))

        # Build feature names
        feature_names = []
        feature_names.extend(candidate_numeric)
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cats = ohe.get_feature_names_out(candidate_categorical)
            feature_names.extend(list(cats))
        except Exception:
            feature_names.extend(candidate_categorical)

        X_df = pd.DataFrame(X, columns=feature_names, index=model_df.index)
        y = model_df[target_col]

        # Train
        res = train_and_evaluate(pd.concat([X_df, y], axis=1), feature_names, target_col, output_dir="outputs")

    st.success("Training complete")
    st.subheader("Metrics")
    st.json(res["results"])

    # Feature importance image
    if res["results"].get("feature_importances") is not None:
        fi_path = "outputs/figures/feature_importance_streamlit.png"
        plot_feature_importance(res["results"]["feature_importances"], feature_names, outpath=fi_path)
        st.image(fi_path, caption="Feature importances")

    # Predictions download
    preds_df = res["predictions"].reset_index()
    csv_bytes = preds_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")


# Breeding Genetics Module
st.header("🧬 Breeding Genetics Evaluator")
st.markdown("Score proposed breeding combinations based on champion bloodlines and genetic diversity.")

evaluator = CombinationEvaluator(
    champions=["King George", "Arasunu", "Don Angel", "Nando", "Franchesco"],
    inbreeding_penalty=15.0,
)

# Get unique lineage values from dataset
lineage_options = ["---"] + sorted(df["combination"].dropna().unique().tolist())

col1, col2 = st.columns(2)
with col1:
    parent_a = st.selectbox("Select Parent A lineage", lineage_options, key="parent_a_select")
with col2:
    parent_b = st.selectbox("Select Parent B lineage", lineage_options, key="parent_b_select")

# Allow custom input
st.subheader("Or enter custom lineages manually")
col1, col2 = st.columns(2)
with col1:
    parent_a_custom = st.text_input("Parent A (custom)", placeholder="e.g., 498 x King George", key="parent_a_custom")
with col2:
    parent_b_custom = st.text_input("Parent B (custom)", placeholder="e.g., 1310 x Arasunu", key="parent_b_custom")

# Use custom if provided, otherwise use dropdown
if parent_a_custom and parent_a_custom.strip():
    parent_a = parent_a_custom
elif parent_a == "---":
    parent_a = None

if parent_b_custom and parent_b_custom.strip():
    parent_b = parent_b_custom
elif parent_b == "---":
    parent_b = None

if st.button("Evaluate Breeding Combination"):
    if parent_a and parent_b:
        with st.spinner("Evaluating genetics..."):
            result = evaluator.score_combination(parent_a, parent_b)

        # Display results
        st.subheader("Evaluation Results")
        
        # Score display (large, colored)
        score_color = "green" if result.score >= 70 else ("orange" if result.score >= 50 else "red")
        st.markdown(
            f"<h2 style='color:{score_color};'><b>Score: {result.score:.1f} / 100</b></h2>",
            unsafe_allow_html=True,
        )

        # Explanation
        st.info(result.explanation)

        # Champion matches
        if result.champion_matches:
            st.subheader("Champion Genetics Detected")
            match_str = ", ".join([f"{c} ({count})" for c, count in result.champion_matches.items()])
            st.success(f"Champions found: {match_str}")
        else:
            st.warning("No known champion genetics detected in this combination.")

        # Inbreeding risk
        if result.inbreeding_risk:
            st.error("⚠️ Inbreeding Risk: Duplicate sires detected.")
        else:
            st.success("✓ Genetic diversity looks healthy.")

        # Parsed lineages (for transparency)
        with st.expander("View parsed lineages"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Parent A (parsed)**")
                st.json(result.parent_a_parsed)
            with col2:
                st.write("**Parent B (parsed)**")
                st.json(result.parent_b_parsed)

    else:
        st.warning("Please select or enter both parent lineages to evaluate.")

st.markdown("---")
st.caption("Note: this demo trains models in-memory; for production use, persist models and preprocessors.")
