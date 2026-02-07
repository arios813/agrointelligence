"""Exploratory data analysis functions.

Plots and summary statistics for cattle weight data.
"""
from typing import List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger(__name__)


def summary_stats(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    return df[numeric_cols].describe().transpose()


def correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str], outpath: str = "outputs/figures/corr_heatmap.png"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag")
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    logger.info("Saved correlation heatmap to %s", outpath)


def plot_growth_curves(df: pd.DataFrame, weight_cols: List[str], id_col: str = "id", outpath: str = "outputs/figures/growth_curves.png"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    # Melt to long format
    long = df[[id_col] + weight_cols].melt(id_vars=[id_col], value_vars=weight_cols, var_name="weigh_time", value_name="weight")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=long, x="weigh_time", y="weight", estimator="mean")
    plt.xticks(rotation=45)
    plt.ylabel("Weight (kg)")
    plt.title("Average growth curve")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    logger.info("Saved growth curves to %s", outpath)


def plot_weight_distributions(df: pd.DataFrame, weight_cols: List[str], outdir: str = "outputs/figures"):
    os.makedirs(outdir, exist_ok=True)
    for c in weight_cols:
        path = os.path.join(outdir, f"dist_{c}.png")
        plt.figure(figsize=(8, 4))
        sns.histplot(df[c].dropna(), kde=True)
        plt.title(f"Distribution of {c}")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logger.info("Saved distribution plot for %s -> %s", c, path)
