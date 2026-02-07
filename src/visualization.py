"""Visualization helpers for model outputs."""
from typing import Sequence
import matplotlib.pyplot as plt
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def plot_feature_importance(importances: Sequence[float], feature_names: Sequence[str], outpath: str = "outputs/figures/feature_importance.png"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    idx = np.argsort(importances)[::-1]
    sorted_imp = np.array(importances)[idx]
    sorted_names = np.array(feature_names)[idx]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_imp)), sorted_imp, align="center")
    plt.xticks(range(len(sorted_imp)), sorted_names, rotation=90)
    plt.title("Feature importances")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    logger.info("Saved feature importance to %s", outpath)


def plot_predictions(y_true, y_pred, outpath: str = "outputs/figures/predictions_parity.png"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    plt.plot(lims, lims, "--", color="gray")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Prediction parity")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    logger.info("Saved prediction parity plot to %s", outpath)
