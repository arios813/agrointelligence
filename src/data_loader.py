"""Data loading utilities.

Functions:
- load_data(path): read CSV into pandas DataFrame with basic normalization
"""
from typing import Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_data(path: str, dayfirst: bool = True, encoding: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV file into a DataFrame and standardize column names.

    Parameters
    ----------
    path : str
        Path to CSV file.
    dayfirst : bool
        Whether dates are day-first when parsing.
    encoding : Optional[str]
        File encoding; if None, pandas will infer.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with standardized column names.
    """
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path, encoding=encoding)

    # Standardize column names: lowercase, strip, replace spaces with underscores
    df.columns = ["_".join(c.strip().lower().split()) for c in df.columns.astype(str)]

    # Attempt to parse obvious date columns later in preprocessing
    return df
