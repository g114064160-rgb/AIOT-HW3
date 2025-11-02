import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


DEFAULT_COLUMN_NAMES = ["label", "message"]


def load_from_url(url: str, header: bool = False) -> pd.DataFrame:
    """Load dataset from a raw CSV URL. The CSV has no header by default.

    Args:
        url: raw CSV URL
        header: whether the CSV has a header row

    Returns:
        DataFrame with columns ['label','message']
    """
    df = pd.read_csv(url, header=0 if header else None, names=DEFAULT_COLUMN_NAMES)
    return df


def load_from_string(csv_text: str) -> pd.DataFrame:
    """Load dataset from a CSV text (useful for tests)."""
    from io import StringIO

    return pd.read_csv(StringIO(csv_text), header=None, names=DEFAULT_COLUMN_NAMES)


def prepare_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Map labels to binary and split into train/test.

    Labels are expected to be strings like 'ham' and 'spam'.
    Returns X_train, X_test, y_train, y_test
    """
    df = df.copy()
    # Normalize label column
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    label_map = {'ham': 0, 'spam': 1}
    df['y'] = df['label'].map(label_map)
    if df['y'].isnull().any():
        # try to coerce numeric labels
        df['y'] = pd.to_numeric(df['label'], errors='coerce').fillna(df['y'])

    X = df['message']
    y = df['y'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test


def ensure_artifacts_dir(path: str = "artifacts") -> str:
    os.makedirs(path, exist_ok=True)
    return path
