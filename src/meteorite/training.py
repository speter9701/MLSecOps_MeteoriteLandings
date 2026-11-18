"""End-to-end training: candidate selection with cross-validation and MLflow tracking."""
import pandas as pd

from meteorite.schemas import FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select the model inputs (numeric columns as float) and the target."""
    features = pd.DataFrame(df[list(FEATURE_COLUMNS)]).copy()
    features[list(NUMERIC_FEATURES)] = features[list(NUMERIC_FEATURES)].astype(float)
    return features, pd.Series(df[TARGET_COLUMN], dtype=float)
