import pandas as pd

from meteorite import training
from meteorite.preprocessing import clean_data


def test_split_features_casts_numeric(raw_frame: pd.DataFrame) -> None:
    x, y = training.split_features(clean_data(raw_frame))
    assert list(x.columns) == ["recclass", "nametype", "fall", "year", "reclat", "reclong"]
    assert x["year"].dtype == float and y.dtype == float and len(x) == len(y)
