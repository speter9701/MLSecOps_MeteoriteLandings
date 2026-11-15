from pathlib import Path

import pandas as pd
import pytest

from meteorite.data import load_raw


def test_load_raw_reads_csv(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(source, index=False)
    assert load_raw(source)["a"].tolist() == [1, 2]


def test_load_raw_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        load_raw(tmp_path / "nope.csv")
