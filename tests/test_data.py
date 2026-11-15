from pathlib import Path

import pandas as pd
import pytest

from meteorite.data import load_raw, sample_dataset


def test_load_raw_reads_csv(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(source, index=False)
    assert load_raw(source)["a"].tolist() == [1, 2]


def test_load_raw_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        load_raw(tmp_path / "nope.csv")


def test_sample_is_reproducible() -> None:
    df = pd.DataFrame({"a": range(100)})
    first, second = sample_dataset(df, 10, 1), sample_dataset(df, 10, 1)
    assert len(first) == 10
    assert first.equals(second)
    assert first.index.tolist() == list(range(10))


def test_sample_depends_on_seed() -> None:
    df = pd.DataFrame({"a": range(100)})
    assert not sample_dataset(df, 10, 1).equals(sample_dataset(df, 10, 2))


def test_sample_returns_all_rows_when_small() -> None:
    assert len(sample_dataset(pd.DataFrame({"a": [1, 2]}), 10, 1)) == 2


def test_sample_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError):
        sample_dataset(pd.DataFrame({"a": [1]}), 0, 1)
