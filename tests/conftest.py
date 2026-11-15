"""Shared fixtures: synthetic data shaped like the NASA export (no real data needed in CI)."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """120 synthetic raw rows with realistic columns and a skewed mass distribution."""
    rng = np.random.default_rng(0)
    n = 120
    lat = rng.uniform(-85, 70, n)
    return pd.DataFrame({
        "name": [f"m{i}" for i in range(n)],
        "id": range(n),
        "nametype": rng.choice(["Valid", "Relict"], n, p=[0.95, 0.05]),
        "recclass": rng.choice(["L6", "H5", "Iron", "LL5"], n),
        "mass (g)": np.exp(rng.normal(5, 2, n) - (lat < -60)),
        "fall": rng.choice(["Fell", "Found"], n),
        "year": rng.integers(1850, 2013, n).astype(float),
        "reclat": lat,
        "reclong": rng.uniform(-180, 180, n),
        "GeoLocation": [f"({a:.2f}, 0.0)" for a in lat],
    })
