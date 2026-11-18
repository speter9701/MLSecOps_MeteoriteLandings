"""Loading and sampling of the raw meteorite landings dataset."""
import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV file of the NASA meteorite landings dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path} (did you run 'dvc pull'?)")
    return pd.read_csv(path)


def sample_dataset(df: pd.DataFrame, n_rows: int, random_state: int) -> pd.DataFrame:
    """Return a reproducible random sample of at most ``n_rows`` rows."""
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if len(df) <= n_rows:
        return df.reset_index(drop=True)
    return df.sample(n=n_rows, random_state=random_state).reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI used by the DVC ``sample`` stage."""
    parser = argparse.ArgumentParser(description="Sample the meteorite dataset.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    sample = sample_dataset(load_raw(args.input), args.n_rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper, main() is tested directly
    raise SystemExit(main())
