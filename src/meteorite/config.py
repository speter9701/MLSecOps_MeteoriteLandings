"""Runtime configuration read from environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Settings shared by training and inference."""

    data_path: Path = Path("data/processed/meteorites_sample.csv")
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "meteorite-mass"
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    min_improvement: float = 0.0

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings, letting environment variables override the defaults."""
        return cls(
            data_path=Path(os.getenv("DATA_PATH", str(cls.data_path))),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", cls.tracking_uri),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", cls.experiment_name),
            min_improvement=float(os.getenv("MIN_IMPROVEMENT", str(cls.min_improvement))),
        )
