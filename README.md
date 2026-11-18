# MLSecOps_MeteoriteLandings

Regression model predicting the mass of a meteorite from the NASA
[Meteorite Landings](https://www.kaggle.com/datasets/nasa/meteorite-landings) dataset,
wrapped in an MLOps toolchain (DVC, MLflow, REST API, Docker, Evidently, Streamlit, Airflow).

## Local development

Requirements: Python 3.13, Git.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
```

### Quality checks

| Check | Command | Rule |
|-------|---------|------|
| PEP8 | `flake8 src tests` | max line length 120 |
| Types | `pyright` | strict mode, `src/` only, 0 errors |
| Tests + coverage | `pytest` | coverage >= 95% over `src/` and `tests/` |
| All hooks | `pre-commit run --all-files` | runs on every commit / push |

## Team conventions

- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `build:`, `ci:`, `refactor:`, `style:`),
  enforced by the `commit-msg` hook.
- **Branches:** `feature/<topic>`, `fix/<topic>`, `docs/<topic>`; `main` only changes through
  reviewed pull requests merged with a merge commit (history of feature branches is kept).
- **Code layout:** all business logic lives in `src/meteorite/`; Streamlit and Airflow code is a thin layer.
- **Dependencies:** exact `==` versions in `requirements.txt` (runtime) and `requirements-dev.txt` (tooling).

## Interface contract (ML core -> REST API / monitoring)

The REST API (`src/meteorite/api/`) and the monitoring layer only use these public objects:

| Module | Object | Used by |
|--------|--------|---------|
| `meteorite.schemas` | `MeteoriteFeatures` (pydantic input model), `FEATURE_COLUMNS`, `TARGET_COLUMN` | API request validation, Evidently column mapping |
| `meteorite.config` | `Settings.from_env()` | everything that needs paths / MLflow URI |
| `meteorite.training` | `train_and_register(settings=None, candidates=None) -> TrainResult` | `/train` endpoint, Airflow DAG |
| `meteorite.inference` | `load_champion(tracking_uri=None)`, `predict(model, records) -> list[float]` | `/predict` endpoint |

Environment variables: `DATA_PATH`, `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `MIN_IMPROVEMENT`.

## Data versioning (DVC)

- Raw data: `data/raw/Meteorite_Landings.csv` - the NASA Meteorite Landings export (45,716 rows,
  the same file that is mirrored on Kaggle), tracked by DVC (`.dvc` pointer file in git, content in the remote).
- Sample: `data/processed/meteorites_sample.csv` - 5,000 rows drawn by the DVC `sample` stage
  (`dvc.yaml`), size and seed live in `params.yaml`.
- Remote: DagsHub (`https://dagshub.com/speter9701/MLSecOps_MeteoriteLandings.dvc`).

```powershell
# one-time: credentials stay local (.dvc/config.local is git-ignored)
dvc remote modify origin --local user <dagshub-username>
dvc remote modify origin --local password <dagshub-token>

dvc pull      # download the data
dvc repro     # re-run the sample stage after changing params.yaml or src/meteorite/data.py
dvc push      # upload new data versions
```

## Preprocessing

Implemented in `src/meteorite/preprocessing.py` (`clean_data`, `add_geo_features`, `build_preprocessor`).

| Problem in the raw data | Handling |
|-------------------------|----------|
| Target column called `mass (g)` | renamed to `mass` |
| Missing (131) or zero mass (19) | row dropped - the target must be known and positive |
| Year as number or date string, impossible values (e.g. 2101) | parsed to a number, outside 800-2016 set to missing, later median-imputed |
| `(0, 0)` coordinates (6,214 rows) mean "unknown location" | set to missing, later median-imputed |
| Longitude outside +-180 (e.g. 354) | set to missing |
| `name`, `id`, `GeoLocation` | dropped (identifiers / redundant with `reclat`, `reclong`) |
| 466 meteorite classes, many rare | one-hot encoding, classes seen fewer than 10 times grouped as "infrequent" |
| Hypothesis: Antarctica matters | engineered features `is_antarctica` (latitude < -60) and `abs_lat` |

The sklearn preprocessor (feature engineering, imputation, scaling, encoding) is part of the model pipeline,
so training and the `/predict` endpoint always apply exactly the same transformation.

## Model

- Task: regression of `mass` (grams) from `recclass`, `nametype`, `fall`, `year`, `reclat`, `reclong`.
- The target spans 0.01 g - 60 t, so every model is a `TransformedTargetRegressor`: it is trained on
  `log1p(mass)` and predicts grams through `expm1`.
- Candidates (`src/meteorite/model.py`): `Ridge`, `RandomForestRegressor`, `HistGradientBoostingRegressor`,
  each behind the same preprocessing pipeline.
- Metrics (`src/meteorite/evaluation.py`): `rmse_log` (model selection and promotion metric), `mae_log`,
  `r2_log`, `median_ae_g`. Model quality itself is not a goal of the project; keeping it from degrading
  automatically is (see *Automatic promotion*).
