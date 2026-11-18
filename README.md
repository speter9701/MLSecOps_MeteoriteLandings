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
