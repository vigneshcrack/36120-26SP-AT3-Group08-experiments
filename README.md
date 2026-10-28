# 36120-26SP-AT3-Group08 – Experimentation Repository

**Data acquisition, analysis, feature engineering and model development for Group 08's Crypto Market Intelligence Platform: four models forecasting cryptocurrency market conditions seven days ahead.**

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/"><img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" /></a>
![Python](https://img.shields.io/badge/python-3.12.13-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/env-uv-DE5FE9)
![Optuna](https://img.shields.io/badge/tuning-Optuna%204.9.0-2a78d6)
![W&B](https://img.shields.io/badge/tracking-Weights%20%26%20Biases-FFBE00)

| | |
|---|---|
| **Subject** | 36120 Advanced Machine Learning Application, Spring 2026, University of Technology Sydney |
| **Assessment** | AT3 – Data Product with Machine Learning |
| **Group** | 08 |
| **Integration contract** | [`CONTRACT.md`](CONTRACT.md): tokens, feature formulas, API format, errors, conventions |
| **Shared feature package** | [`mykrml-25740287` v0.3.0](https://test.pypi.org/project/mykrml-25740287/) (TestPyPI) |

---

## Contents
1. [Group members](#1-group-members)
2. [Purpose of this repository](#2-purpose-of-this-repository)
3. [Repositories and deployments](#3-repositories-and-deployments)
4. [Setup](#4-setup)
5. [Running the experiments](#5-running-the-experiments)
6. [Command-line pipeline](#6-command-line-pipeline)
7. [Shared experimental conventions](#7-shared-experimental-conventions)
8. [Results – Student B (Market Risk Index)](#8-results--student-b-market-risk-index)
9. [Data management](#9-data-management)
10. [Project organisation](#10-project-organisation)
11. [Testing and code quality](#11-testing-and-code-quality)
12. [Collaboration workflow](#12-collaboration-workflow)
13. [Reproducibility checklist](#13-reproducibility-checklist)
14. [Troubleshooting](#14-troubleshooting)
15. [AI usage acknowledgement](#15-ai-usage-acknowledgement)

---

## 1. Group members

| Role | Target | Student name | Student ID |
|---|---|---|---|
| Student A | Asset Performance Index – API | Ryan Cruikshank | 13191263 |
| Student B | Market Risk Index – MRI (LightGBM Regressor) | Vignesh Selvam | 25740287 |
| Student C | Trend Category – TC | Nandeeswari Kumaravelu | 25980436 |
| Student D | Trading Opportunity Rating – TOR | Vinnie Bui | 26160438 |

---

## 2. Purpose of this repository

This is the shared experimentation repository behind Group 08's platform. The platform has two layers:
- four FastAPI services, each forecasting one indicator **seven days ahead (t+7)** from live Kraken market data;
- a Streamlit web application that presents the four forecasts.

This repository contains everything that happens *before* deployment:
- data acquisition and validation
- exploratory data analysis
- target construction from the brief's formulas
- feature engineering and selection
- leakage-safe splitting
- model training with Optuna hyperparameter optimisation
- evaluation and explainability

Each student's selected model is exported to `models/student_x/` and then served by that student's API.

| Indicator | Definition (brief) | Type |
|---|---|---|
| Asset Performance Index | 0–100 score from Return20, TrendStrength, RelativeVolume and Volatility20 | Regression |
| Market Risk Index | 0–100 score from Volatility20, drawdown, negative Return20 and GapRisk | Regression |
| Trend Category | 5 classes from TrendScore = 0.7·(MA5/MA20 − 1) + 0.3·Return20 | Classification |
| Trading Opportunity Rating | 5 classes (Strong Sell … Strong Buy) from momentum, trend, volume and volatility | Classification |

The exact formulas, thresholds and label strings are in `CONTRACT.md` §5.

---

## 3. Repositories and deployments

| Component | Repository | Deployment |
|---|---|---|
| Experiments (this repo) | `36120-26SP-AT3-Group08-experiments` | – |
| Streamlit front-end | `36120-26SP-AT3-Group08-streamlit` | _Streamlit Community Cloud URL – add after deployment_ |
| Student A API | _TBC_ | _TBC_ |
| Student B API | `36120-26SP-AT3-Group08-StudentB-api` | https://group08-studentb-mri-api.onrender.com |
| Student C API | _TBC_ | _TBC_ |
| Student D API | _TBC_ | _TBC_ |

All repositories are private. The teaching staff listed in the brief have administrator access.

---

## 4. Setup

**Requirements:** [Git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/). This works on Windows, macOS and Linux.

```bash
# 1. Clone (private repository – access required)
git clone https://github.com/vigneshcrack/36120-26SP-AT3-Group08-experiments.git
cd 36120-26SP-AT3-Group08-experiments

# 2. Install the pinned Python version and every dependency (exact versions from uv.lock)
uv python install 3.12.13
uv sync

# 3. Experiment tracking (optional): copy .env.example to .env and add your own WANDB_API_KEY
#    .env is git-ignored – never commit it

# 4. Start Jupyter Lab
uv run jupyter lab
```

**Without uv:** create a Python 3.12.13 virtual environment, then run `pip install -r requirements.txt` and `pip install -e .`.

The shared package `mykrml-25740287` is installed from TestPyPI. Only that package uses the TestPyPI index; everything else comes from PyPI. This is configured in `pyproject.toml` (`[tool.uv.sources]`).

**Pinned versions, as required by the brief:**
- Python 3.12.13, pandas 2.2.2, jupyterlab 4.6.1, scikit-learn 1.6.1, matplotlib 3.11.0
- wandb 0.28.0, lightgbm 4.6.0, xgboost 3.3.0, optuna 4.9.0, lime 0.2.0.1, joblib 1.5.3

Streamlit, FastAPI and uvicorn are pinned in the front-end and API repositories, where they are used. TensorFlow 2.21.0 is not installed here because no experiment uses a neural network.

---

## 5. Running the experiments

Each student works in their own notebook folder. To reproduce a notebook:
1. Open it in Jupyter Lab.
2. Run **Kernel → Restart Kernel and Run All Cells**.
3. Every printed number and every saved artefact is regenerated.

| Notebook | Owner | Artefacts written |
|---|---|---|
| `notebooks/student_a/` | Student A | `models/student_a/`, `data/processed/student_a/` |
| `notebooks/student_b/36120_26SP_AT3_08_25740287_experiment.ipynb` | Student B | `models/student_b/`, `data/processed/student_b/`, `reports/figures/student_b/` |
| `notebooks/student_c/` | Student C | `models/student_c/`, `data/processed/student_c/` |
| `notebooks/student_d/` | Student D | `models/student_d/`, `data/processed/student_d/` |

**The notebooks follow the subject template:**
- A – project description
- C – data understanding
- D – feature selection
- E/F – preparation and feature engineering
- G – modelling preparation
- H – saved sets
- I – metrics
- J – training and evaluation
- outcomes and next steps

The Student B notebook reads the dated Kraken snapshot in `data/raw/`, so its results are reproducible. Set `REFRESH_DATA = True` in section C.1 to download a new one.

---

## 6. Command-line pipeline

The `at3_crypto` package provides scripts for each pipeline step, so the main steps can run outside the notebooks. This is useful for retraining and checks.

| Command | What it does |
|---|---|
| `make data` | Downloads a new snapshot of all 8 tokens from Kraken into `data/raw/kraken_daily_ohlcv.csv` |
| `make features` | Builds the labelled Market Risk dataset (5,544 token-days) into `data/interim/mri_dataset.csv` |
| `make evaluate` | Re-fits the selected Student B configuration on train + validation and scores the test period. It reproduces **MAE 3.70 vs 3.91 for persistence** |
| `make train` | Re-fits both Student B models on all labelled data and writes `models/student_b/mri_lightgbm_retrained.joblib`. The official bundle is not touched |
| `make predict` | Forecasts next week's Market Risk Index for every token with the saved bundle |
| `make test` / `make lint` / `make format` | Runs the tests, checks the code style, or reformats the code |

The same commands work without `make`, for example `uv run python -m at3_crypto.modeling.train --evaluate`.

---

## 7. Shared experimental conventions

These rules are agreed in `CONTRACT.md` §8 and apply to every model.

| Area | Rule |
|---|---|
| Data | Kraken public OHLC API, daily USD candles, last 720 complete days. The same source is used for training and serving |
| Features | Computed only by the shared package, so all four models and services use identical definitions |
| Labels | Index at t+7, created within each token. Inputs use information up to day t only |
| Split | Chronological by date across all tokens, about 70/15/15, with a **7-day purge gap** before validation and before test. Never random |
| Tuning | Optuna (TPE, `seed=42`) with 5-fold `TimeSeriesSplit(gap=7)` on the training period only |
| Baselines | Mean or majority class **and** persistence (today's value as the 7-day forecast) |
| Metrics | Regression: MAE (primary), RMSE, R². Classification: macro-F1 (primary), balanced accuracy, per-class recall, log-loss |
| Evaluation | The test set is scored once. The deployed model is refitted on all data with the selected settings |
| Tracking | Optuna trials and final metrics logged to Weights & Biases |
| Robustness | Models must keep predicting when a peer feature is missing (NaN). The degradation is measured |

---

## 8. Results – Student B (Market Risk Index)

| Item | Result |
|---|---|
| Data | 8 tokens × 720 days (2 Nov 2024 – 22 Oct 2026) → 5,544 labelled token-days |
| Split | Train 3,824 · validation 776 · test 832 rows, with 7-day purge gaps |
| Selected inputs | Return20, Volatility20, DR (MaxPrice20 as drawdown), GapRisk, and MRI(t) as the residual anchor |
| Final model | LightGBM **residual-on-persistence**: MRI(t) + 0.678 × predicted 7-day change. Fallback: level LightGBM with missing-feature augmentation |
| Optuna | Residual model: 60 trials, best CV MAE 7.53 vs persistence 7.96 (5.4 % better) |
| **Test** | **MAE 3.70, RMSE 5.10, R² 0.63** vs persistence 3.91 / 5.38 / 0.59 (**5.3 % better**). Training mean: 11.29 |
| Risk bands | 84 % of test days in the correct band (persistence: 82 %) |
| Outcome | Hypothesis partially confirmed. Better than persistence overall and on volatile tokens, worse on BTC (−7 %) and ETH (−9 %) |

**Artefacts:**
- `models/student_b/mri_lightgbm.joblib`: primary and fallback models, shrink factor and feature lists.
- `models/student_b/metrics.json`: all metadata, served by the API's `/model-metadata`.
- `reports/figures/student_b/`: the figures used in the report.

---

## 9. Data management

| Location | Content | In Git? |
|---|---|---|
| `data/raw/kraken_daily_ohlcv.csv` | Dated Kraken snapshot used by the notebooks | **Yes** (force-added, so results are reproducible) |
| `data/interim/` | Intermediate datasets (e.g. `mri_dataset.csv`) | No, regenerated by `make features` |
| `data/processed/student_x/` | Final X/y train, validation and test sets | No, regenerated by the notebook |
| `data/external/` | Third-party files (e.g. the provided CoinMarketCap CSV) | No |
| `models/student_x/` | Final model bundle and `metrics.json` | Yes |

`.gitignore` keeps the `data/` folder structure but ignores the data files. To commit a new snapshot, use `git add -f data/raw/kraken_daily_ohlcv.csv`. The data is public market data and contains no personal information.

---

## 10. Project organisation

```
├── CONTRACT.md          <- Group integration contract shared by all four services
├── README.md            <- This file
├── Makefile             <- Pipeline and quality commands (make help lists them)
├── pyproject.toml       <- Project metadata, pinned dependencies, ruff settings
├── uv.lock              <- Exact resolved dependency versions
├── requirements.txt     <- pip-compatible export of the dependencies
├── .env.example         <- Template for local secrets (WANDB_API_KEY); copy to .env
├── data
│   ├── external         <- Third-party data
│   ├── interim          <- Intermediate datasets
│   ├── processed        <- Final train/validation/test sets, one sub-folder per student
│   └── raw              <- Kraken snapshot
├── docs
│   └── meetings         <- Meeting minutes (YYYY-MM-DD.md from _template.md)
├── models               <- Final model artefacts and metrics, one sub-folder per student
├── notebooks            <- Experiment notebooks, one sub-folder per student
├── references           <- Data dictionaries and API notes
├── reports
│   └── figures          <- Figures used in the final report, one sub-folder per student
├── tests                <- Tests for the shared code and the Student B artefacts
└── at3_crypto           <- Shared project code
    ├── config.py        <- Project paths and .env loading
    ├── dataset.py       <- Download / load the Kraken snapshot
    ├── features.py      <- Build the labelled Market Risk dataset with the shared package
    ├── plots.py         <- Reusable report figures
    └── modeling
        ├── train.py     <- Re-fit and evaluate the selected Student B configuration
        └── predict.py   <- Forecast next week's risk for every token
```

---

## 11. Testing and code quality

```bash
make test      # = uv run pytest tests
make lint      # ruff format --check + ruff check
```

The tests run offline on the committed snapshot and check:
- **The snapshot:** exactly 8 tokens × 720 consecutive days, no duplicates, and valid OHLC invariants.
- **The labels:** every label equals the MRI exactly seven calendar days later.
- **The dataset and split:** 5,544 labelled rows, 8 latest rows set aside, a 3,824 / 776 / 832 split, and purge gaps longer than 7 days.
- **The saved bundle:** it loads and forecasts every token within 0–100.

The shared package has its own suite of 48 tests in its repository.

---

## 12. Collaboration workflow

- **Protected `main`:** direct pushes are blocked. Every change needs a pull request with at least one approving review.
- **Branch names:** `student-<a|b|c|d>/<topic>` (for example `student-b/mri-experiment`), or `setup/<topic>` for shared work.
- **Merging experiments:** each student merges their best experiment into `main` through a pull request.
- **Escalation:** under `CONTRACT.md` §11, if no review arrives within 24 hours and a milestone is at risk, the group lead may merge with the owner's bypass. The reason and the passing checks are recorded on the pull request.
- **Contract changes:** any change to `CONTRACT.md` goes through a pull request, and its change log records every version.
- **Meetings:** minuted in `docs/meetings/YYYY-MM-DD.md`, using [`_template.md`](docs/meetings/_template.md). Each record covers attendance, decisions, and action items with an owner and due date.

---

## 13. Reproducibility checklist

- [x] Pinned Python and package versions (`.python-version`, `pyproject.toml`, `uv.lock`, `requirements.txt`)
- [x] Dated data snapshot committed. Notebooks read it unless `REFRESH_DATA = True`
- [x] Fixed seeds (42) and deterministic LightGBM (`deterministic=True`, `force_row_wise=True`)
- [x] No hard-coded local paths. All paths are built from the project root (`at3_crypto/config.py`)
- [x] Secrets only in `.env`, which is git-ignored
- [x] Test result reproducible from the command line (`make evaluate`)

---

## 14. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `uv sync` cannot find `mykrml-25740287` | TestPyPI is unreachable. Retry, or check that `[tool.uv.index]` in `pyproject.toml` points to `https://test.pypi.org/simple/` |
| W&B asks for a login or fails | Add `WANDB_API_KEY` to `.env`, or set `WANDB_MODE=offline` to log locally |
| Notebook numbers differ from the report | The data was refreshed. Set `REFRESH_DATA = False` and use the committed snapshot |
| `ModuleNotFoundError: at3_crypto` | Run commands with `uv run` from the repository root (uv installs the project in editable mode) |
| Kraken returns an error | Public API rate limit. Wait a minute; the package retries with back-off |

---

## 15. AI usage acknowledgement

Generative AI (Claude) was used to support planning, coding and documentation, in line with the subject's AI usage policy. All generated content was reviewed, executed and validated by the group members, who take full responsibility for it and can explain and justify every design decision.
