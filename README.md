# 36120-26SP-AT3-Group08 — Experimentation Repository

**Subject:** 36120 Advanced Machine Learning Application, Spring 2026 (UTS)
**Assessment:** AT3 – Data Product with Machine Learning
**Group:** 08

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Group members

| Role | Target (model) | Student name | Student ID |
|---|---|---|---|
| Student A | Asset Performance Index – API (XGBoost Regressor) | Ryan Cruikshank | 13191263 |
| Student B | Market Risk Index – MRI (LightGBM Regressor) | Vignesh Selvam | 25740287 |
| Student C | Trend Category – TC (XGBoost Classifier) | Nandeeswari Kumaravelu | 25980436 |
| Student D | Trading Opportunity Rating – TOR (LightGBM Classifier) | Vinnie Bui | 26160438 |

## What this repository is

This is the shared experimentation repository for Group 08's cloud-native
**Financial Intelligence Platform for cryptocurrency investors**. It contains data
acquisition, exploratory data analysis, feature engineering, target construction, model
development and Optuna hyperparameter optimisation for four models that each forecast
market conditions **7 days ahead** (t+7) for eight cryptocurrencies:
BTC, ETH, XRP, SOL, DOGE, XLM, TRX and XMR.

The best model from each student is exported to `models/` and served by that student's
independent FastAPI service on Render, which the group's Streamlit application consumes.

| Component | Repository | Deployment |
|---|---|---|
| Experiments (this repo) | `36120-26SP-AT3-Group08-experiments` | – |
| Streamlit front-end | `36120-26SP-AT3-Group08-streamlit` | _URL TBC_ |
| Student A API | _TBC_ | _URL TBC_ |
| Student B API | `36120-26SP-AT3-Group08-StudentB-api` | https://group08-studentb-mri-api.onrender.com |
| Student C API | _TBC_ | _URL TBC_ |
| Student D API | _TBC_ | _URL TBC_ |

The interface every service must follow (tokens, feature names and formulas, request and
response JSON, error format, timeouts) is defined in **[CONTRACT.md](CONTRACT.md)**.

## Setup (Windows, macOS or Linux)

Requirements: [Git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Clone (private repository – you need access)
git clone https://github.com/vigneshcrack/36120-26SP-AT3-Group08-experiments.git
cd 36120-26SP-AT3-Group08-experiments

# 2. Install the pinned Python version and all dependencies
uv python install 3.12.13
uv sync

# 3. (Optional) experiment tracking with Weights & Biases
#    copy .env.example to .env and paste your own WANDB_API_KEY (never commit .env)

# 4. Launch Jupyter Lab
uv run jupyter lab
```

Alternatively, without uv: create a Python 3.12.13 virtual environment and run
`pip install -r requirements.txt`.

### Pinned versions (as required by the brief)

Python 3.12.13 · pandas 2.2.2 · jupyterlab 4.6.1 · scikit-learn 1.6.1 · matplotlib 3.11.0 ·
wandb 0.28.0 · lightgbm 4.6.0 · xgboost 3.3.0 · optuna 4.9.0 · lime 0.2.0.1 · joblib 1.5.3.

Streamlit 1.59.0, fastapi[standard] 0.139.0 and uvicorn 0.50.0 are pinned in the
front-end and API repositories, where they are used. TensorFlow 2.21.0 is not installed
here because no experiment uses a neural network; it would add a very large download
without affecting results.

## Running the experiments

Each student has their own notebook folder. Run the notebooks top to bottom
(*Kernel → Restart Kernel and Run All Cells*):

| Notebook | Owner | Output artefacts |
|---|---|---|
| `notebooks/student_a/…_experiment.ipynb` | Student A | `models/student_a/` |
| `notebooks/student_b/36120_26SP_AT3_08_25740287_experiment.ipynb` | Student B | `models/student_b/` |
| `notebooks/student_c/…_experiment.ipynb` | Student C | `models/student_c/` |
| `notebooks/student_d/…_experiment.ipynb` | Student D | `models/student_d/` |

Data is fetched live from public APIs (see CONTRACT.md §3). Raw and processed files are
written under `data/` and are not committed to Git (only the folder structure is).

## Project organisation

```
├── CONTRACT.md        <- Group API / feature contract shared by all four services
├── README.md          <- This file
├── Makefile           <- Convenience commands (make lint, make format, make test)
├── pyproject.toml     <- Project metadata and pinned dependencies
├── uv.lock            <- Exact resolved dependency versions (reproducibility)
├── requirements.txt   <- pip-compatible export of the dependencies
├── data
│   ├── external       <- Data from third-party sources (e.g. the provided CSV)
│   ├── interim        <- Intermediate data
│   ├── processed      <- Final train/validation/test sets, one sub-folder per student
│   └── raw            <- Raw API downloads
├── docs
│   └── meetings       <- Meeting minutes (agenda, decisions, action items)
├── models             <- Final model artefacts and metrics, one sub-folder per student
├── notebooks          <- Experiment notebooks, one sub-folder per student
├── references         <- Data dictionaries and API documentation notes
├── reports
│   └── figures        <- Figures used in the final report
├── tests              <- Unit tests
└── at3_crypto         <- Shared project code
```

## Collaboration workflow

- `main` is protected: direct pushes are blocked and every change needs a pull request
  approved by at least one other group member.
- Branch naming: `student-<a|b|c|d>/<topic>` (for example `student-b/mri-experiment`) or
  `setup/<topic>` for shared work.
- Each student merges their best experiment into `main` through a reviewed pull request.
- Meetings are minuted in `docs/meetings/YYYY-MM-DD.md`.

## AI usage acknowledgement

Generative AI (Claude) was used to support planning, coding and documentation, in line
with the subject's AI usage policy. All generated content was reviewed, executed and
validated by the group members, who take full responsibility for it and can explain and
justify every design decision.
