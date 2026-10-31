# Group 08 – Service & Feature Contract (v0.1 – DRAFT for approval)

This document is the single source of truth that lets four independently built FastAPI
services and one Streamlit app work together. **Any change requires a pull request
approved by all four members.** Proposed by Vignesh Selvam (Student B), 20 Oct 2026.

---

## 1. Scope

| Item | Agreed value |
|---|---|
| Supported tokens | `BTC`, `ETH`, `XRP`, `SOL`, `DOGE`, `XLM`, `TRX`, `XMR` (upper-case, exactly these strings) |
| Forecast horizon | `7` days (t+7). Any other value is rejected (see §6). |
| Data frequency | Daily candles, UTC days |
| Quote currency | USD |
| Python / packages | Python 3.12.13; versions exactly as listed in the brief (see §9) |

---

## 2. Roles and ownership

| Role | Target | Algorithm | Owns features | Consumes features from |
|---|---|---|---|---|
| A | Asset Performance Index (API) | XGBoost Regressor | DailyReturn, Return5, Return20, Momentum | B (TrendStrength), C (Volatility20), D (RelativeVolume) |
| B | Market Risk Index (MRI) | LightGBM Regressor | MA5, MA20, TrendStrength | A (Return20), C (Volatility20, MaxPrice20, GapRisk) |
| C | Trend Category (TC) | XGBoost Classifier | Volatility20, MaxPrice20, MaxDrawdown, GapRisk | A (Return20), B (MA5, MA20) |
| D | Trading Opportunity Rating (TOR) | LightGBM Classifier | VolumeSMA20, RelativeVolume | A (Return20), B (MA5, MA20), C (Volatility20) |

`Return20` (A) is used by every model and `Volatility20` (C) by three, so these two
`/features` endpoints are critical and must be deployed first.

---

## 3. Data source

- **Primary source (training and serving): Kraken public OHLC API**
  `GET https://api.kraken.com/0/public/OHLC?pair=<PAIR>&interval=1440` (no API key).
- Pair mapping (all verified online on 20 Oct 2026):

  | Token | Kraken `pair` | Result key in response |
  |---|---|---|
  | BTC | `XBTUSD` | `XXBTZUSD` |
  | ETH | `ETHUSD` | `XETHZUSD` |
  | XRP | `XRPUSD` | `XXRPZUSD` |
  | SOL | `SOLUSD` | `SOLUSD` |
  | DOGE | `XDGUSD` | `XDGUSD` |
  | XLM | `XLMUSD` | `XXLMZUSD` |
  | TRX | `TRXUSD` | `TRXUSD` |
  | XMR | `XMRUSD` | `XXMRZUSD` |

- Each row is `[time, open, high, low, close, vwap, volume, count]`; prices are strings
  → cast to float. `date = UTC date of time`.
- **The last row is the current, incomplete day and MUST be dropped** before computing
  any feature.
- Kraken returns at most the **720 most recent daily candles** (≈ 2 years). This is the
  training window for every model, so training and serving use the same source and
  definitions.
- The provided CoinMarketCap-format CSV may only be used for exploratory comparison
  (e.g. checking that Kraken features agree with CMC features on overlapping dates).
- Responses are cached per token for 15 minutes; on HTTP 429/5xx retry up to 2 times with
  exponential back-off (1 s, 2 s), then return error 503 (§6).

---

## 4. Feature definitions (computed per token, rows sorted by date ascending)

Exact pandas implementation all four services must use (shared in the group package so
there is one code path for training and serving). `min_periods` = window, so the first
rows are `NaN` until enough history exists.

| Name (exact JSON key) | Owner | Formula | pandas |
|---|---|---|---|
| `DailyReturn` | A | (C_t − C_t−1)/C_t−1 | `close.pct_change(1)` |
| `Return5` | A | (C_t − C_t−5)/C_t−5 | `close.pct_change(5)` |
| `Return20` | A | (C_t − C_t−20)/C_t−20 | `close.pct_change(20)` |
| `Momentum` | A | C_t − C_t−20 | `close - close.shift(20)` |
| `MA5` | B | mean(C_t … C_t−4) | `close.rolling(5).mean()` |
| `MA20` | B | mean(C_t … C_t−19) | `close.rolling(20).mean()` |
| `TrendStrength` | B | MA5/MA20 − 1 | `MA5 / MA20 - 1` |
| `Volatility20` | C | STD(DailyReturn_t … t−19) | `DailyReturn.rolling(20).std(ddof=1)` |
| `MaxPrice20` | C | MAX(High_t … t−19) | `high.rolling(20).max()` |
| `MaxDrawdown` | C | (MaxPrice20 − C_t)/MaxPrice20 | `(MaxPrice20 - close) / MaxPrice20` |
| `GapRisk` | C | Mean(\|O_t − C_t−1\|/C_t−1) | `((open - close.shift(1)).abs() / close.shift(1)).rolling(20).mean()` |
| `VolumeSMA20` | D | mean(V_t … V_t−19) | `volume.rolling(20).mean()` |
| `RelativeVolume` | D | V_t / SMA20(V) | `volume / VolumeSMA20` |

Assumptions to state in the report (the brief leaves them open):
- **GapRisk window = 20 days** (the brief gives no window; 20 matches all other features).
- **Volatility20 uses the sample standard deviation (`ddof=1`), not annualised.**
- Expect `GapRisk ≈ 0`: crypto trades 24/7, so a daily open ≈ the previous close.

---

## 5. Targets (all at t+7; features use information up to t only)

For every token, compute the index from the features at each date, then
`target_t = index.shift(-7)` (within the token). The final 7 rows per token have no target
and are dropped from training.

**A – Asset Performance Index (0–100)**
`RS = clip((Return20+0.20)/0.40, 0, 1)`; `TS = clip((TrendStrength+0.10)/0.20, 0, 1)`;
`RV = min(RelativeVolume, 2)/2`; `PS = 1 − Volatility20`;
`API = 100·(0.40·RS + 0.30·TS + 0.20·RV + 0.10·PS)`

**B – Market Risk Index (0–100)**
`DR = (MaxPrice20 − Close)/MaxPrice20`; `RR = |min(Return20, 0)|`;
`VRN = min(Volatility20/0.10, 1)`; `DRN = min(DR/0.40, 1)`; `RRN = min(RR/0.30, 1)`;
`GRN = min(GapRisk/0.10, 1)`; `MRI = 100·(0.40·VRN + 0.30·DRN + 0.20·RRN + 0.10·GRN)`

**C – Trend Category** — `TrendScore = 0.7·(MA5/MA20 − 1) + 0.3·Return20`

| Condition | Label (exact string) |
|---|---|
| TrendScore < −0.10 | `Strong Downtrend` |
| −0.10 ≤ TrendScore < −0.02 | `Downtrend` |
| −0.02 ≤ TrendScore ≤ 0.02 | `Sideways` |
| 0.02 < TrendScore ≤ 0.10 | `Uptrend` |
| TrendScore > 0.10 | `Strong Uptrend` |

**D – Trading Opportunity Rating** — `M = (Return20+0.20)/0.40`; `T = MA5/MA20`;
`V = min(RelativeVolume, 2)/2`; `R = min(Volatility20/0.10, 1)`;
`OS = 0.40·M + 0.30·T + 0.20·V − 0.10·R`

| Condition | Label (exact string) |
|---|---|
| OS < 0.20 | `Strong Sell` |
| 0.20 ≤ OS < 0.40 | `Sell` |
| 0.40 ≤ OS < 0.60 | `Hold` |
| 0.60 ≤ OS < 0.80 | `Buy` |
| OS ≥ 0.80 | `Strong Buy` |

Boundary values must be unit-tested (use `np.select`, not `pd.cut`, for the mixed
inclusive/exclusive rules).

---

## 6. API contract (all four services)

All responses are JSON. Successful calls return **HTTP 200**.

### 6.1 `GET /`
```json
{
  "project": "Market Risk Index API",
  "description": "Predicts the Market Risk Index (MRI) of cryptocurrencies 7 days into the future.",
  "algorithm": "LightGBM Regressor",
  "version": "1.0.0",
  "github": "https://github.com/<user>/<repo>",
  "endpoints": ["/", "/health", "/features", "/predict", "/model-metadata"]
}
```

### 6.2 `GET /health`
```json
{"status": "ok", "model_loaded": true, "model_version": "1.0.0", "timestamp": "2026-10-20T08:00:00Z"}
```

### 6.3 `POST /features`
Request (`forecast_horizon` optional, default 7; `history_days` optional, default 1, max 365):
```json
{"token": "BTC", "forecast_horizon": 7, "history_days": 1}
```
Response – only the features **owned** by that service; values that cannot be computed
are `null` (never `NaN`, which is invalid JSON):
```json
{
  "token": "BTC",
  "as_of_date": "2026-10-19",
  "source": "kraken",
  "features": {"MA5": 67012.4, "MA20": 65230.1, "TrendStrength": 0.0273},
  "history": [
    {"date": "2026-10-18", "close": 66890.0, "MA5": 66800.2, "MA20": 65100.7, "TrendStrength": 0.0261},
    {"date": "2026-10-19", "close": 67300.5, "MA5": 67012.4, "MA20": 65230.1, "TrendStrength": 0.0273}
  ]
}
```
`history` contains the last `history_days` rows (oldest first) and always includes `close`,
so Streamlit can draw charts without calling Kraken itself.

### 6.4 `POST /predict`
Request (exactly as in the brief):
```json
{"token": "BTC", "forecast_horizon": 7}
```
Regression response (A, B) – required keys first, extra keys allowed:
```json
{
  "prediction": 42.7,
  "forecast_horizon": 7,
  "prediction_date": "2026-10-26",
  "token": "BTC",
  "as_of_date": "2026-10-19",
  "model_version": "1.0.0",
  "missing_features": [],
  "feature_sources": {"Return20": "student_a_api", "Volatility20": "student_c_api"}
}
```
Classification response (C, D) – adds `confidence` (probability of the predicted class)
and the full distribution:
```json
{
  "prediction": "Buy",
  "confidence": 0.84,
  "probabilities": {"Strong Sell": 0.01, "Sell": 0.03, "Hold": 0.10, "Buy": 0.84, "Strong Buy": 0.02},
  "forecast_horizon": 7,
  "prediction_date": "2026-10-26",
  "token": "BTC",
  "as_of_date": "2026-10-19",
  "model_version": "1.0.0",
  "missing_features": [],
  "feature_sources": {}
}
```
`as_of_date` = the date of the latest complete candle; `prediction_date = as_of_date + 7 days`.

### 6.5 `GET /model-metadata`
```json
{
  "target": "Market Risk Index",
  "target_description": "0-100 composite risk score ... measured 7 days ahead",
  "algorithm": "LightGBM Regressor",
  "forecast_horizon": 7,
  "features": ["Return20", "Volatility20", "MaxPrice20", "GapRisk"],
  "feature_importance": {"Volatility20": 0.41, "Return20": 0.27, "MaxPrice20": 0.22, "GapRisk": 0.10},
  "classification_labels": null,
  "metrics": {"test_mae": 0.0, "test_rmse": 0.0, "test_r2": 0.0},
  "training_period": {"start": "2024-10-31", "end": "2026-10-12"},
  "model_version": "1.0.0"
}
```
`classification_labels` is the ordered label list for C and D, `null` for A and B. **All
values are read from artefacts saved by the notebook (`metrics.json`), never typed by
hand.**

### 6.6 Errors (same shape everywhere – the brief's format)
```json
{"error": "Unsupported cryptocurrency", "supported_tokens": ["BTC","ETH","XRP","SOL","DOGE","XLM","TRX","XMR"]}
```

| Situation | HTTP status | Body |
|---|---|---|
| Unknown token | 400 | `{"error": "Unsupported cryptocurrency", "supported_tokens": [...]}` |
| `forecast_horizon` ≠ 7 | 422 | `{"error": "Unsupported forecast_horizon", "supported_horizons": [7]}` |
| Malformed / missing JSON fields | 422 | `{"error": "Invalid request", "details": [...]}` |
| Kraken unavailable after retries | 503 | `{"error": "Market data source unavailable", "retry_after_seconds": 60}` |
| Unexpected server error | 500 | `{"error": "Internal server error"}` (never a traceback) |

---

## 7. Calling other services (resilience rules)

- Peer base URLs come from environment variables (`STUDENT_A_URL`, …), never hard-coded.
- Call peers **in parallel** (`httpx.AsyncClient`), timeout **10 s** per call, 1 retry.
- If a peer is unavailable or returns an error, its features become missing (`NaN`)
  and are listed in `missing_features`; the prediction still returns **200** (the brief
  requires services to keep operating with missing values; XGBoost/LightGBM handle NaN
  natively).
- Models must be trained with **missing-feature augmentation** (randomly masking peer
  features) so the NaN path is learned, and each student reports a degradation table
  (performance with each peer down).
- Peer feature responses are cached for 15 minutes per token.
- Render free instances sleep; before demos and marking, call every `/health` endpoint to
  wake the services.

---

## 8. Modelling conventions

- Chronological split across all tokens by date: **train ≈ 70 %, validation ≈ 15 %,
  test ≈ 15 %**, with a **7-day purge gap** before validation and before test (the target
  looks 7 days ahead). Never random splitting.
- Optuna (TPE sampler, `seed=42`) with `TimeSeriesSplit(n_splits=5, gap=7)` on the
  training period only; search space and best parameters documented; trials logged to
  Weights & Biases.
- Baselines for every model: mean/majority class **and persistence** (today's index or
  class as the 7-day forecast).
- Metrics – regression: MAE (primary), RMSE, R²; classification: macro-F1 (primary),
  balanced accuracy, per-class recall, log-loss, calibration (ECE).
- The test set is evaluated once. The deployed model is refitted on all data with the
  selected hyperparameters; this is documented in the notebook.
- Random seed `42` everywhere. Each student writes to `data/processed/student_x/` and
  `models/student_x/` only.

---

## 9. Pinned versions (all repositories)

Python 3.12.13 · pandas 2.2.2 · jupyterlab 4.6.1 · scikit-learn 1.6.1 · tensorflow 2.21.0 ·
matplotlib 3.11.0 · wandb 0.28.0 · lightgbm 4.6.0 · lime 0.2.0.1 · streamlit 1.59.0 ·
fastapi[standard] 0.139.0 · uvicorn 0.50.0 · joblib 1.5.3 · xgboost 3.3.0 · optuna 4.9.0.

---

## 10. Repositories, deployments and milestones

| Component | Repo name | Deployment URL |
|---|---|---|
| Experiments | `36120-26SP-AT3-Group08-experiments` | – |
| Streamlit | `36120-26SP-AT3-Group08-streamlit` | _TBC_ |
| Student A API | `36120-26SP-AT3-Group08-StudentA-api` | _TBC_ |
| Student B API | `36120-26SP-AT3-Group08-StudentB-api` | _TBC_ |
| Student C API | `36120-26SP-AT3-Group08-StudentC-api` |`https://three6120-26sp-at3-group08-studentc-api.onrender.com/`|
| Student D API | `36120-26SP-AT3-Group08-StudentD-api` | _TBC_ |

All six repositories are private, give administrator access to the nine teaching staff
listed in the brief, and protect `main` (pull request + 1 approval).

| Date (2026) | Milestone |
|---|---|
| Wed 21 Oct | Contract approved; roles and GitHub usernames confirmed; repos created |
| Thu 22 Oct | Every `/features` endpoint deployed on Render (unblocks the others) |
| Sun 25 Oct | Models trained and tuned; best experiments merged via PR |
| Tue 27 Oct | All `/predict` endpoints live; Streamlit integrated and deployed |
| Wed 28 Oct | Report draft complete; full end-to-end testing |
| Thu 29 Oct | Final checks, ZIPs and submission (before 23:59) |
