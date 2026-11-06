# CSE 4105 — Impact of Transformations in Modeling and Forecasting with ARIMA

Assignment solution for **CSE 4105: Data Science and Applications** (BAUET), implemented in
Python instead of R as requested.

Dataset: **`AirPassengers`** from
[vincentarelbundock/Rdatasets](https://github.com/vincentarelbundock/Rdatasets)
(selection rule `Student_ID %% 6 == 4`).

## Deliverables

| File | What it is |
|---|---|
| `arima_assignment.ipynb` | Main deliverable — narrative notebook, already executed, with all plots, tables and equations |
| `arima_assignment.py` | The same analysis as a single runnable script (console report + saves all figures/tables) |
| `data/AirPassengers.csv` | Dataset as distributed by Rdatasets (`time` = fractional year, `value` = passengers) |
| `outputs/` | Every figure (`*.png`) and table (`*.csv`) produced by the script |

## Reproduce

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                        # creates .venv from pyproject.toml / uv.lock
uv run python arima_assignment.py    # full console report -> outputs/
uv run jupyter lab             # or open arima_assignment.ipynb
```

Python is pinned to 3.12 because `pmdarima` (the `auto.arima` equivalent) still requires
`numpy < 2`.

## Method

| R (assignment) | Python (this project) |
|---|---|
| `ts()`, `window()` | `pandas.Series` with a monthly `DatetimeIndex` |
| `adf.test()` | `statsmodels.tsa.stattools.adfuller` |
| `kpss.test()` | `statsmodels.tsa.stattools.kpss` |
| `Acf()`, `Pacf()` | `statsmodels.graphics.tsaplots` |
| `arima()` | `statsmodels.tsa.statespace.SARIMAX` |
| `BoxCox.lambda()`, `BoxCox()` | `scipy.stats.boxcox_normmax`, `scipy.stats.boxcox` |
| `auto.arima()` | `pmdarima.auto_arima` |
| `checkresiduals()` | Ljung-Box, Jarque-Bera, ARCH-LM, ACF and Q-Q plots |
| `accuracy()` | MSE, RMSE, MAE, MPE, MAPE |

Chronological 70/30 split: 100 training observations (Jan 1949 – Apr 1957),
44 test observations (May 1957 – Dec 1960).

## Results

| Model | Specification | AIC | AICc | BIC | Test MAPE (%) |
|---|---|---|---|---|---|
| M1 | ARIMA(0,1,1)(0,1,1)[12], original scale | 644.73 | 645.15 | 652.13 | **5.08** |
| M2 | ARIMA(0,1,1)(0,1,1)[12], log | −309.76 | −309.34 | −302.36 | 8.11 |
| M3 | ARIMA(0,1,1)(0,1,1)[12], Box–Cox (λ\* = 0.0576) | −256.54 | −256.11 | −249.14 | 7.58 |
| M4 | ARIMA(1,1,0)(1,1,0)[12], `auto.arima` | 644.20 | 644.62 | 651.60 | 5.21 |

* ADF/KPSS: only `(1−B)(1−B¹²)yₜ` (and its log / Box–Cox versions) is stationary, so
  `d = 1, D = 1` in every model.
* All four models pass the residual checks (independence, normality, constant variance);
  M2/M3 have the largest Ljung–Box p-values and the smallest residual autocorrelation.
* AIC/AICc/BIC are **not** comparable across the transformed models — they are different
  response scales. The test-set accuracy in Table 2 is the fair comparison.
* Best out-of-sample model: **M1**, which is then re-estimated on the full series to
  forecast January–October 1961.

The notebook explains why the untransformed model wins here: on the level scale the
seasonal random walk repeats the last annual increment, while the log/Box–Cox forecasts
compound geometric growth and over-shoot once the growth decelerates.
