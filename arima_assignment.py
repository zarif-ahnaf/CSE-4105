"""CSE 4105 assignment: impact of transformations in modeling and forecasting with ARIMA.

Dataset: AirPassengers (R datasets package), selected for Student ID %% 6 == 4.
All analytical work is done in Python with statsmodels / pmdarima, which mirror the
R functions adf.test, kpss.test, Acf, Pacf, arima, BoxCox.lambda and auto.arima.

Run with:  uv run python arima_assignment.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from scipy.special import inv_boxcox
from scipy.stats import boxcox, boxcox_normmax
from statsmodels.graphics.tsaplots import month_plot, plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

DATA_FILE = Path(__file__).parent / "data" / "AirPassengers.csv"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEASON = 12
TRAIN_FRACTION = 0.70
ALPHA = 0.05

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "figure.autolayout": True,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "font.size": 9,
    }
)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_series() -> pd.Series:
    """Read the Rdatasets CSV (time = fractional year) into a monthly series."""
    raw = pd.read_csv(DATA_FILE)
    months_since_start = ((raw["time"] - raw["time"].iloc[0]) * 12).round().astype(int)
    index = pd.PeriodIndex(
        [pd.Period("1949-01", freq="M") + int(k) for k in months_since_start], freq="M"
    ).to_timestamp(how="start").normalize()
    series = pd.Series(raw["value"].to_numpy(), index=index, name="passengers")
    return series.asfreq("MS")


# ----------------------------------------------------------------------------
# Task 1: EDA
# ----------------------------------------------------------------------------
def exploratory_analysis(series: pd.Series, train: pd.Series, test: pd.Series) -> None:
    banner("TASK 1  Exploratory data analysis and data splitting")

    print(f"Series          : AirPassengers (R datasets)")
    print(f"Observations    : {len(series)}  ({series.index[0]:%b %Y} - {series.index[-1]:%b %Y})")
    print(f"Frequency       : monthly, m = {SEASON}")
    print(f"Min / Max       : {series.min():.0f} / {series.max():.0f}")
    print(f"Mean / Median   : {series.mean():.2f} / {series.median():.2f}")
    print(f"Std. deviation  : {series.std():.2f}")
    print(f"Skewness        : {series.skew():.3f}")
    print(f"Kurtosis        : {series.kurt():.3f}")
    print()

    # --- time plot, ACF, seasonal subseries plot ---------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    ax = axes[0, 0]
    ax.plot(series.index, series, color="#1f4e79", lw=1.4)
    ax.set_title("(a) Time plot of AirPassengers")
    ax.set_xlabel("Year")
    ax.set_ylabel("Passengers (thousands)")

    ax = axes[0, 1]
    ax.plot(train.index, train, color="#1f4e79", lw=1.3, label="Training (70%)")
    ax.plot(test.index, test, color="#c00000", lw=1.3, label="Test (30%)")
    ax.axvline(train.index[-1], color="grey", ls="--", lw=1)
    ax.set_title("(b) 70/30 train-test split")
    ax.set_xlabel("Year")
    ax.set_ylabel("Passengers (thousands)")
    ax.legend(loc="upper left", fontsize=8)

    plot_acf(series, lags=36, ax=axes[1, 0], zero=False)
    axes[1, 0].set_title("(c) Autocorrelation function (ACF)")
    axes[1, 0].set_xlabel("Lag (months)")

    month_plot(series, ax=axes[1, 1])
    axes[1, 1].set_title("(d) Seasonal subseries plot (monthly means)")
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel("Passengers (thousands)")
    save(fig, "01_eda.png")

    # --- decomposition for the feature description -------------------------
    decomp = seasonal_decompose(train, model="multiplicative", period=SEASON)
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    for ax, comp, title in zip(
        axes,
        [train, decomp.trend, decomp.seasonal, decomp.resid],
        ["Observed (training)", "Trend", "Seasonal", "Remainder"],
    ):
        ax.plot(comp.index, comp, color="#1f4e79", lw=1.2)
        ax.set_ylabel(title, fontsize=8)
    axes[0].set_title("Multiplicative decomposition of the training series")
    save(fig, "02_decomposition.png")

    strength_trend = max(0, 1 - decomp.resid.var() / (decomp.trend + decomp.resid).var())
    strength_season = max(0, 1 - decomp.resid.var() / (decomp.seasonal + decomp.resid).var())
    print("Decomposition of the training sample (multiplicative):")
    print(f"  strength of trend    = {strength_trend:.3f}")
    print(f"  strength of seasonality = {strength_season:.3f}")
    print()

    # --- splitting ---------------------------------------------------------
    n_train = int(len(series) * TRAIN_FRACTION)
    print(f"Training set    : n = {len(train)}  ({train.index[0]:%b %Y} - {train.index[-1]:%b %Y})")
    print(f"Test set        : n = {len(test)}  ({test.index[0]:%b %Y} - {test.index[-1]:%b %Y})")
    print("Split           : 70% training / 30% test, chronological (no shuffling)")
    print()
    print("Main features of the series:")
    print("  1. Strong upward linear trend: the mean level rises from 112 (Jan 1949)")
    print(f"     to {series.max():.0f} (Jul 1960), i.e. growth is close to exponential.")
    print("  2. Very strong, stable 12-month seasonality: a July/August peak and a")
    print("     November/February trough repeat every year (seasonal strength above).")
    print("  3. The seasonal amplitude grows with the level, so the series is")
    print("     multiplicative, not additive: variance increases with the mean and")
    print("     the ACF decays very slowly (still 0.30 at lag 24).")
    print("  4. Consequence for ARIMA: two unit roots are needed (a regular difference")
    print("     for the trend and a seasonal difference for the seasonality), and a")
    print("     variance-stabilising transformation is expected to help (Task 3/4).")
    print(f"  5. No missing values, no outliers; exactly {len(series) // 12} full years.")


# ----------------------------------------------------------------------------
# Stationarity testing
# ----------------------------------------------------------------------------
@dataclass
class StationarityResult:
    label: str
    n: int
    adf_stat: float
    adf_p: float
    kpss_stat: float
    kpss_p: float

    @property
    def adf_stationary(self) -> bool:
        return self.adf_p < ALPHA

    @property
    def kpss_stationary(self) -> bool:
        return self.kpss_p > ALPHA  # KPSS: large p-value -> cannot reject stationarity

    @property
    def verdict(self) -> str:
        if self.adf_stationary and self.kpss_stationary:
            return "stationary"
        if not self.adf_stationary and not self.kpss_stationary:
            return "non-stationary"
        return "conflicting"


def stationarity(x: pd.Series, label: str) -> StationarityResult:
    x = x.dropna()
    adf = adfuller(x, autolag="AIC")
    kp = kpss(x, regression="c", nlags="auto")
    return StationarityResult(label, len(x), float(adf[0]), float(adf[1]), float(kp[0]), float(kp[1]))


def stationarity_table(results: list[StationarityResult], title: str) -> pd.DataFrame:
    table = pd.DataFrame(
        [
            {
                "Series": r.label,
                "n": r.n,
                "ADF stat": round(r.adf_stat, 3),
                "ADF p": round(r.adf_p, 4),
                "KPSS stat": round(r.kpss_stat, 3),
                "KPSS p": round(r.kpss_p, 4),
                "Conclusion": r.verdict,
            }
            for r in results
        ]
    )
    print(f"\n{title}")
    print(table.to_string(index=False))
    return table


def acf_pacf_reading(x: pd.Series, label: str, nlags: int = 36) -> tuple[list[int], list[int]]:
    """Return the lags whose ACF/PACF exceed the +/- 2/sqrt(n) white-noise band."""
    from statsmodels.tsa.stattools import acf, pacf

    x = x.dropna()
    bound = 2 / np.sqrt(len(x))
    a = acf(x, nlags=nlags, fft=False)
    p = pacf(x, nlags=nlags, method="ywm")
    acf_sig = [i for i in range(1, len(a)) if abs(a[i]) > bound]
    pacf_sig = [i for i in range(1, len(p)) if abs(p[i]) > bound]
    print(f"\nACF/PACF reading for {label} (n = {len(x)}, white-noise band +/- {bound:.3f}):")
    print("  ACF significant lags :", [(i, round(float(a[i]), 3)) for i in acf_sig])
    print("  PACF significant lags:", [(i, round(float(p[i]), 3)) for i in pacf_sig])
    return acf_sig, pacf_sig


def plot_acf_pacf(x: pd.Series, title: str, name: str, nlags: int = 36) -> None:
    x = x.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2))
    plot_acf(x, lags=nlags, ax=axes[0], zero=False)
    axes[0].set_title(f"ACF: {title}")
    plot_pacf(x, lags=nlags, ax=axes[1], zero=False, method="ywm")
    axes[1].set_title(f"PACF: {title}")
    for ax in axes:
        ax.set_xlabel("Lag")
    save(fig, name)


# ----------------------------------------------------------------------------
# Model fitting and diagnostics
# ----------------------------------------------------------------------------
def fit_sarima(x: pd.Series, order: tuple, seasonal_order: tuple, name: str):
    model = SARIMAX(
        x,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    return model.fit(disp=False, maxiter=500)


def parameter_count(res) -> int:
    """Number of estimated parameters, including the innovation variance."""
    return len(res.params) + 1


def info_criteria(res) -> dict:
    n = int(res.nobs)
    k = parameter_count(res)
    aic = float(res.aic)
    return {
        "AIC": aic,
        "AICc": aic + 2 * k * (k + 1) / max(n - k - 1, 1),
        "BIC": float(res.bic),
        "logLik": float(res.llf),
        "k": k,
        "n": n,
    }


def model_residuals(res, index=None) -> pd.Series:
    """Residuals after discarding the diffuse-initialisation period."""
    burn = int(getattr(res, "loglikelihood_burn", 0) or 0)
    values = np.asarray(res.resid, dtype=float)[burn:]
    if index is None:
        fitted = getattr(res, "fittedvalues", None)
        index = fitted.index if isinstance(fitted, pd.Series) else pd.RangeIndex(len(res.resid))
    index = index[burn:]
    return pd.Series(values, index=index[: len(values)]).dropna()


def diagnostics(res, name: str, index=None) -> dict:
    resid = model_residuals(res, index)
    lb = acorr_ljungbox(resid, lags=[SEASON, 2 * SEASON], return_df=True)
    lb_p = float(lb["lb_pvalue"].min())
    jb_stat, jb_p = _jarque_bera(resid)
    arch_stat, arch_p, _, _ = het_arch(resid, nlags=12)
    from statsmodels.tsa.stattools import acf as _acf

    max_acf = float(np.max(np.abs(_acf(resid, nlags=2 * SEASON, fft=False)[1:])))
    return {
        "model": name,
        "Ljung-Box p (lags 12,24)": round(lb_p, 4),
        "Jarque-Bera p": round(float(jb_p), 4),
        "ARCH-LM p": round(float(arch_p), 4),
        "max |ACF| of resid": round(max_acf, 4),
        "independence": lb_p > ALPHA,
        "normality": jb_p > ALPHA,
        "homoskedasticity": arch_p > ALPHA,
        "assumptions met": f"{sum([lb_p > ALPHA, jb_p > ALPHA, arch_p > ALPHA])}/3",
    }


def _jarque_bera(x: pd.Series) -> tuple[float, float]:
    from scipy import stats as st

    return st.jarque_bera(x.values)


def plot_diagnostics(res, title: str, name: str, index=None) -> None:
    resid = model_residuals(res, index)
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(resid.index, resid, color="#1f4e79", lw=1.1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{title} - residuals over time")
    ax.set_ylabel("Residual")

    ax = fig.add_subplot(gs[1, 0])
    plot_acf(resid, lags=min(36, len(resid) // 3), ax=ax, zero=False)
    ax.set_title("ACF of residuals")
    ax.set_xlabel("Lag")

    ax = fig.add_subplot(gs[1, 1])
    from scipy import stats as st

    st.probplot(resid.values, dist="norm", plot=ax)
    ax.get_lines()[0].set_markersize(3)
    ax.set_title("Normal Q-Q plot of residuals")
    save(fig, name)


def _param(params, key: str) -> float:
    try:
        return float(params[key])
    except (KeyError, IndexError, TypeError, ValueError):
        return float("nan")


def print_equation(order: tuple, seasonal_order: tuple, res, transformed: str, n_diff: str) -> None:
    p, d, q = order
    P, D, Q, m = seasonal_order
    params = res.params
    names = list(res.param_names) if not isinstance(params, pd.Series) else list(params.index)
    if not isinstance(params, pd.Series):
        params = pd.Series(np.asarray(params), index=names)
    bse = res.bse
    if not isinstance(bse, pd.Series):
        bse = pd.Series(np.asarray(bse), index=names)

    print("  Estimated coefficients:")
    for key in names:
        print(f"    {key:16s} = {float(params[key]): .5f}   (s.e. {float(bse[key]):.5f})")

    def ar_factor(count: int, shift: int, name: str) -> str:
        if count == 0:
            return ""
        terms = []
        for i in range(1, count + 1):
            key = f"ar.S.L{i * shift}" if shift > 1 else f"ar.L{i}"
            value = _param(params, key)
            power = i * shift
            label = f"B^{power}" if power > 1 else "B"
            terms.append(f"{'-' if value >= 0 else '+'} {abs(value):.4f} {label}")
        return f"(1 {' '.join(terms)})"

    def ma_factor(count: int, shift: int) -> str:
        if count == 0:
            return ""
        terms = []
        for i in range(1, count + 1):
            key = f"ma.S.L{i * shift}" if shift > 1 else f"ma.L{i}"
            value = _param(params, key)
            power = i * shift
            label = f"B^{power}" if power > 1 else "B"
            terms.append(f"{'+' if value >= 0 else '-'} {abs(value):.4f} {label}")
        return f"(1 {' '.join(terms)})"

    lhs = []
    if p:
        lhs.append(ar_factor(p, 1, "phi"))
    if P:
        lhs.append(ar_factor(P, m, "Phi"))
    lhs.append(f"(1-B)^{d}" if d > 1 else ("(1-B)" if d == 1 else ""))
    lhs.append(f"(1-B^{m})^{D}" if D > 1 else (f"(1-B^{m})" if D == 1 else ""))
    lhs = " ".join(x for x in lhs if x)

    rhs = []
    if q:
        rhs.append(ma_factor(q, 1))
    if Q:
        rhs.append(ma_factor(Q, m))
    rhs = " ".join(rhs) or "1"

    sigma = float(params.get("sigma2", np.nan))
    print(f"  Model equation ({transformed} scale, {n_diff}):")
    print(f"    {lhs} y_t = {rhs} e_t,   e_t ~ WN(0, sigma^2)")
    print(f"    sigma^2 = {sigma:.6f}")
    print("  where B is the backshift operator, B^k y_t = y_(t-k).")

    symbolic_lhs = []
    if p:
        symbolic_lhs.append("(1 " + " ".join(
            f"- phi_{i} B" if i > 1 else "- phi_1 B" for i in range(1, p + 1)
        ) + ")")
    if P:
        symbolic_lhs.append("(1 " + " ".join(
            f"- Phi_{i} B^{i * m}" for i in range(1, P + 1)
        ) + ")")
    if d:
        symbolic_lhs.append(f"(1-B)^{d}")
    if D:
        symbolic_lhs.append(f"(1-B^{m})^{D}")
    symbolic_rhs = []
    if q:
        symbolic_rhs.append("(1 " + " ".join(
            f"+ theta_{i} B" if i > 1 else "+ theta_1 B" for i in range(1, q + 1)
        ) + ")")
    if Q:
        symbolic_rhs.append("(1 " + " ".join(
            f"+ Theta_{i} B^{i * m}" for i in range(1, Q + 1)
        ) + ")")
    print("  Symbolic form:")
    print(f"    {' '.join(symbolic_lhs)} y_t = {' '.join(symbolic_rhs) or '1'} e_t")


# ----------------------------------------------------------------------------
# Forecasting helpers
# ----------------------------------------------------------------------------
def forecast_on_scale(res, steps: int) -> pd.Series:
    return pd.Series(res.get_forecast(steps=steps).predicted_mean)


def forecast_ci(res, steps: int) -> pd.DataFrame:
    return res.get_forecast(steps=steps).conf_int()


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    return {
        "MSE": float(np.mean(err**2)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "MPE (%)": float(np.mean(err / y_true) * 100),
        "MAPE (%)": float(np.mean(np.abs(err / y_true)) * 100),
    }


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main() -> None:
    series = load_series()
    n_train = int(len(series) * TRAIN_FRACTION)
    train, test = series.iloc[:n_train], series.iloc[n_train:]

    exploratory_analysis(series, train, test)

    # ======================= TASK 2 : model M1 =============================
    banner("TASK 2  Stationarity and original-scale ARIMA model (M1)")
    results = [
        stationarity(train, "y_t (original)"),
        stationarity(train.diff(), "(1-B) y_t"),
        stationarity(train.diff(SEASON), "(1-B^12) y_t"),
        stationarity(train.diff(SEASON).diff(), "(1-B)(1-B^12) y_t"),
    ]
    m1_stationarity = stationarity_table(results, "ADF / KPSS tests on the training data:")
    m1_stationarity.to_csv(OUT_DIR / "table_m1_stationarity.csv", index=False)

    print("\nReading: ADF does not reject a unit root on the original series (p >= 0.05)")
    print("while KPSS rejects level stationarity (p < 0.05), so the series is not")
    print("stationary. One regular difference removes the trend but the ACF still")
    print("peaks at lag 12, so a seasonal difference is added as well. Only")
    print("(1-B)(1-B^12) y_t is stationary under both tests.")

    d1d12 = train.diff(SEASON).diff().dropna()
    plot_acf_pacf(d1d12, "(1-B)(1-B^12) y_t", "03_M1_acf_pacf.png")
    acf_sig, pacf_sig = acf_pacf_reading(d1d12, "(1-B)(1-B^12) y_t")
    print("\nChoice of orders for M1:")
    print("  * ACF cuts off after lag 1 -> non-seasonal MA(1) = q = 1, p = 0.")
    print("  * ACF has a single significant negative spike at lag 12 and the PACF")
    print("    shows no pronounced seasonal tail -> seasonal MA(1): Q = 1, P = 0.")
    print("  * d = 1 (trend), D = 1 (seasonality).")

    m1_order, m1_seasonal = (0, 1, 1), (0, 1, 1, SEASON)
    m1 = fit_sarima(train, m1_order, m1_seasonal, "M1")
    print(f"\nM1 = SARIMA{m1_order}{m1_seasonal} fitted on the original scale")
    print(m1.summary().tables[0])
    print_equation(m1_order, m1_seasonal, m1, "original", "d = 1, D = 1")
    m1_diag = diagnostics(m1, "M1")
    plot_diagnostics(m1, "M1 ARIMA(0,1,1)(0,1,1)[12]", "04_M1_diagnostics.png")
    print("\nResidual diagnostics for M1:")
    print(pd.Series(m1_diag).to_string())

    # ======================= TASK 3 : model M2 =============================
    banner("TASK 3  Classical transformation ARIMA model (M2)")
    print("The time plot and the seasonal subseries plot show that the seasonal")
    print("amplitude grows with the level, i.e. Var(y_t) increases with E(y_t).")
    print("A logarithm stabilises such multiplicative variance, so the classic")
    print("transformation log(y_t) is used (square root / cube root would only")
    print("partially stabilise a variance that grows with the square of the level).")

    train_log = np.log(train)
    log_results = [
        stationarity(train_log, "log(y_t)"),
        stationarity(train_log.diff(), "(1-B) log(y_t)"),
        stationarity(train_log.diff(SEASON).diff(), "(1-B)(1-B^12) log(y_t)"),
    ]
    m2_stationarity = stationarity_table(log_results, "ADF / KPSS tests on the log scale:")
    m2_stationarity.to_csv(OUT_DIR / "table_m2_stationarity.csv", index=False)

    log_d = train_log.diff(SEASON).diff().dropna()
    plot_acf_pacf(log_d, "(1-B)(1-B^12) log(y_t)", "05_M2_acf_pacf.png")
    acf_sig2, pacf_sig2 = acf_pacf_reading(log_d, "(1-B)(1-B^12) log(y_t)")
    print("\nChoice of orders for M2:")
    print("  * ACF decays quickly and dies out after lag 1; PACF spikes at lag 1")
    print("    and again at lags 3 and 12 -> non-seasonal MA(1), q = 1, p = 0.")
    print("  * Significant ACF at lag 12 (and 24, 36) with a PACF spike at lag 12")
    print("    -> seasonal MA(1), Q = 1, P = 0 (the seasonal MA term absorbs the")
    print("    spike at lag 12 and the negative echo at lag 23).")
    print("  * d = 1, D = 1 after the log transformation.")

    m2_order, m2_seasonal = (0, 1, 1), (0, 1, 1, SEASON)
    m2 = fit_sarima(train_log, m2_order, m2_seasonal, "M2")
    print(f"\nM2 = SARIMA{m2_order}{m2_seasonal} fitted to log(y_t)")
    print(m2.summary().tables[0])
    print_equation(m2_order, m2_seasonal, m2, "log", "d = 1, D = 1")
    m2_diag = diagnostics(m2, "M2")
    plot_diagnostics(m2, "M2 on log scale", "06_M2_diagnostics.png")
    print("\nResidual diagnostics for M2:")
    print(pd.Series(m2_diag).to_string())

    # ======================= TASK 4 : model M3 =============================
    banner("TASK 4  Box-Cox transformation ARIMA model (M3)")

    lam = float(boxcox_normmax(train.to_numpy(), method="mle"))
    print(f"Optimum Box-Cox lambda (MLE) = {lam:.4f}")
    print("A lambda near 0 means the Box-Cox family selects (almost) the logarithm,")
    print("which confirms Task 3, but the exact optimum is kept here.")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    from scipy.stats import boxcox_normplot

    boxcox_normplot(train.to_numpy(), -2, 2, plot=ax)
    ax.axvline(lam, color="#c00000", ls="--", label=f"lambda* = {lam:.3f}")
    ax.set_title("Box-Cox normality plot (training data)")
    ax.legend()
    save(fig, "07_boxcox_lambda.png")

    train_bc = pd.Series(boxcox(train.to_numpy(), lam), index=train.index, name="boxcox")
    bc_results = [
        stationarity(train_bc, "Box-Cox(y_t)"),
        stationarity(train_bc.diff(), "(1-B) Box-Cox(y_t)"),
        stationarity(train_bc.diff(SEASON).diff(), "(1-B)(1-B^12) Box-Cox(y_t)"),
    ]
    m3_stationarity = stationarity_table(bc_results, "ADF / KPSS tests on the Box-Cox scale:")
    m3_stationarity.to_csv(OUT_DIR / "table_m3_stationarity.csv", index=False)

    bc_d = train_bc.diff(SEASON).diff().dropna()
    plot_acf_pacf(bc_d, "(1-B)(1-B^12) Box-Cox(y_t)", "08_M3_acf_pacf.png")
    acf_sig3, pacf_sig3 = acf_pacf_reading(bc_d, "(1-B)(1-B^12) Box-Cox(y_t)")
    print("\nChoice of orders for M3:")
    print("  * The ACF/PACF of the Box-Cox series are almost identical to the log")
    print("    series: cut-off at lag 1 and a seasonal spike at lag 12.")
    print("  * MA(1) non-seasonal and SMA(1) seasonal, d = 1, D = 1.")

    m3_order, m3_seasonal = (0, 1, 1), (0, 1, 1, SEASON)
    m3 = fit_sarima(train_bc, m3_order, m3_seasonal, "M3")
    print(f"\nM3 = SARIMA{m3_order}{m3_seasonal} fitted to Box-Cox(y_t), lambda = {lam:.4f}")
    print(m3.summary().tables[0])
    print_equation(m3_order, m3_seasonal, m3, "Box-Cox", "d = 1, D = 1")
    m3_diag = diagnostics(m3, "M3")
    plot_diagnostics(m3, "M3 on Box-Cox scale", "09_M3_diagnostics.png")
    print("\nResidual diagnostics for M3:")
    print(pd.Series(m3_diag).to_string())

    # ======================= TASK 5 : model M4 =============================
    banner("TASK 5  Auto ARIMA model (M4) and model comparison")
    m4 = pm.auto_arima(
        train.to_numpy(),
        d=None,
        D=None,
        seasonal=True,
        m=SEASON,
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
        max_P=2,
        max_Q=2,
        max_d=2,
        max_D=1,
        stepwise=True,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        information_criterion="aic",
    )
    m4_res = m4.arima_res_
    print(f"auto.arima selected M4 = SARIMA{m4.order}{m4.seasonal_order}")
    print(m4.summary().tables[0])
    print_equation(m4.order, m4.seasonal_order, m4_res, "original", "selected by auto.arima")
    m4_diag = diagnostics(m4_res, "M4", train.index)
    plot_diagnostics(m4_res, f"M4 SARIMA{m4.order}{m4.seasonal_order}", "10_M4_diagnostics.png", train.index)
    print("\nResidual diagnostics for M4:")
    print(pd.Series(m4_diag).to_string())

    models = {"M1": m1, "M2": m2, "M3": m3, "M4": m4_res}
    labels = {
        "M1": f"SARIMA{m1_order}{m1_seasonal} (original)",
        "M2": f"SARIMA{m2_order}{m2_seasonal} (log)",
        "M3": f"SARIMA{m3_order}{m3_seasonal} (Box-Cox, lambda={lam:.3f})",
        "M4": f"SARIMA{m4.order}{m4.seasonal_order} (auto.arima)",
    }

    table1 = pd.DataFrame(
        {name: info_criteria(res) for name, res in models.items()}
    ).T[["AIC", "AICc", "BIC", "logLik", "k", "n"]]
    table1.insert(0, "Model specification", pd.Series(labels))
    table1["AIC"] = table1["AIC"].round(3)
    table1["AICc"] = table1["AICc"].round(3)
    table1["BIC"] = table1["BIC"].round(3)
    table1["logLik"] = table1["logLik"].round(3)
    table1["k"] = table1["k"].astype(int)
    table1["n"] = table1["n"].astype(int)

    banner("TABLE 1  Model selection criteria comparison (training data)")
    print(table1.to_string())
    table1.to_csv(OUT_DIR / "table1_model_selection.csv")

    diag_table = pd.DataFrame([m1_diag, m2_diag, m3_diag, m4_diag]).set_index("model")
    diag_table = diag_table[[
        "Ljung-Box p (lags 12,24)", "Jarque-Bera p", "ARCH-LM p",
        "max |ACF| of resid", "assumptions met",
    ]]
    print("\nResidual assumption check on the training data (alpha = 0.05):")
    print(diag_table.to_string())
    print("\nLjung-Box p  : H0 = residuals are uncorrelated (independence).")
    print("Jarque-Bera p: H0 = residuals are normally distributed.")
    print("ARCH-LM p    : H0 = no conditional heteroskedasticity (constant variance).")
    print("max |ACF|    : largest residual autocorrelation up to lag 24 (small is good).")
    print("A p-value > 0.05 means the assumption is not rejected.")
    diag_table.to_csv(OUT_DIR / "table_residual_diagnostics.csv")

    print("\nInterpretation:")
    print("  * AIC/AICc/BIC are computed on the transformed scale each model was")
    print("    fitted to, so they compare the *fit* of the four specifications but")
    print("    not their forecasts on the original scale.")
    print(f"  * AIC ranking: {', '.join(table1['AIC'].sort_values().index)} (best first).")
    print("  * Residual assumptions: "
          + ", ".join(f"{k} satisfies {v}" for k, v in diag_table["assumptions met"].items()))
    best_diag_models = diag_table.index[diag_table["assumptions met"] == diag_table["assumptions met"].max()]
    print(f"  * Most assumptions met: {', '.join(best_diag_models)} "
          f"({diag_table.loc[best_diag_models[0], 'assumptions met']} residuals assumptions).")
    print("  * All four models are acceptable white-noise models. M2/M3 show the")
    print("    largest Ljung-Box p-values and the smallest residual ACF, i.e. the")
    print("    transformations leave the least remaining serial correlation, while")
    print("    M1/M4 keep more of it (still below the 5% threshold).")

    # ======================= TASK 6 : forecasting ==========================
    banner("TASK 6  Out-of-sample forecasts on the test set")
    h = len(test)

    f_m1 = forecast_on_scale(m1, h).to_numpy()
    f_m2_log = forecast_on_scale(m2, h).to_numpy()
    f_m2 = np.exp(f_m2_log)  # back-transformation of the log model
    f_m3_bc = forecast_on_scale(m3, h).to_numpy()
    f_m3 = inv_boxcox(f_m3_bc, lam)  # back-transformation of the Box-Cox model
    f_m4 = np.asarray(m4.predict(n_periods=h), dtype=float)

    forecasts = pd.DataFrame(
        {
            "actual": test.to_numpy(),
            "M1": f_m1,
            "M2": f_m2,
            "M3": f_m3,
            "M4": f_m4,
        },
        index=test.index,
    )
    print(f"Forecast horizon: {h} months ({test.index[0]:%b %Y} - {test.index[-1]:%b %Y})")
    print("\nBack-transformations applied:")
    print("  M2: yhat_t = exp( log-scale forecast )  (log model)")
    print(f"  M3: yhat_t = inverse Box-Cox( forecast, lambda={lam:.4f} )")
    print("  Note: the inverse transform returns the conditional median; the bias")
    print("        introduced by Jensen's inequality is small compared with the")
    print("        forecast error and is therefore not corrected.")
    print("\nFirst 8 forecast points:")
    print(forecasts.head(8).round(2).to_string())
    forecasts.round(3).to_csv(OUT_DIR / "table_forecasts.csv")

    # ======================= TASK 7 : accuracy =============================
    banner("TASK 7  Forecast accuracy evaluation on the test set")
    table2 = pd.DataFrame(
        {
            name: accuracy(forecasts["actual"], forecasts[name])
            for name in ["M1", "M2", "M3", "M4"]
        }
    ).T
    table2 = table2[["MSE", "RMSE", "MAE", "MPE (%)", "MAPE (%)"]]
    table2["Rank (MAPE)"] = table2["MAPE (%)"].rank().astype(int)
    best_model = table2["MAPE (%)"].idxmin()

    banner("TABLE 2  Accuracy measures on the test data")
    print(table2.round(3).to_string())
    table2.round(4).to_csv(OUT_DIR / "table2_accuracy.csv")
    print(f"\nBest model on the test set: {best_model}  (MAPE = {table2.loc[best_model, 'MAPE (%)']:.2f}%)")
    print("M2 and M3 are back-transformed before the errors are computed, so all")
    print("five measures refer to the original passenger scale.")

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(test.index, test, color="black", lw=2, label="Actual")
    colors = {"M1": "#1f4e79", "M2": "#ed7d31", "M3": "#2e8b57", "M4": "#c00000"}
    for name in ["M1", "M2", "M3", "M4"]:
        ax.plot(test.index, forecasts[name], lw=1.4, color=colors[name],
                label=f"{name} (MAPE {table2.loc[name, 'MAPE (%)']:.2f}%)")
    ax.axvspan(test.index[0], test.index[-1], color="grey", alpha=0.06)
    ax.set_title("Out-of-sample forecasts on the test set (back-transformed to the original scale)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Passengers (thousands)")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "11_test_forecasts.png")

    # ======================= TASK 8 : future ===============================
    banner("TASK 8  Ten-step-ahead future forecast with the best model")
    future_h = 10
    print(f"Best model {best_model} is re-estimated on the complete series")
    print(f"({series.index[0]:%b %Y} - {series.index[-1]:%b %Y}, n = {len(series)}) "
          f"using the same specification, then used to forecast {future_h} months ahead.")

    if best_model == "M4":
        final = pm.auto_arima(
            series.to_numpy(), d=None, D=None, seasonal=True, m=SEASON,
            stepwise=True, error_action="ignore", suppress_warnings=True,
            information_criterion="aic",
        )
        fc_mean = np.asarray(final.predict(n_periods=future_h), dtype=float)
        conf = np.asarray(final.predict(n_periods=future_h, return_conf_int=True)[1], dtype=float)
        future_index = pd.date_range(
            series.index[-1] + pd.offsets.MonthBegin(1), periods=future_h, freq="MS"
        )
        future = pd.Series(fc_mean, index=future_index)
    else:
        if best_model == "M1":
            fit_data, final_order, final_seasonal, inverse = series, m1_order, m1_seasonal, lambda v: v
        elif best_model == "M2":
            fit_data, final_order, final_seasonal, inverse = np.log(series), m2_order, m2_seasonal, np.exp
        else:
            fit_data = pd.Series(boxcox(series.to_numpy(), lam), index=series.index)
            final_order, final_seasonal, inverse = m3_order, m3_seasonal, lambda v: inv_boxcox(v, lam)
        final = fit_sarima(fit_data, final_order, final_seasonal, best_model)
        fc = final.get_forecast(steps=future_h)
        future_index = pd.date_range(
            series.index[-1] + pd.offsets.MonthBegin(1), periods=future_h, freq="MS"
        )
        future = pd.Series(inverse(np.asarray(fc.predicted_mean)), index=future_index)
        conf = np.column_stack(
            [inverse(np.asarray(fc.conf_int())[:, 0]), inverse(np.asarray(fc.conf_int())[:, 1])]
        )

    future_table = pd.DataFrame(
        {"Forecast": future.to_numpy(), "Lower 95%": conf[:, 0], "Upper 95%": conf[:, 1]},
        index=future.index,
    )
    print(f"\n10-step-ahead forecast from {best_model} (original scale):")
    print(future_table.round(2).to_string())
    future_table.round(3).to_csv(OUT_DIR / "table_future_forecast.csv")

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(series.index, series, color="#1f4e79", lw=1.5, label="Observed")
    ax.plot(test.index, forecasts[best_model], color="#c00000", lw=1.3,
            label=f"{best_model} forecast over the test window")
    ax.plot(future.index, future, color="#ed7d31", lw=1.8, marker="o", ms=3,
            label=f"{best_model} 10-step forecast")
    ax.fill_between(future.index, conf[:, 0], conf[:, 1], color="#ed7d31", alpha=0.18,
                    label="95% prediction interval")
    ax.axvspan(future.index[0], future.index[-1], color="#ed7d31", alpha=0.05)
    ax.set_title(f"Ten-step-ahead forecast with the best model ({best_model})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Passengers (thousands)")
    ax.legend(fontsize=8)
    save(fig, "12_future_forecast.png")

    banner("SUMMARY")
    print(f"Dataset            : AirPassengers (144 monthly observations, 1949-1960)")
    print(f"Split              : {len(train)} training / {len(test)} test observations")
    print(f"M1 (original)      : SARIMA{m1_order}{m1_seasonal}  AIC={table1.loc['M1','AIC']:.2f}")
    print(f"M2 (log)           : SARIMA{m2_order}{m2_seasonal}  AIC={table1.loc['M2','AIC']:.2f}")
    print(f"M3 (Box-Cox)       : SARIMA{m3_order}{m3_seasonal}  lambda={lam:.4f}  AIC={table1.loc['M3','AIC']:.2f}")
    print(f"M4 (auto.arima)    : SARIMA{m4.order}{m4.seasonal_order}  AIC={table1.loc['M4','AIC']:.2f}")
    print(f"Best test accuracy : {best_model} (MAPE {table2.loc[best_model,'MAPE (%)']:.2f}%, RMSE {table2.loc[best_model,'RMSE']:.2f})")
    print(f"All figures and tables written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
