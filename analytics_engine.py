"""
analytics_engine.py
====================
QuantLab Institutional Portfolio Research & Risk Analytics Platform

Provides a comprehensive suite of quantitative analytics for portfolio performance
measurement and risk management. Designed to consume price DataFrames produced by
market_data.py and expose results in a format suitable for downstream reporting,
attribution, and optimisation pipelines.

Conventions
-----------
- Price DataFrames are expected to be indexed by a DatetimeIndex with timezone-aware
  or timezone-naive timestamps (consistent within a single DataFrame).
- Columns represent individual assets identified by their ticker symbol.
- All monetary / percentage results are expressed as decimals unless explicitly noted
  (e.g. 0.08 → 8 %).
- Trading-day counts use ``TRADING_DAYS_PER_YEAR = 252`` by default; this constant
  can be overridden at class instantiation for different asset classes (e.g. 365 for
  24 h crypto markets).

Author : QuantLab Research Engineering
Version: 1.0.0
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats  # optional – guarded with try/except below

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR: int = 252
_EPSILON: float = 1e-10  # guard against division by near-zero volatility


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class AnalyticsEngineError(Exception):
    """Base exception for all analytics_engine errors."""


class InsufficientDataError(AnalyticsEngineError):
    """Raised when the input data contains too few observations for a calculation."""


class InputValidationError(AnalyticsEngineError):
    """Raised when input arguments fail validation checks."""


# ---------------------------------------------------------------------------
# Core Analytics Engine
# ---------------------------------------------------------------------------
class AnalyticsEngine:
    """
    Institutional-grade portfolio analytics engine.

    Computes a comprehensive set of performance and risk metrics from a DataFrame
    of asset prices.  All calculations are vectorised using NumPy/pandas for
    efficiency on large universes.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide-format DataFrame of asset closing prices.
        - Index  : ``pd.DatetimeIndex`` (ascending, no duplicates).
        - Columns: asset identifiers (str ticker symbols).
        - Values : positive float prices (NaN allowed; handled internally).
    trading_days_per_year : int, optional
        Annualisation factor.  Defaults to 252 (equity markets).
        Use 365 for continuous markets such as crypto.
    min_periods : int, optional
        Minimum number of non-NaN return observations required before a metric
        is computed.  Raises ``InsufficientDataError`` if not met.  Default 30.

    Raises
    ------
    InputValidationError
        If ``prices`` is not a non-empty ``pd.DataFrame`` with a ``DatetimeIndex``.

    Examples
    --------
    >>> import pandas as pd
    >>> from market_data import MarketDataLoader          # hypothetical companion module
    >>> from analytics_engine import AnalyticsEngine
    >>>
    >>> loader = MarketDataLoader()
    >>> prices = loader.get_prices(["AAPL", "MSFT", "GOOGL"], start="2020-01-01")
    >>> engine = AnalyticsEngine(prices)
    >>> print(engine.sharpe_ratio())
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
        min_periods: int = 30,
    ) -> None:
        self._validate_prices(prices)
        # Store a clean, sorted copy to avoid mutating the caller's DataFrame
        self._prices: pd.DataFrame = prices.sort_index().copy()
        self.trading_days_per_year: int = trading_days_per_year
        self.min_periods: int = min_periods

        # Lazily computed; reset whenever prices change
        self._simple_returns: Optional[pd.DataFrame] = None
        self._log_returns: Optional[pd.DataFrame] = None

        logger.info(
            "AnalyticsEngine initialised | assets=%d | observations=%d | "
            "annualisation=%d",
            len(self._prices.columns),
            len(self._prices),
            self.trading_days_per_year,
        )

    # ------------------------------------------------------------------
    # Public Price Access
    # ------------------------------------------------------------------
    @property
    def prices(self) -> pd.DataFrame:
        """Read-only view of the cleaned price DataFrame."""
        return self._prices

    def update_prices(self, prices: pd.DataFrame) -> None:
        """
        Replace the internal price series and invalidate cached returns.

        Parameters
        ----------
        prices : pd.DataFrame
            New price DataFrame conforming to the same schema as the constructor.
        """
        self._validate_prices(prices)
        self._prices = prices.sort_index().copy()
        self._simple_returns = None
        self._log_returns = None
        logger.info("Prices updated; return caches invalidated.")

    # ------------------------------------------------------------------
    # 1. Simple Returns
    # ------------------------------------------------------------------
    def calculate_returns(self, fill_method: Optional[str] = None) -> pd.DataFrame:
        """
        Compute period-over-period simple (arithmetic) returns.

        .. math::
            r_t = \\frac{P_t - P_{t-1}}{P_{t-1}}

        Parameters
        ----------
        fill_method : {None, 'ffill', 'bfill'}, optional
            Forward- or backward-fill NaN prices before computing returns.
            ``None`` (default) leaves gaps intact.

        Returns
        -------
        pd.DataFrame
            Simple returns; same shape as ``prices`` minus the first row.

        Raises
        ------
        InsufficientDataError
            If fewer than ``min_periods + 1`` price observations are available.
        """
        if self._simple_returns is not None:
            return self._simple_returns

        prices = self._prepare_prices(fill_method)
        self._check_min_periods(prices, label="calculate_returns")

        returns = prices.pct_change(fill_method=None).iloc[1:]
        self._simple_returns = returns
        logger.debug("Simple returns computed | shape=%s", returns.shape)
        return returns

    # ------------------------------------------------------------------
    # 2. Log Returns
    # ------------------------------------------------------------------
    def calculate_log_returns(self, fill_method: Optional[str] = None) -> pd.DataFrame:
        """
        Compute continuously compounded (log) returns.

        .. math::
            r_t^{\\log} = \\ln\\!\\left(\\frac{P_t}{P_{t-1}}\\right)

        Log returns are time-additive and better suited to statistical modelling
        because they are approximately normally distributed for short horizons.

        Parameters
        ----------
        fill_method : {None, 'ffill', 'bfill'}, optional
            Fill strategy for NaN prices.

        Returns
        -------
        pd.DataFrame
            Log returns; same shape as ``prices`` minus the first row.

        Raises
        ------
        InsufficientDataError
            If fewer than ``min_periods + 1`` price observations are available.
        """
        if self._log_returns is not None:
            return self._log_returns

        prices = self._prepare_prices(fill_method)
        self._check_min_periods(prices, label="calculate_log_returns")

        log_returns = np.log(prices / prices.shift(1)).iloc[1:]
        self._log_returns = log_returns
        logger.debug("Log returns computed | shape=%s", log_returns.shape)
        return log_returns

    # ------------------------------------------------------------------
    # 3. Annualised Return
    # ------------------------------------------------------------------
    def annualized_return(
        self,
        method: Literal["geometric", "arithmetic"] = "geometric",
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute the annualised return for each asset.

        Parameters
        ----------
        method : {'geometric', 'arithmetic'}, optional
            - ``'geometric'`` (default): Compound annual growth rate (CAGR).
              Preferred for performance reporting.

              .. math::
                  \\text{CAGR} = \\left(\\prod_{t=1}^{T}(1+r_t)\\right)^{N/T} - 1

            - ``'arithmetic'``: Simple average scaled by the annualisation factor.

              .. math::
                  \\bar{r}_{\\text{ann}} = \\bar{r} \\times N

        Parameters
        ----------
        returns : pd.DataFrame, optional
            Pre-computed simple returns.  If ``None``, ``calculate_returns()``
            is called internally.

        Returns
        -------
        pd.Series
            Annualised return per asset (decimal).

        Raises
        ------
        InputValidationError
            If ``method`` is not one of the accepted literals.
        """
        self._validate_method(method, {"geometric", "arithmetic"}, "method")
        r = returns if returns is not None else self.calculate_returns()
        n = self.trading_days_per_year

        if method == "geometric":
            total_return = (1 + r).prod()
            periods = r.notna().sum()
            ann_ret = total_return ** (n / periods) - 1
        else:
            ann_ret = r.mean() * n

        ann_ret.name = "annualized_return"
        return ann_ret

    # ------------------------------------------------------------------
    # 4. Annualised Volatility
    # ------------------------------------------------------------------
    def annualized_volatility(
        self,
        ddof: int = 1,
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute annualised return volatility (standard deviation) for each asset.

        .. math::
            \\sigma_{\\text{ann}} = \\sigma_{\\text{daily}} \\times \\sqrt{N}

        Parameters
        ----------
        ddof : int, optional
            Delta degrees of freedom for the standard deviation estimator.
            Defaults to 1 (sample std).  Use 0 for population std.
        returns : pd.DataFrame, optional
            Pre-computed simple returns.

        Returns
        -------
        pd.Series
            Annualised volatility per asset (decimal).
        """
        r = returns if returns is not None else self.calculate_returns()
        vol = r.std(ddof=ddof) * np.sqrt(self.trading_days_per_year)
        vol.name = "annualized_volatility"
        return vol

    # ------------------------------------------------------------------
    # 5. Sharpe Ratio
    # ------------------------------------------------------------------
    def sharpe_ratio(
        self,
        risk_free_rate: float = 0.0,
        ddof: int = 1,
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute the annualised Sharpe ratio for each asset.

        .. math::
            SR = \\frac{\\mu_{\\text{ann}} - r_f}{\\sigma_{\\text{ann}}}

        Parameters
        ----------
        risk_free_rate : float, optional
            Annualised risk-free rate expressed as a decimal (e.g. 0.05 → 5 %).
            Defaults to 0.0.
        ddof : int, optional
            Degrees-of-freedom correction for volatility.  Default 1.
        returns : pd.DataFrame, optional
            Pre-computed simple returns.

        Returns
        -------
        pd.Series
            Sharpe ratio per asset.

        Notes
        -----
        Returns ``np.nan`` for assets with near-zero volatility to avoid
        spurious infinite ratios.
        """
        r = returns if returns is not None else self.calculate_returns()
        ann_ret = self.annualized_return(returns=r)
        ann_vol = self.annualized_volatility(ddof=ddof, returns=r)

        excess = ann_ret - risk_free_rate
        sharpe = excess / ann_vol.where(ann_vol.abs() > _EPSILON, other=np.nan)
        sharpe.name = "sharpe_ratio"
        return sharpe

    # ------------------------------------------------------------------
    # 6. Sortino Ratio
    # ------------------------------------------------------------------
    def sortino_ratio(
        self,
        risk_free_rate: float = 0.0,
        target_return: float = 0.0,
        ddof: int = 1,
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute the annualised Sortino ratio, penalising only downside volatility.

        .. math::
            \\text{Sortino} = \\frac{\\mu_{\\text{ann}} - r_f}{\\sigma_{\\text{down,ann}}}

        where the downside deviation uses only returns below ``target_return``:

        .. math::
            \\sigma_{\\text{down}} = \\sqrt{\\frac{\\sum_{r_t < \\tau}(r_t - \\tau)^2}{n-\\text{ddof}}}

        Parameters
        ----------
        risk_free_rate : float, optional
            Annualised risk-free rate.  Default 0.0.
        target_return : float, optional
            Minimum acceptable daily return (MAR).  Default 0.0.
        ddof : int, optional
            Degrees-of-freedom correction.  Default 1.
        returns : pd.DataFrame, optional
            Pre-computed simple returns.

        Returns
        -------
        pd.Series
            Sortino ratio per asset.
        """
        r = returns if returns is not None else self.calculate_returns()
        ann_ret = self.annualized_return(returns=r)

        downside = r.copy()
        downside[downside >= target_return] = 0.0
        downside_sq = downside ** 2

        n = downside.notna().sum()
        downside_std = np.sqrt(downside_sq.sum() / (n - ddof).clip(lower=1))
        ann_downside_std = downside_std * np.sqrt(self.trading_days_per_year)

        excess = ann_ret - risk_free_rate
        sortino = excess / ann_downside_std.where(
            ann_downside_std.abs() > _EPSILON, other=np.nan
        )
        sortino.name = "sortino_ratio"
        return sortino

    # ------------------------------------------------------------------
    # 7. Maximum Drawdown
    # ------------------------------------------------------------------
    def max_drawdown(
        self,
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Compute the maximum peak-to-trough drawdown for each asset.

        .. math::
            \\text{MDD} = \\min_t \\frac{W_t - \\max_{s \\le t} W_s}{\\max_{s \\le t} W_s}

        where :math:`W_t` is the cumulative wealth index.

        Parameters
        ----------
        returns : pd.DataFrame, optional
            Pre-computed simple returns.

        Returns
        -------
        pd.Series
            Maximum drawdown per asset (negative decimal, e.g. −0.35 → −35 %).

        Notes
        -----
        The function also logs the drawdown period (peak date and trough date)
        at DEBUG level for audit purposes.
        """
        r = returns if returns is not None else self.calculate_returns()
        wealth = (1 + r).cumprod()
        rolling_max = wealth.cummax()
        drawdown = wealth / rolling_max - 1
        mdd = drawdown.min()
        mdd.name = "max_drawdown"

        # Log peak/trough dates at debug level
        for asset in drawdown.columns:
            trough_idx = drawdown[asset].idxmin()
            if pd.notna(trough_idx):
                peak_mask = wealth[asset][:trough_idx]
                if not peak_mask.empty:
                    peak_idx = peak_mask.idxmax()
                    logger.debug(
                        "MDD | asset=%s | peak=%s | trough=%s | mdd=%.4f",
                        asset,
                        peak_idx,
                        trough_idx,
                        mdd[asset],
                    )
        return mdd

    # ------------------------------------------------------------------
    # 8. Value at Risk (VaR)
    # ------------------------------------------------------------------
    def value_at_risk(
        self,
        confidence_level: float = 0.95,
        method: Literal["historical", "parametric"] = "historical",
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Estimate the Value at Risk (VaR) at a given confidence level.

        VaR represents the loss threshold that is *not exceeded* with probability
        equal to ``confidence_level`` over a single holding period (one day).

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level in (0, 1).  Default 0.95.
        method : {'historical', 'parametric'}, optional
            - ``'historical'``: Empirical quantile of the return distribution.
            - ``'parametric'``: Gaussian approximation using sample mean and
              standard deviation.
        returns : pd.DataFrame, optional
            Pre-computed simple returns.

        Returns
        -------
        pd.Series
            VaR per asset (positive value represents a loss, e.g. 0.02 → 2 % loss).

        Raises
        ------
        InputValidationError
            If ``confidence_level`` is not in (0, 1).
        """
        self._validate_confidence(confidence_level)
        self._validate_method(method, {"historical", "parametric"}, "method")
        r = returns if returns is not None else self.calculate_returns()
        alpha = 1.0 - confidence_level

        if method == "historical":
            var = -r.quantile(alpha)
        else:
            mu = r.mean()
            sigma = r.std(ddof=1)
            z = stats.norm.ppf(alpha)
            var = -(mu + z * sigma)

        var.name = f"var_{int(confidence_level * 100)}"
        return var

    # ------------------------------------------------------------------
    # 9. Conditional Value at Risk (CVaR / Expected Shortfall)
    # ------------------------------------------------------------------
    def conditional_value_at_risk(
        self,
        confidence_level: float = 0.95,
        method: Literal["historical", "parametric"] = "historical",
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Estimate the Conditional Value at Risk (CVaR), also known as Expected
        Shortfall (ES).

        CVaR is the expected loss *given* that the loss exceeds VaR.  It is a
        coherent risk measure and is preferred by Basel III / IV frameworks.

        .. math::
            \\text{CVaR}_{\\alpha} = -\\mathbb{E}\\left[r_t \\mid r_t < \\text{VaR}_{\\alpha}\\right]

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level in (0, 1).  Default 0.95.
        method : {'historical', 'parametric'}, optional
            - ``'historical'``: Average of returns in the tail below VaR.
            - ``'parametric'``: Closed-form Gaussian ES.
        returns : pd.DataFrame, optional
            Pre-computed simple returns.

        Returns
        -------
        pd.Series
            CVaR per asset (positive value represents expected tail loss).

        Raises
        ------
        InputValidationError
            If ``confidence_level`` is not in (0, 1).
        """
        self._validate_confidence(confidence_level)
        self._validate_method(method, {"historical", "parametric"}, "method")
        r = returns if returns is not None else self.calculate_returns()
        alpha = 1.0 - confidence_level

        if method == "historical":
            threshold = r.quantile(alpha)
            # Average of returns strictly below the VaR threshold per asset
            cvar = r.apply(
                lambda col: -col[col <= threshold[col.name]].mean()
                if col.notna().sum() > 0
                else np.nan
            )
        else:
            mu = r.mean()
            sigma = r.std(ddof=1)
            z = stats.norm.ppf(alpha)
            # Closed-form Gaussian ES: ES = -(mu - sigma * phi(z) / alpha)
            phi_z = stats.norm.pdf(z)
            cvar = -(mu - sigma * phi_z / alpha)

        cvar.name = f"cvar_{int(confidence_level * 100)}"
        return cvar

    # ------------------------------------------------------------------
    # 10. Correlation Matrix
    # ------------------------------------------------------------------
    def correlation_matrix(
        self,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        min_periods: Optional[int] = None,
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute the pairwise return correlation matrix.

        Parameters
        ----------
        method : {'pearson', 'spearman', 'kendall'}, optional
            Correlation estimator.  Defaults to ``'pearson'`` (linear).
            Use ``'spearman'`` or ``'kendall'`` for rank-based estimates robust
            to heavy tails and outliers.
        min_periods : int, optional
            Minimum number of overlapping observations required for each pair.
            Defaults to ``self.min_periods``.
        returns : pd.DataFrame, optional
            Pre-computed simple returns.

        Returns
        -------
        pd.DataFrame
            Square correlation matrix with asset names as both index and columns.
            Diagonal elements are 1.0 by construction.

        Raises
        ------
        InputValidationError
            If ``method`` is not one of the accepted literals.
        """
        self._validate_method(method, {"pearson", "spearman", "kendall"}, "method")
        r = returns if returns is not None else self.calculate_returns()
        mp = min_periods if min_periods is not None else self.min_periods
        corr = r.corr(method=method, min_periods=mp)
        logger.debug("Correlation matrix computed | method=%s | shape=%s", method, corr.shape)
        return corr

    # ------------------------------------------------------------------
    # 11. Covariance Matrix
    # ------------------------------------------------------------------
    def covariance_matrix(
        self,
        annualized: bool = True,
        ddof: int = 1,
        min_periods: Optional[int] = None,
        returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute the pairwise return covariance matrix.

        Parameters
        ----------
        annualized : bool, optional
            If ``True`` (default), multiply by ``trading_days_per_year`` to
            produce an annualised covariance matrix (required for mean-variance
            optimisation at an annual horizon).
        ddof : int, optional
            Delta degrees of freedom.  Default 1 (unbiased sample estimate).
        min_periods : int, optional
            Minimum overlapping observations per pair.
        returns : pd.DataFrame, optional
            Pre-computed simple returns.

        Returns
        -------
        pd.DataFrame
            (Annualised) covariance matrix.

        Notes
        -----
        The diagonal of an annualised covariance matrix equals the squared
        annualised volatility for each asset.
        """
        r = returns if returns is not None else self.calculate_returns()
        mp = min_periods if min_periods is not None else self.min_periods
        cov = r.cov(ddof=ddof, min_periods=mp)

        if annualized:
            cov = cov * self.trading_days_per_year

        label = "annualised" if annualized else "daily"
        logger.debug("Covariance matrix computed | %s | shape=%s", label, cov.shape)
        return cov

    # ------------------------------------------------------------------
    # Convenience: Full Risk Summary
    # ------------------------------------------------------------------
    def risk_summary(
        self,
        risk_free_rate: float = 0.0,
        confidence_level: float = 0.95,
        var_method: Literal["historical", "parametric"] = "historical",
    ) -> pd.DataFrame:
        """
        Produce a consolidated risk and performance summary table.

        Computes all scalar metrics in a single pass to avoid redundant return
        calculations.

        Parameters
        ----------
        risk_free_rate : float, optional
            Annualised risk-free rate.  Default 0.0.
        confidence_level : float, optional
            Confidence level for VaR and CVaR.  Default 0.95.
        var_method : {'historical', 'parametric'}, optional
            VaR/CVaR estimation method.  Default ``'historical'``.

        Returns
        -------
        pd.DataFrame
            Summary DataFrame with metrics as rows and assets as columns.
        """
        r = self.calculate_returns()

        metrics = {
            "annualized_return": self.annualized_return(returns=r),
            "annualized_volatility": self.annualized_volatility(returns=r),
            "sharpe_ratio": self.sharpe_ratio(risk_free_rate=risk_free_rate, returns=r),
            "sortino_ratio": self.sortino_ratio(risk_free_rate=risk_free_rate, returns=r),
            "max_drawdown": self.max_drawdown(returns=r),
            f"var_{int(confidence_level * 100)}": self.value_at_risk(
                confidence_level=confidence_level, method=var_method, returns=r
            ),
            f"cvar_{int(confidence_level * 100)}": self.conditional_value_at_risk(
                confidence_level=confidence_level, method=var_method, returns=r
            ),
        }
        summary = pd.DataFrame(metrics).T
        summary.index.name = "metric"
        return summary

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_prices(prices: pd.DataFrame) -> None:
        """Raise ``InputValidationError`` for malformed price inputs."""
        if not isinstance(prices, pd.DataFrame):
            raise InputValidationError(
                f"prices must be a pd.DataFrame, got {type(prices).__name__}."
            )
        if prices.empty:
            raise InputValidationError("prices DataFrame is empty.")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise InputValidationError(
                "prices must have a DatetimeIndex. "
                f"Got {type(prices.index).__name__}."
            )
        if prices.index.duplicated().any():
            raise InputValidationError(
                "prices index contains duplicate timestamps. "
                "Deduplicate before passing to AnalyticsEngine."
            )
        if (prices.select_dtypes(include=[np.number]) < 0).any().any():
            warnings.warn(
                "prices contains negative values; verify this is intentional.",
                UserWarning,
                stacklevel=3,
            )

    def _prepare_prices(self, fill_method: Optional[str]) -> pd.DataFrame:
        """Apply optional fill strategy to the internal price DataFrame."""
        if fill_method == "ffill":
            return self._prices.ffill()
        elif fill_method == "bfill":
            return self._prices.bfill()
        elif fill_method is None:
            return self._prices
        else:
            raise InputValidationError(
                f"fill_method must be None, 'ffill', or 'bfill'; got '{fill_method}'."
            )

    def _check_min_periods(self, prices: pd.DataFrame, label: str) -> None:
        """Raise ``InsufficientDataError`` if price history is too short."""
        min_obs = prices.notna().sum().min()
        required = self.min_periods + 1  # +1 because pct_change drops first row
        if min_obs < required:
            raise InsufficientDataError(
                f"{label}: insufficient data. "
                f"Need at least {required} observations; "
                f"minimum across assets is {min_obs}."
            )

    @staticmethod
    def _validate_confidence(confidence_level: float) -> None:
        """Ensure confidence level is strictly within (0, 1)."""
        if not (0.0 < confidence_level < 1.0):
            raise InputValidationError(
                f"confidence_level must be in (0, 1); got {confidence_level}."
            )

    @staticmethod
    def _validate_method(value: str, valid: set, param_name: str) -> None:
        """Ensure a method argument is one of the accepted options."""
        if value not in valid:
            raise InputValidationError(
                f"Invalid {param_name}='{value}'. Must be one of {sorted(valid)}."
            )


# ---------------------------------------------------------------------------
# Demonstration main()
# ---------------------------------------------------------------------------
def _generate_synthetic_prices(
    tickers: list[str],
    n_days: int = 756,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic GBM price paths for demonstration purposes.

    Uses Geometric Brownian Motion with asset-specific drift and volatility
    parameters to produce realistic-looking price series.

    Parameters
    ----------
    tickers : list[str]
        List of asset identifiers.
    n_days : int
        Number of trading-day observations to generate.
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic price DataFrame with a DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    params = {
        "AAPL":  {"mu": 0.25, "sigma": 0.28, "S0": 150.0},
        "MSFT":  {"mu": 0.22, "sigma": 0.24, "S0": 280.0},
        "GOOGL": {"mu": 0.18, "sigma": 0.26, "S0": 2800.0},
        "AMZN":  {"mu": 0.20, "sigma": 0.30, "S0": 3400.0},
        "TSLA":  {"mu": 0.30, "sigma": 0.55, "S0": 700.0},
    }

    dt = 1 / TRADING_DAYS_PER_YEAR
    dates = pd.bdate_range("2021-01-04", periods=n_days, freq="B")
    data: dict[str, np.ndarray] = {}

    for ticker in tickers:
        p = params.get(ticker, {"mu": 0.15, "sigma": 0.25, "S0": 100.0})
        z = rng.standard_normal(n_days)
        log_returns = (p["mu"] - 0.5 * p["sigma"] ** 2) * dt + p["sigma"] * np.sqrt(dt) * z
        prices = p["S0"] * np.exp(np.cumsum(log_returns))
        data[ticker] = prices

    return pd.DataFrame(data, index=dates)


def _print_section(title: str) -> None:
    width = 72
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def main() -> None:
    """
    Demonstration of the AnalyticsEngine with a synthetic five-asset portfolio.

    Exercises every public method and prints formatted results to stdout.
    Suitable as an integration test and as documentation-by-example.
    """
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    print("\n" + "╔" + "═" * 70 + "╗")
    print("║" + "  QuantLab – Analytics Engine Demo".center(70) + "║")
    print("╚" + "═" * 70 + "╝")

    # ── 0. Synthetic prices (stand-in for market_data.py) ──────────────────
    prices = _generate_synthetic_prices(tickers, n_days=756)
    print(f"\n› Synthetic prices generated | shape={prices.shape}")
    print(prices.tail(3).to_string())

    # ── Instantiate engine ──────────────────────────────────────────────────
    engine = AnalyticsEngine(prices, trading_days_per_year=252, min_periods=30)

    # ── 1. Simple returns ───────────────────────────────────────────────────
    _print_section("1. Simple Returns (last 3 rows)")
    r = engine.calculate_returns()
    print(r.tail(3).map(lambda x: f"{x:+.4%}").to_string())

    # ── 2. Log returns ──────────────────────────────────────────────────────
    _print_section("2. Log Returns (last 3 rows)")
    lr = engine.calculate_log_returns()
    print(lr.tail(3).map(lambda x: f"{x:+.4%}").to_string())

    # ── 3. Annualised return ────────────────────────────────────────────────
    _print_section("3. Annualised Return (Geometric CAGR)")
    ann_ret = engine.annualized_return(returns=r)
    for ticker, val in ann_ret.items():
        print(f"  {ticker:6s}  {val:+.2%}")

    # ── 4. Annualised volatility ────────────────────────────────────────────
    _print_section("4. Annualised Volatility")
    ann_vol = engine.annualized_volatility(returns=r)
    for ticker, val in ann_vol.items():
        print(f"  {ticker:6s}  {val:.2%}")

    # ── 5. Sharpe ratio ─────────────────────────────────────────────────────
    _print_section("5. Sharpe Ratio (Rf = 4.5 %)")
    sharpe = engine.sharpe_ratio(risk_free_rate=0.045, returns=r)
    for ticker, val in sharpe.items():
        print(f"  {ticker:6s}  {val:.4f}")

    # ── 6. Sortino ratio ────────────────────────────────────────────────────
    _print_section("6. Sortino Ratio (Rf = 4.5 %)")
    sortino = engine.sortino_ratio(risk_free_rate=0.045, returns=r)
    for ticker, val in sortino.items():
        print(f"  {ticker:6s}  {val:.4f}")

    # ── 7. Maximum drawdown ─────────────────────────────────────────────────
    _print_section("7. Maximum Drawdown")
    mdd = engine.max_drawdown(returns=r)
    for ticker, val in mdd.items():
        print(f"  {ticker:6s}  {val:.2%}")

    # ── 8. Value at Risk ────────────────────────────────────────────────────
    _print_section("8. Value at Risk (95 %, Historical)")
    var_95 = engine.value_at_risk(confidence_level=0.95, method="historical", returns=r)
    for ticker, val in var_95.items():
        print(f"  {ticker:6s}  {val:.4%}")

    # ── 9. Conditional VaR ─────────────────────────────────────────────────
    _print_section("9. Conditional VaR / Expected Shortfall (95 %, Historical)")
    cvar_95 = engine.conditional_value_at_risk(
        confidence_level=0.95, method="historical", returns=r
    )
    for ticker, val in cvar_95.items():
        print(f"  {ticker:6s}  {val:.4%}")

    # ── 10. Correlation matrix ──────────────────────────────────────────────
    _print_section("10. Pearson Correlation Matrix")
    corr = engine.correlation_matrix(method="pearson", returns=r)
    print(corr.map(lambda x: f"{x:.4f}").to_string())

    # ── 11. Covariance matrix ───────────────────────────────────────────────
    _print_section("11. Annualised Covariance Matrix")
    cov = engine.covariance_matrix(annualized=True, returns=r)
    print(cov.map(lambda x: f"{x:.6f}").to_string())

    # ── Full risk summary ───────────────────────────────────────────────────
    _print_section("12. Full Risk Summary (Rf = 4.5 %, VaR 95 %)")
    summary = engine.risk_summary(risk_free_rate=0.045, confidence_level=0.95)
    fmt = summary.map(lambda x: f"{x:+.4f}" if pd.notna(x) else "N/A")
    print(fmt.to_string())

    print("\n✔  Demo complete.\n")


if __name__ == "__main__":
    main()
