"""
regime_engine.py
=================

Market Regime Detection Engine for QuantLab.

This module analyzes historical price data for a basket of assets and
classifies the prevailing market regime (Bull, Bear, High Volatility,
Recovery, or Sideways) based on trailing return, volatility, drawdown,
and momentum statistics computed on an equal-weight portfolio.

The engine is designed for institutional portfolio analytics workflows:
it is deterministic, side-effect free, and returns a structured,
auditable result (`RegimeResult`) suitable for downstream reporting,
risk dashboards, or automated allocation logic.

Typical usage
-------------
    import pandas as pd
    from regime_engine import detect_market_regime, regime_color, regime_description

    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
    result = detect_market_regime(prices)

    print(result.regime, result.confidence)
    print(regime_color(result.regime))
    print(regime_description(result.regime))
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
import pandas as pd

__all__ = [
    "MarketRegime",
    "RegimeResult",
    "detect_market_regime",
    "regime_color",
    "regime_description",
]


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: Final[int] = 252
MOMENTUM_SHORT_WINDOW: Final[int] = 63   # ~3 months
MOMENTUM_LONG_WINDOW: Final[int] = 252   # ~1 year

# Classification thresholds (kept as named constants for transparency
# and so they can be tuned/audited without touching classification logic)
BULL_RETURN_THRESHOLD: Final[float] = 0.10
BULL_VOL_THRESHOLD: Final[float] = 0.25

BEAR_RETURN_THRESHOLD: Final[float] = -0.05
BEAR_DRAWDOWN_THRESHOLD: Final[float] = -0.20

HIGH_VOL_THRESHOLD: Final[float] = 0.30


# --------------------------------------------------------------------------
# Enums and Data Structures
# --------------------------------------------------------------------------

class MarketRegime(Enum):
    """
    Enumeration of supported market regime classifications.

    Members
    -------
    BULL : str
        Sustained uptrend with controlled volatility.
    BEAR : str
        Sustained downtrend with significant drawdown.
    HIGH_VOL : str
        Elevated realized volatility, regardless of directional trend.
    RECOVERY : str
        Positive short-term momentum following negative long-term momentum,
        indicative of a market rebounding from a prior decline.
    SIDEWAYS : str
        No dominant trend, volatility, or drawdown signal is present.
    """

    BULL = "Bull Market"
    BEAR = "Bear Market"
    HIGH_VOL = "High Volatility"
    RECOVERY = "Recovery"
    SIDEWAYS = "Sideways"


@dataclass(frozen=True)
class RegimeResult:
    """
    Structured output of a market regime detection run.

    Attributes
    ----------
    regime : MarketRegime
        The classified market regime.
    confidence : float
        A score in [0.0, 1.0] indicating how strongly the underlying
        statistics support the assigned regime. Higher values indicate
        the metrics are further past the relevant classification
        threshold(s); lower values indicate a borderline classification.
    annual_return : float
        Annualized return of the equal-weight portfolio over the
        observed sample period (e.g. 0.12 == 12%).
    annual_volatility : float
        Annualized volatility (standard deviation of daily returns,
        scaled by sqrt(252)) of the equal-weight portfolio.
    drawdown : float
        Maximum drawdown of the equal-weight portfolio over the sample
        period, expressed as a negative fraction (e.g. -0.25 == -25%).
    momentum_63d : float
        Trailing ~3-month (63 trading day) price momentum, expressed
        as a simple return over that window.
    momentum_252d : float
        Trailing ~1-year (252 trading day) price momentum, expressed
        as a simple return over that window.
    """

    regime: MarketRegime
    confidence: float
    annual_return: float
    annual_volatility: float
    drawdown: float
    momentum_63d: float
    momentum_252d: float


# --------------------------------------------------------------------------
# Core Statistics
# --------------------------------------------------------------------------

def _build_equal_weight_portfolio(prices: pd.DataFrame) -> pd.Series:
    """
    Construct an equal-weight portfolio price index from a multi-asset
    price DataFrame.

    Each asset's price series is normalized to a base value of 1.0 at
    the first observation, the normalized series are averaged
    cross-sectionally (equal weight), and the result is treated as a
    synthetic portfolio price index.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide-format DataFrame of asset prices, indexed by date, with
        one column per asset. Missing values are forward-filled then
        back-filled to handle non-overlapping listing histories.

    Returns
    -------
    pd.Series
        Synthetic equal-weight portfolio price index, indexed by date.

    Raises
    ------
    ValueError
        If `prices` is empty or contains no usable numeric columns.
    """
    if prices.empty:
        raise ValueError("`prices` DataFrame is empty; cannot detect regime.")

    numeric_prices = prices.select_dtypes(include=[np.number])
    if numeric_prices.empty:
        raise ValueError("`prices` DataFrame has no numeric price columns.")

    # Handle gaps (e.g. assets with different listing histories or holidays)
    clean_prices = numeric_prices.ffill().bfill()

    if clean_prices.isna().any().any():
        # Entire columns are NaN (e.g. an all-NaN asset column) -- drop them.
        clean_prices = clean_prices.dropna(axis=1, how="all")

    if clean_prices.shape[1] == 0:
        raise ValueError("No valid asset price series remain after cleaning.")

    # Normalize each asset to start at 1.0, then equal-weight average.
    normalized = clean_prices / clean_prices.iloc[0]
    portfolio_index = normalized.mean(axis=1)
    portfolio_index.name = "equal_weight_portfolio"

    return portfolio_index


def _compute_daily_returns(portfolio_index: pd.Series) -> pd.Series:
    """
    Compute simple daily returns from a portfolio price index.

    Parameters
    ----------
    portfolio_index : pd.Series
        Synthetic portfolio price index, indexed by date.

    Returns
    -------
    pd.Series
        Daily simple returns, with the leading NaN (from the first
        differencing operation) dropped.
    """
    daily_returns = portfolio_index.pct_change().dropna()
    return daily_returns


def _annualized_return(daily_returns: pd.Series) -> float:
    """
    Compute the annualized return from a series of daily returns,
    using geometric compounding over the observed sample length.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily simple returns.

    Returns
    -------
    float
        Annualized return (e.g. 0.10 == 10% per year). Returns 0.0 if
        there is insufficient data to compute a meaningful figure.
    """
    n_obs = len(daily_returns)
    if n_obs == 0:
        return 0.0

    cumulative_growth = float((1.0 + daily_returns).prod())
    if cumulative_growth <= 0:
        # Portfolio value went to zero or negative (degenerate case)
        return -1.0

    years = n_obs / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0

    annual_return = cumulative_growth ** (1.0 / years) - 1.0
    return float(annual_return)


def _annualized_volatility(daily_returns: pd.Series) -> float:
    """
    Compute annualized volatility (standard deviation) from daily returns.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily simple returns.

    Returns
    -------
    float
        Annualized volatility (e.g. 0.20 == 20% per year). Returns 0.0
        if there is insufficient data (fewer than 2 observations).
    """
    if len(daily_returns) < 2:
        return 0.0

    daily_vol = float(daily_returns.std(ddof=1))
    return daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)


def _max_drawdown(portfolio_index: pd.Series) -> float:
    """
    Compute the maximum peak-to-trough drawdown of a portfolio price index.

    Parameters
    ----------
    portfolio_index : pd.Series
        Synthetic portfolio price index, indexed by date.

    Returns
    -------
    float
        Maximum drawdown expressed as a negative fraction
        (e.g. -0.30 == -30% peak-to-trough decline). Returns 0.0 if
        there is insufficient data.
    """
    if len(portfolio_index) < 2:
        return 0.0

    running_max = portfolio_index.cummax()
    drawdown_series = portfolio_index / running_max - 1.0
    return float(drawdown_series.min())


def _momentum(portfolio_index: pd.Series, window: int) -> float:
    """
    Compute trailing price momentum over a given lookback window.

    Momentum is defined as the simple return from the price observed
    `window` trading days ago to the most recent price.

    Parameters
    ----------
    portfolio_index : pd.Series
        Synthetic portfolio price index, indexed by date.
    window : int
        Lookback window length, in trading days.

    Returns
    -------
    float
        Simple return over the lookback window. Returns 0.0 if there
        is insufficient history to cover the full window.
    """
    if len(portfolio_index) <= window:
        return 0.0

    current_price = float(portfolio_index.iloc[-1])
    past_price = float(portfolio_index.iloc[-1 - window])

    if past_price == 0:
        return 0.0

    return (current_price / past_price) - 1.0


# --------------------------------------------------------------------------
# Confidence Scoring
# --------------------------------------------------------------------------

def _scale_distance(value: float, threshold: float, scale: float) -> float:
    """
    Convert a raw distance-from-threshold into a [0, 1] confidence
    contribution using a saturating linear scale.

    Parameters
    ----------
    value : float
        Observed metric value.
    threshold : float
        Classification threshold the metric was compared against.
    scale : float
        Normalization constant representing the distance at which
        confidence saturates to 1.0. Must be positive.

    Returns
    -------
    float
        A value in [0.0, 1.0] proportional to how far `value` is past
        `threshold`, relative to `scale`.
    """
    if scale <= 0:
        return 0.0
    distance = abs(value - threshold)
    return float(np.clip(distance / scale, 0.0, 1.0))


def _confidence_bull(annual_return: float, momentum_63d: float, annual_volatility: float) -> float:
    """
    Confidence score for a BULL classification, derived from how far
    return is above its threshold, momentum is above zero, and
    volatility is below its ceiling. Combined via simple averaging.
    """
    return_score = _scale_distance(annual_return, BULL_RETURN_THRESHOLD, scale=0.20)
    momentum_score = _scale_distance(momentum_63d, 0.0, scale=0.10)
    vol_score = _scale_distance(annual_volatility, BULL_VOL_THRESHOLD, scale=0.15)
    return float(np.mean([return_score, momentum_score, vol_score]))


def _confidence_bear(annual_return: float, drawdown: float) -> float:
    """
    Confidence score for a BEAR classification, derived from how far
    return is below its threshold and drawdown exceeds its threshold.
    """
    return_score = _scale_distance(annual_return, BEAR_RETURN_THRESHOLD, scale=0.20)
    drawdown_score = _scale_distance(drawdown, BEAR_DRAWDOWN_THRESHOLD, scale=0.20)
    return float(np.mean([return_score, drawdown_score]))


def _confidence_high_vol(annual_volatility: float) -> float:
    """
    Confidence score for a HIGH_VOL classification, derived from how
    far volatility exceeds its threshold.
    """
    return _scale_distance(annual_volatility, HIGH_VOL_THRESHOLD, scale=0.25)


def _confidence_recovery(momentum_63d: float, momentum_252d: float) -> float:
    """
    Confidence score for a RECOVERY classification, derived from the
    strength of the short-term rebound and the depth of the prior
    long-term decline.
    """
    momentum_score = _scale_distance(momentum_63d, 0.0, scale=0.10)
    decline_score = _scale_distance(momentum_252d, 0.0, scale=0.20)
    return float(np.mean([momentum_score, decline_score]))


def _confidence_sideways(
    annual_return: float,
    annual_volatility: float,
    drawdown: float,
    momentum_63d: float,
) -> float:
    """
    Confidence score for a SIDEWAYS classification. Since SIDEWAYS is a
    "none of the above" classification, confidence is highest when all
    metrics sit comfortably *within* their respective bull/bear/high-vol
    boundaries (i.e. far from every other regime's threshold).
    """
    return_margin = _scale_distance(
        np.clip(annual_return, BEAR_RETURN_THRESHOLD, BULL_RETURN_THRESHOLD),
        annual_return,
        scale=0.10,
    )
    # Simpler, robust formulation: average of "headroom" before nearest
    # competing threshold is breached, normalized to [0, 1].
    headroom_bull = max(BULL_RETURN_THRESHOLD - annual_return, 0.0) / 0.20
    headroom_bear = max(annual_return - BEAR_RETURN_THRESHOLD, 0.0) / 0.20
    headroom_vol = max(HIGH_VOL_THRESHOLD - annual_volatility, 0.0) / 0.25
    headroom_dd = max(drawdown - BEAR_DRAWDOWN_THRESHOLD, 0.0) / 0.20

    score = np.mean([
        np.clip(headroom_bull, 0.0, 1.0),
        np.clip(headroom_bear, 0.0, 1.0),
        np.clip(headroom_vol, 0.0, 1.0),
        np.clip(headroom_dd, 0.0, 1.0),
    ])
    return float(score)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def _classify(
    annual_return: float,
    annual_volatility: float,
    drawdown: float,
    momentum_63d: float,
    momentum_252d: float,
) -> tuple[MarketRegime, float]:
    """
    Apply the rule-based regime classification logic to a set of
    portfolio statistics.

    Rules are evaluated in the following priority order: BULL, BEAR,
    HIGH_VOL, RECOVERY, then SIDEWAYS as the default. Priority order
    matters because more than one rule's conditions can technically be
    satisfied simultaneously (e.g. a high-return, high-vol market);
    BULL and BEAR are checked first since they represent the strongest,
    most economically meaningful directional signals.

    Parameters
    ----------
    annual_return : float
        Annualized portfolio return.
    annual_volatility : float
        Annualized portfolio volatility.
    drawdown : float
        Maximum drawdown (negative fraction).
    momentum_63d : float
        63-day trailing momentum.
    momentum_252d : float
        252-day trailing momentum.

    Returns
    -------
    tuple[MarketRegime, float]
        The classified regime and its associated confidence score.
    """
    is_bull = (
        annual_return > BULL_RETURN_THRESHOLD
        and momentum_63d > 0
        and annual_volatility < BULL_VOL_THRESHOLD
    )
    if is_bull:
        confidence = _confidence_bull(annual_return, momentum_63d, annual_volatility)
        return MarketRegime.BULL, confidence

    is_bear = (
        annual_return < BEAR_RETURN_THRESHOLD
        and drawdown < BEAR_DRAWDOWN_THRESHOLD
    )
    if is_bear:
        confidence = _confidence_bear(annual_return, drawdown)
        return MarketRegime.BEAR, confidence

    is_high_vol = annual_volatility > HIGH_VOL_THRESHOLD
    if is_high_vol:
        confidence = _confidence_high_vol(annual_volatility)
        return MarketRegime.HIGH_VOL, confidence

    is_recovery = momentum_63d > 0 and momentum_252d < 0
    if is_recovery:
        confidence = _confidence_recovery(momentum_63d, momentum_252d)
        return MarketRegime.RECOVERY, confidence

    confidence = _confidence_sideways(annual_return, annual_volatility, drawdown, momentum_63d)
    return MarketRegime.SIDEWAYS, confidence


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def detect_market_regime(prices: pd.DataFrame) -> RegimeResult:
    """
    Detect the prevailing market regime from historical asset prices.

    An equal-weight portfolio is constructed from all asset columns in
    `prices`. Annualized return, annualized volatility, maximum
    drawdown, and trailing 63-day / 252-day momentum are computed on
    that portfolio, and a rule-based classifier maps these statistics
    to a `MarketRegime`.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide-format DataFrame of historical asset prices. Must be
        indexed by date (ascending order is assumed) with one column
        per asset and at least 2 rows of price history. Non-numeric
        columns are ignored. Missing values are forward/backward
        filled to accommodate assets with differing listing histories.

    Returns
    -------
    RegimeResult
        Structured result containing the classified regime, a
        confidence score in [0, 1], and the underlying portfolio
        statistics used to derive the classification.

    Raises
    ------
    ValueError
        If `prices` is empty, contains no numeric columns, or has
        fewer than 2 valid rows of price data.

    Notes
    -----
    - Momentum figures default to 0.0 when the price history is
      shorter than the corresponding lookback window (63 or 252
      trading days), since momentum cannot be meaningfully computed
      over an incomplete window.
    - This function is purely deterministic and has no side effects;
      it does not mutate the input DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> dates = pd.date_range("2023-01-01", periods=300, freq="B")
    >>> rng = np.random.default_rng(42)
    >>> data = {
    ...     "AAPL": 150 * (1 + rng.normal(0.0006, 0.01, len(dates))).cumprod(),
    ...     "MSFT": 250 * (1 + rng.normal(0.0006, 0.01, len(dates))).cumprod(),
    ... }
    >>> prices = pd.DataFrame(data, index=dates)
    >>> result = detect_market_regime(prices)
    >>> isinstance(result.regime, MarketRegime)
    True
    """
    portfolio_index = _build_equal_weight_portfolio(prices)

    if len(portfolio_index) < 2:
        raise ValueError(
            "Insufficient price history: at least 2 valid observations "
            "are required to detect a market regime."
        )

    daily_returns = _compute_daily_returns(portfolio_index)

    annual_return = _annualized_return(daily_returns)
    annual_volatility = _annualized_volatility(daily_returns)
    drawdown = _max_drawdown(portfolio_index)
    momentum_63d = _momentum(portfolio_index, MOMENTUM_SHORT_WINDOW)
    momentum_252d = _momentum(portfolio_index, MOMENTUM_LONG_WINDOW)

    regime, confidence = _classify(
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        drawdown=drawdown,
        momentum_63d=momentum_63d,
        momentum_252d=momentum_252d,
    )

    return RegimeResult(
        regime=regime,
        confidence=round(confidence, 4),
        annual_return=round(annual_return, 6),
        annual_volatility=round(annual_volatility, 6),
        drawdown=round(drawdown, 6),
        momentum_63d=round(momentum_63d, 6),
        momentum_252d=round(momentum_252d, 6),
    )


def regime_color(regime: MarketRegime) -> str:
    """
    Return a display color associated with a given market regime, for
    use in dashboards, charts, and reporting UIs.

    Parameters
    ----------
    regime : MarketRegime
        The market regime to look up.

    Returns
    -------
    str
        A lowercase color name:
        - BULL      -> "green"
        - BEAR      -> "red"
        - HIGH_VOL  -> "orange"
        - RECOVERY  -> "blue"
        - SIDEWAYS  -> "gray"

    Raises
    ------
    ValueError
        If `regime` is not a recognized `MarketRegime` member.
    """
    color_map: dict[MarketRegime, str] = {
        MarketRegime.BULL: "green",
        MarketRegime.BEAR: "red",
        MarketRegime.HIGH_VOL: "orange",
        MarketRegime.RECOVERY: "blue",
        MarketRegime.SIDEWAYS: "gray",
    }
    try:
        return color_map[regime]
    except KeyError as exc:
        raise ValueError(f"Unrecognized MarketRegime: {regime!r}") from exc


def regime_description(regime: MarketRegime) -> str:
    """
    Return an institutional-quality narrative description of a given
    market regime, suitable for inclusion in client-facing reports,
    risk commentary, or portfolio review decks.

    Parameters
    ----------
    regime : MarketRegime
        The market regime to describe.

    Returns
    -------
    str
        A multi-sentence description summarizing the regime's
        characteristics and typical portfolio implications.

    Raises
    ------
    ValueError
        If `regime` is not a recognized `MarketRegime` member.
    """
    description_map: dict[MarketRegime, str] = {
        MarketRegime.BULL: (
            "Bull Market: The portfolio is exhibiting sustained positive "
            "returns alongside positive short-term momentum and contained "
            "volatility. Risk assets are broadly favored in this regime, "
            "and historical drawdowns tend to be shallow and short-lived. "
            "Strategic positioning typically emphasizes growth exposure "
            "while maintaining discipline around valuation and concentration risk."
        ),
        MarketRegime.BEAR: (
            "Bear Market: The portfolio is experiencing a sustained decline "
            "in value, marked by negative annualized returns and a "
            "significant peak-to-trough drawdown. Capital preservation, "
            "downside hedging, and reduced gross exposure are typically "
            "prioritized in this regime. Elevated correlation across risk "
            "assets often limits the effectiveness of traditional diversification."
        ),
        MarketRegime.HIGH_VOL: (
            "High Volatility: Realized volatility is elevated well beyond "
            "historical norms, irrespective of the prevailing directional "
            "trend. This regime is often associated with macroeconomic "
            "uncertainty, liquidity stress, or rapid repricing of risk. "
            "Position sizing, tail-risk hedging, and stress testing warrant "
            "heightened attention until volatility normalizes."
        ),
        MarketRegime.RECOVERY: (
            "Recovery: Short-term momentum has turned positive following a "
            "period of negative longer-term performance, suggesting the "
            "portfolio may be rebounding from a prior drawdown or correction. "
            "This regime often precedes a renewed uptrend but can also "
            "represent a temporary relief rally within a broader downtrend; "
            "confirmation from breadth and volume indicators is advisable "
            "before increasing risk materially."
        ),
        MarketRegime.SIDEWAYS: (
            "Sideways: No dominant directional trend, elevated volatility, "
            "or significant drawdown signal is currently present. Returns "
            "are range-bound and the portfolio lacks a clear regime bias. "
            "This environment often favors relative-value, carry, or "
            "mean-reversion strategies over directional risk-taking."
        ),
    }
    try:
        return description_map[regime]
    except KeyError as exc:
        raise ValueError(f"Unrecognized MarketRegime: {regime!r}") from exc

