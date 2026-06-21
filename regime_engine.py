"""
regime_engine.py
=================

Institutional Market Regime Detection Engine for QuantLab.

This module analyzes historical price data for a basket of assets and
classifies the prevailing market regime — Bull, Bear, Sideways,
Recovery, or High Volatility — based on trailing annualized return,
annualized volatility, maximum drawdown, and dual-horizon (63-day /
252-day) momentum statistics computed on an equal-weight synthetic
portfolio.

The engine is designed for institutional portfolio analytics
workflows: it is deterministic, side-effect free, strongly typed, and
returns a single structured, auditable result (`RegimeResult`)
suitable for downstream reporting, risk dashboards, or automated
allocation logic. The result bundles not only the headline
classification but full diagnostics: the supporting evidence for the
call, a plain-language explanation, a risk assessment, portfolio
implications, and guidance for Monte Carlo simulation and stress
testing calibration.

Streamlit / dashboard compatibility
------------------------------------
`RegimeResult` is intentionally over-populated with convenience
aliases (`volatility`, `trend`) alongside the canonical, fully-named
fields (`annual_volatility`, `momentum_63d`) so that callers — such as
`dashboard_app.py` — can reference whichever naming convention they
expect without raising `AttributeError`. All fields are plain Python
floats / strs / dicts (no numpy scalar types) so they serialize and
format cleanly inside Streamlit widgets (`st.metric`, f-strings, etc.).

Typical usage
-------------
    import pandas as pd
    from regime_engine import detect_market_regime, regime_color, regime_description

    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
    result = detect_market_regime(prices)

    print(result.regime, result.confidence)
    print(regime_color(result.regime))
    print(regime_description(result.regime))
    print(result.explanation)
    print(result.risk_assessment)
    print(result.portfolio_implications)
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
MOMENTUM_SHORT_WINDOW: Final[int] = 63    # ~3 months
MOMENTUM_LONG_WINDOW: Final[int] = 252    # ~1 year

# Classification thresholds (named constants for transparency and so
# they can be tuned/audited without touching classification logic).
BULL_RETURN_THRESHOLD: Final[float] = 0.10
BULL_VOL_THRESHOLD: Final[float] = 0.25

BEAR_RETURN_THRESHOLD: Final[float] = -0.05
BEAR_DRAWDOWN_THRESHOLD: Final[float] = -0.20

HIGH_VOL_THRESHOLD: Final[float] = 0.30

# Confidence-scaling constants (distance-from-threshold at which a
# component's confidence contribution saturates to 1.0).
_SCALE_RETURN: Final[float] = 0.20
_SCALE_MOMENTUM_SHORT: Final[float] = 0.10
_SCALE_MOMENTUM_LONG: Final[float] = 0.20
_SCALE_VOL: Final[float] = 0.15
_SCALE_VOL_HIGH: Final[float] = 0.25
_SCALE_DRAWDOWN: Final[float] = 0.20

# Monte Carlo calibration defaults, keyed by regime. These are
# starting-point simulation parameters (annualized drift / vol
# multipliers and suggested path count) intended to seed a downstream
# Monte Carlo engine with regime-appropriate assumptions rather than
# naive historical-mean assumptions that ignore the current backdrop.
_MC_GUIDANCE_TEMPLATE: Final[dict[str, dict[str, object]]] = {
    "Bull Market": {
        "drift_adjustment": "Use trailing annual return as drift; haircut by 20-30% for mean reversion.",
        "volatility_multiplier": 1.0,
        "suggested_paths": 5000,
        "horizon_guidance": "12-24 month horizons are most informative; tail risk is understated in calm bull regimes.",
        "fat_tail_adjustment": "Apply Student-t innovations (df ~5-7) to compensate for understated left-tail risk.",
    },
    "Bear Market": {
        "drift_adjustment": "Blend trailing return with a long-run equilibrium drift to avoid extrapolating the decline indefinitely.",
        "volatility_multiplier": 1.25,
        "suggested_paths": 10000,
        "horizon_guidance": "Shorter 3-6 month horizons reduce compounding of an unsustainable trend; revisit assumptions monthly.",
        "fat_tail_adjustment": "Apply Student-t innovations (df ~3-4) and negative skew to capture crash clustering.",
    },
    "High Volatility": {
        "drift_adjustment": "Set drift near zero / long-run neutral; current trend signal is unreliable under elevated vol.",
        "volatility_multiplier": 1.5,
        "suggested_paths": 15000,
        "horizon_guidance": "Favor short horizons (1-3 months) and frequent re-simulation as the regime evolves.",
        "fat_tail_adjustment": "Use a regime-switching or jump-diffusion model; volatility clustering dominates path outcomes.",
    },
    "Recovery": {
        "drift_adjustment": "Blend short-term momentum with a fade toward long-run mean; avoid extrapolating the rebound.",
        "volatility_multiplier": 1.15,
        "suggested_paths": 10000,
        "horizon_guidance": "6-12 month horizons help distinguish a durable recovery from a relief rally.",
        "fat_tail_adjustment": "Apply asymmetric vol (higher downside vol) to capture relapse risk.",
    },
    "Sideways": {
        "drift_adjustment": "Use long-run historical drift; trend signal is weak so anchor to structural assumptions.",
        "volatility_multiplier": 0.9,
        "suggested_paths": 5000,
        "horizon_guidance": "Range-bound dynamics favor 6-18 month horizons with mean-reverting drift assumptions.",
        "fat_tail_adjustment": "Standard normal or mild Student-t (df ~8-10) innovations are typically adequate.",
    },
}

# Stress-testing calibration guidance, keyed by regime. Suggests which
# canonical shock scenarios deserve the most weight given the current
# regime, since the marginal value of a stress test depends on how far
# the current state already is from the shock being tested.
_STRESS_GUIDANCE_TEMPLATE: Final[dict[str, dict[str, object]]] = {
    "Bull Market": {
        "priority_scenarios": ["Market Crash (-30%)", "Volatility Spike (2x)", "Rate Shock (+200bps)"],
        "rationale": "Complacency risk is highest after sustained gains; prioritize tail-risk and reversal scenarios that this regime has not recently experienced.",
        "suggested_shock_magnitude": "Severe (2-3 standard deviations) — current realized vol is depressed and underrepresents tail risk.",
    },
    "Bear Market": {
        "priority_scenarios": ["Further Drawdown (-15% incremental)", "Liquidity Crisis", "Correlation Breakdown"],
        "rationale": "Test for capitulation/continuation risk and whether diversification assumptions still hold under continued stress.",
        "suggested_shock_magnitude": "Moderate-to-severe — the portfolio may already be partially repricing tail scenarios.",
    },
    "High Volatility": {
        "priority_scenarios": ["Volatility Spike (2x)", "Liquidity Crisis", "Correlation Breakdown", "Rate Shock (+200bps)"],
        "rationale": "Elevated realized volatility is itself the dominant risk factor; stress liquidity and cross-asset correlation assumptions specifically.",
        "suggested_shock_magnitude": "Severe — current conditions already exhibit fat-tail behavior; design scenarios around regime persistence.",
    },
    "Recovery": {
        "priority_scenarios": ["Relapse Scenario (-10%)", "Volatility Spike (2x)", "Rate Shock (+200bps)"],
        "rationale": "Validate that the rebound survives a renewed shock before increasing risk budget; recoveries can fail.",
        "suggested_shock_magnitude": "Moderate — focused on testing the durability of the nascent uptrend.",
    },
    "Sideways": {
        "priority_scenarios": ["Market Crash (-30%)", "Rate Shock (+200bps)", "Volatility Spike (2x)"],
        "rationale": "Range-bound regimes can mask building risk; use broad scenario coverage since no single directional risk dominates.",
        "suggested_shock_magnitude": "Standard — balanced coverage across canonical shock scenarios.",
    },
}


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

    Core classification
    --------------------
    regime : MarketRegime
        The classified market regime.
    confidence : float
        A score in [0.0, 1.0] indicating how strongly the underlying
        statistics support the assigned regime. Higher values indicate
        the metrics are further past the relevant classification
        threshold(s); lower values indicate a borderline classification.

    Canonical analytics
    --------------------
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

    Dashboard-compatibility aliases
    --------------------------------
    These duplicate canonical fields above under shorter names expected
    by some downstream callers (e.g. `dashboard_app.py`'s Regime
    Intelligence page). They are always numerically identical to their
    canonical counterparts.

    volatility : float
        Alias for `annual_volatility`.
    trend : float
        Alias for `momentum_63d` (the shorter-horizon momentum figure
        is used as the headline "trend" indicator since it is more
        responsive to regime shifts than the 252-day figure).

    Diagnostics & narrative
    ------------------------
    diagnostics : dict[str, float]
        Flat dictionary of every raw statistic used in classification,
        keyed by name, for audit/logging/export purposes.
    evidence : list[str]
        Ordered list of plain-language evidentiary statements
        justifying the regime call (e.g. "Annualized return of 14.2%
        exceeds the +10% bull threshold.").
    explanation : str
        A synthesized, narrative explanation of why this regime was
        selected over the alternatives, referencing the evidence.
    risk_assessment : str
        A narrative risk assessment specific to the detected regime
        and the magnitude of its supporting statistics.
    portfolio_implications : str
        Actionable, narrative guidance on portfolio construction and
        positioning appropriate to the detected regime.
    monte_carlo_guidance : dict[str, object]
        Regime-conditioned parameters for seeding a downstream Monte
        Carlo simulation engine (drift adjustment guidance, volatility
        multiplier, suggested path count, horizon guidance, and
        fat-tail adjustment guidance).
    stress_test_guidance : dict[str, object]
        Regime-conditioned guidance for prioritizing and calibrating
        stress-test scenarios (priority scenario list, rationale, and
        suggested shock magnitude).
    """

    regime: MarketRegime
    confidence: float

    annual_return: float
    annual_volatility: float
    drawdown: float
    momentum_63d: float
    momentum_252d: float

    # Dashboard-compatibility aliases
    volatility: float
    trend: float

    diagnostics: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    explanation: str = ""
    risk_assessment: str = ""
    portfolio_implications: str = ""
    monte_carlo_guidance: dict[str, object] = field(default_factory=dict)
    stress_test_guidance: dict[str, object] = field(default_factory=dict)


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
    if prices is None or prices.empty:
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

    # Guard against any remaining non-positive prices that would corrupt
    # normalization (e.g. a corrupted feed reporting 0.0).
    first_row = clean_prices.iloc[0]
    invalid_cols = first_row[(first_row <= 0) | first_row.isna()].index.tolist()
    if invalid_cols:
        clean_prices = clean_prices.drop(columns=invalid_cols)

    if clean_prices.shape[1] == 0:
        raise ValueError(
            "No valid asset price series remain after removing non-positive "
            "or invalid initial observations."
        )

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
    if np.isnan(daily_vol):
        return 0.0
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
    result = float(drawdown_series.min())
    return result if not np.isnan(result) else 0.0


def _current_drawdown(portfolio_index: pd.Series) -> float:
    """
    Compute the *current* (as-of-latest-observation) drawdown from the
    running peak, as distinct from the maximum historical drawdown.

    Parameters
    ----------
    portfolio_index : pd.Series
        Synthetic portfolio price index, indexed by date.

    Returns
    -------
    float
        Current drawdown from peak, expressed as a negative fraction
        (or 0.0 if the series is currently at a new high).
    """
    if len(portfolio_index) < 2:
        return 0.0
    running_max = portfolio_index.cummax()
    current = float(portfolio_index.iloc[-1] / running_max.iloc[-1] - 1.0)
    return current if not np.isnan(current) else 0.0


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
    return_score = _scale_distance(annual_return, BULL_RETURN_THRESHOLD, scale=_SCALE_RETURN)
    momentum_score = _scale_distance(momentum_63d, 0.0, scale=_SCALE_MOMENTUM_SHORT)
    vol_score = _scale_distance(annual_volatility, BULL_VOL_THRESHOLD, scale=_SCALE_VOL)
    return float(np.mean([return_score, momentum_score, vol_score]))


def _confidence_bear(annual_return: float, drawdown: float) -> float:
    """
    Confidence score for a BEAR classification, derived from how far
    return is below its threshold and drawdown exceeds its threshold.
    """
    return_score = _scale_distance(annual_return, BEAR_RETURN_THRESHOLD, scale=_SCALE_RETURN)
    drawdown_score = _scale_distance(drawdown, BEAR_DRAWDOWN_THRESHOLD, scale=_SCALE_DRAWDOWN)
    return float(np.mean([return_score, drawdown_score]))


def _confidence_high_vol(annual_volatility: float) -> float:
    """
    Confidence score for a HIGH_VOL classification, derived from how
    far volatility exceeds its threshold.
    """
    return _scale_distance(annual_volatility, HIGH_VOL_THRESHOLD, scale=_SCALE_VOL_HIGH)


def _confidence_recovery(momentum_63d: float, momentum_252d: float) -> float:
    """
    Confidence score for a RECOVERY classification, derived from the
    strength of the short-term rebound and the depth of the prior
    long-term decline.
    """
    momentum_score = _scale_distance(momentum_63d, 0.0, scale=_SCALE_MOMENTUM_SHORT)
    decline_score = _scale_distance(momentum_252d, 0.0, scale=_SCALE_MOMENTUM_LONG)
    return float(np.mean([momentum_score, decline_score]))


def _confidence_sideways(
    annual_return: float,
    annual_volatility: float,
    drawdown: float,
) -> float:
    """
    Confidence score for a SIDEWAYS classification. Since SIDEWAYS is a
    "none of the above" classification, confidence is highest when all
    metrics sit comfortably *within* their respective bull/bear/high-vol
    boundaries (i.e. far from every other regime's threshold).
    """
    headroom_bull = max(BULL_RETURN_THRESHOLD - annual_return, 0.0) / _SCALE_RETURN
    headroom_bear = max(annual_return - BEAR_RETURN_THRESHOLD, 0.0) / _SCALE_RETURN
    headroom_vol = max(HIGH_VOL_THRESHOLD - annual_volatility, 0.0) / _SCALE_VOL_HIGH
    headroom_dd = max(drawdown - BEAR_DRAWDOWN_THRESHOLD, 0.0) / _SCALE_DRAWDOWN

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

    confidence = _confidence_sideways(annual_return, annual_volatility, drawdown)
    return MarketRegime.SIDEWAYS, confidence


# --------------------------------------------------------------------------
# Diagnostics, Evidence & Narrative Generation
# --------------------------------------------------------------------------

def _build_diagnostics(
    annual_return: float,
    annual_volatility: float,
    drawdown: float,
    current_drawdown: float,
    momentum_63d: float,
    momentum_252d: float,
    n_observations: int,
) -> dict[str, float]:
    """
    Assemble the flat diagnostics dictionary bundled with every
    `RegimeResult`, for audit logging, export, or programmatic
    downstream consumption.
    """
    return {
        "annual_return": round(annual_return, 6),
        "annual_volatility": round(annual_volatility, 6),
        "max_drawdown": round(drawdown, 6),
        "current_drawdown": round(current_drawdown, 6),
        "momentum_63d": round(momentum_63d, 6),
        "momentum_252d": round(momentum_252d, 6),
        "n_observations": float(n_observations),
        "bull_return_threshold": BULL_RETURN_THRESHOLD,
        "bull_vol_threshold": BULL_VOL_THRESHOLD,
        "bear_return_threshold": BEAR_RETURN_THRESHOLD,
        "bear_drawdown_threshold": BEAR_DRAWDOWN_THRESHOLD,
        "high_vol_threshold": HIGH_VOL_THRESHOLD,
    }


def _build_evidence(
    regime: MarketRegime,
    annual_return: float,
    annual_volatility: float,
    drawdown: float,
    momentum_63d: float,
    momentum_252d: float,
) -> list[str]:
    """
    Construct an ordered list of plain-language evidentiary statements
    that justify the classified regime, referencing the specific
    statistics and thresholds involved.
    """
    evidence: list[str] = []

    if regime is MarketRegime.BULL:
        evidence.append(
            f"Annualized return of {annual_return:.1%} exceeds the bull "
            f"threshold of {BULL_RETURN_THRESHOLD:.0%}."
        )
        evidence.append(
            f"63-day momentum is positive ({momentum_63d:.1%}), confirming "
            f"the uptrend is intact in the near term."
        )
        evidence.append(
            f"Annualized volatility of {annual_volatility:.1%} remains below "
            f"the {BULL_VOL_THRESHOLD:.0%} ceiling, indicating the rally is "
            f"not accompanied by excessive risk."
        )
    elif regime is MarketRegime.BEAR:
        evidence.append(
            f"Annualized return of {annual_return:.1%} is below the bear "
            f"threshold of {BEAR_RETURN_THRESHOLD:.0%}."
        )
        evidence.append(
            f"Maximum drawdown of {drawdown:.1%} breaches the "
            f"{BEAR_DRAWDOWN_THRESHOLD:.0%} threshold, confirming a "
            f"significant peak-to-trough decline."
        )
        if momentum_63d < 0:
            evidence.append(
                f"63-day momentum remains negative ({momentum_63d:.1%}), "
                f"showing no near-term stabilization yet."
            )
    elif regime is MarketRegime.HIGH_VOL:
        evidence.append(
            f"Annualized volatility of {annual_volatility:.1%} exceeds the "
            f"{HIGH_VOL_THRESHOLD:.0%} high-volatility threshold."
        )
        evidence.append(
            f"This elevated volatility dominates the regime signal "
            f"regardless of the {annual_return:.1%} annualized return, "
            f"since risk magnitude — not direction — is the binding constraint."
        )
    elif regime is MarketRegime.RECOVERY:
        evidence.append(
            f"63-day momentum has turned positive ({momentum_63d:.1%}) "
            f"following a negative 252-day momentum reading "
            f"({momentum_252d:.1%}), the defining signature of a rebound "
            f"from a prior decline."
        )
        evidence.append(
            f"Annualized return of {annual_return:.1%} and volatility of "
            f"{annual_volatility:.1%} did not independently trigger a "
            f"Bull, Bear, or High Volatility classification."
        )
    else:  # SIDEWAYS
        evidence.append(
            f"Annualized return of {annual_return:.1%} sits between the "
            f"bear threshold ({BEAR_RETURN_THRESHOLD:.0%}) and bull "
            f"threshold ({BULL_RETURN_THRESHOLD:.0%})."
        )
        evidence.append(
            f"Annualized volatility of {annual_volatility:.1%} is below the "
            f"{HIGH_VOL_THRESHOLD:.0%} high-volatility threshold."
        )
        evidence.append(
            f"Maximum drawdown of {drawdown:.1%} does not breach the bear "
            f"threshold of {BEAR_DRAWDOWN_THRESHOLD:.0%}, and momentum does "
            f"not exhibit a recovery pattern."
        )

    return evidence


def _build_explanation(regime: MarketRegime, confidence: float, evidence: list[str]) -> str:
    """
    Synthesize a narrative explanation of the classification decision,
    referencing the supporting evidence and confidence level.
    """
    confidence_band = (
        "high" if confidence >= 0.66 else "moderate" if confidence >= 0.33 else "low"
    )
    evidence_text = " ".join(evidence)
    return (
        f"The portfolio is classified as {regime.value} with {confidence_band} "
        f"confidence ({confidence:.0%}). {evidence_text} This classification "
        f"reflects a rule-based, deterministic evaluation of trailing return, "
        f"volatility, drawdown, and dual-horizon momentum statistics; it is "
        f"not a forecast of future regime persistence."
    )


def _build_risk_assessment(
    regime: MarketRegime,
    annual_volatility: float,
    drawdown: float,
    current_drawdown: float,
) -> str:
    """
    Construct a narrative risk assessment specific to the detected
    regime, incorporating the magnitude of current drawdown relative
    to the historical maximum.
    """
    recovery_pct = (
        0.0 if drawdown == 0 else float(np.clip(1.0 - (current_drawdown / drawdown), 0.0, 1.0))
    )

    base: dict[MarketRegime, str] = {
        MarketRegime.BULL: (
            f"Primary risk is complacency: realized volatility ({annual_volatility:.1%}) "
            f"is contained, but valuations and crowding can build quietly during "
            f"extended uptrends. Maximum drawdown over the sample was "
            f"{drawdown:.1%}, underscoring that even bull regimes are not "
            f"immune to sharp corrections."
        ),
        MarketRegime.BEAR: (
            f"Primary risk is continued capital impairment and correlation "
            f"breakdown across risk assets. The portfolio is currently "
            f"{current_drawdown:.1%} from its peak against a maximum observed "
            f"drawdown of {drawdown:.1%} ({recovery_pct:.0%} of the way back "
            f"from the trough). Liquidity and funding risk should be "
            f"monitored closely alongside price risk."
        ),
        MarketRegime.HIGH_VOL: (
            f"Primary risk is magnitude and unpredictability of price swings: "
            f"annualized volatility of {annual_volatility:.1%} implies daily "
            f"moves materially larger than historical norms. Tail risk, "
            f"gap risk, and margin/liquidity stress are elevated; "
            f"directional conviction should be discounted accordingly."
        ),
        MarketRegime.RECOVERY: (
            f"Primary risk is a false signal: short-term strength can fail to "
            f"confirm into a durable uptrend. The portfolio remains "
            f"{current_drawdown:.1%} below its historical peak ({drawdown:.1%} "
            f"maximum drawdown), so downside risk from a relapse is still "
            f"material until the recovery broadens and persists."
        ),
        MarketRegime.SIDEWAYS: (
            f"Primary risk is opportunity cost and false breakouts within a "
            f"range-bound market, rather than acute capital loss. Realized "
            f"volatility ({annual_volatility:.1%}) and drawdown ({drawdown:.1%}) "
            f"are both contained, but a breakout in either direction can "
            f"develop quickly once a catalyst emerges."
        ),
    }
    return base[regime]


def _build_portfolio_implications(regime: MarketRegime) -> str:
    """
    Construct narrative, actionable portfolio-construction guidance
    appropriate to the detected regime.
    """
    implications: dict[MarketRegime, str] = {
        MarketRegime.BULL: (
            "Favor growth and risk-asset exposure while maintaining discipline "
            "around valuation, concentration, and position sizing. Trailing "
            "stops or partial hedges can lock in gains without fully exiting "
            "the trend. Rebalancing toward target weights helps avoid "
            "unintended drift into concentrated winners."
        ),
        MarketRegime.BEAR: (
            "Prioritize capital preservation: reduce gross and net exposure, "
            "add downside hedges (puts, inverse exposure, or low-beta "
            "substitutes), and increase allocation to high-quality fixed "
            "income or cash equivalents. Avoid catching falling-knife "
            "positions without a confirmed stabilization signal."
        ),
        MarketRegime.HIGH_VOL: (
            "Reduce position sizing to account for the wider distribution of "
            "outcomes, increase tail-risk hedging (options, volatility "
            "exposure), and tighten risk limits. Increase the frequency of "
            "portfolio monitoring and stress testing until volatility "
            "normalizes toward historical levels."
        ),
        MarketRegime.RECOVERY: (
            "Consider scaling into risk incrementally rather than all at "
            "once, using the strength of breadth, volume, or fundamental "
            "confirmation as a gating signal. Maintain a partial hedge until "
            "the recovery demonstrates persistence across multiple "
            "confirming indicators."
        ),
        MarketRegime.SIDEWAYS: (
            "Favor relative-value, carry, and mean-reversion strategies over "
            "directional bets. Range-bound conditions can also be an "
            "efficient window for rebalancing, tax-loss harvesting, or "
            "restructuring exposures ahead of the next directional move."
        ),
    }
    return implications[regime]


def _build_monte_carlo_guidance(regime: MarketRegime) -> dict[str, object]:
    """
    Return regime-conditioned Monte Carlo simulation guidance, as a
    plain dict suitable for direct use as keyword seeds into a
    downstream simulation engine or for display in a dashboard.
    """
    return dict(_MC_GUIDANCE_TEMPLATE[regime.value])


def _build_stress_test_guidance(regime: MarketRegime) -> dict[str, object]:
    """
    Return regime-conditioned stress-testing guidance, as a plain dict
    suitable for prioritizing scenario selection in a downstream
    stress-testing module.
    """
    return dict(_STRESS_GUIDANCE_TEMPLATE[regime.value])


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
    to a `MarketRegime`. The returned `RegimeResult` additionally
    bundles full diagnostics, evidence, narrative explanation, risk
    assessment, portfolio implications, and Monte Carlo / stress-test
    guidance appropriate to the detected regime.

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
        Structured, fully-populated result containing the classified
        regime, confidence score, canonical statistics, dashboard
        compatibility aliases, diagnostics, evidence, and narrative
        guidance.

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
    cur_drawdown = _current_drawdown(portfolio_index)
    momentum_63d = _momentum(portfolio_index, MOMENTUM_SHORT_WINDOW)
    momentum_252d = _momentum(portfolio_index, MOMENTUM_LONG_WINDOW)

    regime, confidence = _classify(
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        drawdown=drawdown,
        momentum_63d=momentum_63d,
        momentum_252d=momentum_252d,
    )

    diagnostics = _build_diagnostics(
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        drawdown=drawdown,
        current_drawdown=cur_drawdown,
        momentum_63d=momentum_63d,
        momentum_252d=momentum_252d,
        n_observations=len(daily_returns),
    )
    evidence = _build_evidence(
        regime=regime,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        drawdown=drawdown,
        momentum_63d=momentum_63d,
        momentum_252d=momentum_252d,
    )
    explanation = _build_explanation(regime, confidence, evidence)
    risk_assessment = _build_risk_assessment(regime, annual_volatility, drawdown, cur_drawdown)
    portfolio_implications = _build_portfolio_implications(regime)
    monte_carlo_guidance = _build_monte_carlo_guidance(regime)
    stress_test_guidance = _build_stress_test_guidance(regime)

    annual_return_r = round(annual_return, 6)
    annual_volatility_r = round(annual_volatility, 6)
    drawdown_r = round(drawdown, 6)
    momentum_63d_r = round(momentum_63d, 6)
    momentum_252d_r = round(momentum_252d, 6)

    return RegimeResult(
        regime=regime,
        confidence=round(confidence, 4),
        annual_return=annual_return_r,
        annual_volatility=annual_volatility_r,
        drawdown=drawdown_r,
        momentum_63d=momentum_63d_r,
        momentum_252d=momentum_252d_r,
        volatility=annual_volatility_r,
        trend=momentum_63d_r,
        diagnostics=diagnostics,
        evidence=evidence,
        explanation=explanation,
        risk_assessment=risk_assessment,
        portfolio_implications=portfolio_implications,
        monte_carlo_guidance=monte_carlo_guidance,
        stress_test_guidance=stress_test_guidance,
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
