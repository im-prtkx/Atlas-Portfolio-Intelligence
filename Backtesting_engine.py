"""
backtesting_engine.py

A production-quality module implementing the core infrastructure for
backtesting trading strategies on historical price/signal data: input
validation, vectorized strategy return calculation, a comprehensive
suite of portfolio performance metrics, three rule-based trading
strategies (moving average crossover, momentum, mean reversion), a
cross-strategy comparison utility, and rolling walk-forward backtesting.

Author: Quantitative Research Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------
# Custom Exceptions
# --------------------------------------------------------------------------
class BacktestingEngineError(Exception):
    """Base exception for all BacktestingEngine-related errors."""


class InvalidDataError(BacktestingEngineError):
    """Raised when input price or signal data is malformed or invalid."""


class InsufficientDataError(BacktestingEngineError):
    """Raised when there is not enough historical data to run a backtest."""


class InvalidSignalError(BacktestingEngineError):
    """Raised when strategy signal data is malformed or out of range."""


class MetricCalculationError(BacktestingEngineError):
    """Raised when a performance metric cannot be computed reliably."""


# --------------------------------------------------------------------------
# BacktestResult Dataclass
# --------------------------------------------------------------------------
@dataclass
class BacktestResult:
    """
    Container for the full output of a backtest run.

    Attributes
    ----------
    strategy_returns : pd.Series
        Periodic (e.g. daily) returns earned by the strategy, indexed
        by date.
    cumulative_returns : pd.Series
        Cumulative growth of $1 invested in the strategy over time,
        indexed by date (i.e. (1 + strategy_returns).cumprod()).
    total_return : float
        Total cumulative return over the full backtest period
        (e.g. 0.25 for +25%).
    cagr : float
        Compound Annual Growth Rate.
    annualized_volatility : float
        Annualized standard deviation of strategy returns.
    sharpe_ratio : float
        Annualized Sharpe ratio.
    sortino_ratio : float
        Annualized Sortino ratio (downside-risk-adjusted return).
    max_drawdown : float
        Maximum peak-to-trough drawdown, expressed as a negative
        decimal (e.g. -0.35 for a 35% drawdown).
    calmar_ratio : float
        CAGR divided by the absolute value of maximum drawdown.
    win_rate : float
        Fraction of periods with strictly positive strategy returns,
        expressed as a decimal in [0, 1].
    metadata : dict
        Optional free-form metadata about the backtest run (e.g.
        strategy name, parameters, run timestamp). Defaults to an
        empty dict.
    """

    strategy_returns: pd.Series
    cumulative_returns: pd.Series
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    metadata: dict = field(default_factory=dict)

    def summary(self) -> pd.Series:
        """
        Produce a condensed, human-readable summary of scalar metrics.

        Returns
        -------
        pd.Series
            Series of the scalar performance metrics (excludes the
            return time series and metadata), suitable for printing
            or tabular comparison across multiple backtests.
        """
        return pd.Series(
            {
                "Total Return": self.total_return,
                "CAGR": self.cagr,
                "Annualized Volatility": self.annualized_volatility,
                "Sharpe Ratio": self.sharpe_ratio,
                "Sortino Ratio": self.sortino_ratio,
                "Maximum Drawdown": self.max_drawdown,
                "Calmar Ratio": self.calmar_ratio,
                "Win Rate": self.win_rate,
            },
            name="Backtest Summary",
        )


# --------------------------------------------------------------------------
# StrategyResult Dataclass
# --------------------------------------------------------------------------
@dataclass
class StrategyResult:
    """
    Container for the full output of a single trading strategy backtest,
    including the raw signals that generated it.

    Attributes
    ----------
    strategy_name : str
        Human-readable name of the strategy (e.g. "MA Crossover (20/50)").
    buy_signals : pd.Series
        Boolean series, True on dates where a new long entry signal
        fired, indexed by date.
    sell_signals : pd.Series
        Boolean series, True on dates where an exit/short signal fired,
        indexed by date.
    positions : pd.Series
        Target position weight series in [-1, 1] (or [0, 1] for
        long-only strategies) used to drive strategy returns.
    backtest_result : BacktestResult
        Full performance evaluation (returns, equity curve, and metrics)
        produced by `BacktestingEngine.evaluate_performance`.
    """

    strategy_name: str
    buy_signals: pd.Series
    sell_signals: pd.Series
    positions: pd.Series
    backtest_result: BacktestResult

    @property
    def equity_curve(self) -> pd.Series:
        """
        Convenience accessor for the strategy's cumulative equity curve.

        Returns
        -------
        pd.Series
            Cumulative growth of $1 invested in the strategy, indexed
            by date.
        """
        return self.backtest_result.cumulative_returns


# --------------------------------------------------------------------------
# BacktestingEngine
# --------------------------------------------------------------------------
class BacktestingEngine:
    """
    Core engine for backtesting trading strategies on historical price
    data.

    The engine consumes a price series and a signal/position series
    (long-only or long/short, expressed as position weights), computes
    realized strategy returns, and reports a comprehensive suite of
    risk-adjusted performance metrics.

    Attributes
    ----------
    prices : pd.Series
        Historical asset prices, indexed by date, sorted ascending.
    risk_free_rate : float
        Annualized risk-free rate used in Sharpe/Sortino calculations.
    periods_per_year : int
        Number of return periods per year, used for annualization
        (e.g. 252 for daily, 12 for monthly).
    asset_returns : pd.Series
        Periodic simple returns derived from `prices`.
    """

    def __init__(
        self,
        prices: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> None:
        """
        Initialize the BacktestingEngine.

        Parameters
        ----------
        prices : pd.Series
            Historical asset prices indexed by date (e.g. daily close
            prices). Must contain at least 2 observations and be free
            of non-positive or non-finite values.
        risk_free_rate : float, optional
            Annualized risk-free rate, by default 0.02 (2%).
        periods_per_year : int, optional
            Number of periods in a year used to annualize return and
            volatility statistics, by default 252 (daily data).

        Raises
        ------
        InvalidDataError
            If `prices` is not a valid, non-empty, numeric Series of
            strictly positive, finite values.
        InsufficientDataError
            If `prices` has fewer than 2 observations.
        """
        logger.info("Initializing BacktestingEngine.")

        self._validate_prices(prices)
        self._validate_risk_free_rate(risk_free_rate)
        self._validate_periods_per_year(periods_per_year)

        # Ensure chronological order regardless of input ordering.
        sorted_prices = prices.sort_index()

        self.prices: pd.Series = sorted_prices.astype(float)
        self.risk_free_rate: float = float(risk_free_rate)
        self.periods_per_year: int = int(periods_per_year)

        self.asset_returns: pd.Series = self.prices.pct_change().dropna()

        if self.asset_returns.empty:
            raise InsufficientDataError(
                "No valid returns could be derived from `prices`; at "
                "least 2 price observations are required."
            )

        logger.info(
            "BacktestingEngine initialized with %d price observations "
            "(%d derived returns) spanning %s to %s.",
            len(self.prices),
            len(self.asset_returns),
            self.prices.index.min(),
            self.prices.index.max(),
        )

    # ----------------------------------------------------------------
    # Data Validation Methods
    # ----------------------------------------------------------------
    @staticmethod
    def _validate_prices(prices: pd.Series) -> None:
        """
        Validate the price series supplied at construction time.

        Parameters
        ----------
        prices : pd.Series
            Candidate historical price series.

        Raises
        ------
        InvalidDataError
            If `prices` is not a Series, is empty, contains non-numeric,
            non-positive, NaN, or infinite values, or has a duplicated
            index.
        InsufficientDataError
            If `prices` has fewer than 2 observations.
        """
        if not isinstance(prices, pd.Series):
            raise InvalidDataError(
                f"`prices` must be a pandas Series, got {type(prices).__name__}."
            )

        if prices.empty:
            raise InvalidDataError("`prices` Series is empty.")

        if not pd.api.types.is_numeric_dtype(prices):
            raise InvalidDataError(
                f"`prices` must contain numeric data, got dtype "
                f"{prices.dtype}."
            )

        if prices.isnull().any():
            raise InvalidDataError("`prices` contains NaN values.")

        if np.isinf(prices.values).any():
            raise InvalidDataError("`prices` contains infinite values.")

        if (prices.values <= 0).any():
            raise InvalidDataError(
                "`prices` contains non-positive values; prices must be "
                "strictly positive."
            )

        if prices.index.duplicated().any():
            raise InvalidDataError("`prices` has a duplicated date index.")

        if len(prices) < 2:
            raise InsufficientDataError(
                f"At least 2 price observations are required; got {len(prices)}."
            )

    @staticmethod
    def _validate_risk_free_rate(risk_free_rate: float) -> None:
        """
        Validate the risk-free rate.

        Parameters
        ----------
        risk_free_rate : float
            Annualized risk-free rate.

        Raises
        ------
        InvalidDataError
            If `risk_free_rate` is not numeric, non-finite, or
            unreasonably extreme.
        """
        if not isinstance(risk_free_rate, (int, float)) or isinstance(
            risk_free_rate, bool
        ):
            raise InvalidDataError(
                f"`risk_free_rate` must be numeric, got "
                f"{type(risk_free_rate).__name__}."
            )
        if not np.isfinite(risk_free_rate):
            raise InvalidDataError("`risk_free_rate` must be finite.")
        if abs(risk_free_rate) > 1.0:
            raise InvalidDataError(
                f"`risk_free_rate` of {risk_free_rate} looks unrealistic "
                f"(expected a decimal, e.g. 0.02 for 2%)."
            )

    @staticmethod
    def _validate_periods_per_year(periods_per_year: int) -> None:
        """
        Validate the annualization factor.

        Parameters
        ----------
        periods_per_year : int
            Number of return periods per year.

        Raises
        ------
        InvalidDataError
            If `periods_per_year` is not a positive integer.
        """
        if not isinstance(periods_per_year, (int, np.integer)) or isinstance(
            periods_per_year, bool
        ):
            raise InvalidDataError(
                f"`periods_per_year` must be an integer, got "
                f"{type(periods_per_year).__name__}."
            )
        if periods_per_year <= 0:
            raise InvalidDataError(
                f"`periods_per_year` must be positive, got {periods_per_year}."
            )

    def _validate_signals(self, signals: pd.Series) -> pd.Series:
        """
        Validate a strategy signal/position series against the engine's
        price data.

        Signals represent target portfolio position weights at each
        timestamp. Long-only strategies should use values in [0, 1];
        long/short strategies may use values in [-1, 1]. This method
        enforces the latter, broader range and leaves narrower
        long-only constraints to the caller/strategy layer.

        Parameters
        ----------
        signals : pd.Series
            Candidate position weights indexed by date.

        Returns
        -------
        pd.Series
            The signal series reindexed to align with `self.prices.index`,
            forward-filled, and with any leading NaNs filled to 0
            (flat/no position).

        Raises
        ------
        InvalidSignalError
            If `signals` is not a Series, is empty, contains non-numeric,
            non-finite values, or values outside [-1, 1].
        """
        if not isinstance(signals, pd.Series):
            raise InvalidSignalError(
                f"`signals` must be a pandas Series, got {type(signals).__name__}."
            )

        if signals.empty:
            raise InvalidSignalError("`signals` Series is empty.")

        if not pd.api.types.is_numeric_dtype(signals):
            raise InvalidSignalError(
                f"`signals` must contain numeric data, got dtype "
                f"{signals.dtype}."
            )

        finite_mask = np.isfinite(signals.values)
        if not finite_mask.all():
            raise InvalidSignalError(
                "`signals` contains NaN or infinite values."
            )

        if (signals.values < -1 - 1e-9).any() or (signals.values > 1 + 1e-9).any():
            raise InvalidSignalError(
                "`signals` contains values outside the allowable range "
                "[-1, 1]: "
                f"min={signals.values.min():.6f}, max={signals.values.max():.6f}."
            )

        # Align signals to the engine's price index. Use forward-fill so
        # a position persists until the next signal change, then treat
        # any remaining leading gaps as "flat" (no position).
        aligned = signals.reindex(self.prices.index).ffill()
        aligned = aligned.fillna(0.0)

        return aligned

    # ----------------------------------------------------------------
    # Strategy Return Calculation Engine
    # ----------------------------------------------------------------
    def calculate_strategy_returns(
        self,
        signals: pd.Series,
        transaction_cost: float = 0.0,
        shift_signals: bool = True,
    ) -> pd.Series:
        """
        Compute periodic strategy returns from a position-weight signal
        series applied to the engine's price data.

        Strategy return at time t is computed as:

            r_t = position_(t-1) * asset_return_t - cost_t

        where `position_(t-1)` is the position held entering period t
        (i.e. signals are shifted by one period by default to avoid
        lookahead bias), and `cost_t` is a proportional transaction
        cost applied whenever the position changes.

        Parameters
        ----------
        signals : pd.Series
            Target position weights indexed by date. Long-only
            strategies should use [0, 1]; long/short strategies may
            use [-1, 1].
        transaction_cost : float, optional
            Proportional transaction cost charged on the absolute
            change in position at each rebalance, by default 0.0
            (no costs). For example, 0.001 represents 10 bps per
            unit of turnover.
        shift_signals : bool, optional
            If True (default), shifts signals forward by one period
            before applying them to returns, so that a signal observed
            at time t is only acted upon starting at time t+1. Set to
            False only if `signals` have already been shifted/lagged
            appropriately by the caller.

        Returns
        -------
        pd.Series
            Periodic net strategy returns, indexed by date, aligned to
            `self.asset_returns.index`.

        Raises
        ------
        InvalidSignalError
            If `signals` fails validation.
        InvalidDataError
            If `transaction_cost` is negative or non-finite.
        """
        logger.info("Calculating strategy returns from signal series.")

        if not isinstance(transaction_cost, (int, float)) or isinstance(
            transaction_cost, bool
        ):
            raise InvalidDataError(
                f"`transaction_cost` must be numeric, got "
                f"{type(transaction_cost).__name__}."
            )
        if not np.isfinite(transaction_cost) or transaction_cost < 0:
            raise InvalidDataError(
                f"`transaction_cost` must be a finite, non-negative "
                f"number, got {transaction_cost}."
            )

        aligned_signals = self._validate_signals(signals)

        positions = (
            aligned_signals.shift(1).fillna(0.0)
            if shift_signals
            else aligned_signals
        )

        # Align positions to the return index (positions has one more
        # leading observation than asset_returns, since returns start
        # one period after prices).
        positions = positions.reindex(self.asset_returns.index).fillna(0.0)

        gross_returns = positions * self.asset_returns

        if transaction_cost > 0:
            turnover = positions.diff().abs()
            if len(positions) > 0:
                turnover.iloc[0] = abs(positions.iloc[0])
            costs = turnover * transaction_cost
            net_returns = gross_returns - costs
        else:
            net_returns = gross_returns

        net_returns = net_returns.rename("strategy_returns")

        logger.info(
            "Strategy returns calculated: %d periods, mean=%.6f, std=%.6f.",
            len(net_returns),
            net_returns.mean(),
            net_returns.std(),
        )
        return net_returns

    # ----------------------------------------------------------------
    # Portfolio Performance Metrics
    # ----------------------------------------------------------------
    def _validate_returns_series(self, returns: pd.Series) -> None:
        """
        Validate a returns series prior to metric computation.

        Parameters
        ----------
        returns : pd.Series
            Candidate periodic returns series.

        Raises
        ------
        InvalidDataError
            If `returns` is not a Series, is empty, contains
            non-numeric or non-finite values.
        InsufficientDataError
            If `returns` has fewer than 2 observations.
        """
        if not isinstance(returns, pd.Series):
            raise InvalidDataError(
                f"`returns` must be a pandas Series, got {type(returns).__name__}."
            )
        if returns.empty:
            raise InvalidDataError("`returns` Series is empty.")
        if not pd.api.types.is_numeric_dtype(returns):
            raise InvalidDataError(
                f"`returns` must contain numeric data, got dtype "
                f"{returns.dtype}."
            )
        if not np.isfinite(returns.values).all():
            raise InvalidDataError("`returns` contains NaN or infinite values.")
        if len(returns) < 2:
            raise InsufficientDataError(
                f"At least 2 return observations are required for "
                f"performance metrics; got {len(returns)}."
            )

    def calculate_total_return(self, returns: pd.Series) -> float:
        """
        Compute the total cumulative return over the full period.

        Parameters
        ----------
        returns : pd.Series
            Periodic strategy returns.

        Returns
        -------
        float
            Total return as a decimal (e.g. 0.25 for +25%), computed
            as the product of (1 + r_t) across all periods, minus 1.

        Raises
        ------
        InvalidDataError
            If `returns` fails validation.
        """
        self._validate_returns_series(returns)
        total_return = float((1.0 + returns).prod() - 1.0)
        return total_return

    def calculate_cagr(self, returns: pd.Series) -> float:
        """
        Compute the Compound Annual Growth Rate (CAGR).

        Parameters
        ----------
        returns : pd.Series
            Periodic strategy returns.

        Returns
        -------
        float
            Annualized compound growth rate as a decimal. Returns
            -1.0 (total loss) if cumulative growth is non-positive,
            since CAGR is undefined for a portfolio that has lost
            all value.

        Raises
        ------
        InvalidDataError
            If `returns` fails validation.
        MetricCalculationError
            If the implied number of years is non-positive.
        """
        self._validate_returns_series(returns)

        n_periods = len(returns)
        years = n_periods / self.periods_per_year

        if years <= 0:
            raise MetricCalculationError(
                "Cannot compute CAGR: implied number of years is non-positive."
            )

        cumulative_growth = float((1.0 + returns).prod())

        if cumulative_growth <= 0:
            logger.warning(
                "Cumulative growth is non-positive (%.6f); CAGR set to -1.0 "
                "(total loss).",
                cumulative_growth,
            )
            return -1.0

        cagr = cumulative_growth ** (1.0 / years) - 1.0
        return float(cagr)

    def calculate_annualized_volatility(self, returns: pd.Series) -> float:
        """
        Compute the annualized volatility (standard deviation) of
        periodic returns.

        Parameters
        ----------
        returns : pd.Series
            Periodic strategy returns.

        Returns
        -------
        float
            Annualized volatility as a decimal.

        Raises
        ------
        InvalidDataError
            If `returns` fails validation.
        """
        self._validate_returns_series(returns)
        periodic_std = float(returns.std(ddof=1))
        annualized_vol = periodic_std * np.sqrt(self.periods_per_year)
        return float(annualized_vol)

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Compute the annualized Sharpe ratio.

        Defined as the annualized excess return over the risk-free
        rate, divided by annualized volatility.

        Parameters
        ----------
        returns : pd.Series
            Periodic strategy returns.

        Returns
        -------
        float
            Annualized Sharpe ratio. Returns NaN if annualized
            volatility is zero (undefined ratio), with a logged
            warning.

        Raises
        ------
        InvalidDataError
            If `returns` fails validation.
        """
        self._validate_returns_series(returns)

        periodic_mean = float(returns.mean())
        annualized_return = periodic_mean * self.periods_per_year
        annualized_vol = self.calculate_annualized_volatility(returns)

        if annualized_vol == 0:
            logger.warning(
                "Annualized volatility is zero; Sharpe ratio is undefined "
                "(returning NaN)."
            )
            return float("nan")

        sharpe = (annualized_return - self.risk_free_rate) / annualized_vol
        return float(sharpe)

    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """
        Compute the annualized Sortino ratio.

        Like the Sharpe ratio, but penalizes only downside volatility
        (returns below the periodic risk-free rate), rather than total
        volatility.

        Parameters
        ----------
        returns : pd.Series
            Periodic strategy returns.

        Returns
        -------
        float
            Annualized Sortino ratio. Returns NaN if there is no
            downside deviation (e.g. no periods underperformed the
            risk-free rate), with a logged warning.

        Raises
        ------
        InvalidDataError
            If `returns` fails validation.
        """
        self._validate_returns_series(returns)

        periodic_rf = self.risk_free_rate / self.periods_per_year
        periodic_mean = float(returns.mean())
        annualized_return = periodic_mean * self.periods_per_year

        downside_diff = returns[returns < periodic_rf] - periodic_rf
        if downside_diff.empty:
            logger.warning(
                "No downside periods relative to the risk-free rate; "
                "Sortino ratio is undefined (returning NaN)."
            )
            return float("nan")

        downside_deviation = float(
            np.sqrt((downside_diff ** 2).mean()) * np.sqrt(self.periods_per_year)
        )

        if downside_deviation == 0:
            logger.warning(
                "Downside deviation is zero; Sortino ratio is undefined "
                "(returning NaN)."
            )
            return float("nan")

        sortino = (annualized_return - self.risk_free_rate) / downside_deviation
        return float(sortino)

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Compute the maximum peak-to-trough drawdown of the cumulative
        return series.

        Parameters
        ----------
        returns : pd.Series
            Periodic strategy returns.

        Returns
        -------
        float
            Maximum drawdown expressed as a negative decimal (e.g.
            -0.35 for a 35% decline from peak), or 0.0 if the
            cumulative series never declines from its running maximum.

        Raises
        ------
        InvalidDataError
            If `returns` fails validation.
        """
        self._validate_returns_series(returns)

        cumulative = (1.0 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown_series = cumulative / running_max - 1.0

        max_drawdown = float(drawdown_series.min())
        # Guard against floating point noise producing a tiny positive value.
        max_drawdown = min(max_drawdown, 0.0)
        return max_drawdown

    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """
        Compute the Calmar ratio: CAGR divided by the absolute value
        of maximum drawdown.

        Parameters
        ----------
        returns : pd.Series
            Periodic strategy returns.

        Returns
        -------
        float
            Calmar ratio. Returns NaN if maximum drawdown is zero
            (undefined ratio), with a logged warning.

        Raises
        ------
        InvalidDataError
            If `returns` fails validation.
        """
        self._validate_returns_series(returns)

        cagr = self.calculate_cagr(returns)
        max_drawdown = self.calculate_max_drawdown(returns)

        if max_drawdown == 0:
            logger.warning(
                "Maximum drawdown is zero; Calmar ratio is undefined "
                "(returning NaN)."
            )
            return float("nan")

        calmar = cagr / abs(max_drawdown)
        return float(calmar)

    def calculate_win_rate(self, returns: pd.Series) -> float:
        """
        Compute the win rate: the fraction of periods with strictly
        positive returns.

        Parameters
        ----------
        returns : pd.Series
            Periodic strategy returns.

        Returns
        -------
        float
            Win rate as a decimal in [0, 1].

        Raises
        ------
        InvalidDataError
            If `returns` fails validation.
        """
        self._validate_returns_series(returns)

        n_periods = len(returns)
        n_wins = int((returns > 0).sum())
        win_rate = n_wins / n_periods
        return float(win_rate)

    def evaluate_performance(
        self,
        returns: pd.Series,
        metadata: Optional[dict] = None,
    ) -> BacktestResult:
        """
        Compute the full suite of performance metrics for a strategy
        return series and package the results into a `BacktestResult`.

        Parameters
        ----------
        returns : pd.Series
            Periodic strategy returns (e.g. as produced by
            `calculate_strategy_returns`).
        metadata : Optional[dict], optional
            Free-form metadata to attach to the result (e.g. strategy
            name or parameters), by default None (stored as an empty
            dict).

        Returns
        -------
        BacktestResult
            Fully populated backtest result object.

        Raises
        ------
        InvalidDataError
            If `returns` fails validation.
        InsufficientDataError
            If `returns` has too few observations to compute metrics.
        MetricCalculationError
            If a metric cannot be computed due to a degenerate return
            series (e.g. non-positive implied duration).
        """
        logger.info("Evaluating full performance metric suite.")

        self._validate_returns_series(returns)

        cumulative_returns = (1.0 + returns).cumprod()
        cumulative_returns = cumulative_returns.rename("cumulative_returns")

        result = BacktestResult(
            strategy_returns=returns.rename("strategy_returns"),
            cumulative_returns=cumulative_returns,
            total_return=self.calculate_total_return(returns),
            cagr=self.calculate_cagr(returns),
            annualized_volatility=self.calculate_annualized_volatility(returns),
            sharpe_ratio=self.calculate_sharpe_ratio(returns),
            sortino_ratio=self.calculate_sortino_ratio(returns),
            max_drawdown=self.calculate_max_drawdown(returns),
            calmar_ratio=self.calculate_calmar_ratio(returns),
            win_rate=self.calculate_win_rate(returns),
            metadata=metadata if metadata is not None else {},
        )

        logger.info(
            "Performance evaluation complete: CAGR=%.4f%%, Sharpe=%.4f, "
            "MaxDD=%.4f%%, WinRate=%.4f%%.",
            result.cagr * 100,
            result.sharpe_ratio,
            result.max_drawdown * 100,
            result.win_rate * 100,
        )
        return result

    # ----------------------------------------------------------------
    # Moving Average Crossover Strategy
    # ----------------------------------------------------------------
    def moving_average_crossover_strategy(
        self,
        short_window: int = 20,
        long_window: int = 50,
        transaction_cost: float = 0.0,
    ) -> StrategyResult:
        """
        Construct and backtest a long/flat moving average crossover
        strategy.

        Generates a buy signal when the short-window simple moving
        average (SMA) crosses above the long-window SMA, and a sell
        (exit-to-flat) signal when it crosses back below. The strategy
        is long-only: it holds a full position (weight = 1.0) whenever
        the short SMA is above the long SMA, and is flat (weight = 0.0)
        otherwise.

        Parameters
        ----------
        short_window : int, optional
            Lookback period (in periods) for the short-term SMA, by
            default 20.
        long_window : int, optional
            Lookback period (in periods) for the long-term SMA, by
            default 50. Must be strictly greater than `short_window`.
        transaction_cost : float, optional
            Proportional transaction cost charged on each change in
            position, by default 0.0.

        Returns
        -------
        StrategyResult
            Buy/sell signals, position series, and full backtest
            results (returns, equity curve, performance metrics).

        Raises
        ------
        InvalidDataError
            If `short_window` or `long_window` are not positive
            integers, if `short_window >= long_window`, or if there is
            insufficient price history to compute the long SMA.
        """
        logger.info(
            "Running Moving Average Crossover strategy (short=%d, long=%d).",
            short_window,
            long_window,
        )

        for name, value in (("short_window", short_window), ("long_window", long_window)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise InvalidDataError(f"`{name}` must be an integer, got {type(value).__name__}.")
            if value <= 0:
                raise InvalidDataError(f"`{name}` must be positive, got {value}.")

        if short_window >= long_window:
            raise InvalidDataError(
                f"`short_window` ({short_window}) must be strictly less than "
                f"`long_window` ({long_window})."
            )

        if len(self.prices) <= long_window:
            raise InsufficientDataError(
                f"At least {long_window + 1} price observations are required "
                f"to compute a {long_window}-period moving average; got "
                f"{len(self.prices)}."
            )

        short_sma = self.prices.rolling(window=short_window, min_periods=short_window).mean()
        long_sma = self.prices.rolling(window=long_window, min_periods=long_window).mean()

        is_bullish = (short_sma > long_sma).fillna(False)

        # A buy signal fires on the bar where the regime first turns
        # bullish; a sell signal fires where it first turns bearish.
        regime_change = is_bullish.astype(int).diff()
        buy_signals = regime_change == 1
        sell_signals = regime_change == -1
        buy_signals.iloc[0] = bool(is_bullish.iloc[0])
        sell_signals.iloc[0] = False

        positions = is_bullish.astype(float).rename("position")

        strategy_returns = self.calculate_strategy_returns(
            signals=positions, transaction_cost=transaction_cost
        )
        backtest_result = self.evaluate_performance(
            strategy_returns,
            metadata={
                "strategy": "Moving Average Crossover",
                "short_window": short_window,
                "long_window": long_window,
                "transaction_cost": transaction_cost,
            },
        )

        logger.info(
            "MA Crossover complete: %d buy signals, %d sell signals.",
            int(buy_signals.sum()),
            int(sell_signals.sum()),
        )

        return StrategyResult(
            strategy_name=f"MA Crossover ({short_window}/{long_window})",
            buy_signals=buy_signals.rename("buy_signal"),
            sell_signals=sell_signals.rename("sell_signal"),
            positions=positions,
            backtest_result=backtest_result,
        )

    # ----------------------------------------------------------------
    # Momentum Strategy
    # ----------------------------------------------------------------
    def momentum_strategy(
        self,
        lookback_window: int = 60,
        holding_threshold: float = 0.0,
        transaction_cost: float = 0.0,
    ) -> StrategyResult:
        """
        Construct and backtest a long/flat time-series momentum
        strategy.

        At each date, computes the trailing total return over
        `lookback_window` periods. A buy signal fires (and a full long
        position is held) whenever trailing momentum exceeds
        `holding_threshold`; otherwise the strategy is flat.

        Parameters
        ----------
        lookback_window : int, optional
            Number of periods over which trailing momentum is measured,
            by default 60.
        holding_threshold : float, optional
            Minimum trailing return required to hold a long position,
            by default 0.0 (i.e. go long whenever trailing momentum is
            positive).
        transaction_cost : float, optional
            Proportional transaction cost charged on each change in
            position, by default 0.0.

        Returns
        -------
        StrategyResult
            Buy/sell signals, position series, and full backtest
            results (returns, equity curve, performance metrics).

        Raises
        ------
        InvalidDataError
            If `lookback_window` is not a positive integer, if
            `holding_threshold` is not finite, or if there is
            insufficient price history.
        """
        logger.info(
            "Running Momentum strategy (lookback=%d, threshold=%.4f).",
            lookback_window,
            holding_threshold,
        )

        if not isinstance(lookback_window, (int, np.integer)) or isinstance(
            lookback_window, bool
        ):
            raise InvalidDataError(
                f"`lookback_window` must be an integer, got "
                f"{type(lookback_window).__name__}."
            )
        if lookback_window <= 0:
            raise InvalidDataError(
                f"`lookback_window` must be positive, got {lookback_window}."
            )
        if not isinstance(holding_threshold, (int, float)) or not np.isfinite(
            holding_threshold
        ):
            raise InvalidDataError(
                f"`holding_threshold` must be a finite number, got "
                f"{holding_threshold}."
            )

        if len(self.prices) <= lookback_window:
            raise InsufficientDataError(
                f"At least {lookback_window + 1} price observations are "
                f"required to compute {lookback_window}-period momentum; "
                f"got {len(self.prices)}."
            )

        trailing_momentum = self.prices.pct_change(periods=lookback_window)

        is_long = (trailing_momentum > holding_threshold).fillna(False)

        regime_change = is_long.astype(int).diff()
        buy_signals = regime_change == 1
        sell_signals = regime_change == -1
        buy_signals.iloc[0] = bool(is_long.iloc[0])
        sell_signals.iloc[0] = False

        positions = is_long.astype(float).rename("position")

        strategy_returns = self.calculate_strategy_returns(
            signals=positions, transaction_cost=transaction_cost
        )
        backtest_result = self.evaluate_performance(
            strategy_returns,
            metadata={
                "strategy": "Momentum",
                "lookback_window": lookback_window,
                "holding_threshold": holding_threshold,
                "transaction_cost": transaction_cost,
            },
        )

        logger.info(
            "Momentum strategy complete: %d buy signals, %d sell signals.",
            int(buy_signals.sum()),
            int(sell_signals.sum()),
        )

        return StrategyResult(
            strategy_name=f"Momentum ({lookback_window})",
            buy_signals=buy_signals.rename("buy_signal"),
            sell_signals=sell_signals.rename("sell_signal"),
            positions=positions,
            backtest_result=backtest_result,
        )

    # ----------------------------------------------------------------
    # Mean Reversion Strategy
    # ----------------------------------------------------------------
    def mean_reversion_strategy(
        self,
        lookback_window: int = 20,
        entry_z_score: float = 1.0,
        exit_z_score: float = 0.0,
        transaction_cost: float = 0.0,
    ) -> StrategyResult:
        """
        Construct and backtest a long/flat mean-reversion strategy
        based on rolling Z-scores of price.

        Computes a rolling Z-score of price relative to its trailing
        mean and standard deviation. A buy signal fires (entering a
        full long position) when the Z-score falls below
        `-entry_z_score` (price is unusually depressed); the position
        is exited (sell signal) once the Z-score recovers to
        `exit_z_score` or above.

        Parameters
        ----------
        lookback_window : int, optional
            Lookback period (in periods) for the rolling mean and
            standard deviation, by default 20.
        entry_z_score : float, optional
            Absolute Z-score threshold below which a long entry is
            triggered, by default 1.0. Must be positive.
        exit_z_score : float, optional
            Z-score level at or above which an open position is exited,
            by default 0.0 (exit once price reverts to its rolling
            mean). Must be less than `entry_z_score`.
        transaction_cost : float, optional
            Proportional transaction cost charged on each change in
            position, by default 0.0.

        Returns
        -------
        StrategyResult
            Buy/sell signals, position series, and full backtest
            results (returns, equity curve, performance metrics).

        Raises
        ------
        InvalidDataError
            If `lookback_window` is not a positive integer, if
            `entry_z_score` is not positive, if `exit_z_score` is not
            less than `entry_z_score`, or if there is insufficient
            price history.
        """
        logger.info(
            "Running Mean Reversion strategy (lookback=%d, entry_z=%.2f, "
            "exit_z=%.2f).",
            lookback_window,
            entry_z_score,
            exit_z_score,
        )

        if not isinstance(lookback_window, (int, np.integer)) or isinstance(
            lookback_window, bool
        ):
            raise InvalidDataError(
                f"`lookback_window` must be an integer, got "
                f"{type(lookback_window).__name__}."
            )
        if lookback_window <= 1:
            raise InvalidDataError(
                f"`lookback_window` must be greater than 1, got "
                f"{lookback_window}."
            )
        if not np.isfinite(entry_z_score) or entry_z_score <= 0:
            raise InvalidDataError(
                f"`entry_z_score` must be a positive finite number, got "
                f"{entry_z_score}."
            )
        if not np.isfinite(exit_z_score):
            raise InvalidDataError(
                f"`exit_z_score` must be a finite number, got {exit_z_score}."
            )
        if exit_z_score >= entry_z_score:
            raise InvalidDataError(
                f"`exit_z_score` ({exit_z_score}) must be less than "
                f"`entry_z_score` ({entry_z_score})."
            )

        if len(self.prices) <= lookback_window:
            raise InsufficientDataError(
                f"At least {lookback_window + 1} price observations are "
                f"required to compute a {lookback_window}-period rolling "
                f"Z-score; got {len(self.prices)}."
            )

        rolling_mean = self.prices.rolling(window=lookback_window, min_periods=lookback_window).mean()
        rolling_std = self.prices.rolling(window=lookback_window, min_periods=lookback_window).std(ddof=1)

        # Avoid division by zero on flat-price windows.
        safe_std = rolling_std.replace(0.0, np.nan)
        z_score = (self.prices - rolling_mean) / safe_std
        z_score = z_score.fillna(0.0)

        # Stateful long/flat logic: enter when z_score < -entry_z_score,
        # exit when z_score >= exit_z_score, otherwise hold prior state.
        is_long = pd.Series(False, index=self.prices.index)
        position_state = False
        for date, z in z_score.items():
            if not position_state and z < -entry_z_score:
                position_state = True
            elif position_state and z >= exit_z_score:
                position_state = False
            is_long.loc[date] = position_state

        regime_change = is_long.astype(int).diff()
        buy_signals = regime_change == 1
        sell_signals = regime_change == -1
        buy_signals.iloc[0] = bool(is_long.iloc[0])
        sell_signals.iloc[0] = False

        positions = is_long.astype(float).rename("position")

        strategy_returns = self.calculate_strategy_returns(
            signals=positions, transaction_cost=transaction_cost
        )
        backtest_result = self.evaluate_performance(
            strategy_returns,
            metadata={
                "strategy": "Mean Reversion",
                "lookback_window": lookback_window,
                "entry_z_score": entry_z_score,
                "exit_z_score": exit_z_score,
                "transaction_cost": transaction_cost,
            },
        )

        logger.info(
            "Mean Reversion strategy complete: %d buy signals, %d sell signals.",
            int(buy_signals.sum()),
            int(sell_signals.sum()),
        )

        return StrategyResult(
            strategy_name=f"Mean Reversion ({lookback_window}, {entry_z_score:.1f}σ)",
            buy_signals=buy_signals.rename("buy_signal"),
            sell_signals=sell_signals.rename("sell_signal"),
            positions=positions,
            backtest_result=backtest_result,
        )

    # ----------------------------------------------------------------
    # Strategy Comparison Utility
    # ----------------------------------------------------------------
    @staticmethod
    def compare_strategies(strategy_results: List["StrategyResult"]) -> pd.DataFrame:
        """
        Build a side-by-side comparison table of performance metrics
        across multiple strategy backtests.

        Parameters
        ----------
        strategy_results : list of StrategyResult
            Strategy results to compare (e.g. as returned by
            `moving_average_crossover_strategy`, `momentum_strategy`,
            `mean_reversion_strategy`).

        Returns
        -------
        pd.DataFrame
            One row per strategy (indexed by `strategy_name`), one
            column per performance metric, sorted by descending
            Sharpe ratio.

        Raises
        ------
        InvalidDataError
            If `strategy_results` is empty or contains objects that
            are not `StrategyResult` instances.
        """
        if not strategy_results:
            raise InvalidDataError(
                "`strategy_results` must contain at least one StrategyResult."
            )

        if not all(isinstance(item, StrategyResult) for item in strategy_results):
            raise InvalidDataError(
                "All items in `strategy_results` must be StrategyResult instances."
            )

        logger.info(
            "Comparing %d strategies: %s.",
            len(strategy_results),
            ", ".join(s.strategy_name for s in strategy_results),
        )

        rows = {}
        for strategy_result in strategy_results:
            rows[strategy_result.strategy_name] = strategy_result.backtest_result.summary()

        comparison_df = pd.DataFrame(rows).T
        comparison_df = comparison_df.sort_values("Sharpe Ratio", ascending=False)
        comparison_df.index.name = "Strategy"

        return comparison_df

    # ----------------------------------------------------------------
    # Walk Forward Backtesting
    # ----------------------------------------------------------------
    def walk_forward_backtest(
        self,
        strategy_fn: Callable[..., "StrategyResult"],
        strategy_kwargs: Optional[dict] = None,
        train_window: int = 252,
        test_window: int = 63,
        step_window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run a rolling walk-forward backtest, re-evaluating a strategy
        on successive out-of-sample test windows.

        The full price history is split into a sequence of
        (train_window, test_window) blocks. For each block, a fresh
        `BacktestingEngine` is constructed on the train+test slice
        (so rolling indicators have sufficient lookback history at the
        start of the test window), the strategy is run via
        `strategy_fn`, and only the portion of the resulting strategy
        returns falling within the test window is retained. This
        avoids both lookahead bias and indicator warm-up artifacts at
        each fold boundary.

        Parameters
        ----------
        strategy_fn : Callable[..., StrategyResult]
            A bound strategy method of this class (e.g.
            `self.moving_average_crossover_strategy`) to be re-invoked
            on each fold's `BacktestingEngine` instance. Must accept
            `transaction_cost` among its keyword arguments and return
            a `StrategyResult`.
        strategy_kwargs : Optional[dict], optional
            Keyword arguments to pass to `strategy_fn` on every fold
            (e.g. `{"short_window": 20, "long_window": 50}`), by
            default None (no extra kwargs).
        train_window : int, optional
            Number of periods of history required before each test
            window begins (used only to size the lookback slice, not
            as a separate fitting step, since these strategies are
            rule-based rather than fitted), by default 252.
        test_window : int, optional
            Number of out-of-sample periods evaluated per fold, by
            default 63.
        step_window : Optional[int], optional
            Number of periods to advance between folds, by default
            None (defaults to `test_window`, i.e. non-overlapping
            folds).

        Returns
        -------
        pd.DataFrame
            One row per fold with columns: "fold", "test_start",
            "test_end", and all scalar metrics from
            `BacktestResult.summary()`. A final row with `fold="ALL"`
            reports performance metrics computed on the full
            concatenated out-of-sample return series.

        Raises
        ------
        InvalidDataError
            If `train_window`, `test_window`, or `step_window` are not
            positive integers, or if `strategy_fn` is not callable.
        InsufficientDataError
            If there is not enough price history to form at least one
            complete fold.
        """
        logger.info(
            "Running walk-forward backtest (train=%d, test=%d, step=%s).",
            train_window,
            test_window,
            step_window if step_window is not None else test_window,
        )

        if not callable(strategy_fn):
            raise InvalidDataError("`strategy_fn` must be callable.")

        for name, value in (
            ("train_window", train_window),
            ("test_window", test_window),
        ):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise InvalidDataError(f"`{name}` must be an integer, got {type(value).__name__}.")
            if value <= 0:
                raise InvalidDataError(f"`{name}` must be positive, got {value}.")

        if step_window is None:
            step_window = test_window
        else:
            if not isinstance(step_window, (int, np.integer)) or isinstance(
                step_window, bool
            ):
                raise InvalidDataError(
                    f"`step_window` must be an integer, got "
                    f"{type(step_window).__name__}."
                )
            if step_window <= 0:
                raise InvalidDataError(
                    f"`step_window` must be positive, got {step_window}."
                )

        strategy_kwargs = strategy_kwargs or {}
        n_obs = len(self.prices)
        fold_size = train_window + test_window

        if n_obs < fold_size:
            raise InsufficientDataError(
                f"At least {fold_size} price observations are required for "
                f"one walk-forward fold (train={train_window} + "
                f"test={test_window}); got {n_obs}."
            )

        fold_records = []
        all_oos_returns = []

        fold_index = 0
        train_start = 0
        while train_start + fold_size <= n_obs:
            train_end = train_start + train_window
            test_end = train_end + test_window

            window_prices = self.prices.iloc[train_start:test_end]
            test_start_date = self.prices.index[train_end]
            test_end_date = self.prices.index[test_end - 1]

            try:
                fold_engine = BacktestingEngine(
                    prices=window_prices,
                    risk_free_rate=self.risk_free_rate,
                    periods_per_year=self.periods_per_year,
                )
                bound_method = getattr(fold_engine, strategy_fn.__name__)
                fold_strategy_result = bound_method(**strategy_kwargs)
            except BacktestingEngineError as exc:
                logger.warning(
                    "Fold %d (%s to %s) failed and was skipped: %s",
                    fold_index,
                    test_start_date,
                    test_end_date,
                    exc,
                )
                train_start += step_window
                fold_index += 1
                continue

            full_returns = fold_strategy_result.backtest_result.strategy_returns
            oos_returns = full_returns[
                (full_returns.index >= test_start_date)
                & (full_returns.index <= test_end_date)
            ]

            if oos_returns.empty:
                logger.warning(
                    "Fold %d (%s to %s) produced no out-of-sample returns; "
                    "skipped.",
                    fold_index,
                    test_start_date,
                    test_end_date,
                )
                train_start += step_window
                fold_index += 1
                continue

            fold_metrics = fold_engine.evaluate_performance(oos_returns).summary()
            record = {
                "fold": fold_index,
                "test_start": test_start_date,
                "test_end": test_end_date,
            }
            record.update(fold_metrics.to_dict())
            fold_records.append(record)
            all_oos_returns.append(oos_returns)

            train_start += step_window
            fold_index += 1

        if not fold_records:
            raise InsufficientDataError(
                "Walk-forward backtest produced no valid folds; check "
                "`train_window`/`test_window` against the available history."
            )

        concatenated_oos = pd.concat(all_oos_returns).sort_index()
        concatenated_oos = concatenated_oos[~concatenated_oos.index.duplicated(keep="first")]

        overall_metrics = self.evaluate_performance(concatenated_oos).summary()
        overall_record = {
            "fold": "ALL",
            "test_start": concatenated_oos.index.min(),
            "test_end": concatenated_oos.index.max(),
        }
        overall_record.update(overall_metrics.to_dict())
        fold_records.append(overall_record)

        results_df = pd.DataFrame(fold_records).set_index("fold")

        logger.info(
            "Walk-forward backtest complete: %d folds evaluated.",
            len(fold_records) - 1,
        )
        return results_df


# --------------------------------------------------------------------------
# Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    """
    Demonstrate the BacktestingEngine's strategy suite end-to-end on
    simulated price data.

    Generates a synthetic price history, then:
        1. Runs the Moving Average Crossover, Momentum, and Mean
           Reversion strategies.
        2. Compares their performance side-by-side via
           `compare_strategies`.
        3. Runs a rolling walk-forward backtest of the Moving Average
           Crossover strategy to evaluate out-of-sample robustness.

    This function is intended as a runnable, illustrative entry point
    and is not part of the public backtesting API.

    Raises
    ------
    BacktestingEngineError
        Propagates any unhandled error raised during construction or
        evaluation, after logging it for diagnostic purposes.
    """
    try:
        rng = np.random.default_rng(seed=11)

        dates = pd.date_range(start="2019-01-01", periods=1500, freq="B")
        # Simulate a price series with a mild trend, mean-reverting noise,
        # and a regime shift, so all three strategies have something to do.
        trend = np.linspace(0, 0.35, len(dates))
        noise = rng.normal(loc=0.0, scale=0.012, size=len(dates))
        mean_reverting = 0.05 * np.sin(np.linspace(0, 40, len(dates)))
        log_returns = np.diff(trend, prepend=0.0) + noise + np.diff(
            mean_reverting, prepend=0.0
        )
        prices = pd.Series(
            100.0 * np.exp(np.cumsum(log_returns)), index=dates, name="price"
        )

        engine = BacktestingEngine(
            prices=prices, risk_free_rate=0.02, periods_per_year=252
        )

        # --- Run all three strategies ---
        ma_result = engine.moving_average_crossover_strategy(
            short_window=20, long_window=50, transaction_cost=0.0005
        )
        momentum_result = engine.momentum_strategy(
            lookback_window=60, holding_threshold=0.0, transaction_cost=0.0005
        )
        mean_reversion_result = engine.mean_reversion_strategy(
            lookback_window=20,
            entry_z_score=1.0,
            exit_z_score=0.0,
            transaction_cost=0.0005,
        )

        for strategy_result in (ma_result, momentum_result, mean_reversion_result):
            print(f"\n=== {strategy_result.strategy_name} ===")
            print(strategy_result.backtest_result.summary().to_string())
            print(
                f"Buy signals: {int(strategy_result.buy_signals.sum())} | "
                f"Sell signals: {int(strategy_result.sell_signals.sum())}"
            )

        # --- Strategy comparison ---
        comparison = BacktestingEngine.compare_strategies(
            [ma_result, momentum_result, mean_reversion_result]
        )
        print("\n=== Strategy Comparison (sorted by Sharpe Ratio) ===")
        print(comparison.to_string())

        # --- Walk-forward backtest of the MA Crossover strategy ---
        wf_results = engine.walk_forward_backtest(
            strategy_fn=engine.moving_average_crossover_strategy,
            strategy_kwargs={
                "short_window": 20,
                "long_window": 50,
                "transaction_cost": 0.0005,
            },
            train_window=252,
            test_window=63,
        )
        print("\n=== Walk-Forward Backtest: MA Crossover (20/50) ===")
        print(wf_results.to_string())

    except BacktestingEngineError as exc:
        logger.error("Demonstration failed: %s", exc)
        raise


if __name__ == "__main__":
    main()

