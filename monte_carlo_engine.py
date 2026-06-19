"""
monte_carlo_engine.py
=====================
QuantLab Institutional Portfolio Research & Risk Analytics Platform
-------------------------------------------------------------------
Monte Carlo simulation engine for asset price paths and portfolio
return distributions using Geometric Brownian Motion (GBM).

Integrates with:
    - market_data.py        : Asset price and return feeds
    - analytics_engine.py  : Covariance / correlation estimation
    - portfolio_optimizer.py: Weight vectors
    - backtesting_engine.py : Historical calibration inputs

Author : QuantLab Research Team
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())   # library-safe; callers configure handlers


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class MonteCarloError(Exception):
    """Base exception for all MonteCarloEngine errors."""


class InputValidationError(MonteCarloError):
    """Raised when caller-supplied parameters fail validation."""


class SimulationNotRunError(MonteCarloError):
    """Raised when result properties are accessed before a simulation has been run."""


class NumericalInstabilityError(MonteCarloError):
    """Raised when the simulation produces non-finite values."""


# ---------------------------------------------------------------------------
# SimulationResult Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Immutable container for all outputs produced by a Monte Carlo simulation run.

    Attributes
    ----------
    price_paths : np.ndarray, shape (n_assets, n_simulations, n_steps + 1)
        Simulated asset price paths for every asset, simulation, and time step.
        Index 0 along the last axis corresponds to t=0 (the seed price).

    portfolio_paths : np.ndarray, shape (n_simulations, n_steps + 1)
        Dollar value of the portfolio along each simulated path.
        Index 0 along the last axis is the initial portfolio value.

    terminal_values : np.ndarray, shape (n_simulations,)
        Portfolio dollar value at the end of the horizon for each simulation.
        Equivalent to ``portfolio_paths[:, -1]``.

    return_distribution : np.ndarray, shape (n_simulations,)
        Total portfolio return (as a decimal) for each simulation,
        i.e. ``(terminal_values / initial_portfolio_value) - 1``.

    asset_names : list[str]
        Ordered list of asset tickers / identifiers matching axis-0 of
        ``price_paths``.

    n_simulations : int
        Number of Monte Carlo paths that were generated.

    n_steps : int
        Number of discrete time steps within the investment horizon.

    horizon_years : float
        Investment horizon expressed in years.

    confidence_levels : list[float]
        Confidence levels (e.g. [0.95, 0.99]) supplied at construction time.

    percentiles : dict[float, float]
        Mapping from each confidence level to the corresponding terminal
        portfolio value at that percentile of the return distribution.

    simulation_metadata : dict
        Auxiliary information recorded at simulation time (seed, dt, elapsed
        wall-clock seconds, etc.).
    """

    price_paths: np.ndarray
    portfolio_paths: np.ndarray
    terminal_values: np.ndarray
    return_distribution: np.ndarray
    asset_names: list[str]
    n_simulations: int
    n_steps: int
    horizon_years: float
    confidence_levels: list[float]
    percentiles: dict[float, float] = field(default_factory=dict)
    simulation_metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def mean_terminal_value(self) -> float:
        """Mean terminal portfolio value across all simulations."""
        return float(np.mean(self.terminal_values))

    @property
    def median_terminal_value(self) -> float:
        """Median terminal portfolio value across all simulations."""
        return float(np.median(self.terminal_values))

    @property
    def std_terminal_value(self) -> float:
        """Standard deviation of terminal portfolio values."""
        return float(np.std(self.terminal_values, ddof=1))

    @property
    def mean_return(self) -> float:
        """Mean total portfolio return across all simulations."""
        return float(np.mean(self.return_distribution))

    @property
    def std_return(self) -> float:
        """Standard deviation of total portfolio returns."""
        return float(np.std(self.return_distribution, ddof=1))

    def summary_frame(self) -> pd.DataFrame:
        """
        Return a tidy ``pd.DataFrame`` summarising key distribution statistics.

        Returns
        -------
        pd.DataFrame
            Single-column frame indexed by statistic name.
        """
        rows = {
            "n_simulations": self.n_simulations,
            "horizon_years": self.horizon_years,
            "mean_terminal_value": self.mean_terminal_value,
            "median_terminal_value": self.median_terminal_value,
            "std_terminal_value": self.std_terminal_value,
            "mean_return": self.mean_return,
            "std_return": self.std_return,
        }
        for cl, pv in self.percentiles.items():
            rows[f"pct_{int(cl * 100)}"] = pv

        return pd.DataFrame.from_dict(rows, orient="index", columns=["value"])


# ---------------------------------------------------------------------------
# MonteCarloEngine
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    """
    Geometric Brownian Motion Monte Carlo simulation engine.

    This engine simulates future asset price paths under GBM dynamics and
    aggregates them into portfolio value trajectories.  It is designed for
    institutional-scale workloads and supports thousands of simultaneous
    paths across multi-asset portfolios.

    GBM Dynamics
    ------------
    Each asset follows:

        S(t+dt) = S(t) * exp((mu - 0.5 * sigma^2) * dt
                              + sigma * sqrt(dt) * Z)

    where Z ~ N(0,1) and correlations across assets are introduced via a
    Cholesky decomposition of the supplied covariance matrix.

    Parameters
    ----------
    mu : array-like, shape (n_assets,)
        Annualised expected returns (drift) for each asset.

    sigma : array-like, shape (n_assets,)
        Annualised volatility for each asset.

    correlation_matrix : array-like, shape (n_assets, n_assets)
        Correlation matrix used to couple Brownian motions across assets.
        Must be symmetric positive semi-definite with unit diagonal.

    n_simulations : int, default 10_000
        Number of Monte Carlo paths to generate.

    horizon_years : float, default 1.0
        Investment horizon in years.

    n_steps : int, default 252
        Number of discrete time steps within the horizon.  A value of 252
        corresponds to daily steps for a one-year horizon.

    confidence_levels : sequence of float, default (0.95, 0.99)
        Confidence levels at which terminal value percentiles are reported.
        Each value must lie strictly in (0, 1).

    random_seed : int or None, default None
        Seed for the NumPy random number generator.  Providing a fixed seed
        guarantees reproducible results.

    Examples
    --------
    >>> mu    = np.array([0.10, 0.08, 0.12])
    >>> sigma = np.array([0.20, 0.15, 0.25])
    >>> corr  = np.array([[1.0, 0.4, 0.3],
    ...                   [0.4, 1.0, 0.2],
    ...                   [0.3, 0.2, 1.0]])
    >>> engine = MonteCarloEngine(mu, sigma, corr, n_simulations=5_000)
    >>> prices = pd.Series([100.0, 80.0, 50.0],
    ...                     index=["AAPL", "MSFT", "GOOG"])
    >>> weights = pd.Series([0.5, 0.3, 0.2],
    ...                      index=["AAPL", "MSFT", "GOOG"])
    >>> result = engine.simulate_portfolio(
    ...     current_prices=prices,
    ...     weights=weights,
    ...     portfolio_value=1_000_000.0,
    ... )
    >>> print(result.summary_frame())
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        mu: np.ndarray | Sequence[float],
        sigma: np.ndarray | Sequence[float],
        correlation_matrix: np.ndarray | Sequence[Sequence[float]],
        *,
        n_simulations: int = 10_000,
        horizon_years: float = 1.0,
        n_steps: int = 252,
        confidence_levels: Sequence[float] = (0.95, 0.99),
        random_seed: Optional[int] = None,
    ) -> None:
        # --- Convert and cache raw inputs ---
        self._mu: np.ndarray = np.asarray(mu, dtype=np.float64)
        self._sigma: np.ndarray = np.asarray(sigma, dtype=np.float64)
        self._corr: np.ndarray = np.asarray(correlation_matrix, dtype=np.float64)
        self._n_simulations: int = int(n_simulations)
        self._horizon_years: float = float(horizon_years)
        self._n_steps: int = int(n_steps)
        self._confidence_levels: list[float] = list(confidence_levels)
        self._random_seed: Optional[int] = random_seed

        # --- Validate all inputs eagerly ---
        self._validate_parameters()

        # --- Derived constants (computed once, reused across simulations) ---
        self._n_assets: int = len(self._mu)
        self._dt: float = self._horizon_years / self._n_steps

        # Cholesky factor L such that L @ L.T == covariance matrix
        self._cov_matrix: np.ndarray = self._build_covariance_matrix()
        self._cholesky_L: np.ndarray = self._compute_cholesky()

        # Seeded RNG (NumPy >= 1.17 Generator API)
        self._rng: np.random.Generator = np.random.default_rng(self._random_seed)

        logger.info(
            "MonteCarloEngine initialised | assets=%d | simulations=%d | "
            "horizon=%.2f yr | steps=%d | seed=%s",
            self._n_assets,
            self._n_simulations,
            self._horizon_years,
            self._n_steps,
            self._random_seed,
        )

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def n_assets(self) -> int:
        """Number of assets in the universe."""
        return self._n_assets

    @property
    def n_simulations(self) -> int:
        """Number of Monte Carlo paths per simulation run."""
        return self._n_simulations

    @property
    def horizon_years(self) -> float:
        """Investment horizon in years."""
        return self._horizon_years

    @property
    def n_steps(self) -> int:
        """Number of discrete time steps within the horizon."""
        return self._n_steps

    @property
    def dt(self) -> float:
        """Length of each time step in years (horizon / n_steps)."""
        return self._dt

    @property
    def covariance_matrix(self) -> np.ndarray:
        """Annualised covariance matrix derived from sigma and correlation."""
        return self._cov_matrix.copy()

    # ------------------------------------------------------------------
    # Input Validation
    # ------------------------------------------------------------------

    def _validate_parameters(self) -> None:
        """
        Validate all constructor parameters.

        Raises
        ------
        InputValidationError
            If any parameter is outside acceptable bounds or is malformed.
        """
        self._validate_drift_and_volatility()
        self._validate_correlation_matrix()
        self._validate_simulation_settings()
        self._validate_confidence_levels()

    def _validate_drift_and_volatility(self) -> None:
        """Validate mu and sigma arrays."""
        if self._mu.ndim != 1:
            raise InputValidationError(
                f"mu must be a 1-D array; received shape {self._mu.shape}."
            )
        if self._sigma.ndim != 1:
            raise InputValidationError(
                f"sigma must be a 1-D array; received shape {self._sigma.shape}."
            )
        if len(self._mu) != len(self._sigma):
            raise InputValidationError(
                f"mu (len={len(self._mu)}) and sigma (len={len(self._sigma)}) "
                "must have the same length."
            )
        if len(self._mu) == 0:
            raise InputValidationError("mu and sigma must contain at least one asset.")
        if not np.all(np.isfinite(self._mu)):
            raise InputValidationError("mu contains non-finite values (NaN or Inf).")
        if not np.all(np.isfinite(self._sigma)):
            raise InputValidationError("sigma contains non-finite values (NaN or Inf).")
        if np.any(self._sigma <= 0.0):
            raise InputValidationError(
                "All volatility values in sigma must be strictly positive."
            )

    def _validate_correlation_matrix(self) -> None:
        """Validate the correlation matrix for shape, symmetry, and PSD-ness."""
        n = len(self._mu)
        if self._corr.shape != (n, n):
            raise InputValidationError(
                f"correlation_matrix must have shape ({n}, {n}); "
                f"received {self._corr.shape}."
            )
        if not np.all(np.isfinite(self._corr)):
            raise InputValidationError(
                "correlation_matrix contains non-finite values."
            )
        if not np.allclose(self._corr, self._corr.T, atol=1e-8):
            raise InputValidationError("correlation_matrix is not symmetric.")
        if not np.allclose(np.diag(self._corr), 1.0, atol=1e-8):
            raise InputValidationError(
                "correlation_matrix diagonal entries must all equal 1.0."
            )
        if np.any(self._corr < -1.0 - 1e-8) or np.any(self._corr > 1.0 + 1e-8):
            raise InputValidationError(
                "All correlation coefficients must lie in [-1, 1]."
            )
        eigenvalues = np.linalg.eigvalsh(self._corr)
        if np.any(eigenvalues < -1e-8):
            raise InputValidationError(
                "correlation_matrix is not positive semi-definite "
                f"(min eigenvalue={eigenvalues.min():.6f})."
            )

    def _validate_simulation_settings(self) -> None:
        """Validate simulation counts, horizon, and step count."""
        if self._n_simulations < 1:
            raise InputValidationError(
                f"n_simulations must be >= 1; received {self._n_simulations}."
            )
        if self._n_simulations < 100:
            warnings.warn(
                f"n_simulations={self._n_simulations} is very low; "
                "statistical estimates may be unreliable. "
                "Consider using at least 1,000 simulations.",
                UserWarning,
                stacklevel=4,
            )
        if self._horizon_years <= 0.0:
            raise InputValidationError(
                f"horizon_years must be > 0; received {self._horizon_years}."
            )
        if self._n_steps < 1:
            raise InputValidationError(
                f"n_steps must be >= 1; received {self._n_steps}."
            )

    def _validate_confidence_levels(self) -> None:
        """Validate confidence levels."""
        for cl in self._confidence_levels:
            if not (0.0 < cl < 1.0):
                raise InputValidationError(
                    f"Each confidence level must lie strictly in (0, 1); "
                    f"received {cl}."
                )

    def _validate_prices(
        self,
        current_prices: pd.Series,
        asset_names: list[str],
    ) -> None:
        """
        Validate a ``pd.Series`` of current asset prices.

        Parameters
        ----------
        current_prices : pd.Series
            Current market prices indexed by asset name.
        asset_names : list[str]
            Expected asset names (must match series index).

        Raises
        ------
        InputValidationError
        """
        if not isinstance(current_prices, pd.Series):
            raise InputValidationError(
                "current_prices must be a pd.Series indexed by asset name."
            )
        missing = set(asset_names) - set(current_prices.index)
        if missing:
            raise InputValidationError(
                f"current_prices is missing entries for assets: {sorted(missing)}."
            )
        prices = current_prices[asset_names].values.astype(np.float64)
        if not np.all(np.isfinite(prices)):
            raise InputValidationError(
                "current_prices contains non-finite values (NaN or Inf)."
            )
        if np.any(prices <= 0.0):
            raise InputValidationError(
                "All asset prices must be strictly positive."
            )

    def _validate_weights(
        self,
        weights: pd.Series,
        asset_names: list[str],
    ) -> None:
        """
        Validate portfolio weight vector.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by asset name.  Weights need not sum
            to exactly 1.0 but will trigger a warning if they deviate by
            more than 1 basis point.
        asset_names : list[str]
            Expected asset names.

        Raises
        ------
        InputValidationError
        """
        if not isinstance(weights, pd.Series):
            raise InputValidationError(
                "weights must be a pd.Series indexed by asset name."
            )
        missing = set(asset_names) - set(weights.index)
        if missing:
            raise InputValidationError(
                f"weights is missing entries for assets: {sorted(missing)}."
            )
        w = weights[asset_names].values.astype(np.float64)
        if not np.all(np.isfinite(w)):
            raise InputValidationError(
                "weights contains non-finite values (NaN or Inf)."
            )
        total = w.sum()
        if abs(total - 1.0) > 1e-4:
            warnings.warn(
                f"Portfolio weights sum to {total:.6f}, not 1.0. "
                "Paths will be scaled accordingly.",
                UserWarning,
                stacklevel=4,
            )

    def _validate_portfolio_value(self, portfolio_value: float) -> None:
        """
        Validate the initial portfolio value.

        Raises
        ------
        InputValidationError
        """
        if not np.isfinite(portfolio_value):
            raise InputValidationError(
                "portfolio_value must be a finite number."
            )
        if portfolio_value <= 0.0:
            raise InputValidationError(
                f"portfolio_value must be strictly positive; received {portfolio_value}."
            )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _build_covariance_matrix(self) -> np.ndarray:
        """
        Construct the annualised covariance matrix from sigma and correlation.

        Returns
        -------
        np.ndarray, shape (n_assets, n_assets)
        """
        sigma_diag = np.diag(self._sigma)
        return sigma_diag @ self._corr @ sigma_diag

    def _compute_cholesky(self) -> np.ndarray:
        """
        Compute the lower Cholesky factor of the covariance matrix.

        A small ridge (1e-10 * I) is added to guard against floating-point
        near-singularity in the PSD covariance matrix.

        Returns
        -------
        np.ndarray, shape (n_assets, n_assets)
            Lower-triangular Cholesky factor L such that L @ L.T == cov.

        Raises
        ------
        NumericalInstabilityError
            If Cholesky decomposition fails even after regularisation.
        """
        ridge = 1e-10 * np.eye(self._n_assets)
        try:
            L = np.linalg.cholesky(self._cov_matrix + ridge)
        except np.linalg.LinAlgError as exc:
            raise NumericalInstabilityError(
                "Cholesky decomposition of the covariance matrix failed. "
                "Ensure the correlation matrix is positive definite."
            ) from exc
        logger.debug("Cholesky decomposition succeeded.")
        return L

    # ------------------------------------------------------------------
    # Geometric Brownian Motion Simulation
    # ------------------------------------------------------------------

    def simulate_gbm(
        self,
        current_prices: pd.Series,
        asset_names: Optional[list[str]] = None,
    ) -> np.ndarray:
        """
        Simulate future asset price paths under Geometric Brownian Motion.

        Each asset price follows:

            S(t + dt) = S(t) * exp((mu_i - 0.5 * sigma_i^2) * dt
                                    + [L @ Z]_i * sqrt(dt))

        where ``Z ~ N(0, I)`` and ``L`` is the Cholesky factor of the
        covariance matrix, introducing inter-asset correlations.

        Parameters
        ----------
        current_prices : pd.Series
            Current market prices indexed by asset name.  The series index
            determines the ordering of assets used internally.
        asset_names : list[str] or None, default None
            Subset and ordering of assets to simulate.  If ``None``, all
            assets in ``current_prices`` are used in index order.

        Returns
        -------
        np.ndarray, shape (n_assets, n_simulations, n_steps + 1)
            Simulated price paths.  Axis 0 indexes assets, axis 1 indexes
            simulations, axis 2 indexes time steps (0 = today).

        Raises
        ------
        InputValidationError
            If current_prices is invalid or asset_names mismatch.
        NumericalInstabilityError
            If the simulation produces non-finite values.
        """
        if asset_names is None:
            asset_names = list(current_prices.index)

        self._validate_prices(current_prices, asset_names)

        n_a = len(asset_names)
        if n_a != self._n_assets:
            raise InputValidationError(
                f"Number of assets in current_prices ({n_a}) does not match "
                f"the engine's configured n_assets ({self._n_assets})."
            )

        S0 = current_prices[asset_names].values.astype(np.float64)  # (n_assets,)
        dt = self._dt
        sqrt_dt = np.sqrt(dt)

        # Drift term:  (mu - 0.5 * sigma^2) * dt   -> shape (n_assets,)
        drift = (self._mu - 0.5 * self._sigma ** 2) * dt

        # Allocate output array (n_assets, n_simulations, n_steps + 1)
        paths = np.empty(
            (self._n_assets, self._n_simulations, self._n_steps + 1),
            dtype=np.float64,
        )
        paths[:, :, 0] = S0[:, np.newaxis]  # seed column

        logger.info(
            "Starting GBM simulation | assets=%d | simulations=%d | steps=%d",
            self._n_assets, self._n_simulations, self._n_steps,
        )

        # --- Vectorised time-stepping ---
        # Z_all : shape (n_steps, n_assets, n_simulations)
        # We draw all random numbers in a single call for performance.
        Z_all = self._rng.standard_normal(
            size=(self._n_steps, self._n_assets, self._n_simulations)
        )

        # Correlate shocks via Cholesky:  corr_Z[t] = L @ Z_all[t]
        # L : (n_assets, n_assets),  Z_all[t] : (n_assets, n_simulations)
        # -> (n_assets, n_simulations)  for each t
        for t in range(self._n_steps):
            corr_Z = self._cholesky_L @ Z_all[t]          # (n_assets, n_sims)
            log_ret = drift[:, np.newaxis] + sqrt_dt * corr_Z
            paths[:, :, t + 1] = paths[:, :, t] * np.exp(log_ret)

        # --- Numerical integrity check ---
        if not np.all(np.isfinite(paths)):
            raise NumericalInstabilityError(
                "GBM simulation produced non-finite values. "
                "Check drift and volatility parameters for extreme values."
            )

        logger.info("GBM simulation complete.")
        return paths   # (n_assets, n_simulations, n_steps + 1)

    # ------------------------------------------------------------------
    # Portfolio Path Simulation
    # ------------------------------------------------------------------

    def _compute_portfolio_paths(
        self,
        price_paths: np.ndarray,
        weights: np.ndarray,
        S0: np.ndarray,
        portfolio_value: float,
    ) -> np.ndarray:
        """
        Aggregate per-asset price paths into portfolio dollar-value paths.

        The portfolio value at time t is computed as:

            V(t) = portfolio_value * sum_i [ w_i * S_i(t) / S_i(0) ]

        where w_i is the weight of asset i.

        Parameters
        ----------
        price_paths : np.ndarray, shape (n_assets, n_simulations, n_steps + 1)
            Simulated price paths from ``simulate_gbm``.
        weights : np.ndarray, shape (n_assets,)
            Portfolio weight for each asset.
        S0 : np.ndarray, shape (n_assets,)
            Initial (seed) asset prices.
        portfolio_value : float
            Initial portfolio dollar value.

        Returns
        -------
        np.ndarray, shape (n_simulations, n_steps + 1)
            Portfolio dollar-value paths.
        """
        # Normalised price relatives: S_i(t) / S_i(0)
        # price_paths shape: (n_assets, n_simulations, n_steps + 1)
        price_relatives = price_paths / S0[:, np.newaxis, np.newaxis]

        # Weighted sum across assets
        # weights: (n_assets,) -> broadcast to (n_assets, 1, 1)
        weighted = weights[:, np.newaxis, np.newaxis] * price_relatives

        # portfolio_paths: (n_simulations, n_steps + 1)
        portfolio_paths = portfolio_value * weighted.sum(axis=0)
        return portfolio_paths

    # ------------------------------------------------------------------
    # Portfolio Return Distribution
    # ------------------------------------------------------------------

    def _compute_return_distribution(
        self,
        terminal_values: np.ndarray,
        portfolio_value: float,
    ) -> np.ndarray:
        """
        Compute total portfolio returns from terminal portfolio values.

        Parameters
        ----------
        terminal_values : np.ndarray, shape (n_simulations,)
        portfolio_value : float
            Initial portfolio dollar value.

        Returns
        -------
        np.ndarray, shape (n_simulations,)
            Total return as a decimal (e.g. 0.12 for +12%).
        """
        return (terminal_values / portfolio_value) - 1.0

    # ------------------------------------------------------------------
    # Primary Public Interface
    # ------------------------------------------------------------------

    def simulate_portfolio(
        self,
        current_prices: pd.Series,
        weights: pd.Series,
        portfolio_value: float,
        asset_names: Optional[list[str]] = None,
    ) -> SimulationResult:
        """
        Run a full Monte Carlo simulation and return a ``SimulationResult``.

        This is the primary entry point for consumers of this engine.  It
        chains GBM path generation, portfolio aggregation, and return
        distribution computation into a single call.

        Parameters
        ----------
        current_prices : pd.Series
            Current market prices indexed by asset ticker / identifier.
        weights : pd.Series
            Portfolio weights indexed by asset ticker / identifier.
            Weights are used as supplied (not renormalised); a warning is
            issued if they do not sum to 1.0.
        portfolio_value : float
            Initial portfolio dollar (or base-currency) value.
        asset_names : list[str] or None, default None
            Explicit asset ordering.  If ``None``, the union of
            ``current_prices.index`` is used in its natural order.

        Returns
        -------
        SimulationResult
            Fully populated result container with price paths, portfolio
            paths, terminal values, return distribution, and percentiles.

        Raises
        ------
        InputValidationError
            If any input fails validation.
        NumericalInstabilityError
            If the simulation produces non-finite values.

        Examples
        --------
        >>> result = engine.simulate_portfolio(
        ...     current_prices=pd.Series({"SPY": 450.0, "AGG": 100.0}),
        ...     weights=pd.Series({"SPY": 0.6, "AGG": 0.4}),
        ...     portfolio_value=1_000_000.0,
        ... )
        >>> result.summary_frame()
        """
        import time
        t_start = time.perf_counter()

        # --- Resolve asset universe ---
        if asset_names is None:
            asset_names = list(current_prices.index)

        # --- Validate all portfolio-level inputs ---
        self._validate_prices(current_prices, asset_names)
        self._validate_weights(weights, asset_names)
        self._validate_portfolio_value(portfolio_value)

        logger.info(
            "simulate_portfolio | assets=%s | portfolio_value=%.2f",
            asset_names, portfolio_value,
        )

        # --- GBM price path simulation ---
        price_paths = self.simulate_gbm(current_prices, asset_names)
        # shape: (n_assets, n_simulations, n_steps + 1)

        # --- Portfolio path aggregation ---
        S0 = current_prices[asset_names].values.astype(np.float64)
        w = weights[asset_names].values.astype(np.float64)

        portfolio_paths = self._compute_portfolio_paths(
            price_paths, w, S0, portfolio_value
        )
        # shape: (n_simulations, n_steps + 1)

        # --- Terminal values & return distribution ---
        terminal_values = portfolio_paths[:, -1]

        if not np.all(np.isfinite(terminal_values)):
            raise NumericalInstabilityError(
                "Portfolio simulation produced non-finite terminal values."
            )

        return_distribution = self._compute_return_distribution(
            terminal_values, portfolio_value
        )

        # --- Percentiles at requested confidence levels ---
        percentiles: dict[float, float] = {
            cl: float(np.percentile(terminal_values, cl * 100))
            for cl in self._confidence_levels
        }

        elapsed = time.perf_counter() - t_start

        metadata = {
            "random_seed": self._random_seed,
            "dt_years": self._dt,
            "elapsed_seconds": round(elapsed, 4),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        }

        logger.info(
            "simulate_portfolio complete | elapsed=%.3f s | "
            "mean_terminal=%.2f | std_terminal=%.2f",
            elapsed,
            float(np.mean(terminal_values)),
            float(np.std(terminal_values, ddof=1)),
        )

        return SimulationResult(
            price_paths=price_paths,
            portfolio_paths=portfolio_paths,
            terminal_values=terminal_values,
            return_distribution=return_distribution,
            asset_names=asset_names,
            n_simulations=self._n_simulations,
            n_steps=self._n_steps,
            horizon_years=self._horizon_years,
            confidence_levels=self._confidence_levels,
            percentiles=percentiles,
            simulation_metadata=metadata,
        )

    # ======================================================================
    # VALUE AT RISK  (Monte Carlo)
    # ======================================================================

    def compute_var(
        self,
        result: SimulationResult,
        portfolio_value: float,
        confidence_level: float = 0.95,
    ) -> float:
        """
        Compute Monte Carlo Value at Risk (VaR) from a ``SimulationResult``.

        VaR is defined here as the **loss** (positive number) such that the
        probability of the portfolio losing *more* than VaR over the horizon
        equals ``1 - confidence_level``.  It is computed directly from the
        empirical return distribution produced by ``simulate_portfolio``.

        Formula
        -------
            VaR_α = -portfolio_value × Q_{1-α}(return_distribution)

        where Q_{1-α} is the (1-α) quantile of the simulated return
        distribution (a negative number in loss scenarios).

        Parameters
        ----------
        result : SimulationResult
            Output of a prior ``simulate_portfolio`` call.
        portfolio_value : float
            Initial portfolio value used as the dollar base for VaR.
        confidence_level : float, default 0.95
            Confidence level α ∈ (0, 1).  Typical institutional values
            are 0.95 (regulatory) and 0.99 (internal risk limits).

        Returns
        -------
        float
            VaR expressed as a positive dollar loss.  A return of 50,000
            means "with 95% confidence, losses will not exceed $50,000."

        Raises
        ------
        InputValidationError
            If ``confidence_level`` is outside (0, 1) or ``portfolio_value``
            is non-positive.
        """
        if not (0.0 < confidence_level < 1.0):
            raise InputValidationError(
                f"confidence_level must be in (0, 1); received {confidence_level}."
            )
        if portfolio_value <= 0.0:
            raise InputValidationError(
                f"portfolio_value must be positive; received {portfolio_value}."
            )

        loss_quantile = float(
            np.percentile(result.return_distribution, (1.0 - confidence_level) * 100)
        )
        var_dollar = -portfolio_value * loss_quantile

        logger.info(
            "VaR(%.0f%%) = $%.2f  [quantile=%.4f%%]",
            confidence_level * 100,
            var_dollar,
            loss_quantile * 100,
        )
        return var_dollar

    # ======================================================================
    # CONDITIONAL VALUE AT RISK  (Monte Carlo CVaR / Expected Shortfall)
    # ======================================================================

    def compute_cvar(
        self,
        result: SimulationResult,
        portfolio_value: float,
        confidence_level: float = 0.95,
    ) -> float:
        """
        Compute Monte Carlo Conditional Value at Risk (CVaR).

        CVaR — also known as Expected Shortfall (ES) — is the expected loss
        *conditional on* losses exceeding the VaR threshold.  It is a
        coherent risk measure and is increasingly mandated by Basel III/IV
        and FRTB frameworks as the primary regulatory risk metric.

        Formula
        -------
            CVaR_α = -portfolio_value
                     × E[return | return < Q_{1-α}(return_distribution)]

        Parameters
        ----------
        result : SimulationResult
            Output of a prior ``simulate_portfolio`` call.
        portfolio_value : float
            Initial portfolio value used as the dollar base for CVaR.
        confidence_level : float, default 0.95
            Confidence level α ∈ (0, 1).

        Returns
        -------
        float
            CVaR expressed as a positive dollar loss.  Always >= VaR at the
            same confidence level by construction.

        Raises
        ------
        InputValidationError
            If ``confidence_level`` is outside (0, 1).
        MonteCarloError
            If no simulations fall in the tail (degenerate distribution).
        """
        if not (0.0 < confidence_level < 1.0):
            raise InputValidationError(
                f"confidence_level must be in (0, 1); received {confidence_level}."
            )
        if portfolio_value <= 0.0:
            raise InputValidationError(
                f"portfolio_value must be positive; received {portfolio_value}."
            )

        loss_quantile_value = float(
            np.percentile(result.return_distribution, (1.0 - confidence_level) * 100)
        )
        tail_mask = result.return_distribution <= loss_quantile_value
        tail_returns = result.return_distribution[tail_mask]

        if tail_returns.size == 0:
            raise MonteCarloError(
                "No simulations fall in the tail; cannot compute CVaR. "
                "Increase n_simulations."
            )

        cvar_dollar = -portfolio_value * float(np.mean(tail_returns))

        logger.info(
            "CVaR(%.0f%%) = $%.2f  [tail_sims=%d / %d]",
            confidence_level * 100,
            cvar_dollar,
            tail_returns.size,
            result.n_simulations,
        )
        return cvar_dollar

    # ======================================================================
    # STRESS TESTING FRAMEWORK
    # ======================================================================

    def run_stress_test(
        self,
        current_prices: pd.Series,
        weights: pd.Series,
        portfolio_value: float,
        scenario_name: str,
        drift_shock: float = 0.0,
        vol_multiplier: float = 1.0,
        asset_names: Optional[list[str]] = None,
    ) -> SimulationResult:
        """
        Run a single stress scenario by temporarily overriding drift and/or
        volatility parameters before re-simulating the portfolio.

        The engine's original calibration (``_mu``, ``_sigma``, ``_corr``)
        is **not mutated** — overrides are applied to transient local copies
        used only for this call.

        Stress Mechanics
        ----------------
        * ``drift_shock``     : Additive annualised return shock applied
          uniformly to all assets (e.g. -0.20 for a 20% crash regime).
        * ``vol_multiplier``  : Multiplicative scale applied to all asset
          volatilities (e.g. 2.0 for a high-volatility regime).  Must be
          strictly positive.

        Parameters
        ----------
        current_prices : pd.Series
            Current market prices indexed by asset name.
        weights : pd.Series
            Portfolio weights indexed by asset name.
        portfolio_value : float
            Initial portfolio dollar value.
        scenario_name : str
            Human-readable label recorded in ``simulation_metadata``.
        drift_shock : float, default 0.0
            Additive shock to annualised expected returns (all assets).
        vol_multiplier : float, default 1.0
            Multiplicative factor applied to annualised volatilities.
        asset_names : list[str] or None, default None
            Explicit asset ordering; if ``None``, inferred from
            ``current_prices.index``.

        Returns
        -------
        SimulationResult
            Full simulation result under the stressed parameters, with
            ``scenario_name`` embedded in ``simulation_metadata``.

        Raises
        ------
        InputValidationError
            If ``vol_multiplier`` is non-positive.
        """
        if vol_multiplier <= 0.0:
            raise InputValidationError(
                f"vol_multiplier must be > 0; received {vol_multiplier}."
            )

        logger.info(
            "Stress test | scenario='%s' | drift_shock=%.4f | vol_mult=%.4f",
            scenario_name, drift_shock, vol_multiplier,
        )

        # --- Build stressed parameter copies (no mutation of self) ---
        stressed_mu    = self._mu    + drift_shock
        stressed_sigma = self._sigma * vol_multiplier

        # Rebuild Cholesky on stressed covariance
        stressed_cov = np.diag(stressed_sigma) @ self._corr @ np.diag(stressed_sigma)
        ridge = 1e-10 * np.eye(self._n_assets)
        try:
            stressed_L = np.linalg.cholesky(stressed_cov + ridge)
        except np.linalg.LinAlgError as exc:
            raise NumericalInstabilityError(
                f"Cholesky decomposition failed for scenario '{scenario_name}'."
            ) from exc

        # --- Temporarily swap in stressed parameters ---
        orig_mu, orig_sigma = self._mu, self._sigma
        orig_cov, orig_L    = self._cov_matrix, self._cholesky_L
        try:
            self._mu          = stressed_mu
            self._sigma       = stressed_sigma
            self._cov_matrix  = stressed_cov
            self._cholesky_L  = stressed_L
            result = self.simulate_portfolio(
                current_prices, weights, portfolio_value, asset_names
            )
        finally:
            # Guaranteed restore even if simulate_portfolio raises
            self._mu         = orig_mu
            self._sigma      = orig_sigma
            self._cov_matrix = orig_cov
            self._cholesky_L = orig_L

        result.simulation_metadata["scenario_name"] = scenario_name
        result.simulation_metadata["drift_shock"]   = drift_shock
        result.simulation_metadata["vol_multiplier"] = vol_multiplier
        return result

    def run_standard_stress_suite(
        self,
        current_prices: pd.Series,
        weights: pd.Series,
        portfolio_value: float,
        asset_names: Optional[list[str]] = None,
    ) -> dict[str, SimulationResult]:
        """
        Execute the four canonical QuantLab stress scenarios.

        Scenarios
        ---------
        ``market_crash``
            Uniform -20% drift shock across all assets.  Models a moderate
            equity bear market (e.g. 2001 dot-com, 2022 rate-shock).

        ``severe_crash``
            Uniform -40% drift shock.  Models a severe dislocation event
            (e.g. 2008–09 GFC trough, COVID-19 March 2020).

        ``bull_market``
            Uniform +20% drift shock.  Models a strong risk-on regime
            (e.g. 2013, 2019, 2023 equity rallies).

        ``high_volatility``
            Volatility scaled to 2× baseline with no drift shock.  Models a
            VIX-spike regime (e.g. Aug 2015, Q4 2018, Mar 2020) where
            directional outcome is uncertain but dispersion is elevated.

        Parameters
        ----------
        current_prices : pd.Series
        weights : pd.Series
        portfolio_value : float
        asset_names : list[str] or None

        Returns
        -------
        dict[str, SimulationResult]
            Mapping from scenario label to its ``SimulationResult``.
        """
        _STANDARD_SCENARIOS: list[dict] = [
            {
                "scenario_name": "market_crash",
                "drift_shock":   -0.20,
                "vol_multiplier": 1.0,
            },
            {
                "scenario_name": "severe_crash",
                "drift_shock":   -0.40,
                "vol_multiplier": 1.0,
            },
            {
                "scenario_name": "bull_market",
                "drift_shock":   +0.20,
                "vol_multiplier": 1.0,
            },
            {
                "scenario_name": "high_volatility",
                "drift_shock":    0.0,
                "vol_multiplier": 2.0,
            },
        ]

        logger.info("Running standard stress suite (%d scenarios).", len(_STANDARD_SCENARIOS))
        results: dict[str, SimulationResult] = {}
        for spec in _STANDARD_SCENARIOS:
            name = spec["scenario_name"]
            results[name] = self.run_stress_test(
                current_prices=current_prices,
                weights=weights,
                portfolio_value=portfolio_value,
                asset_names=asset_names,
                **spec,
            )
            logger.info(
                "  [%s] mean_terminal=$%.2f  std=$%.2f",
                name,
                results[name].mean_terminal_value,
                results[name].std_terminal_value,
            )
        return results

    # ======================================================================
    # SCENARIO ANALYSIS ENGINE
    # ======================================================================

    def run_scenario_analysis(
        self,
        current_prices: pd.Series,
        weights: pd.Series,
        portfolio_value: float,
        custom_scenarios: Optional[list[dict]] = None,
        include_baseline: bool = True,
        asset_names: Optional[list[str]] = None,
    ) -> dict[str, SimulationResult]:
        """
        Run a configurable set of named market scenarios and collect results.

        This method combines the standard stress suite with any caller-defined
        scenarios, optionally prepending an unshocked baseline simulation.
        Results are keyed by scenario name and are suitable for direct input
        to ``build_scenario_comparison_table``.

        Parameters
        ----------
        current_prices : pd.Series
            Current market prices indexed by asset name.
        weights : pd.Series
            Portfolio weights indexed by asset name.
        portfolio_value : float
            Initial portfolio dollar value.
        custom_scenarios : list[dict] or None, default None
            Additional scenarios beyond the standard four.  Each dict must
            contain at least:

            .. code-block:: python

                {
                    "scenario_name": "my_scenario",   # str, required
                    "drift_shock":   -0.10,            # float, required
                    "vol_multiplier": 1.5,             # float, required
                }

        include_baseline : bool, default True
            If ``True``, a baseline run (no shock) is inserted first under
            the key ``"baseline"``.
        asset_names : list[str] or None, default None

        Returns
        -------
        dict[str, SimulationResult]
            Ordered mapping from scenario name → ``SimulationResult``.
            Insertion order is: baseline (if included), standard suite,
            then custom scenarios.

        Raises
        ------
        InputValidationError
            If a custom scenario dict is malformed.
        """
        scenario_results: dict[str, SimulationResult] = {}

        # --- Baseline ---
        if include_baseline:
            logger.info("Scenario analysis: running baseline.")
            baseline = self.run_stress_test(
                current_prices, weights, portfolio_value,
                scenario_name="baseline",
                drift_shock=0.0,
                vol_multiplier=1.0,
                asset_names=asset_names,
            )
            scenario_results["baseline"] = baseline

        # --- Standard stress suite ---
        stress_results = self.run_standard_stress_suite(
            current_prices, weights, portfolio_value, asset_names
        )
        scenario_results.update(stress_results)

        # --- Custom scenarios ---
        if custom_scenarios:
            _REQUIRED_KEYS = {"scenario_name", "drift_shock", "vol_multiplier"}
            for i, spec in enumerate(custom_scenarios):
                missing = _REQUIRED_KEYS - set(spec.keys())
                if missing:
                    raise InputValidationError(
                        f"Custom scenario at index {i} is missing keys: {missing}."
                    )
                name = spec["scenario_name"]
                logger.info("Scenario analysis: running custom scenario '%s'.", name)
                scenario_results[name] = self.run_stress_test(
                    current_prices=current_prices,
                    weights=weights,
                    portfolio_value=portfolio_value,
                    asset_names=asset_names,
                    scenario_name=name,
                    drift_shock=float(spec["drift_shock"]),
                    vol_multiplier=float(spec["vol_multiplier"]),
                )

        logger.info("Scenario analysis complete | %d scenarios.", len(scenario_results))
        return scenario_results

    # ======================================================================
    # RISK SUMMARY UTILITY
    # ======================================================================

    def build_risk_summary(
        self,
        result: SimulationResult,
        portfolio_value: float,
        label: str = "portfolio",
    ) -> pd.DataFrame:
        """
        Produce a comprehensive risk summary table for a single simulation.

        The table consolidates VaR / CVaR at both 95% and 99% confidence
        levels alongside key distribution statistics into a single
        ``pd.DataFrame`` suitable for reporting, logging, or downstream
        analytics.

        Metrics Included
        ----------------
        * VaR 95% / 99% (dollar loss, positive)
        * CVaR 95% / 99% (dollar loss, positive)
        * Expected terminal value & return
        * Worst-case terminal value & return (min simulation)
        * Best-case terminal value & return (max simulation)
        * Standard deviation of terminal value
        * Skewness of the return distribution
        * Excess kurtosis of the return distribution

        Parameters
        ----------
        result : SimulationResult
            Simulation whose risk metrics are to be summarised.
        portfolio_value : float
            Initial dollar value; used as the denominator for return
            calculations and the base for VaR / CVaR.
        label : str, default "portfolio"
            Column header in the returned frame (useful when concatenating
            summaries from multiple scenarios).

        Returns
        -------
        pd.DataFrame
            Single-column frame with metric names as index and numeric
            values in column ``label``.
        """
        returns = result.return_distribution
        terminal = result.terminal_values

        # Central moments of the return distribution
        mean_r   = float(np.mean(returns))
        std_r    = float(np.std(returns, ddof=1))
        skew_r   = float(
            np.mean(((returns - mean_r) / std_r) ** 3) if std_r > 0 else 0.0
        )
        kurt_r   = float(
            np.mean(((returns - mean_r) / std_r) ** 4) - 3.0 if std_r > 0 else 0.0
        )

        metrics: dict[str, float] = {
            # ---- VaR ----
            "VaR_95_dollar":        self.compute_var(result, portfolio_value, 0.95),
            "VaR_99_dollar":        self.compute_var(result, portfolio_value, 0.99),
            "VaR_95_pct":           self.compute_var(result, portfolio_value, 0.95) / portfolio_value,
            "VaR_99_pct":           self.compute_var(result, portfolio_value, 0.99) / portfolio_value,
            # ---- CVaR ----
            "CVaR_95_dollar":       self.compute_cvar(result, portfolio_value, 0.95),
            "CVaR_99_dollar":       self.compute_cvar(result, portfolio_value, 0.99),
            "CVaR_95_pct":          self.compute_cvar(result, portfolio_value, 0.95) / portfolio_value,
            "CVaR_99_pct":          self.compute_cvar(result, portfolio_value, 0.99) / portfolio_value,
            # ---- Distribution ----
            "expected_terminal":    float(np.mean(terminal)),
            "expected_return_pct":  mean_r,
            "worst_case_terminal":  float(np.min(terminal)),
            "worst_case_return_pct": float(np.min(returns)),
            "best_case_terminal":   float(np.max(terminal)),
            "best_case_return_pct": float(np.max(returns)),
            "std_terminal":         float(np.std(terminal, ddof=1)),
            "std_return_pct":       std_r,
            "skewness":             skew_r,
            "excess_kurtosis":      kurt_r,
        }

        frame = pd.DataFrame.from_dict(
            metrics, orient="index", columns=[label]
        )
        logger.info("Risk summary built for label='%s'.", label)
        return frame

    def build_scenario_comparison_table(
        self,
        scenario_results: dict[str, SimulationResult],
        portfolio_value: float,
    ) -> pd.DataFrame:
        """
        Compare risk summaries across multiple scenarios in a single table.

        Each scenario column is generated by ``build_risk_summary`` and
        joined horizontally, producing a metric × scenario matrix that
        is ready for export to CSV, Excel, or an HTML report.

        Parameters
        ----------
        scenario_results : dict[str, SimulationResult]
            Mapping from scenario name to ``SimulationResult``, typically
            the output of ``run_scenario_analysis``.
        portfolio_value : float
            Initial portfolio value used as the base for all risk metrics.

        Returns
        -------
        pd.DataFrame
            Multi-column frame: rows are risk metrics, columns are scenario
            names.  Scenarios are ordered as they appear in the input dict.

        Raises
        ------
        InputValidationError
            If ``scenario_results`` is empty.
        """
        if not scenario_results:
            raise InputValidationError(
                "scenario_results must contain at least one scenario."
            )

        frames = [
            self.build_risk_summary(result, portfolio_value, label=name)
            for name, result in scenario_results.items()
        ]
        comparison = pd.concat(frames, axis=1)
        logger.info(
            "Scenario comparison table built | scenarios=%s | metrics=%d",
            list(scenario_results.keys()),
            len(comparison),
        )
        return comparison


# ===========================================================================
# DEMONSTRATION  main()
# ===========================================================================

def main() -> None:
    """
    End-to-end demonstration of the QuantLab Monte Carlo risk analytics suite.

    Demonstrates
    ------------
    1. Engine construction with a realistic 3-asset equity/bond/commodity portfolio.
    2. Baseline GBM portfolio simulation (10,000 paths, 1-year horizon).
    3. Monte Carlo VaR and CVaR at 95% and 99% confidence.
    4. Full standard stress suite (crash / bull / high-vol).
    5. Scenario analysis with one custom scenario.
    6. Scenario comparison table rendered as a formatted console report.

    This function is intentionally self-contained: all parameters are defined
    inline so the module can be executed directly for QA and demonstration
    purposes without any external data source.
    """
    # ------------------------------------------------------------------
    # 0.  Logging configuration  (console handler for demo only)
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)
    _SEPARATOR = "=" * 72

    # ------------------------------------------------------------------
    # 1.  Portfolio specification
    # ------------------------------------------------------------------
    PORTFOLIO_VALUE = 10_000_000.0   # $10 million AUM

    asset_names = ["SPY", "AGG", "GLD"]

    current_prices = pd.Series(
        {"SPY": 450.00, "AGG": 98.50, "GLD": 185.00}
    )
    weights = pd.Series(
        {"SPY": 0.60, "AGG": 0.30, "GLD": 0.10}
    )

    # Annualised calibration (representative long-run estimates)
    mu    = np.array([0.10,  0.04, 0.06])   # expected returns
    sigma = np.array([0.18,  0.06, 0.15])   # volatilities

    corr = np.array([                        # correlation matrix
        [1.00, -0.15,  0.12],
        [-0.15, 1.00, -0.05],
        [ 0.12,-0.05,  1.00],
    ])

    # ------------------------------------------------------------------
    # 2.  Engine construction
    # ------------------------------------------------------------------
    print(f"\n{_SEPARATOR}")
    print("  QuantLab  |  Monte Carlo Risk Analytics Engine  |  Demo")
    print(_SEPARATOR)
    print(f"  Portfolio : {dict(zip(asset_names, weights.values))}")
    print(f"  AUM       : ${PORTFOLIO_VALUE:,.0f}")
    print(f"  Horizon   : 1 year  |  Steps: 252 (daily)  |  Paths: 10,000")
    print(_SEPARATOR)

    engine = MonteCarloEngine(
        mu=mu,
        sigma=sigma,
        correlation_matrix=corr,
        n_simulations=10_000,
        horizon_years=1.0,
        n_steps=252,
        confidence_levels=[0.05, 0.50, 0.95, 0.99],
        random_seed=2024,
    )

    # ------------------------------------------------------------------
    # 3.  Baseline simulation
    # ------------------------------------------------------------------
    print("\n[1/5]  Running baseline simulation …")
    baseline_result = engine.simulate_portfolio(
        current_prices=current_prices,
        weights=weights,
        portfolio_value=PORTFOLIO_VALUE,
    )
    print(f"       price_paths shape    : {baseline_result.price_paths.shape}")
    print(f"       portfolio_paths shape: {baseline_result.portfolio_paths.shape}")
    print(f"       terminal_values shape: {baseline_result.terminal_values.shape}")

    # ------------------------------------------------------------------
    # 4.  VaR and CVaR
    # ------------------------------------------------------------------
    print(f"\n[2/5]  Computing VaR & CVaR …")

    var_95  = engine.compute_var(baseline_result,  PORTFOLIO_VALUE, 0.95)
    var_99  = engine.compute_var(baseline_result,  PORTFOLIO_VALUE, 0.99)
    cvar_95 = engine.compute_cvar(baseline_result, PORTFOLIO_VALUE, 0.95)
    cvar_99 = engine.compute_cvar(baseline_result, PORTFOLIO_VALUE, 0.99)

    print(f"\n  {'Metric':<28}  {'Dollar Loss':>14}  {'% of AUM':>10}")
    print(f"  {'-'*28}  {'-'*14}  {'-'*10}")
    for label, val in [
        ("VaR  (95%)", var_95),
        ("VaR  (99%)", var_99),
        ("CVaR (95%)", cvar_95),
        ("CVaR (99%)", cvar_99),
    ]:
        print(f"  {label:<28}  ${val:>13,.0f}  {val/PORTFOLIO_VALUE:>9.2%}")

    # ------------------------------------------------------------------
    # 5.  Risk summary
    # ------------------------------------------------------------------
    print(f"\n[3/5]  Building risk summary …")
    risk_summary = engine.build_risk_summary(
        baseline_result, PORTFOLIO_VALUE, label="baseline"
    )

    # Pretty-print selected metrics
    _DISPLAY_METRICS = [
        "expected_terminal", "expected_return_pct",
        "worst_case_terminal", "worst_case_return_pct",
        "best_case_terminal",  "best_case_return_pct",
        "std_terminal", "skewness", "excess_kurtosis",
    ]
    subset = risk_summary.loc[_DISPLAY_METRICS]
    print(f"\n  {'Metric':<30}  {'Baseline':>18}")
    print(f"  {'-'*30}  {'-'*18}")
    for metric, row in subset.iterrows():
        val = row["baseline"]
        if "pct" in str(metric) or metric in ("skewness", "excess_kurtosis"):
            print(f"  {metric:<30}  {val:>17.4f}")
        else:
            print(f"  {metric:<30}  ${val:>16,.0f}")

    # ------------------------------------------------------------------
    # 6.  Scenario analysis (standard + 1 custom)
    # ------------------------------------------------------------------
    print(f"\n[4/5]  Running scenario analysis …")
    custom_scenarios = [
        {
            "scenario_name": "stagflation",
            "drift_shock":   -0.10,   # moderate growth headwind
            "vol_multiplier": 1.50,   # elevated uncertainty
        }
    ]

    scenario_results = engine.run_scenario_analysis(
        current_prices=current_prices,
        weights=weights,
        portfolio_value=PORTFOLIO_VALUE,
        custom_scenarios=custom_scenarios,
        include_baseline=True,
    )
    print(f"       Scenarios completed: {list(scenario_results.keys())}")

    # ------------------------------------------------------------------
    # 7.  Scenario comparison table
    # ------------------------------------------------------------------
    print(f"\n[5/5]  Building scenario comparison table …")
    comparison = engine.build_scenario_comparison_table(
        scenario_results, PORTFOLIO_VALUE
    )

    # Select the most decision-relevant rows for console display
    _COMPARISON_ROWS = [
        "VaR_95_dollar", "VaR_99_dollar",
        "CVaR_95_dollar", "CVaR_99_dollar",
        "expected_terminal", "expected_return_pct",
        "worst_case_terminal", "worst_case_return_pct",
        "best_case_terminal",
    ]
    display = comparison.loc[_COMPARISON_ROWS]

    print(f"\n{_SEPARATOR}")
    print("  SCENARIO COMPARISON — KEY RISK METRICS")
    print(_SEPARATOR)

    col_w = 16
    header_cols = "".join(f"  {col:>{col_w}}" for col in display.columns)
    print(f"  {'Metric':<30}{header_cols}")
    print(f"  {'-'*30}" + "".join(f"  {'-'*col_w}" for _ in display.columns))

    for metric, row in display.iterrows():
        fmt_vals = []
        for val in row:
            if "pct" in str(metric):
                fmt_vals.append(f"  {val:>{col_w}.2%}")
            else:
                fmt_vals.append(f"  ${val:>{col_w-1},.0f}")
        print(f"  {metric:<30}" + "".join(fmt_vals))

    print(f"\n{_SEPARATOR}")
    print("  Demo complete.  All risk modules executed successfully.")
    print(_SEPARATOR + "\n")


if __name__ == "__main__":
    main()
