"""
portfolio_optimizer.py

A production-quality module implementing core portfolio optimization
primitives: input validation, performance evaluation, an equal-weight
portfolio, a minimum-variance portfolio, a maximum-Sharpe-ratio portfolio,
and efficient frontier generation (all long-only, fully invested).

Author: Quantitative Research Team
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult


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
class PortfolioOptimizerError(Exception):
    """Base exception for all PortfolioOptimizer-related errors."""


class InvalidDataError(PortfolioOptimizerError):
    """Raised when input return data is malformed, empty, or otherwise invalid."""


class InsufficientDataError(PortfolioOptimizerError):
    """Raised when there is not enough data (e.g. too few observations/assets)."""


class OptimizationError(PortfolioOptimizerError):
    """Raised when a numerical optimization routine fails to converge."""


class InvalidWeightsError(PortfolioOptimizerError):
    """Raised when computed or supplied portfolio weights are invalid."""


# --------------------------------------------------------------------------
# PortfolioOptimizer
# --------------------------------------------------------------------------
class PortfolioOptimizer:
    """
    A portfolio optimization engine supporting long-only, fully-invested
    portfolios subject to the constraints:

        - sum(weights) == 1
        - 0 <= weight_i <= 1  for all assets i

    The optimizer operates on a historical returns matrix and derives the
    annualized expected return vector and covariance matrix internally.

    Attributes
    ----------
    returns : pd.DataFrame
        Historical periodic (e.g. daily) returns, assets as columns.
    risk_free_rate : float
        Annualized risk-free rate used in Sharpe ratio calculations.
    periods_per_year : int
        Number of return periods per year, used for annualization
        (e.g. 252 for daily, 12 for monthly).
    asset_names : pd.Index
        Column labels (tickers) of the returns DataFrame.
    n_assets : int
        Number of assets in the universe.
    expected_returns : pd.Series
        Annualized mean return per asset.
    cov_matrix : pd.DataFrame
        Annualized covariance matrix of asset returns.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> None:
        """
        Initialize the PortfolioOptimizer.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical periodic returns with assets as columns and
            observations (dates) as rows. Values should be simple
            (not log) returns, e.g. 0.01 for 1%.
        risk_free_rate : float, optional
            Annualized risk-free rate, by default 0.02 (2%).
        periods_per_year : int, optional
            Number of periods in a year used to annualize return and
            volatility statistics, by default 252 (daily data).

        Raises
        ------
        InvalidDataError
            If `returns` is not a non-empty DataFrame with valid numeric data.
        InsufficientDataError
            If there are fewer than 2 assets or fewer than 2 observations.
        """
        logger.info("Initializing PortfolioOptimizer.")

        self._validate_returns(returns)
        self._validate_risk_free_rate(risk_free_rate)
        self._validate_periods_per_year(periods_per_year)

        # Drop any rows that are entirely NaN, then validate again for safety.
        cleaned_returns = returns.dropna(how="all")
        if cleaned_returns.isnull().values.any():
            logger.warning(
                "Returns data contains partial NaNs; forward/back-filling "
                "within assets is NOT performed automatically. Consider "
                "cleaning your data prior to optimization."
            )

        self.returns: pd.DataFrame = cleaned_returns.astype(float)
        self.risk_free_rate: float = float(risk_free_rate)
        self.periods_per_year: int = int(periods_per_year)

        self.asset_names: pd.Index = self.returns.columns
        self.n_assets: int = self.returns.shape[1]

        # Annualized statistics derived once at construction time.
        self.expected_returns: pd.Series = (
            self.returns.mean() * self.periods_per_year
        )
        self.cov_matrix: pd.DataFrame = (
            self.returns.cov() * self.periods_per_year
        )

        logger.info(
            "PortfolioOptimizer initialized with %d assets and %d observations.",
            self.n_assets,
            self.returns.shape[0],
        )

    # ----------------------------------------------------------------
    # Input Validation Methods
    # ----------------------------------------------------------------
    @staticmethod
    def _validate_returns(returns: pd.DataFrame) -> None:
        """
        Validate the returns DataFrame supplied at construction time.

        Parameters
        ----------
        returns : pd.DataFrame
            Candidate historical returns matrix.

        Raises
        ------
        InvalidDataError
            If `returns` is not a DataFrame, is empty, contains non-numeric
            columns, or contains infinite values.
        InsufficientDataError
            If there are fewer than 2 assets or fewer than 2 observations.
        """
        if not isinstance(returns, pd.DataFrame):
            raise InvalidDataError(
                f"`returns` must be a pandas DataFrame, got {type(returns).__name__}."
            )

        if returns.empty:
            raise InvalidDataError("`returns` DataFrame is empty.")

        non_numeric = returns.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise InvalidDataError(
                f"`returns` contains non-numeric columns: {non_numeric}."
            )

        if np.isinf(returns.values).any():
            raise InvalidDataError("`returns` contains infinite values.")

        if returns.shape[1] < 2:
            raise InsufficientDataError(
                f"At least 2 assets are required for optimization; "
                f"got {returns.shape[1]}."
            )

        if returns.shape[0] < 2:
            raise InsufficientDataError(
                f"At least 2 return observations are required; "
                f"got {returns.shape[0]}."
            )

        if returns.columns.duplicated().any():
            dupes = returns.columns[returns.columns.duplicated()].tolist()
            raise InvalidDataError(f"`returns` has duplicate asset columns: {dupes}.")

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
            If `risk_free_rate` is not numeric or is unreasonably extreme.
        """
        if not isinstance(risk_free_rate, (int, float)) or isinstance(
            risk_free_rate, bool
        ):
            raise InvalidDataError(
                f"`risk_free_rate` must be numeric, got {type(risk_free_rate).__name__}."
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
                f"`periods_per_year` must be an integer, "
                f"got {type(periods_per_year).__name__}."
            )
        if periods_per_year <= 0:
            raise InvalidDataError(
                f"`periods_per_year` must be positive, got {periods_per_year}."
            )

    def _validate_weights(self, weights: np.ndarray) -> None:
        """
        Validate a candidate weight vector against portfolio constraints.

        Parameters
        ----------
        weights : np.ndarray
            Candidate portfolio weights.

        Raises
        ------
        InvalidWeightsError
            If `weights` has the wrong shape, contains NaN/inf values,
            violates the [0, 1] bound, or does not sum to 1 (within
            numerical tolerance).
        """
        weights = np.asarray(weights, dtype=float)

        if weights.shape != (self.n_assets,):
            raise InvalidWeightsError(
                f"Weights must have shape ({self.n_assets},), got {weights.shape}."
            )

        if not np.all(np.isfinite(weights)):
            raise InvalidWeightsError("Weights contain NaN or infinite values.")

        if np.any(weights < -1e-6) or np.any(weights > 1 + 1e-6):
            raise InvalidWeightsError(
                "Weights violate long-only bounds [0, 1]: "
                f"min={weights.min():.6f}, max={weights.max():.6f}."
            )

        weight_sum = weights.sum()
        if not np.isclose(weight_sum, 1.0, atol=1e-4):
            raise InvalidWeightsError(
                f"Weights must sum to 1.0 (got {weight_sum:.6f})."
            )

    # ----------------------------------------------------------------
    # Portfolio Performance Evaluation
    # ----------------------------------------------------------------
    def evaluate_portfolio(self, weights: np.ndarray) -> Dict[str, object]:
        """
        Evaluate the performance of a portfolio given a set of weights.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights, shape (n_assets,). Must be long-only and
            sum to 1.

        Returns
        -------
        Dict[str, object]
            Dictionary with keys:
                - "weights" (pd.Series): weights indexed by asset name
                - "expected_return" (float): annualized expected return
                - "volatility" (float): annualized volatility (std dev)
                - "sharpe_ratio" (float): annualized Sharpe ratio

        Raises
        ------
        InvalidWeightsError
            If `weights` fails validation (wrong shape, out of bounds,
            does not sum to 1, contains NaN/inf).
        """
        weights = np.asarray(weights, dtype=float)
        self._validate_weights(weights)

        expected_return = float(np.dot(weights, self.expected_returns.values))
        variance = float(np.dot(weights, np.dot(self.cov_matrix.values, weights)))

        if variance < 0:
            # Should not occur with a valid covariance matrix, but guard
            # against numerical noise from near-singular matrices.
            logger.warning(
                "Computed negative variance (%.10f); clipping to 0.0.", variance
            )
            variance = 0.0

        volatility = float(np.sqrt(variance))

        if volatility > 0:
            sharpe_ratio = (expected_return - self.risk_free_rate) / volatility
        else:
            logger.warning(
                "Portfolio volatility is zero; Sharpe ratio set to NaN."
            )
            sharpe_ratio = float("nan")

        result = {
            "weights": pd.Series(weights, index=self.asset_names, name="weight"),
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": float(sharpe_ratio),
        }

        logger.info(
            "Evaluated portfolio: return=%.4f%%, volatility=%.4f%%, sharpe=%.4f",
            expected_return * 100,
            volatility * 100,
            sharpe_ratio,
        )
        return result

    # ----------------------------------------------------------------
    # Equal Weight Portfolio
    # ----------------------------------------------------------------
    def equal_weight_portfolio(self) -> Dict[str, object]:
        """
        Construct the naive 1/N equal-weight portfolio.

        Returns
        -------
        Dict[str, object]
            Dictionary with keys "weights", "expected_return",
            "volatility", and "sharpe_ratio" (see `evaluate_portfolio`).

        Raises
        ------
        InvalidWeightsError
            If the resulting equal weights somehow fail validation
            (should not occur under normal conditions).
        """
        logger.info("Constructing equal-weight (1/N) portfolio.")

        weights = np.full(shape=self.n_assets, fill_value=1.0 / self.n_assets)
        result = self.evaluate_portfolio(weights)

        logger.info("Equal-weight portfolio constructed successfully.")
        return result

    # ----------------------------------------------------------------
    # Minimum Variance Portfolio
    # ----------------------------------------------------------------
    def minimum_variance_portfolio(
        self,
        initial_weights: Optional[np.ndarray] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-9,
    ) -> Dict[str, object]:
        """
        Solve for the long-only, fully-invested minimum-variance portfolio.

        Solves:
            minimize    w' * Cov * w
            subject to  sum(w) == 1
                        0 <= w_i <= 1  for all i

        using Sequential Least Squares Programming (SLSQP).

        Parameters
        ----------
        initial_weights : Optional[np.ndarray], optional
            Starting point for the optimizer, shape (n_assets,). Defaults
            to the equal-weight portfolio if not provided.
        max_iterations : int, optional
            Maximum number of SLSQP iterations, by default 1000.
        tolerance : float, optional
            Optimizer convergence tolerance, by default 1e-9.

        Returns
        -------
        Dict[str, object]
            Dictionary with keys "weights", "expected_return",
            "volatility", and "sharpe_ratio" (see `evaluate_portfolio`).

        Raises
        ------
        InvalidWeightsError
            If `initial_weights` is provided but invalid in shape.
        OptimizationError
            If the SLSQP solver fails to converge to a valid solution.
        """
        logger.info("Solving for minimum-variance portfolio.")

        if initial_weights is None:
            x0 = np.full(shape=self.n_assets, fill_value=1.0 / self.n_assets)
        else:
            x0 = np.asarray(initial_weights, dtype=float)
            if x0.shape != (self.n_assets,):
                raise InvalidWeightsError(
                    f"`initial_weights` must have shape ({self.n_assets},), "
                    f"got {x0.shape}."
                )

        cov_values = self.cov_matrix.values

        def _portfolio_variance(w: np.ndarray) -> float:
            return float(np.dot(w, np.dot(cov_values, w)))

        def _portfolio_variance_grad(w: np.ndarray) -> np.ndarray:
            return 2.0 * np.dot(cov_values, w)

        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)},
        )
        bounds = tuple((0.0, 1.0) for _ in range(self.n_assets))

        try:
            opt_result: OptimizeResult = minimize(
                fun=_portfolio_variance,
                x0=x0,
                jac=_portfolio_variance_grad,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": max_iterations, "ftol": tolerance, "disp": False},
            )
        except Exception as exc:  # noqa: BLE001 - re-raise as domain exception
            logger.error("SLSQP solver raised an exception: %s", exc)
            raise OptimizationError(
                f"Minimum-variance optimization failed due to a solver error: {exc}"
            ) from exc

        if not opt_result.success:
            logger.error(
                "Minimum-variance optimization did not converge: %s",
                opt_result.message,
            )
            raise OptimizationError(
                f"Minimum-variance optimization failed to converge: "
                f"{opt_result.message}"
            )

        optimal_weights = np.asarray(opt_result.x, dtype=float)

        # Clip tiny negative weights from floating point noise, then
        # renormalize so the sum-to-one constraint holds exactly.
        optimal_weights = np.clip(optimal_weights, 0.0, 1.0)
        weight_sum = optimal_weights.sum()
        if weight_sum <= 0:
            raise OptimizationError(
                "Minimum-variance optimization produced a degenerate "
                "(all-zero) weight vector."
            )
        optimal_weights = optimal_weights / weight_sum

        try:
            self._validate_weights(optimal_weights)
        except InvalidWeightsError as exc:
            logger.error("Optimizer produced invalid weights: %s", exc)
            raise OptimizationError(
                f"Minimum-variance optimization produced invalid weights: {exc}"
            ) from exc

        result = self.evaluate_portfolio(optimal_weights)

        logger.info(
            "Minimum-variance portfolio converged in %d iterations.",
            opt_result.nit,
        )
        return result

    # ----------------------------------------------------------------
    # Optimization Utility Functions
    # ----------------------------------------------------------------
    def _portfolio_return(self, weights: np.ndarray) -> float:
        """
        Compute the annualized expected return for a given weight vector.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights, shape (n_assets,).

        Returns
        -------
        float
            Annualized expected portfolio return.
        """
        return float(np.dot(weights, self.expected_returns.values))

    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Compute the annualized volatility (standard deviation) for a
        given weight vector.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights, shape (n_assets,).

        Returns
        -------
        float
            Annualized portfolio volatility. Clipped at 0.0 to guard
            against negative variance arising from numerical noise.
        """
        variance = float(np.dot(weights, np.dot(self.cov_matrix.values, weights)))
        if variance < 0:
            variance = 0.0
        return float(np.sqrt(variance))

    def _negative_sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Compute the negative Sharpe ratio for a given weight vector.

        Used as the objective function for the maximum-Sharpe optimizer,
        since `scipy.optimize.minimize` minimizes by convention and we
        want to maximize the Sharpe ratio.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights, shape (n_assets,).

        Returns
        -------
        float
            The negative Sharpe ratio. Returns 0.0 (a "neutral" value
            for the minimizer) if volatility is zero, to avoid a
            division-by-zero error during line search.
        """
        port_return = self._portfolio_return(weights)
        port_vol = self._portfolio_volatility(weights)

        if port_vol == 0:
            return 0.0

        return -((port_return - self.risk_free_rate) / port_vol)

    @staticmethod
    def _default_bounds(n_assets: int) -> tuple:
        """
        Build long-only weight bounds for `n_assets`.

        Parameters
        ----------
        n_assets : int
            Number of assets in the optimization universe.

        Returns
        -------
        tuple
            Tuple of (0.0, 1.0) bound pairs, one per asset.
        """
        return tuple((0.0, 1.0) for _ in range(n_assets))

    @staticmethod
    def _sum_to_one_constraint() -> Dict[str, object]:
        """
        Build the SLSQP equality constraint enforcing full investment
        (weights sum to 1).

        Returns
        -------
        Dict[str, object]
            A scipy.optimize constraint dictionary of type "eq".
        """
        return {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
            "jac": lambda w: np.ones_like(w),
        }

    # ----------------------------------------------------------------
    # Maximum Sharpe Ratio Portfolio
    # ----------------------------------------------------------------
    def maximum_sharpe_portfolio(
        self,
        initial_weights: Optional[np.ndarray] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-9,
    ) -> Dict[str, object]:
        """
        Solve for the long-only, fully-invested portfolio that maximizes
        the Sharpe ratio.

        Solves:
            maximize    (w' * mu - r_f) / sqrt(w' * Cov * w)
            subject to  sum(w) == 1
                        0 <= w_i <= 1  for all i

        Implemented as the minimization of the negative Sharpe ratio
        using Sequential Least Squares Programming (SLSQP).

        Parameters
        ----------
        initial_weights : Optional[np.ndarray], optional
            Starting point for the optimizer, shape (n_assets,). Defaults
            to the equal-weight portfolio if not provided.
        max_iterations : int, optional
            Maximum number of SLSQP iterations, by default 1000.
        tolerance : float, optional
            Optimizer convergence tolerance, by default 1e-9.

        Returns
        -------
        Dict[str, object]
            Dictionary with keys "weights", "expected_return",
            "volatility", and "sharpe_ratio" (see `evaluate_portfolio`).

        Raises
        ------
        InvalidWeightsError
            If `initial_weights` is provided but invalid in shape.
        OptimizationError
            If the SLSQP solver fails to converge to a valid solution.
        """
        logger.info("Solving for maximum-Sharpe-ratio portfolio.")

        if initial_weights is None:
            x0 = np.full(shape=self.n_assets, fill_value=1.0 / self.n_assets)
        else:
            x0 = np.asarray(initial_weights, dtype=float)
            if x0.shape != (self.n_assets,):
                raise InvalidWeightsError(
                    f"`initial_weights` must have shape ({self.n_assets},), "
                    f"got {x0.shape}."
                )

        constraints = (self._sum_to_one_constraint(),)
        bounds = self._default_bounds(self.n_assets)

        try:
            opt_result: OptimizeResult = minimize(
                fun=self._negative_sharpe_ratio,
                x0=x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": max_iterations, "ftol": tolerance, "disp": False},
            )
        except Exception as exc:  # noqa: BLE001 - re-raise as domain exception
            logger.error("SLSQP solver raised an exception: %s", exc)
            raise OptimizationError(
                f"Maximum-Sharpe optimization failed due to a solver error: {exc}"
            ) from exc

        if not opt_result.success:
            logger.error(
                "Maximum-Sharpe optimization did not converge: %s",
                opt_result.message,
            )
            raise OptimizationError(
                f"Maximum-Sharpe optimization failed to converge: "
                f"{opt_result.message}"
            )

        optimal_weights = np.asarray(opt_result.x, dtype=float)

        # Clip tiny negative weights from floating point noise, then
        # renormalize so the sum-to-one constraint holds exactly.
        optimal_weights = np.clip(optimal_weights, 0.0, 1.0)
        weight_sum = optimal_weights.sum()
        if weight_sum <= 0:
            raise OptimizationError(
                "Maximum-Sharpe optimization produced a degenerate "
                "(all-zero) weight vector."
            )
        optimal_weights = optimal_weights / weight_sum

        try:
            self._validate_weights(optimal_weights)
        except InvalidWeightsError as exc:
            logger.error("Optimizer produced invalid weights: %s", exc)
            raise OptimizationError(
                f"Maximum-Sharpe optimization produced invalid weights: {exc}"
            ) from exc

        result = self.evaluate_portfolio(optimal_weights)

        logger.info(
            "Maximum-Sharpe portfolio converged in %d iterations "
            "(Sharpe ratio = %.4f).",
            opt_result.nit,
            result["sharpe_ratio"],
        )
        return result

    # ----------------------------------------------------------------
    # Efficient Frontier Generation
    # ----------------------------------------------------------------
    def efficient_frontier(
        self,
        n_portfolios: int = 50,
        max_iterations: int = 1000,
        tolerance: float = 1e-9,
    ) -> pd.DataFrame:
        """
        Generate the long-only efficient frontier by solving a series of
        minimum-variance optimizations across a range of target returns.

        For each target return, solves:
            minimize    w' * Cov * w
            subject to  sum(w) == 1
                        w' * mu == target_return
                        0 <= w_i <= 1  for all i

        Parameters
        ----------
        n_portfolios : int, optional
            Number of points to sample along the frontier, by default 50.
        max_iterations : int, optional
            Maximum number of SLSQP iterations per sub-problem, by
            default 1000.
        tolerance : float, optional
            Optimizer convergence tolerance, by default 1e-9.

        Returns
        -------
        pd.DataFrame
            One row per frontier point (sorted by ascending target
            return), with columns:
                - "expected_return" (float)
                - "volatility" (float)
                - "sharpe_ratio" (float)
                - one column per asset, named "weight_<asset>", giving
                  that asset's allocation at that frontier point.
            Target returns for which the optimizer fails to converge are
            skipped (with a logged warning) rather than raising, so that
            a single infeasible target does not abort the entire frontier.

        Raises
        ------
        InvalidDataError
            If `n_portfolios` is not a positive integer.
        OptimizationError
            If every target return fails to converge, leaving zero
            valid frontier points.
        """
        logger.info("Generating efficient frontier with %d points.", n_portfolios)

        if not isinstance(n_portfolios, (int, np.integer)) or n_portfolios <= 0:
            raise InvalidDataError(
                f"`n_portfolios` must be a positive integer, got {n_portfolios}."
            )

        min_return = float(self.expected_returns.min())
        max_return = float(self.expected_returns.max())
        target_returns = np.linspace(min_return, max_return, n_portfolios)

        cov_values = self.cov_matrix.values
        mu_values = self.expected_returns.values
        x0 = np.full(shape=self.n_assets, fill_value=1.0 / self.n_assets)
        bounds = self._default_bounds(self.n_assets)

        def _portfolio_variance(w: np.ndarray) -> float:
            return float(np.dot(w, np.dot(cov_values, w)))

        def _portfolio_variance_grad(w: np.ndarray) -> np.ndarray:
            return 2.0 * np.dot(cov_values, w)

        frontier_records: list = []

        for target_return in target_returns:
            constraints = (
                self._sum_to_one_constraint(),
                {
                    "type": "eq",
                    "fun": lambda w, tr=target_return: float(np.dot(w, mu_values)) - tr,
                    "jac": lambda w, tr=target_return: mu_values,
                },
            )

            try:
                opt_result: OptimizeResult = minimize(
                    fun=_portfolio_variance,
                    x0=x0,
                    jac=_portfolio_variance_grad,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        "maxiter": max_iterations,
                        "ftol": tolerance,
                        "disp": False,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Solver error at target return %.4f%%; skipping point: %s",
                    target_return * 100,
                    exc,
                )
                continue

            if not opt_result.success:
                logger.warning(
                    "Frontier point at target return %.4f%% did not converge; "
                    "skipping: %s",
                    target_return * 100,
                    opt_result.message,
                )
                continue

            weights = np.clip(np.asarray(opt_result.x, dtype=float), 0.0, 1.0)
            weight_sum = weights.sum()
            if weight_sum <= 0:
                logger.warning(
                    "Frontier point at target return %.4f%% produced degenerate "
                    "weights; skipping.",
                    target_return * 100,
                )
                continue
            weights = weights / weight_sum

            try:
                self._validate_weights(weights)
            except InvalidWeightsError as exc:
                logger.warning(
                    "Frontier point at target return %.4f%% produced invalid "
                    "weights; skipping: %s",
                    target_return * 100,
                    exc,
                )
                continue

            point = self.evaluate_portfolio(weights)
            record = {
                "expected_return": point["expected_return"],
                "volatility": point["volatility"],
                "sharpe_ratio": point["sharpe_ratio"],
            }
            for asset, w in point["weights"].items():
                record[f"weight_{asset}"] = w
            frontier_records.append(record)

            # Warm-start the next sub-problem from this solution for
            # faster, more stable convergence along the frontier.
            x0 = weights

        if not frontier_records:
            logger.error("No frontier points converged successfully.")
            raise OptimizationError(
                "Efficient frontier generation failed: no target return "
                "produced a converged solution."
            )

        frontier_df = pd.DataFrame(frontier_records).sort_values(
            "expected_return"
        ).reset_index(drop=True)

        logger.info(
            "Efficient frontier generated with %d/%d valid points.",
            len(frontier_df),
            n_portfolios,
        )
        return frontier_df


# --------------------------------------------------------------------------
# Demonstration
# --------------------------------------------------------------------------
def main() -> None:
    """
    Demonstrate the PortfolioOptimizer end-to-end on simulated asset
    returns.

    Generates a synthetic 5-asset returns history, then constructs and
    prints:
        - the equal-weight portfolio
        - the minimum-variance portfolio
        - the maximum-Sharpe-ratio portfolio
        - a sampled efficient frontier

    This function is intended as a runnable, illustrative entry point
    and is not part of the public optimization API.

    Raises
    ------
    PortfolioOptimizerError
        Propagates any unhandled error raised during construction or
        optimization, after logging it for diagnostic purposes.
    """
    try:
        rng = np.random.default_rng(seed=42)

        dates = pd.date_range(start="2023-01-01", periods=500, freq="B")
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TLT"]

        mean_daily_returns = np.array([0.0007, 0.0006, 0.0005, 0.0008, 0.0002])
        daily_vol = np.array([0.018, 0.016, 0.017, 0.022, 0.006])
        correlation = np.array(
            [
                [1.00, 0.70, 0.65, 0.60, -0.10],
                [0.70, 1.00, 0.68, 0.58, -0.05],
                [0.65, 0.68, 1.00, 0.55, -0.05],
                [0.60, 0.58, 0.55, 1.00, -0.08],
                [-0.10, -0.05, -0.05, -0.08, 1.00],
            ]
        )
        cov = np.outer(daily_vol, daily_vol) * correlation

        simulated_returns = rng.multivariate_normal(
            mean=mean_daily_returns, cov=cov, size=len(dates)
        )
        sample_returns = pd.DataFrame(
            simulated_returns, index=dates, columns=tickers
        )

        optimizer = PortfolioOptimizer(
            returns=sample_returns, risk_free_rate=0.02, periods_per_year=252
        )

        ew = optimizer.equal_weight_portfolio()
        print("\n=== Equal-Weight Portfolio ===")
        print(ew["weights"])
        print(f"Expected Return: {ew['expected_return']:.4%}")
        print(f"Volatility:      {ew['volatility']:.4%}")
        print(f"Sharpe Ratio:    {ew['sharpe_ratio']:.4f}")

        mvp = optimizer.minimum_variance_portfolio()
        print("\n=== Minimum-Variance Portfolio ===")
        print(mvp["weights"])
        print(f"Expected Return: {mvp['expected_return']:.4%}")
        print(f"Volatility:      {mvp['volatility']:.4%}")
        print(f"Sharpe Ratio:    {mvp['sharpe_ratio']:.4f}")

        msr = optimizer.maximum_sharpe_portfolio()
        print("\n=== Maximum-Sharpe-Ratio Portfolio ===")
        print(msr["weights"])
        print(f"Expected Return: {msr['expected_return']:.4%}")
        print(f"Volatility:      {msr['volatility']:.4%}")
        print(f"Sharpe Ratio:    {msr['sharpe_ratio']:.4f}")

        frontier = optimizer.efficient_frontier(n_portfolios=20)
        print("\n=== Efficient Frontier (sample) ===")
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(frontier[["expected_return", "volatility", "sharpe_ratio"]])

    except PortfolioOptimizerError as exc:
        logger.error("Demonstration failed: %s", exc)
        raise


if __name__ == "__main__":
    main()

