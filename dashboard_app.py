"""
dashboard_app.py
================
QuantLab — Institutional Portfolio Research & Risk Analytics Platform
---------------------------------------------------------------------
Streamlit dashboard providing three core research surfaces:

    • Portfolio Builder   — define tickers, weights, and view allocation
    • Risk Analytics      — compute and visualise institutional risk metrics
    • Portfolio Optimizer — compare Equal-Weight, Min-Variance, Max-Sharpe

Run with:
    streamlit run dashboard_app.py

Dependencies:
    streamlit, pandas, numpy, plotly, scipy
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize
from regime_engine import (
    MarketRegime,
    RegimeResult,
    detect_market_regime,
    regime_color,
    regime_description,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
# Palette: deep navy field, cool slate surface, electric cobalt accent,
# gold signal, soft white text — Bloomberg terminal meets Figma-era precision.
NAVY    = "#0A0F1E"
SLATE   = "#111827"
PANEL   = "#1A2235"
BORDER  = "#253047"
COBALT  = "#2563EB"
COBALT2 = "#3B82F6"
GOLD    = "#F59E0B"
GREEN   = "#10B981"
RED     = "#EF4444"
MUTED   = "#64748B"
TEXT    = "#F1F5F9"
SUBTEXT = "#94A3B8"

PLOTLY_TEMPLATE = "plotly_dark"
CHART_BG        = "rgba(17,24,37,0)"   # transparent → inherits panel


# ===========================================================================
# PAGE CONFIG  (must be first Streamlit call)
# ===========================================================================
st.set_page_config(
    page_title="QuantLab",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# GLOBAL STYLES
# ===========================================================================
def inject_styles() -> None:
    """Inject custom CSS that establishes the QuantLab visual identity."""
    st.markdown(
        f"""
        <style>
        /* ── Base ── */
        html, body, [data-testid="stAppViewContainer"] {{
            background-color: {NAVY};
            color: {TEXT};
            font-family: 'Inter', 'IBM Plex Sans', system-ui, sans-serif;
        }}
        [data-testid="stSidebar"] {{
            background-color: {SLATE} !important;
            border-right: 1px solid {BORDER};
        }}
        [data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

        /* ── Sidebar nav buttons ── */
        .nav-btn {{
            display: flex; align-items: center; gap: 10px;
            width: 100%; padding: 10px 14px; margin: 3px 0;
            background: transparent; border: none;
            border-radius: 8px; cursor: pointer;
            color: {SUBTEXT}; font-size: 0.88rem; font-weight: 500;
            letter-spacing: 0.01em; transition: all 0.15s ease;
            text-align: left;
        }}
        .nav-btn:hover  {{ background: {PANEL}; color: {TEXT}; }}
        .nav-btn.active {{ background: {COBALT}20; color: {COBALT2};
                           border-left: 3px solid {COBALT2}; }}

        /* ── Metric cards ── */
        .ql-card {{
            background: {PANEL}; border: 1px solid {BORDER};
            border-radius: 12px; padding: 20px 24px; margin-bottom: 8px;
        }}
        .ql-card-label {{
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
            text-transform: uppercase; color: {SUBTEXT}; margin-bottom: 4px;
        }}
        .ql-card-value {{
            font-size: 1.65rem; font-weight: 700; color: {TEXT};
            letter-spacing: -0.02em; line-height: 1.1;
        }}
        .ql-card-sub {{
            font-size: 0.78rem; color: {SUBTEXT}; margin-top: 4px;
        }}
        .ql-card-value.positive {{ color: {GREEN}; }}
        .ql-card-value.negative {{ color: {RED}; }}
        .ql-card-value.neutral  {{ color: {GOLD}; }}

        /* ── Section headers ── */
        .ql-section-header {{
            font-size: 0.70rem; font-weight: 700; letter-spacing: 0.12em;
            text-transform: uppercase; color: {MUTED};
            border-bottom: 1px solid {BORDER}; padding-bottom: 8px;
            margin: 28px 0 16px;
        }}

        /* ── Page title ── */
        .ql-page-title {{
            font-size: 1.55rem; font-weight: 700; color: {TEXT};
            letter-spacing: -0.02em; margin-bottom: 2px;
        }}
        .ql-page-sub {{
            font-size: 0.85rem; color: {SUBTEXT}; margin-bottom: 24px;
        }}

        /* ── Alert boxes ── */
        .ql-alert-error {{
            background: {RED}18; border: 1px solid {RED}55;
            border-radius: 8px; padding: 12px 16px;
            color: #FCA5A5; font-size: 0.84rem;
        }}
        .ql-alert-info {{
            background: {COBALT}18; border: 1px solid {COBALT}55;
            border-radius: 8px; padding: 12px 16px;
            color: #93C5FD; font-size: 0.84rem;
        }}
        .ql-alert-success {{
            background: {GREEN}18; border: 1px solid {GREEN}55;
            border-radius: 8px; padding: 12px 16px;
            color: #6EE7B7; font-size: 0.84rem;
        }}

        /* ── Dataframe overrides ── */
        [data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}

        /* ── Divider ── */
        .ql-divider {{
            border: none; border-top: 1px solid {BORDER};
            margin: 20px 0;
        }}

        /* ── Weight badge ── */
        .weight-badge {{
            display: inline-block; background: {COBALT}28;
            color: {COBALT2}; font-size: 0.78rem; font-weight: 600;
            padding: 2px 10px; border-radius: 20px;
        }}

        /* hide default streamlit chrome ── */
        #MainMenu, footer, header {{ visibility: hidden; }}
        .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# HELPER UI COMPONENTS
# ===========================================================================

def metric_card(
    label: str,
    value: str,
    sub: str = "",
    sentiment: str = "neutral",   # "positive" | "negative" | "neutral"
) -> None:
    """Render a styled metric card."""
    st.markdown(
        f"""
        <div class="ql-card">
            <div class="ql-card-label">{label}</div>
            <div class="ql-card-value {sentiment}">{value}</div>
            {"<div class='ql-card-sub'>" + sub + "</div>" if sub else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str) -> None:
    st.markdown(f'<div class="ql-section-header">{title}</div>', unsafe_allow_html=True)


def alert(message: str, kind: str = "info") -> None:
    """kind: 'info' | 'error' | 'success'"""
    st.markdown(
        f'<div class="ql-alert-{kind}">{message}</div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="ql-page-title">{title}</div>'
        f'<div class="ql-page-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# SESSION STATE BOOTSTRAP
# ===========================================================================

def init_session_state() -> None:
    """Initialise all session-state keys with safe defaults."""
    defaults: dict = {
        "page":          "Portfolio Builder",
        "tickers":       ["AAPL", "MSFT", "GOOGL", "AMZN", "BRK-B"],
        "weights":       [0.25, 0.25, 0.20, 0.20, 0.10],
        "returns_df":    None,   # pd.DataFrame of daily returns
        "prices_df":     None,   # pd.DataFrame of daily prices
        "portfolio_validated": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ===========================================================================
# DATA SIMULATION  (stands in for market_data.py feed)
# ===========================================================================

def simulate_price_history(
    tickers: list[str],
    n_days: int = 756,        # ~3 years
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily price histories via correlated GBM.

    In production this function is replaced by a call to
    ``market_data.fetch_prices(tickers, start, end)``.

    Parameters
    ----------
    tickers : list[str]
    n_days  : int
    seed    : int

    Returns
    -------
    pd.DataFrame, shape (n_days, n_tickers)
        Daily adjusted-close prices indexed by business-day dates.
    """
    rng = np.random.default_rng(seed)
    n   = len(tickers)

    mu    = rng.uniform(0.06, 0.18, n)
    sigma = rng.uniform(0.15, 0.35, n)

    # Random positive-definite correlation matrix
    A    = rng.standard_normal((n, n))
    corr = A @ A.T
    corr /= np.sqrt(np.outer(np.diag(corr), np.diag(corr)))
    np.fill_diagonal(corr, 1.0)

    cov_daily = np.diag(sigma / np.sqrt(252)) @ corr @ np.diag(sigma / np.sqrt(252))
    L = np.linalg.cholesky(cov_daily + 1e-10 * np.eye(n))

    dt       = 1.0 / 252
    drift    = (mu - 0.5 * sigma ** 2) * dt
    Z        = rng.standard_normal((n_days, n))
    shocks   = Z @ L.T
    log_rets = drift + shocks
    prices   = 100.0 * np.exp(np.cumsum(log_rets, axis=0))

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=tickers)


@st.cache_data(ttl=300, show_spinner=False)
def get_price_data(tickers: list[str]) -> pd.DataFrame:
    """Cached wrapper around price simulation / market data fetch."""
    logger.info("Fetching price data for: %s", tickers)
    return simulate_price_history(tickers)


# ===========================================================================
# ANALYTICS LAYER  (mirrors analytics_engine.py interface)
# ===========================================================================

def compute_portfolio_returns(
    prices: pd.DataFrame,
    weights: list[float],
) -> pd.Series:
    """Compute daily portfolio returns from price DataFrame and weights."""
    daily_returns = prices.pct_change().dropna()
    w = np.array(weights, dtype=float)
    w /= w.sum()
    return daily_returns @ w


def annualised_return(returns: pd.Series) -> float:
    return float(returns.mean() * 252)


def annualised_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.04) -> float:
    excess = returns.mean() - risk_free / 252
    std    = returns.std()
    return float((excess / std) * np.sqrt(252)) if std > 0 else 0.0


def sortino_ratio(returns: pd.Series, risk_free: float = 0.04) -> float:
    excess    = returns - risk_free / 252
    downside  = returns[returns < 0].std()
    return float((excess.mean() / downside) * np.sqrt(252)) if downside > 0 else 0.0


def max_drawdown(returns: pd.Series) -> float:
    cum   = (1 + returns).cumprod()
    peak  = cum.cummax()
    dd    = (cum - peak) / peak
    return float(dd.min())


def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical simulation VaR (positive = loss)."""
    return float(-np.percentile(returns, (1 - confidence) * 100))


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical simulation CVaR / Expected Shortfall (positive = loss)."""
    var    = compute_var(returns, confidence)
    tail   = returns[returns <= -var]
    return float(-tail.mean()) if len(tail) > 0 else var


# ===========================================================================
# OPTIMISATION LAYER  (mirrors portfolio_optimizer.py interface)
# ===========================================================================

def equal_weight_portfolio(n: int) -> np.ndarray:
    return np.ones(n) / n


def min_variance_portfolio(cov: np.ndarray) -> np.ndarray:
    """Solve the minimum-variance portfolio (long-only, fully invested)."""
    n = cov.shape[0]
    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds      = [(0.0, 1.0)] * n
    result      = minimize(
        objective,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        logger.warning("Min-variance optimisation did not converge: %s", result.message)
    w = np.clip(result.x, 0.0, 1.0)
    return w / w.sum()


def max_sharpe_portfolio(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    risk_free: float = 0.04,
) -> np.ndarray:
    """Maximise the Sharpe ratio (long-only, fully invested)."""
    n = len(mean_returns)

    def neg_sharpe(w: np.ndarray) -> float:
        r   = float(w @ mean_returns) * 252
        vol = float(np.sqrt(w @ cov @ w)) * np.sqrt(252)
        return -(r - risk_free) / vol if vol > 0 else 0.0

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds      = [(0.0, 1.0)] * n
    result      = minimize(
        neg_sharpe,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        logger.warning("Max-Sharpe optimisation did not converge: %s", result.message)
    w = np.clip(result.x, 0.0, 1.0)
    return w / w.sum()


# ===========================================================================
# CHART BUILDERS
# ===========================================================================

def _base_layout(title: str = "") -> dict:
    """Shared Plotly layout tokens."""
    return dict(
        title=dict(text=title, font=dict(size=13, color=SUBTEXT), x=0),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=11),
        margin=dict(l=0, r=0, t=36 if title else 12, b=0),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, showgrid=True),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font_size=12),
    )


def chart_donut(tickers: list[str], weights: list[float]) -> go.Figure:
    """Allocation donut chart."""
    colors = [
        COBALT, GOLD, GREEN, "#8B5CF6", "#EC4899",
        "#14B8A6", "#F97316", "#06B6D4", "#84CC16", "#A78BFA",
    ]
    fig = go.Figure(go.Pie(
        labels=tickers,
        values=weights,
        hole=0.62,
        marker=dict(colors=colors[:len(tickers)], line=dict(color=NAVY, width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color=TEXT),
        hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(),
        showlegend=False,
        annotations=[dict(
            text="Allocation", x=0.5, y=0.5, font_size=13,
            font_color=SUBTEXT, showarrow=False,
        )],
    )
    return fig


def chart_cumulative_returns(
    portfolio_returns: pd.Series,
    prices: pd.DataFrame,
    tickers: list[str],
) -> go.Figure:
    """Cumulative return lines: portfolio vs individual assets."""
    fig = go.Figure()
    colors_asset = [BORDER] * len(tickers)

    asset_rets = prices.pct_change().dropna()
    for i, ticker in enumerate(tickers):
        cum = (1 + asset_rets[ticker]).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values * 100,
            name=ticker, line=dict(width=1, color=colors_asset[i]),
            opacity=0.45,
            hovertemplate=f"<b>{ticker}</b>: %{{y:.1f}}%<extra></extra>",
        ))

    cum_port = (1 + portfolio_returns).cumprod() - 1
    fig.add_trace(go.Scatter(
        x=cum_port.index, y=cum_port.values * 100,
        name="Portfolio", line=dict(width=2.5, color=COBALT2),
        hovertemplate="<b>Portfolio</b>: %{y:.1f}}%<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout("Cumulative Return (%)"),
        yaxis_ticksuffix="%",
        hovermode="x unified",
    )
    return fig


def chart_drawdown(portfolio_returns: pd.Series) -> go.Figure:
    """Underwater / drawdown chart."""
    cum  = (1 + portfolio_returns).cumprod()
    peak = cum.cummax()
    dd   = ((cum - peak) / peak) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.15)",
        line=dict(color=RED, width=1.5),
        name="Drawdown",
        hovertemplate="%{y:.2f}%<extra>Drawdown</extra>",
    ))
    fig.update_layout(
        **_base_layout("Portfolio Drawdown (%)"),
        yaxis_ticksuffix="%",
    )
    return fig


def chart_return_distribution(portfolio_returns: pd.Series) -> go.Figure:
    """Return histogram with VaR overlays."""
    r   = portfolio_returns.values * 100
    v95 = -compute_var(portfolio_returns, 0.95) * 100
    v99 = -compute_var(portfolio_returns, 0.99) * 100

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r, nbinsx=80,
        marker_color=COBALT, opacity=0.75,
        name="Daily Returns",
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))
    for val, label, color in [
        (v95, "VaR 95%", GOLD),
        (v99, "VaR 99%", RED),
    ]:
        fig.add_vline(
            x=val, line_width=1.5, line_dash="dash", line_color=color,
            annotation_text=label, annotation_position="top left",
            annotation_font=dict(color=color, size=10),
        )
    fig.update_layout(
        **_base_layout("Daily Return Distribution"),
        xaxis_ticksuffix="%",
        bargap=0.02,
    )
    return fig


def chart_rolling_volatility(portfolio_returns: pd.Series, window: int = 21) -> go.Figure:
    """21-day rolling annualised volatility."""
    roll_vol = portfolio_returns.rolling(window).std() * np.sqrt(252) * 100
    fig = go.Figure(go.Scatter(
        x=roll_vol.index, y=roll_vol.values,
        line=dict(color=GOLD, width=1.8),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.12)",
        name=f"{window}d Rolling Vol",
        hovertemplate="%{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(f"{window}-Day Rolling Volatility (Annualised %)"),
        yaxis_ticksuffix="%",
    )
    return fig


def chart_correlation_heatmap(prices: pd.DataFrame) -> go.Figure:
    """Asset correlation heatmap."""
    corr = prices.pct_change().dropna().corr()
    fig  = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[
            [0.0, RED], [0.5, PANEL], [1.0, COBALT2],
        ],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="<b>%{y} × %{x}</b><br>ρ = %{z:.3f}<extra></extra>",
        showscale=True,
        colorbar=dict(thickness=12, len=0.9, tickfont=dict(size=10)),
    ))
    fig.update_layout(
    **_base_layout("Pairwise Correlation Matrix")
    )

    fig.update_xaxes(side="bottom")
    return fig


def chart_optimised_weights(
    tickers: list[str],
    w_eq: np.ndarray,
    w_mv: np.ndarray,
    w_ms: np.ndarray,
) -> go.Figure:
    """Grouped bar chart comparing three optimised weight vectors."""
    categories = ["Equal Weight", "Min Variance", "Max Sharpe"]
    color_map  = [MUTED, COBALT, GOLD]
    fig = go.Figure()
    for weights, name, color in zip([w_eq, w_mv, w_ms], categories, color_map):
        fig.add_trace(go.Bar(
            name=name,
            x=tickers,
            y=weights * 100,
            marker_color=color,
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        **_base_layout("Portfolio Weights by Strategy (%)"),
        barmode="group",
        yaxis_ticksuffix="%",
        bargap=0.18,
        bargroupgap=0.06,
    )
    return fig


def chart_efficient_frontier(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    tickers: list[str],
    w_mv: np.ndarray,
    w_ms: np.ndarray,
    risk_free: float = 0.04,
    n_portfolios: int = 2500,
    seed: int = 77,
) -> go.Figure:
    """Monte Carlo efficient frontier scatter with overlay portfolios."""
    rng = np.random.default_rng(seed)
    n   = len(tickers)

    vols, rets, sharpes = [], [], []
    for _ in range(n_portfolios):
        w   = rng.dirichlet(np.ones(n))
        r   = float(w @ mean_returns) * 252
        vol = float(np.sqrt(w @ cov @ w)) * np.sqrt(252)
        sr  = (r - risk_free) / vol if vol > 0 else 0
        rets.append(r * 100)
        vols.append(vol * 100)
        sharpes.append(sr)

    fig = go.Figure()

    # Random portfolios
    fig.add_trace(go.Scatter(
        x=vols, y=rets,
        mode="markers",
        marker=dict(
            color=sharpes, colorscale="Blues",
            size=3, opacity=0.5,
            colorbar=dict(title="Sharpe", thickness=10, len=0.7),
        ),
        name="Random Portfolios",
        hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<extra></extra>",
    ))

    # Annotated special portfolios
    for w_opt, label, color, symbol in [
        (w_mv, "Min Variance", GREEN,  "diamond"),
        (w_ms, "Max Sharpe",   GOLD,   "star"),
    ]:
        r   = float(w_opt @ mean_returns) * 252 * 100
        vol = float(np.sqrt(w_opt @ cov @ w_opt)) * np.sqrt(252) * 100
        fig.add_trace(go.Scatter(
            x=[vol], y=[r],
            mode="markers+text",
            marker=dict(color=color, size=14, symbol=symbol,
                        line=dict(color=NAVY, width=1.5)),
            text=[label], textposition="top center",
            textfont=dict(size=10, color=color),
            name=label,
            hovertemplate=f"<b>{label}</b><br>Vol: {vol:.1f}%<br>Ret: {r:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout("Efficient Frontier (Monte Carlo Simulation)"),
        xaxis_title="Annualised Volatility (%)",
        yaxis_title="Annualised Return (%)",
        showlegend=True,
    )

    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(0,0,0,0)"
        )
    )

    return fig


# ===========================================================================
# INPUT VALIDATION
# ===========================================================================

def validate_portfolio_inputs(
    raw_tickers: str,
    raw_weights: str,
) -> tuple[bool, Optional[list[str]], Optional[list[float]], str]:
    """
    Parse and validate raw text inputs from the Portfolio Builder form.

    Returns
    -------
    (ok, tickers, weights, error_message)
    """
    tickers_raw = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    weights_raw = [w.strip() for w in raw_weights.split(",") if w.strip()]

    if not tickers_raw:
        return False, None, None, "Enter at least one ticker symbol."

    if len(tickers_raw) != len(weights_raw):
        return (
            False, None, None,
            f"Ticker count ({len(tickers_raw)}) must match weight count "
            f"({len(weights_raw)}).",
        )

    try:
        weights = [float(w) for w in weights_raw]
    except ValueError:
        return False, None, None, "Weights must be numeric values (e.g. 0.25, 25)."

    if any(w < 0 for w in weights):
        return False, None, None, "Weights must be non-negative."

    if sum(weights) == 0:
        return False, None, None, "At least one weight must be non-zero."

    # Normalise to sum-to-one
    total   = sum(weights)
    weights = [w / total for w in weights]

    if len(tickers_raw) > 15:
        return False, None, None, "Maximum 15 assets supported per portfolio."

    logger.info("Portfolio validated: %s", dict(zip(tickers_raw, weights)))
    return True, tickers_raw, weights, ""


# ===========================================================================
# SIDEBAR NAVIGATION
# ===========================================================================

def render_sidebar() -> None:
    """Render the QuantLab sidebar with logo, navigation, and portfolio state."""
    with st.sidebar:
        # Logo / wordmark
        st.markdown(
            f"""
            <div style="padding: 8px 0 24px;">
                <div style="font-size:1.35rem; font-weight:800;
                            letter-spacing:-0.03em; color:{TEXT};">
                    Quant<span style="color:{COBALT2};">Lab</span>
                </div>
                <div style="font-size:0.68rem; color:{MUTED};
                            letter-spacing:0.10em; text-transform:uppercase;
                            margin-top:2px;">
                    Portfolio Analytics Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="ql-section-header" style="margin-top:16px;">Simulation</div>',
            unsafe_allow_html=True,
        )

        simulation_pages = [
            ("📈", "Backtesting"),
            ("🎲", "Monte Carlo"),
            ("⚡", "Stress Testing"),
        ]

        for icon, page in simulation_pages:
            if st.button(
                f"{icon} {page}",
                key=f"nav_{page}",
                use_container_width=True,
            ):
                st.session_state.page = page
                st.rerun()

        st.markdown(
            f'<div class="ql-section-header" style="margin-top:16px;">Intelligence</div>',
            unsafe_allow_html=True,
        )

        intelligence_pages = [
            ("🧭", "Regime Intelligence"),
        ]

        for icon, page in intelligence_pages:
            if st.button(
                f"{icon} {page}",
                key=f"nav_{page}",
                use_container_width=True,
            ):
                st.session_state.page = page
                st.rerun()

        # Portfolio state summary
        if st.session_state.portfolio_validated:
            st.markdown(f'<hr class="ql-divider">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="ql-section-header">Active Portfolio</div>',
                unsafe_allow_html=True,
            )
            tickers = st.session_state.tickers
            weights = st.session_state.weights
            for t, w in zip(tickers, weights):
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between;
                                align-items:center; padding: 4px 0;
                                font-size:0.82rem; color:{SUBTEXT};">
                        <span>{t}</span>
                        <span class="weight-badge">{w:.1%}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Footer
        st.markdown(
            f"""
            <div style="position:absolute; bottom:24px; left:0; right:0;
                        padding: 0 16px;">
                <div style="font-size:0.65rem; color:{MUTED};
                            border-top:1px solid {BORDER}; padding-top:12px;">
                    QuantLab v1.0 · Research Use Only<br>
                    Not financial advice.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# PAGE 1 — PORTFOLIO BUILDER
# ===========================================================================

def render_portfolio_builder() -> None:
    """Portfolio Builder: define tickers, weights, and review allocation."""
    page_header(
        "Portfolio Builder",
        "Define your asset universe and target weights to begin analysis.",
    )

    # ── Input form ──────────────────────────────────────────────────────────
    section_header("Define Portfolio")

    col_form, col_gap, col_preview = st.columns([5, 0.4, 4])

    with col_form:
        default_tickers = ", ".join(st.session_state.tickers)
        default_weights = ", ".join(f"{w:.2f}" for w in st.session_state.weights)

        raw_tickers = st.text_area(
            "Tickers  (comma-separated)",
            value=default_tickers,
            height=80,
            placeholder="AAPL, MSFT, GOOGL, AMZN",
            help="Enter valid ticker symbols separated by commas.",
        )
        raw_weights = st.text_area(
            "Weights  (comma-separated, need not sum to 1)",
            value=default_weights,
            height=80,
            placeholder="0.25, 0.25, 0.25, 0.25",
            help="Weights are automatically normalised to sum to 1.",
        )

        col_btn, col_hint = st.columns([2, 3])
        with col_btn:
            build_clicked = st.button(
                "Build Portfolio →",
                type="primary",
                use_container_width=True,
            )
        with col_hint:
            st.markdown(
                f'<div style="color:{SUBTEXT}; font-size:0.78rem; '
                f'padding-top:10px;">Weights are normalised automatically.</div>',
                unsafe_allow_html=True,
            )

    if build_clicked:
        ok, tickers, weights, err = validate_portfolio_inputs(raw_tickers, raw_weights)
        if not ok:
            with col_form:
                alert(f"⚠ {err}", "error")
            st.session_state.portfolio_validated = False
        else:
            st.session_state.tickers  = tickers
            st.session_state.weights  = weights
            st.session_state.portfolio_validated = True
            # Clear cached data so new tickers reload
            st.session_state.returns_df = None
            st.session_state.prices_df  = None
            st.rerun()

    # ── Allocation preview ───────────────────────────────────────────────────
    tickers = st.session_state.tickers
    weights = st.session_state.weights

    with col_preview:
        section_header("Allocation")
        st.plotly_chart(
            chart_donut(tickers, weights),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── Weights table ────────────────────────────────────────────────────────
    section_header("Weight Table")
    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": [f"{w:.2%}" for w in weights],
        "Notional @ $1M": [f"${w * 1_000_000:,.0f}" for w in weights],
    })
    st.dataframe(
        weights_df,
        hide_index=True,
        use_container_width=True,
    )

    # ── Historical price data ────────────────────────────────────────────────
    section_header("Historical Prices  (3 Years)")

    with st.spinner("Loading price data …"):
        prices = get_price_data(tuple(tickers))

    st.session_state.prices_df  = prices
    port_rets = compute_portfolio_returns(prices, weights)
    st.session_state.returns_df = port_rets

    # Normalised price chart
    norm = prices / prices.iloc[0] * 100
    fig_prices = go.Figure()
    palette = [COBALT2, GOLD, GREEN, "#8B5CF6", "#EC4899",
               "#14B8A6", "#F97316", "#84CC16"]
    for i, col in enumerate(norm.columns):
        fig_prices.add_trace(go.Scatter(
            x=norm.index, y=norm[col],
            name=col,
            line=dict(width=1.6, color=palette[i % len(palette)]),
            hovertemplate=f"<b>{col}</b>: %{{y:.1f}}<extra></extra>",
        ))
    fig_prices.update_layout(
        **_base_layout("Indexed Price Performance (Base = 100)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_prices, use_container_width=True, config={"displayModeBar": False})

    # ── Correlation heatmap ──────────────────────────────────────────────────
    section_header("Asset Correlations")
    st.plotly_chart(
        chart_correlation_heatmap(prices),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    if st.session_state.portfolio_validated:
        alert("✓ Portfolio validated. Navigate to Risk Analytics or Optimisation.", "success")


# ===========================================================================
# PAGE 2 — RISK ANALYTICS
# ===========================================================================

def render_risk_analytics() -> None:
    """Risk Analytics: institutional risk metrics with Plotly visualisations."""
    page_header(
        "Risk Analytics",
        "Institutional risk metrics computed from simulated historical returns.",
    )

    tickers = st.session_state.tickers
    weights = st.session_state.weights

    # Load data
    with st.spinner("Computing risk metrics …"):
        if st.session_state.prices_df is None:
            prices = get_price_data(tuple(tickers))
            st.session_state.prices_df = prices
        else:
            prices = st.session_state.prices_df

        port_rets = compute_portfolio_returns(prices, weights)

    # ── Metric cards ─────────────────────────────────────────────────────────
    section_header("Key Risk Metrics")

    ann_ret  = annualised_return(port_rets)
    ann_vol  = annualised_volatility(port_rets)
    sr       = sharpe_ratio(port_rets)
    srt      = sortino_ratio(port_rets)
    mdd      = max_drawdown(port_rets)
    var_95   = compute_var(port_rets, 0.95)
    var_99   = compute_var(port_rets, 0.99)
    cvar_95  = compute_cvar(port_rets, 0.95)
    cvar_99  = compute_cvar(port_rets, 0.99)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Annualised Return",  f"{ann_ret:.2%}",
                    "Geometric mean × 252",
                    "positive" if ann_ret >= 0 else "negative")
        metric_card("Sharpe Ratio",        f"{sr:.3f}",
                    "Risk-free = 4%",
                    "positive" if sr >= 1 else "neutral" if sr >= 0 else "negative")
    with c2:
        metric_card("Annualised Vol",     f"{ann_vol:.2%}",
                    "Daily σ × √252", "neutral")
        metric_card("Sortino Ratio",       f"{srt:.3f}",
                    "Downside deviation",
                    "positive" if srt >= 1 else "neutral" if srt >= 0 else "negative")
    with c3:
        metric_card("VaR 95%",   f"{var_95:.2%}",
                    "1-day historical", "negative")
        metric_card("CVaR 95%",  f"{cvar_95:.2%}",
                    "Expected Shortfall", "negative")
    with c4:
        metric_card("VaR 99%",   f"{var_99:.2%}",
                    "1-day historical", "negative")
        metric_card("CVaR 99% / Max DD",
                    f"{cvar_99:.2%}  /  {mdd:.2%}",
                    "ES @ 99% · Peak-to-trough", "negative")

    # ── Risk metric table ────────────────────────────────────────────────────
    section_header("Full Risk Summary")
    risk_df = pd.DataFrame({
        "Metric": [
            "Annualised Return", "Annualised Volatility",
            "Sharpe Ratio (Rf=4%)", "Sortino Ratio (Rf=4%)",
            "Max Drawdown",
            "VaR 95% (1-day)", "VaR 99% (1-day)",
            "CVaR 95% (ES)", "CVaR 99% (ES)",
        ],
        "Value": [
            f"{ann_ret:.4%}", f"{ann_vol:.4%}",
            f"{sr:.4f}",      f"{srt:.4f}",
            f"{mdd:.4%}",
            f"{var_95:.4%}",  f"{var_99:.4%}",
            f"{cvar_95:.4%}", f"{cvar_99:.4%}",
        ],
        "Interpretation": [
            "Expected annual gain",
            "Return dispersion",
            ">1 is considered good",
            "Penalises downside only",
            "Largest peak-to-trough loss",
            "Loss exceeded 5% of days",
            "Loss exceeded 1% of days",
            "Avg loss in worst 5% of days",
            "Avg loss in worst 1% of days",
        ],
    })
    st.dataframe(risk_df, hide_index=True, use_container_width=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    section_header("Return Analysis")

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(
            chart_cumulative_returns(port_rets, prices, tickers),
            use_container_width=True, config={"displayModeBar": False},
        )
    with col_r:
        st.plotly_chart(
            chart_return_distribution(port_rets),
            use_container_width=True, config={"displayModeBar": False},
        )

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.plotly_chart(
            chart_drawdown(port_rets),
            use_container_width=True, config={"displayModeBar": False},
        )
    with col_r2:
        st.plotly_chart(
            chart_rolling_volatility(port_rets),
            use_container_width=True, config={"displayModeBar": False},
        )

    # ── Per-asset risk breakdown ─────────────────────────────────────────────
    section_header("Per-Asset Risk Breakdown")
    asset_rets = prices.pct_change().dropna()
    asset_rows = []
    for ticker in tickers:
        r = asset_rets[ticker]
        asset_rows.append({
            "Ticker":       ticker,
            "Ann. Return":  f"{annualised_return(r):.2%}",
            "Ann. Vol":     f"{annualised_volatility(r):.2%}",
            "Sharpe":       f"{sharpe_ratio(r):.3f}",
            "Max DD":       f"{max_drawdown(r):.2%}",
            "VaR 95%":      f"{compute_var(r, 0.95):.2%}",
            "CVaR 95%":     f"{compute_cvar(r, 0.95):.2%}",
        })
    st.dataframe(
        pd.DataFrame(asset_rows),
        hide_index=True,
        use_container_width=True,
    )


# ===========================================================================
# PAGE 3 — PORTFOLIO OPTIMIZATION
# ===========================================================================

def render_portfolio_optimization() -> None:
    """Portfolio Optimisation: Equal-Weight, Min-Variance, Max-Sharpe."""
    page_header(
        "Portfolio Optimisation",
        "Compare three allocation strategies on risk-adjusted performance.",
    )

    tickers = st.session_state.tickers
    weights = st.session_state.weights

    with st.spinner("Running optimisation …"):
        if st.session_state.prices_df is None:
            prices = get_price_data(tuple(tickers))
            st.session_state.prices_df = prices
        else:
            prices = st.session_state.prices_df

        daily_rets  = prices.pct_change().dropna()
        mean_daily  = daily_rets.mean().values
        cov_daily   = daily_rets.cov().values
        n           = len(tickers)

        w_eq = equal_weight_portfolio(n)
        w_mv = min_variance_portfolio(cov_daily)
        w_ms = max_sharpe_portfolio(mean_daily, cov_daily)

    # Helper: compute metrics for a weight vector
    def portfolio_metrics(w: np.ndarray) -> dict:
        r_series = daily_rets @ w
        return {
            "Ann. Return":   annualised_return(r_series),
            "Ann. Vol":      annualised_volatility(r_series),
            "Sharpe":        sharpe_ratio(r_series),
            "Sortino":       sortino_ratio(r_series),
            "Max DD":        max_drawdown(r_series),
            "VaR 95%":       compute_var(r_series, 0.95),
            "CVaR 95%":      compute_cvar(r_series, 0.95),
        }

    m_eq = portfolio_metrics(w_eq)
    m_mv = portfolio_metrics(w_mv)
    m_ms = portfolio_metrics(w_ms)

    # ── Strategy metric cards ────────────────────────────────────────────────
    section_header("Strategy Comparison")

    col_eq, col_mv, col_ms = st.columns(3)
    strategy_cols = [
        (col_eq, "Equal Weight",   m_eq, MUTED),
        (col_mv, "Min Variance",   m_mv, COBALT2),
        (col_ms, "Max Sharpe",     m_ms, GOLD),
    ]
    for col, name, m, color in strategy_cols:
        with col:
            st.markdown(
                f"""
                <div class="ql-card" style="border-color:{color}44;">
                    <div style="font-size:0.70rem; font-weight:700;
                                letter-spacing:0.10em; text-transform:uppercase;
                                color:{color}; margin-bottom:12px;">{name}</div>
                    <table style="width:100%; font-size:0.82rem;
                                  border-collapse:collapse;">
                        <tr><td style="color:{SUBTEXT}; padding:3px 0;">Return</td>
                            <td style="text-align:right; color:{GREEN}; font-weight:600;">
                                {m['Ann. Return']:.2%}</td></tr>
                        <tr><td style="color:{SUBTEXT}; padding:3px 0;">Volatility</td>
                            <td style="text-align:right; color:{TEXT}; font-weight:600;">
                                {m['Ann. Vol']:.2%}</td></tr>
                        <tr><td style="color:{SUBTEXT}; padding:3px 0;">Sharpe</td>
                            <td style="text-align:right; color:{GOLD}; font-weight:600;">
                                {m['Sharpe']:.3f}</td></tr>
                        <tr><td style="color:{SUBTEXT}; padding:3px 0;">Sortino</td>
                            <td style="text-align:right; color:{TEXT}; font-weight:600;">
                                {m['Sortino']:.3f}</td></tr>
                        <tr><td style="color:{SUBTEXT}; padding:3px 0;">Max DD</td>
                            <td style="text-align:right; color:{RED}; font-weight:600;">
                                {m['Max DD']:.2%}</td></tr>
                        <tr><td style="color:{SUBTEXT}; padding:3px 0;">VaR 95%</td>
                            <td style="text-align:right; color:{RED}; font-weight:600;">
                                {m['VaR 95%']:.2%}</td></tr>
                        <tr><td style="color:{SUBTEXT}; padding:3px 0;">CVaR 95%</td>
                            <td style="text-align:right; color:{RED}; font-weight:600;">
                                {m['CVaR 95%']:.2%}</td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Weights table ────────────────────────────────────────────────────────
    section_header("Optimised Weights")
    weights_opt_df = pd.DataFrame({
        "Ticker":        tickers,
        "Equal Weight":  [f"{w:.2%}" for w in w_eq],
        "Min Variance":  [f"{w:.2%}" for w in w_mv],
        "Max Sharpe":    [f"{w:.2%}" for w in w_ms],
    })
    st.dataframe(weights_opt_df, hide_index=True, use_container_width=True)

    # ── Grouped bar chart ────────────────────────────────────────────────────
    section_header("Weight Distribution")
    st.plotly_chart(
        chart_optimised_weights(tickers, w_eq, w_mv, w_ms),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ── Efficient frontier ───────────────────────────────────────────────────
    section_header("Efficient Frontier")
    with st.spinner("Sampling 2,500 portfolios …"):
        fig_ef = chart_efficient_frontier(mean_daily, cov_daily, tickers, w_mv, w_ms)
    st.plotly_chart(fig_ef, use_container_width=True, config={"displayModeBar": False})

    # ── Cumulative returns comparison ────────────────────────────────────────
    section_header("Cumulative Return Comparison")
    cum_fig = go.Figure()
    strategy_palette = [
        (w_eq, "Equal Weight",  MUTED,   "dash"),
        (w_mv, "Min Variance",  COBALT2, "solid"),
        (w_ms, "Max Sharpe",    GOLD,    "solid"),
    ]
    for w, label, color, dash in strategy_palette:
        r_ser = daily_rets @ w
        cum   = (1 + r_ser).cumprod() - 1
        cum_fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values * 100,
            name=label,
            line=dict(width=2, color=color, dash=dash),
            hovertemplate=f"<b>{label}</b>: %{{y:.1f}}%<extra></extra>",
        ))
    cum_fig.update_layout(
        **_base_layout("Cumulative Return by Strategy (%)"),
        yaxis_ticksuffix="%",
        hovermode="x unified",
    )
    st.plotly_chart(cum_fig, use_container_width=True, config={"displayModeBar": False})


# ===========================================================================
# VISUALIZATION UTILITIES
# ===========================================================================

def chart_equity_curve(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Equity Curve",
) -> go.Figure:
    """
    Plot a dollar-value equity curve with an optional benchmark overlay.

    Parameters
    ----------
    equity    : pd.Series  — cumulative portfolio value (indexed by date)
    benchmark : pd.Series  — optional benchmark equity curve (same index)
    title     : str
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Portfolio",
        line=dict(color=COBALT2, width=2.2),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.10)",
        hovertemplate="<b>Portfolio</b>: $%{y:,.0f}<extra></extra>",
    ))
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index, y=benchmark.values,
            name="Benchmark (EW)",
            line=dict(color=MUTED, width=1.5, dash="dot"),
            hovertemplate="<b>Benchmark</b>: $%{y:,.0f}<extra></extra>",
        ))
    fig.update_layout(
        **_base_layout(title),
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        hovermode="x unified",
    )
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.98,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        )
    )
            
    return fig


def chart_bt_drawdown(equity: pd.Series, title: str = "Drawdown (%)") -> go.Figure:
    """Underwater chart derived from an equity curve."""
    peak = equity.cummax()
    dd   = ((equity - peak) / peak) * 100
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        line=dict(color=RED, width=1.5),
        name="Drawdown",
        hovertemplate="%{y:.2f}%<extra>Drawdown</extra>",
    ))
    fig.update_layout(**_base_layout(title), yaxis_ticksuffix="%")
    return fig


def chart_monthly_returns_heatmap(returns: pd.Series) -> go.Figure:
    """
    Calendar heatmap of monthly returns — rows = years, columns = months.
    """
    monthly = (
        returns.resample("ME").apply(lambda r: (1 + r).prod() - 1) * 100
    )
    df = monthly.to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month

    years  = sorted(df["year"].unique(), reverse=True)
    months = list(range(1, 13))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    z    = []
    text = []
    for yr in years:
        row_z, row_t = [], []
        for mo in months:
            val = df.loc[(df["year"] == yr) & (df["month"] == mo), "ret"]
            v   = float(val.iloc[0]) if len(val) else float("nan")
            row_z.append(v)
            row_t.append(f"{v:.1f}%" if not np.isnan(v) else "—")
        z.append(row_z)
        text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=month_labels,
        y=[str(yr) for yr in years],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale=[[0.0, RED], [0.5, PANEL], [1.0, GREEN]],
        zmid=0,
        colorbar=dict(title="%", thickness=12, len=0.8, ticksuffix="%"),
        hovertemplate="<b>%{y} %{x}</b>: %{text}<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout("Monthly Return Heatmap (%)"),
    )
    fig.update_xaxes(side="top")
    return fig


def chart_mc_paths(
    paths: np.ndarray,
    dates: pd.DatetimeIndex,
    portfolio_value: float,
    n_display: int = 200,
    title: str = "Simulated Portfolio Paths",
) -> go.Figure:
    """
    Plot a random subsample of Monte Carlo paths with percentile bands.

    Parameters
    ----------
    paths           : np.ndarray, shape (n_sims, n_steps+1)
    dates           : pd.DatetimeIndex length n_steps+1
    portfolio_value : float  — initial value for reference line
    n_display       : int    — max paths to draw (performance guard)
    """
    rng     = np.random.default_rng(0)
    n_sims  = paths.shape[0]
    indices = rng.choice(n_sims, size=min(n_display, n_sims), replace=False)

    fig = go.Figure()

    # Thin individual paths
    for i in indices:
        fig.add_trace(go.Scatter(
            x=dates, y=paths[i],
            mode="lines",
            line=dict(color="rgba(37,99,235,0.35)", width=0.6),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Percentile fan
    pct_specs = [
        (5, 95,  "rgba(16,185,129,0.18)"),
        (10, 90, "rgba(37,99,235,0.12)"),
        (25, 75, "rgba(37,99,235,0.08)"),
    ]
    for lo, hi, fill_color in pct_specs:
        lo_vals = np.percentile(paths, lo, axis=0)
        hi_vals = np.percentile(paths, hi, axis=0)
        fig.add_trace(go.Scatter(
            x=list(dates) + list(dates[::-1]),
            y=list(hi_vals) + list(lo_vals[::-1]),
            fill="toself",
            fillcolor=fill_color,
            line=dict(width=0),
            showlegend=True,
            name=f"P{lo}–P{hi}",
            hoverinfo="skip",
        ))

    # Median path
    median_path = np.median(paths, axis=0)
    fig.add_trace(go.Scatter(
        x=dates, y=median_path,
        line=dict(color=GOLD, width=2.2),
        name="Median",
        hovertemplate="Median: $%{y:,.0f}<extra></extra>",
    ))

    # Initial value reference
    fig.add_hline(
        y=portfolio_value,
        line_dash="dot", line_color=MUTED, line_width=1,
        annotation_text="Initial Value",
        annotation_font=dict(color=MUTED, size=10),
    )

    fig.update_layout(
        **_base_layout(title),
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
        hovermode="x",
    )
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.98,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        )
    )
    return fig


def chart_terminal_distribution(
    terminal_values: np.ndarray,
    portfolio_value: float,
    title: str = "Terminal Value Distribution",
) -> go.Figure:
    """
    Histogram of terminal portfolio values with VaR / mean / median overlays.
    """
    pct_5  = float(np.percentile(terminal_values, 5))
    pct_25 = float(np.percentile(terminal_values, 25))
    median = float(np.median(terminal_values))
    mean   = float(np.mean(terminal_values))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=terminal_values,
        nbinsx=80,
        marker_color=COBALT,
        opacity=0.78,
        name="Terminal Value",
        hovertemplate="Value: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))

    # Vertical annotations
    for val, label, color in [
        (portfolio_value, "Initial",  MUTED),
        (pct_5,           "P5",       RED),
        (pct_25,          "P25",      GOLD),
        (median,          "Median",   GREEN),
        (mean,            "Mean",     COBALT2),
    ]:
        fig.add_vline(
            x=val, line_width=1.5, line_dash="dash", line_color=color,
            annotation_text=label, annotation_position="top right",
            annotation_font=dict(color=color, size=10),
        )

    fig.update_layout(
        **_base_layout(title),
        xaxis_tickprefix="$", xaxis_tickformat=",.0f",
        bargap=0.02,
    )
    return fig


def chart_scenario_bars(scenario_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing key metrics across stress scenarios.

    Parameters
    ----------
    scenario_df : pd.DataFrame
        Rows = scenarios, columns include at least:
        "Expected Return (%)", "VaR 95% (%)", "CVaR 95% (%)", "Max DD (%)"
    """
    metrics = ["Expected Return (%)", "VaR 95% (%)", "CVaR 95% (%)", "Max DD (%)"]
    colors  = [GREEN, GOLD, RED, "#EF4444"]
    fig     = go.Figure()

    for metric, color in zip(metrics, colors):
        if metric not in scenario_df.columns:
            continue
        fig.add_trace(go.Bar(
            name=metric,
            x=scenario_df.index.tolist(),
            y=scenario_df[metric].values,
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.2f}}%<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout("Scenario Metric Comparison (%)"),
        barmode="group",
        yaxis_ticksuffix="%",
        bargap=0.20,
        bargroupgap=0.08,
    )
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.98,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        )
    )
    return fig


def chart_scenario_equity_curves(scenario_curves: dict[str, pd.Series]) -> go.Figure:
    """
    Overlay equity curves for each stress scenario.

    Parameters
    ----------
    scenario_curves : dict mapping scenario label → cumulative return Series
    """
    palette = {
        "Baseline":        COBALT2,
        "Market Crash":    GOLD,
        "Severe Crash":    RED,
        "Bull Market":     GREEN,
        "High Volatility": "#8B5CF6",
    }
    dash_map = {
        "Baseline":        "solid",
        "Market Crash":    "dash",
        "Severe Crash":    "dot",
        "Bull Market":     "solid",
        "High Volatility": "dashdot",
    }
    fig = go.Figure()
    for label, series in scenario_curves.items():
        color = palette.get(label, MUTED)
        dash  = dash_map.get(label, "solid")
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values * 100,
            name=label,
            line=dict(color=color, width=2, dash=dash),
            hovertemplate=f"<b>{label}</b>: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        **_base_layout("Scenario Cumulative Returns (%)"),
        yaxis_ticksuffix="%",
        hovermode="x unified",
    )
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=0.98,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        )
    )
    return fig


# ===========================================================================
# BACKTESTING ANALYTICS
# ===========================================================================

def _bt_simulate_gbm_returns(
    mean_daily: np.ndarray,
    cov_daily: np.ndarray,
    weights: np.ndarray,
    n_days: int,
    seed: int = 0,
) -> pd.Series:
    """
    Generate a synthetic daily return series for backtesting via correlated GBM.

    In production, replace with ``backtesting_engine.run(strategy, prices)``.
    """
    rng = np.random.default_rng(seed)
    n   = len(weights)
    L   = np.linalg.cholesky(cov_daily + 1e-10 * np.eye(n))
    Z   = rng.standard_normal((n_days, n)) @ L.T
    mu_dt = mean_daily - 0.5 * np.diag(cov_daily)
    log_rets = mu_dt + Z
    asset_simple_rets = np.expm1(log_rets)   # exp(log_ret) - 1
    port_rets = asset_simple_rets @ weights
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.Series(port_rets, index=dates, name="strategy")


def bt_cagr(equity: pd.Series) -> float:
    """Compound Annual Growth Rate from an equity curve."""
    n_years = len(equity) / 252
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1)


def bt_sharpe(returns: pd.Series, risk_free: float = 0.04) -> float:
    excess = returns.mean() - risk_free / 252
    std    = returns.std()
    return float((excess / std) * np.sqrt(252)) if std > 0 else 0.0


def bt_sortino(returns: pd.Series, risk_free: float = 0.04) -> float:
    excess   = returns - risk_free / 252
    downside = returns[returns < 0].std()
    return float((excess.mean() / downside) * np.sqrt(252)) if downside > 0 else 0.0


def bt_max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float(((equity - peak) / peak).min())


def bt_calmar(equity: pd.Series, returns: pd.Series) -> float:
    """CAGR / |Max Drawdown|."""
    mdd = abs(bt_max_drawdown(equity))
    return float(bt_cagr(equity) / mdd) if mdd > 0 else 0.0


def bt_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio: probability-weighted gains above threshold / losses below."""
    gains  = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    return float(gains.sum() / losses.sum()) if losses.sum() > 0 else float("inf")


# ===========================================================================
# PAGE 4 — BACKTESTING
# ===========================================================================

def render_backtesting() -> None:
    """
    Backtesting Page.

    Simulates a strategy return series over a configurable look-back window,
    computes institutional performance metrics, and renders equity curve,
    drawdown, monthly return heatmap, and rolling Sharpe.
    """
    page_header(
        "Backtesting",
        "Evaluate historical strategy performance with institutional-grade metrics.",
    )

    tickers = st.session_state.tickers
    weights = st.session_state.weights

    # ── Controls ─────────────────────────────────────────────────────────────
    section_header("Backtest Parameters")
    col_p, col_rf, col_seed, col_gap = st.columns([2, 2, 2, 3])
    with col_p:
        lookback_years = st.selectbox(
            "Look-back period",
            options=[1, 2, 3, 5, 7, 10],
            index=2,
            help="Number of years of simulated history.",
        )
    with col_rf:
        risk_free = st.number_input(
            "Risk-free rate (%)",
            min_value=0.0, max_value=20.0,
            value=4.0, step=0.25,
            help="Annualised risk-free rate used in Sharpe / Sortino.",
        ) / 100
    with col_seed:
        seed = st.number_input(
            "Random seed",
            min_value=0, max_value=9999,
            value=42, step=1,
            help="Fix the seed for reproducible synthetic histories.",
        )

    n_days = int(lookback_years * 252)

    # ── Compute ───────────────────────────────────────────────────────────────
    with st.spinner("Running backtest …"):
        prices = (
            st.session_state.prices_df
            if st.session_state.prices_df is not None
            else get_price_data(tuple(tickers))
        )
        daily_rets = prices.pct_change().dropna()
        mean_daily = daily_rets.mean().values
        cov_daily  = daily_rets.cov().values
        w          = np.array(weights, dtype=float)
        w         /= w.sum()

        strategy_rets = _bt_simulate_gbm_returns(
            mean_daily, cov_daily, w, n_days, seed=int(seed)
        )
        # Equal-weight benchmark
        w_bench    = np.ones(len(tickers)) / len(tickers)
        bench_rets = _bt_simulate_gbm_returns(
            mean_daily, cov_daily, w_bench, n_days, seed=int(seed) + 1
        )

    initial_value  = 1_000_000.0
    equity         = initial_value * (1 + strategy_rets).cumprod()
    bench_equity   = initial_value * (1 + bench_rets).cumprod()

    cagr     = bt_cagr(equity)
    sr       = bt_sharpe(strategy_rets, risk_free)
    srt      = bt_sortino(strategy_rets, risk_free)
    mdd      = bt_max_drawdown(equity)
    calmar   = bt_calmar(equity, strategy_rets)
    omega    = bt_omega_ratio(strategy_rets)
    ann_vol  = float(strategy_rets.std() * np.sqrt(252))
    total_rt = float(equity.iloc[-1] / equity.iloc[0] - 1)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    section_header("Performance Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("CAGR",         f"{cagr:.2%}",   f"Over {lookback_years}Y",
                    "positive" if cagr >= 0 else "negative")
        metric_card("Total Return", f"{total_rt:.2%}", "Cumulative",
                    "positive" if total_rt >= 0 else "negative")
    with c2:
        metric_card("Sharpe Ratio",  f"{sr:.3f}",  f"Rf = {risk_free:.1%}",
                    "positive" if sr >= 1 else "neutral" if sr >= 0 else "negative")
        metric_card("Sortino Ratio", f"{srt:.3f}", "Downside deviation",
                    "positive" if srt >= 1 else "neutral" if srt >= 0 else "negative")
    with c3:
        metric_card("Max Drawdown",      f"{mdd:.2%}",    "Peak-to-trough", "negative")
        metric_card("Annualised Vol",    f"{ann_vol:.2%}", "Daily σ × √252",  "neutral")
    with c4:
        metric_card("Calmar Ratio", f"{calmar:.3f}", "CAGR / |Max DD|",
                    "positive" if calmar >= 0.5 else "neutral")
        metric_card("Omega Ratio",  f"{omega:.3f}",  "Gains / Losses",
                    "positive" if omega >= 1 else "negative")

    # ── Full statistics table ─────────────────────────────────────────────────
    section_header("Full Statistics")
    var_95   = float(-np.percentile(strategy_rets, 5))
    cvar_95  = float(-strategy_rets[strategy_rets <= -var_95].mean())
    win_rate = float((strategy_rets > 0).mean())
    avg_win  = float(strategy_rets[strategy_rets > 0].mean())
    avg_loss = float(strategy_rets[strategy_rets <= 0].mean())

    stats_df = pd.DataFrame({
        "Metric": [
            "CAGR", "Total Return", "Annualised Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Omega Ratio",
            "Maximum Drawdown",
            "VaR 95% (1-day)", "CVaR 95% (1-day)",
            "Win Rate", "Avg Winning Day", "Avg Losing Day",
            "Best Day", "Worst Day",
        ],
        "Strategy": [
            f"{cagr:.4%}", f"{total_rt:.4%}", f"{ann_vol:.4%}",
            f"{sr:.4f}", f"{srt:.4f}", f"{calmar:.4f}", f"{omega:.4f}",
            f"{mdd:.4%}",
            f"{var_95:.4%}", f"{cvar_95:.4%}",
            f"{win_rate:.2%}", f"{avg_win:.4%}", f"{avg_loss:.4%}",
            f"{strategy_rets.max():.4%}", f"{strategy_rets.min():.4%}",
        ],
    })
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    section_header("Equity Curve")
    st.plotly_chart(
        chart_equity_curve(equity, bench_equity, "Portfolio vs Benchmark Equity Curve ($)"),
        use_container_width=True, config={"displayModeBar": False},
    )

    col_l, col_r = st.columns(2)
    with col_l:
        section_header("Drawdown")
        st.plotly_chart(
            chart_bt_drawdown(equity),
            use_container_width=True, config={"displayModeBar": False},
        )
    with col_r:
        section_header("Monthly Return Heatmap")
        st.plotly_chart(
            chart_monthly_returns_heatmap(strategy_rets),
            use_container_width=True, config={"displayModeBar": False},
        )

    # Rolling 63-day Sharpe
    section_header("Rolling 63-Day Sharpe Ratio")
    roll_sr = (
        strategy_rets.rolling(63)
        .apply(lambda r: bt_sharpe(r, risk_free / 252 * 63), raw=False)
    )
    fig_rsr = go.Figure(go.Scatter(
        x=roll_sr.index, y=roll_sr.values,
        line=dict(color=COBALT2, width=1.8),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.12)",
        name="Rolling Sharpe",
        hovertemplate="%{y:.3f}<extra>63d Sharpe</extra>",
    ))
    fig_rsr.add_hline(y=0, line_dash="dot", line_color=MUTED, line_width=1)
    fig_rsr.add_hline(y=1, line_dash="dash", line_color=GREEN, line_width=1,
                      annotation_text="Sharpe = 1",
                      annotation_font=dict(color=GREEN, size=10))
    fig_rsr.update_layout(**_base_layout("Rolling 63-Day Sharpe Ratio"))
    st.plotly_chart(fig_rsr, use_container_width=True, config={"displayModeBar": False})

    logger.info("Backtesting page rendered | CAGR=%.4f | Sharpe=%.4f", cagr, sr)


# ===========================================================================
# MONTE CARLO ANALYTICS
# ===========================================================================

def _mc_run_gbm_paths(
    mean_daily: np.ndarray,
    cov_daily: np.ndarray,
    weights: np.ndarray,
    n_sims: int,
    n_steps: int,
    portfolio_value: float,
    seed: int = 2024,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate correlated GBM portfolio paths.

    Returns
    -------
    paths         : np.ndarray, shape (n_sims, n_steps+1) — dollar values
    terminal_vals : np.ndarray, shape (n_sims,)
    """
    rng    = np.random.default_rng(seed)
    n      = len(weights)
    L      = np.linalg.cholesky(cov_daily + 1e-10 * np.eye(n))
    mu_dt  = mean_daily - 0.5 * np.diag(cov_daily)  # drift correction

    # Shape: (n_sims, n_steps, n_assets)
    Z = rng.standard_normal((n_sims, n_steps, n)) @ L.T
    log_asset_rets = mu_dt[np.newaxis, np.newaxis, :] + Z
    asset_rets     = np.expm1(log_asset_rets)                # (n_sims, n_steps, n)
    port_rets      = asset_rets @ weights                     # (n_sims, n_steps)

    # Build equity paths: (n_sims, n_steps+1) — seed column + cumulative product
    paths        = np.empty((n_sims, n_steps + 1), dtype=np.float64)
    paths[:, 0]  = portfolio_value
    cum_factors  = np.cumprod(1 + port_rets, axis=1)
    paths[:, 1:] = portfolio_value * cum_factors

    return paths, paths[:, -1]


# ===========================================================================
# PAGE 5 — MONTE CARLO SIMULATION
# ===========================================================================

def render_monte_carlo() -> None:
    """
    Monte Carlo Simulation Page.

    Runs correlated GBM across thousands of paths, displays the fan chart,
    terminal value distribution, and key forward-looking statistics.
    """
    page_header(
        "Monte Carlo Simulation",
        "Forward-looking path simulation via correlated Geometric Brownian Motion.",
    )

    tickers = st.session_state.tickers
    weights = st.session_state.weights

    # ── Controls ──────────────────────────────────────────────────────────────
    section_header("Simulation Parameters")
    col_s, col_h, col_pv, col_seed = st.columns(4)
    with col_s:
        n_sims = st.select_slider(
            "Simulations",
            options=[500, 1_000, 2_500, 5_000, 10_000],
            value=2_500,
        )
    with col_h:
        horizon_years = st.selectbox(
            "Horizon (years)", options=[1, 2, 3, 5, 7, 10], index=2
        )
    with col_pv:
        portfolio_value = st.number_input(
            "Initial Portfolio ($)",
            min_value=10_000,
            max_value=100_000_000,
            value=1_000_000,
            step=100_000,
            format="%d",
        )
    with col_seed:
        mc_seed = st.number_input(
            "Random Seed", min_value=0, max_value=9999, value=42, step=1
        )

    n_steps = int(horizon_years * 252)

    # ── Run simulation ────────────────────────────────────────────────────────
    with st.spinner(f"Simulating {n_sims:,} paths over {horizon_years}Y …"):
        prices = (
            st.session_state.prices_df
            if st.session_state.prices_df is not None
            else get_price_data(tuple(tickers))
        )
        daily_rets = prices.pct_change().dropna()
        mean_daily = daily_rets.mean().values
        cov_daily  = daily_rets.cov().values
        w          = np.array(weights, dtype=float)
        w         /= w.sum()

        paths, terminal_vals = _mc_run_gbm_paths(
            mean_daily, cov_daily, w,
            n_sims=int(n_sims),
            n_steps=n_steps,
            portfolio_value=float(portfolio_value),
            seed=int(mc_seed),
        )

    terminal_returns = terminal_vals / portfolio_value - 1
    dates = pd.bdate_range(
        start=pd.Timestamp.today().normalize(),
        periods=n_steps + 1,
    )

    # ── KPI cards ─────────────────────────────────────────────────────────────
    section_header("Forward-Looking Statistics")

    pct_5   = float(np.percentile(terminal_vals, 5))
    pct_25  = float(np.percentile(terminal_vals, 25))
    pct_75  = float(np.percentile(terminal_vals, 75))
    pct_95  = float(np.percentile(terminal_vals, 95))
    mean_tv = float(np.mean(terminal_vals))
    med_tv  = float(np.median(terminal_vals))
    mean_r  = float(np.mean(terminal_returns))
    std_r   = float(np.std(terminal_returns, ddof=1))
    prob_loss = float((terminal_vals < portfolio_value).mean())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Expected Portfolio Value",   f"${mean_tv:,.0f}",
                    f"Mean over {n_sims:,} paths", "positive" if mean_tv >= portfolio_value else "negative")
        metric_card("Median Portfolio Value",      f"${med_tv:,.0f}", "50th percentile", "neutral")
    with c2:
        metric_card("Expected Total Return",       f"{mean_r:.2%}",
                    f"Over {horizon_years}Y horizon",
                    "positive" if mean_r >= 0 else "negative")
        metric_card("Return Std Dev",              f"{std_r:.2%}", "Cross-path dispersion", "neutral")
    with c3:
        metric_card("P5 Terminal Value",           f"${pct_5:,.0f}",  "Worst 5% of paths",  "negative")
        metric_card("P95 Terminal Value",          f"${pct_95:,.0f}", "Best 5% of paths",   "positive")
    with c4:
        metric_card("Probability of Loss",         f"{prob_loss:.2%}", "Paths below initial", "negative")
        metric_card("Interquartile Range",
                    f"${pct_25:,.0f} – ${pct_75:,.0f}",
                    "P25 to P75", "neutral")

    # ── Distribution table ────────────────────────────────────────────────────
    section_header("Terminal Value Percentiles")
    pct_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals   = np.percentile(terminal_vals, pct_levels)
    pct_rets   = pct_vals / portfolio_value - 1
    pct_df = pd.DataFrame({
        "Percentile":          [f"P{p}" for p in pct_levels],
        "Terminal Value":      [f"${v:,.0f}" for v in pct_vals],
        "Total Return":        [f"{r:.2%}" for r in pct_rets],
        "Ann. Return (approx)": [
            f"{((v / portfolio_value) ** (1 / horizon_years) - 1):.2%}"
            for v in pct_vals
        ],
    })
    st.dataframe(pct_df, hide_index=True, use_container_width=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    section_header("Simulated Portfolio Paths")
    st.plotly_chart(
        chart_mc_paths(paths, dates, float(portfolio_value),
                       title=f"Monte Carlo Paths — {horizon_years}Y Horizon"),
        use_container_width=True, config={"displayModeBar": False},
    )

    col_l, col_r = st.columns(2)
    with col_l:
        section_header("Terminal Value Distribution")
        st.plotly_chart(
            chart_terminal_distribution(terminal_vals, float(portfolio_value)),
            use_container_width=True, config={"displayModeBar": False},
        )
    with col_r:
        section_header("Return Distribution")
        ret_vals = terminal_returns * 100
        var_5    = float(np.percentile(ret_vals, 5))
        fig_ret  = go.Figure()
        fig_ret.add_trace(go.Histogram(
            x=ret_vals, nbinsx=80,
            marker_color=COBALT, opacity=0.78,
            name="Total Returns",
            hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra></extra>",
        ))
        for val, label, color in [(var_5, "P5", RED), (0, "Break-even", MUTED)]:
            fig_ret.add_vline(
                x=val, line_dash="dash", line_color=color, line_width=1.5,
                annotation_text=label,
                annotation_font=dict(color=color, size=10),
            )
        fig_ret.update_layout(
            **_base_layout(f"Total Return Distribution over {horizon_years}Y"),
            xaxis_ticksuffix="%", bargap=0.02,
        )
        st.plotly_chart(fig_ret, use_container_width=True,
                        config={"displayModeBar": False})

    # Rolling path convergence (mean and ±1σ band over time)
    section_header("Path Convergence — Mean ± 1σ")
    path_mean = paths.mean(axis=0)
    path_std  = paths.std(axis=0)
    fig_conv  = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(path_mean + path_std) + list((path_mean - path_std)[::-1]),
        fill="toself", fillcolor="rgba(37,99,235,0.12)",
        line=dict(width=0), showlegend=True, name="Mean ± 1σ",
        hoverinfo="skip",
    ))
    fig_conv.add_trace(go.Scatter(
        x=dates, y=path_mean,
        line=dict(color=COBALT2, width=2),
        name="Mean Path",
        hovertemplate="Mean: $%{y:,.0f}<extra></extra>",
    ))
    fig_conv.add_hline(y=float(portfolio_value), line_dash="dot",
                       line_color=MUTED, line_width=1)
    fig_conv.update_layout(
        **_base_layout("Mean Path ± 1σ Band"),
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
    )
    st.plotly_chart(fig_conv, use_container_width=True,
                    config={"displayModeBar": False})

    logger.info(
        "Monte Carlo rendered | n_sims=%d | horizon=%dY | mean_tv=$%.0f",
        n_sims, horizon_years, mean_tv,
    )


# ===========================================================================
# STRESS TESTING ANALYTICS
# ===========================================================================

# Canonical scenario definitions — mirrors monte_carlo_engine.run_standard_stress_suite
_HISTORICAL_SCENARIOS = {
    "COVID Crash": (
        "2020-02-15",
        "2020-03-31",
    ),

    "Financial Crisis": (
        "2008-09-01",
        "2009-03-31",
    ),

    "Inflation Shock": (
        "2022-01-01",
        "2022-12-31",
    ),

    "AI Bull Market": (
        "2023-01-01",
        "2024-12-31",
    ),
}
_STRESS_SCENARIOS: list[dict] = [
    {
        "label":          "Baseline",
        "drift_shock":    0.0,
        "vol_multiplier": 1.0,
        "color":          COBALT2,
        "description":    "No shock. Current calibration.",
    },
    {
        "label":          "Market Crash",
        "drift_shock":    -0.20,
        "vol_multiplier": 1.0,
        "color":          GOLD,
        "description":    "−20% drift shock. Moderate bear market.",
    },
    {
        "label":          "Severe Crash",
        "drift_shock":    -0.40,
        "vol_multiplier": 1.0,
        "color":          RED,
        "description":    "−40% drift shock. GFC / COVID-trough severity.",
    },
    {
        "label":          "Bull Market",
        "drift_shock":    +0.20,
        "vol_multiplier": 1.0,
        "color":          GREEN,
        "description":    "+20% drift shock. Strong risk-on regime.",
    },
    {
        "label":          "High Volatility",
        "drift_shock":    0.0,
        "vol_multiplier": 2.0,
        "color":          "#8B5CF6",
        "description":    "2× volatility. VIX-spike / dislocation regime.",
    },
]


def _stress_simulate_scenario(
    mean_daily: np.ndarray,
    cov_daily: np.ndarray,
    weights: np.ndarray,
    drift_shock: float,
    vol_multiplier: float,
    n_days: int,
    portfolio_value: float,
    seed: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Apply shock parameters to the calibration and run a single GBM scenario.

    Returns
    -------
    equity  : pd.Series — daily portfolio dollar values
    returns : pd.Series — daily portfolio simple returns
    """
    n = len(weights)
    # Shock drift and rebuild covariance with scaled vol
    stressed_mu  = mean_daily + drift_shock / 252
    scale        = np.sqrt(np.diag(cov_daily)) * vol_multiplier
    orig_scale   = np.sqrt(np.diag(cov_daily))
    # Re-extract correlation then rebuild cov with stressed vols
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov_daily / np.outer(orig_scale, orig_scale)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    stressed_cov = np.outer(scale, scale) * corr

    rng   = np.random.default_rng(seed)
    L     = np.linalg.cholesky(stressed_cov + 1e-10 * np.eye(n))
    mu_dt = stressed_mu - 0.5 * np.diag(stressed_cov)
    Z     = rng.standard_normal((n_days, n)) @ L.T
    asset_rets = np.expm1(mu_dt + Z)
    port_rets  = pd.Series(asset_rets @ weights)
    dates      = pd.bdate_range(
        start=pd.Timestamp.today().normalize(), periods=n_days
    )
    port_rets.index = dates
    equity = portfolio_value * (1 + port_rets).cumprod()
    return equity, port_rets


def _stress_scenario_metrics(
    equity: pd.Series,
    returns: pd.Series,
    portfolio_value: float,
    risk_free: float,
) -> dict:
    """Compute the headline metrics for one stress scenario."""
    mdd    = float(((equity - equity.cummax()) / equity.cummax()).min())
    var_95 = float(-np.percentile(returns, 5))
    tail   = returns[returns <= -var_95]
    cvar_95 = float(-tail.mean()) if len(tail) else var_95
    return {
        "Expected Return (%)":   float(returns.mean() * 252) * 100,
        "Ann. Vol (%)":          float(returns.std() * np.sqrt(252)) * 100,
        "Sharpe":                bt_sharpe(returns, risk_free),
        "Max DD (%)":            mdd * 100,
        "VaR 95% (%)":           var_95 * 100,
        "CVaR 95% (%)":          cvar_95 * 100,
        "Terminal Value ($)":    float(equity.iloc[-1]),
        "Total Return (%)":      float(equity.iloc[-1] / portfolio_value - 1) * 100,
    }


# ===========================================================================
# PAGE 6 — STRESS TESTING
# ===========================================================================

def render_stress_testing() -> None:
    """
    Stress Testing Page.

    Applies four canonical market-shock scenarios plus a baseline,
    renders per-scenario equity curves, a grouped metric comparison table,
    and scenario bar charts.
    """
    page_header(
        "Stress Testing",
        "Evaluate portfolio resilience across four canonical market-shock regimes.",
    )

    tickers = st.session_state.tickers
    weights = st.session_state.weights

    # ── Controls ──────────────────────────────────────────────────────────────
    section_header("Scenario Parameters")
    col_h, col_pv, col_rf, col_seed = st.columns(4)
    with col_h:
        horizon_years = st.selectbox(
            "Horizon (years)", options=[1, 2, 3, 5], index=0
        )
    with col_pv:
        portfolio_value = st.number_input(
            "Initial Portfolio ($)",
            min_value=10_000, max_value=100_000_000,
            value=1_000_000, step=100_000, format="%d",
        )
    with col_rf:
        risk_free = st.number_input(
            "Risk-free rate (%)",
            min_value=0.0, max_value=20.0,
            value=4.0, step=0.25,
        ) / 100
    with col_seed:
        st_seed = st.number_input(
            "Random Seed", min_value=0, max_value=9999, value=99, step=1
        )

    n_days = int(horizon_years * 252)

    # Optional custom scenario
    with st.expander("➕  Add a custom scenario", expanded=False):
        col_cs1, col_cs2, col_cs3 = st.columns(3)
        with col_cs1:
            custom_name = st.text_input("Scenario name", value="Custom")
        with col_cs2:
            custom_drift = st.number_input(
                "Drift shock (annualised, e.g. −0.30)", value=-0.10,
                min_value=-1.0, max_value=1.0, step=0.05,
            )
        with col_cs3:
            custom_vol = st.number_input(
                "Vol multiplier (e.g. 1.5)", value=1.5,
                min_value=0.1, max_value=5.0, step=0.1,
            )
        add_custom = st.button("Add scenario", type="secondary")

    # Build active scenario list
    scenarios = [dict(s) for s in _STRESS_SCENARIOS]
    if add_custom and custom_name.strip():
        scenarios.append({
            "label":          custom_name.strip(),
            "drift_shock":    float(custom_drift),
            "vol_multiplier": float(custom_vol),
            "color":          "#EC4899",
            "description":    f"Custom: drift {custom_drift:+.0%}, vol ×{custom_vol:.1f}",
        })

    # ── Run all scenarios ─────────────────────────────────────────────────────
    with st.spinner("Running stress scenarios …"):
        prices = (
            st.session_state.prices_df
            if st.session_state.prices_df is not None
            else get_price_data(tuple(tickers))
        )
        daily_rets = prices.pct_change().dropna()
        mean_daily = daily_rets.mean().values
        cov_daily  = daily_rets.cov().values
        w          = np.array(weights, dtype=float)
        w         /= w.sum()

        results: dict[str, dict] = {}
        equity_curves: dict[str, pd.Series] = {}
        cum_return_curves: dict[str, pd.Series] = {}

        for i, spec in enumerate(scenarios):
            eq, rets = _stress_simulate_scenario(
                mean_daily, cov_daily, w,
                drift_shock=spec["drift_shock"],
                vol_multiplier=spec["vol_multiplier"],
                n_days=n_days,
                portfolio_value=float(portfolio_value),
                seed=int(st_seed) + i,
            )
            results[spec["label"]]            = _stress_scenario_metrics(
                eq, rets, float(portfolio_value), risk_free
            )
            equity_curves[spec["label"]]      = eq
            cum_return_curves[spec["label"]]  = (1 + rets).cumprod() - 1

    # ── Scenario description cards ────────────────────────────────────────────
    section_header("Active Scenarios")
    cols = st.columns(len(scenarios))
    for col, spec in zip(cols, scenarios):
        m = results[spec["label"]]
        sentiment = "positive" if m["Expected Return (%)"] >= 0 else "negative"
        with col:
            st.markdown(
                f"""
                <div class="ql-card" style="border-color:{spec['color']}55; min-height:160px;">
                    <div style="font-size:0.68rem; font-weight:700; letter-spacing:0.10em;
                                text-transform:uppercase; color:{spec['color']};
                                margin-bottom:8px;">{spec['label']}</div>
                    <div style="font-size:0.75rem; color:{SUBTEXT};
                                margin-bottom:12px;">{spec['description']}</div>
                    <div style="font-size:1.25rem; font-weight:700;
                                color:{'#10B981' if m['Expected Return (%)'] >= 0 else '#EF4444'};">
                        {m['Expected Return (%)']:.1f}%
                    </div>
                    <div style="font-size:0.70rem; color:{SUBTEXT};">Expected Ann. Return</div>
                    <div style="margin-top:8px; font-size:0.82rem;">
                        <span style="color:{SUBTEXT};">Max DD </span>
                        <span style="color:#EF4444; font-weight:600;">
                            {m['Max DD (%)']:.1f}%
                        </span>
                        &nbsp;&nbsp;
                        <span style="color:{SUBTEXT};">Sharpe </span>
                        <span style="color:{GOLD}; font-weight:600;">
                            {m['Sharpe']:.2f}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Comparison table ──────────────────────────────────────────────────────
    section_header("Scenario Comparison Table")
    metric_keys = [
        "Expected Return (%)", "Ann. Vol (%)", "Sharpe",
        "Max DD (%)", "VaR 95% (%)", "CVaR 95% (%)",
        "Terminal Value ($)", "Total Return (%)",
    ]
    rows = []
    for key in metric_keys:
        row = {"Metric": key}
        for spec in scenarios:
            val = results[spec["label"]][key]
            if "$" in key:
                row[spec["label"]] = f"${val:,.0f}"
            elif key == "Sharpe":
                row[spec["label"]] = f"{val:.3f}"
            else:
                row[spec["label"]] = f"{val:.2f}%"
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)

    # ── Equity curve overlay ──────────────────────────────────────────────────
    section_header("Equity Curves by Scenario")
    fig_eq = go.Figure()
    for spec in scenarios:
        eq = equity_curves[spec["label"]]
        fig_eq.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            name=spec["label"],
            line=dict(color=spec["color"], width=2),
            hovertemplate=f"<b>{spec['label']}</b>: $%{{y:,.0f}}<extra></extra>",
        ))
    fig_eq.add_hline(
        y=float(portfolio_value), line_dash="dot",
        line_color=MUTED, line_width=1,
        annotation_text="Initial Value",
        annotation_font=dict(color=MUTED, size=10),
    )
    fig_eq.update_layout(
        **_base_layout("Stressed Equity Curves ($)"),
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
        hovermode="x unified",
    )
    fig_eq.update_layout(
        legend=dict(
            x=0.01,
            y=0.02,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        )
    )
    st.plotly_chart(fig_eq, use_container_width=True,
                    config={"displayModeBar": False})

    col_l, col_r = st.columns(2)
    with col_l:
        section_header("Cumulative Return Comparison")
        st.plotly_chart(
            chart_scenario_equity_curves(cum_return_curves),
            use_container_width=True, config={"displayModeBar": False},
        )
    with col_r:
        section_header("Metric Comparison")
        scenario_idx_df = pd.DataFrame(results).T   # scenarios as rows
        st.plotly_chart(
            chart_scenario_bars(scenario_idx_df),
            use_container_width=True, config={"displayModeBar": False},
        )

    # ── Per-scenario drawdown subplots ────────────────────────────────────────
    section_header("Drawdown by Scenario")
    from plotly.subplots import make_subplots
    n_scen = len(scenarios)
    n_cols = min(3, n_scen)
    n_rows = (n_scen + n_cols - 1) // n_cols
    fig_dd = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[s["label"] for s in scenarios],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )
    for idx, spec in enumerate(scenarios):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        eq  = equity_curves[spec["label"]]
        dd  = ((eq - eq.cummax()) / eq.cummax()) * 100
        fig_dd.add_trace(
            go.Scatter(
                x=dd.index, y=dd.values,
                fill="tozeroy", fillcolor="rgba(37,99,235,0.15)",
                line=dict(color=spec["color"], width=1.3),
                name=spec["label"], showlegend=False,
                hovertemplate="%{y:.2f}%<extra></extra>",
            ),
            row=row, col=col,
        )
    fig_dd.update_layout(
        **_base_layout("Portfolio Drawdown Under Each Scenario (%)"),
        height=280 * n_rows,
        showlegend=False,
    )
    fig_dd.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_dd, use_container_width=True,
                    config={"displayModeBar": False})

    logger.info(
        "Stress testing page rendered | %d scenarios | horizon=%dY",
        len(scenarios), horizon_years,
    )
def render_regime_intelligence() -> None:
    """
    Regime Intelligence Dashboard
    """

    page_header(
        "Regime Intelligence",
        "Detect current market regime and adapt portfolio decisions."
    )

    tickers = st.session_state.tickers

    if not tickers:
        st.warning("Build a portfolio first.")
        return

    try:
        prices = (
            st.session_state.prices_df
            if st.session_state.prices_df is not None
            else get_price_data(tuple(tickers))
        )

        close_prices = prices.copy()

        result = detect_market_regime(close_prices)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Current Regime", result.regime.value)

        with col2:
            st.metric("Volatility", f"{result.volatility:.2%}")

        with col3:
            st.metric("Trend", f"{result.trend:.2%}")

        st.markdown("---")

        st.subheader("Regime Description")
        st.info(regime_description(result.regime))

        st.subheader("Confidence")
        st.progress(float(result.confidence))

        st.caption(f"Confidence: {result.confidence:.0%}")

    except Exception as e:
        st.error(f"Regime detection failed: {e}")
# ===========================================================================
# APPLICATION ENTRY POINT
# ===========================================================================

def main() -> None:
    """Bootstrap and route the QuantLab Streamlit application."""
    inject_styles()
    init_session_state()
    render_sidebar()

    page = st.session_state.page
    logger.info("Rendering page: %s", page)

    if page == "Portfolio Builder":
        render_portfolio_builder()
    elif page == "Risk Analytics":
        render_risk_analytics()
    elif page == "Portfolio Optimization":
        render_portfolio_optimization()
    elif page == "Backtesting":
        render_backtesting()
    elif page == "Monte Carlo":
        render_monte_carlo()
    elif page == "Stress Testing":
        render_stress_testing()
    elif page == "Regime Intelligence":
        render_regime_intelligence()
    else:
        st.error(f"Unknown page: {page}")


if __name__ == "__main__":
    main()
