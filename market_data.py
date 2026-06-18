"""
QuantLab - Market Data Module
==============================
Institutional-grade market data acquisition layer for historical price data,
ETF analytics, and index benchmarks using the yfinance API.

Author: QuantLab Engineering
License: MIT
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants & Enumerations
# ---------------------------------------------------------------------------

DEFAULT_START = "2010-01-01"
DEFAULT_INTERVAL = "1d"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

# Well-known benchmark indices (Yahoo Finance tickers)
BENCHMARK_INDICES: dict[str, str] = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "RUSSELL2000": "^RUT",
    "VIX": "^VIX",
    "FTSE100": "^FTSE",
    "DAX": "^GDAXI",
    "NIKKEI": "^N225",
    "HANGSENG": "^HSI",
    "MSCI_WORLD": "URTH",   # ETF proxy
}

# Commonly referenced ETFs by category
ETF_UNIVERSE: dict[str, list[str]] = {
    "us_equity": ["SPY", "QQQ", "IWM", "DIA", "VTI"],
    "fixed_income": ["AGG", "BND", "TLT", "IEF", "SHY"],
    "international": ["EFA", "EEM", "VEU", "IEFA", "VWO"],
    "sector": ["XLF", "XLK", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLRE"],
    "commodities": ["GLD", "SLV", "USO", "DBC", "PDBC"],
    "volatility": ["UVXY", "SVXY", "VXX"],
    "alternatives": ["BITO", "IAU", "PDBC"],
}


class AssetClass(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    INDEX = "index"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DownloadResult:
    """
    Container for the result of a market data download request.

    Attributes
    ----------
    tickers : list[str]
        Requested tickers.
    data : pd.DataFrame
        OHLCV price data with a MultiIndex (field, ticker) or flat columns.
    metadata : dict
        Per-ticker metadata (sector, currency, shortName, etc.).
    failed : list[str]
        Tickers for which data could not be retrieved.
    asset_class : AssetClass
        Classification of the asset group.
    download_time : datetime
        UTC timestamp of the download.
    """

    tickers: list[str]
    data: pd.DataFrame
    metadata: dict = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)
    asset_class: AssetClass = AssetClass.UNKNOWN
    download_time: datetime = field(default_factory=datetime.utcnow)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def prices(self, price_type: str = "Close") -> pd.DataFrame:
        """
        Return a single price-type DataFrame (e.g. 'Close', 'Adj Close').

        Parameters
        ----------
        price_type : str
            Column level to extract when data has a MultiIndex.

        Returns
        -------
        pd.DataFrame
            Date-indexed DataFrame of requested prices, one column per ticker.
        """
        if isinstance(self.data.columns, pd.MultiIndex):
            try:
                return self.data[price_type].copy()
            except KeyError:
                available = self.data.columns.get_level_values(0).unique().tolist()
                raise KeyError(
                    f"'{price_type}' not found. Available: {available}"
                ) from None
        return self.data[[price_type]].copy() if price_type in self.data.columns else self.data.copy()

    def returns(self, price_type: str = "Close", periods: int = 1) -> pd.DataFrame:
        """
        Compute period-over-period percentage returns.

        Parameters
        ----------
        price_type : str
            Price series to use for return calculation.
        periods : int
            Lag for pct_change (default 1 = daily returns).

        Returns
        -------
        pd.DataFrame
            Returns DataFrame, NaN-dropped.
        """
        return self.prices(price_type).pct_change(periods).dropna(how="all")

    def log_returns(self, price_type: str = "Close") -> pd.DataFrame:
        """Return log-returns (ln(P_t / P_{t-1}))."""
        import numpy as np
        px = self.prices(price_type)
        return np.log(px / px.shift(1)).dropna(how="all")

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the underlying DataFrame."""
        return self.data.shape

    def __repr__(self) -> str:
        return (
            f"DownloadResult("
            f"tickers={self.tickers}, "
            f"shape={self.shape}, "
            f"failed={self.failed}, "
            f"asset_class={self.asset_class.value})"
        )


# ---------------------------------------------------------------------------
# Core downloader
# ---------------------------------------------------------------------------

class MarketDataClient:
    """
    Institutional market data client wrapping yfinance.

    Supports bulk downloads of equities, ETFs, and indices with retry logic,
    metadata enrichment, and a clean DataFrame output contract.

    Parameters
    ----------
    auto_adjust : bool
        If True, prices are adjusted for splits and dividends (default True).
    progress : bool
        Show yfinance download progress bar (default False for server use).
    threads : bool | int
        Use multi-threading in yfinance bulk downloads (default True).

    Examples
    --------
    >>> client = MarketDataClient()
    >>> result = client.download_equities(["AAPL", "MSFT", "GOOGL"])
    >>> result.prices().tail()
    """

    def __init__(
        self,
        auto_adjust: bool = True,
        progress: bool = False,
        threads: bool | int = True,
    ) -> None:
        self.auto_adjust = auto_adjust
        self.progress = progress
        self.threads = threads

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_equities(
        self,
        tickers: list[str],
        start: str | date = DEFAULT_START,
        end: Optional[str | date] = None,
        interval: str = DEFAULT_INTERVAL,
    ) -> DownloadResult:
        """
        Download historical OHLCV data for one or more equity tickers.

        Parameters
        ----------
        tickers : list[str]
            List of Yahoo Finance equity ticker symbols (e.g. ["AAPL", "MSFT"]).
        start : str | date
            Start date, inclusive (default "2010-01-01").
        end : str | date | None
            End date, inclusive. Defaults to today.
        interval : str
            Data frequency: "1d", "1wk", "1mo", "1h", etc.

        Returns
        -------
        DownloadResult
            Container with OHLCV data and per-ticker metadata.
        """
        logger.info("Downloading equities: %s  [%s → %s]", tickers, start, end or "today")
        return self._download(tickers, start, end, interval, AssetClass.EQUITY)

    def download_etfs(
        self,
        tickers: list[str],
        start: str | date = DEFAULT_START,
        end: Optional[str | date] = None,
        interval: str = DEFAULT_INTERVAL,
    ) -> DownloadResult:
        """
        Download historical OHLCV data for one or more ETF tickers.

        Parameters
        ----------
        tickers : list[str]
            List of Yahoo Finance ETF ticker symbols (e.g. ["SPY", "QQQ"]).
        start : str | date
            Start date, inclusive.
        end : str | date | None
            End date, inclusive. Defaults to today.
        interval : str
            Data frequency.

        Returns
        -------
        DownloadResult
        """
        logger.info("Downloading ETFs: %s  [%s → %s]", tickers, start, end or "today")
        return self._download(tickers, start, end, interval, AssetClass.ETF)

    def download_indices(
        self,
        tickers: list[str],
        start: str | date = DEFAULT_START,
        end: Optional[str | date] = None,
        interval: str = DEFAULT_INTERVAL,
    ) -> DownloadResult:
        """
        Download historical data for market indices.

        Parameters
        ----------
        tickers : list[str]
            List of Yahoo Finance index symbols (e.g. ["^GSPC", "^IXIC"]).
            Use the ``BENCHMARK_INDICES`` mapping for convenient aliases.
        start : str | date
            Start date, inclusive.
        end : str | date | None
            End date, inclusive. Defaults to today.
        interval : str
            Data frequency.

        Returns
        -------
        DownloadResult
        """
        logger.info("Downloading indices: %s  [%s → %s]", tickers, start, end or "today")
        return self._download(tickers, start, end, interval, AssetClass.INDEX)

    def download_benchmarks(
        self,
        keys: Optional[list[str]] = None,
        start: str | date = DEFAULT_START,
        end: Optional[str | date] = None,
    ) -> DownloadResult:
        """
        Download a preset basket of benchmark indices.

        Parameters
        ----------
        keys : list[str] | None
            Subset of keys from ``BENCHMARK_INDICES``. If None, all are fetched.
        start : str | date
            Start date, inclusive.
        end : str | date | None
            End date, inclusive.

        Returns
        -------
        DownloadResult
        """
        selection = {k: v for k, v in BENCHMARK_INDICES.items() if k in (keys or BENCHMARK_INDICES)}
        tickers = list(selection.values())
        logger.info("Downloading benchmark indices: %s", list(selection.keys()))
        return self._download(tickers, start, end, DEFAULT_INTERVAL, AssetClass.INDEX)

    def download_etf_universe(
        self,
        categories: Optional[list[str]] = None,
        start: str | date = DEFAULT_START,
        end: Optional[str | date] = None,
    ) -> dict[str, DownloadResult]:
        """
        Download ETFs grouped by category from the built-in ETF universe.

        Parameters
        ----------
        categories : list[str] | None
            Subset of ``ETF_UNIVERSE`` keys (e.g. ["us_equity", "sector"]).
            If None, all categories are downloaded.
        start : str | date
            Start date, inclusive.
        end : str | date | None
            End date, inclusive.

        Returns
        -------
        dict[str, DownloadResult]
            Mapping of category name → DownloadResult.
        """
        selected = {
            cat: tickers
            for cat, tickers in ETF_UNIVERSE.items()
            if categories is None or cat in categories
        }
        results: dict[str, DownloadResult] = {}
        for cat, tickers in selected.items():
            logger.info("Downloading ETF category '%s': %s", cat, tickers)
            results[cat] = self.download_etfs(tickers, start=start, end=end)
        return results

    def get_ticker_info(self, ticker: str) -> dict:
        """
        Fetch metadata for a single ticker via yfinance.

        Parameters
        ----------
        ticker : str
            Yahoo Finance ticker symbol.

        Returns
        -------
        dict
            Metadata dict (sector, industry, currency, marketCap, etc.).
            Returns an empty dict if the ticker is invalid or unavailable.
        """
        try:
            info = yf.Ticker(ticker).info
            return info if isinstance(info, dict) else {}
        except Exception as exc:
            logger.warning("Could not fetch info for '%s': %s", ticker, exc)
            return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download(
        self,
        tickers: list[str],
        start: str | date,
        end: Optional[str | date],
        interval: str,
        asset_class: AssetClass,
    ) -> DownloadResult:
        """
        Core download routine with retry logic and validation.

        Parameters
        ----------
        tickers : list[str]
        start : str | date
        end : str | date | None
        interval : str
        asset_class : AssetClass

        Returns
        -------
        DownloadResult
        """
        tickers = self._normalise_tickers(tickers)
        end_str = str(end) if end else str(date.today())
        start_str = str(start)

        raw = self._fetch_with_retry(tickers, start_str, end_str, interval)

        if raw is None or raw.empty:
            logger.error("No data returned for tickers: %s", tickers)
            return DownloadResult(
                tickers=tickers,
                data=pd.DataFrame(),
                failed=tickers,
                asset_class=asset_class,
            )

        data, failed = self._validate_and_clean(raw, tickers)
        metadata = self._fetch_metadata(tickers)

        return DownloadResult(
            tickers=tickers,
            data=data,
            metadata=metadata,
            failed=failed,
            asset_class=asset_class,
        )

    def _fetch_with_retry(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """
        Attempt yfinance download with exponential backoff retries.

        Parameters
        ----------
        tickers : list[str]
        start : str
        end : str
        interval : str

        Returns
        -------
        pd.DataFrame | None
        """
        ticker_str = " ".join(tickers)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = yf.download(
                    tickers=ticker_str,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=self.auto_adjust,
                    progress=self.progress,
                    threads=self.threads,
                    group_by="column",
                )
                if df is not None and not df.empty:
                    return df
                logger.warning("Attempt %d/%d returned empty DataFrame.", attempt, MAX_RETRIES)
            except Exception as exc:
                logger.warning(
                    "Attempt %d/%d failed for tickers %s: %s",
                    attempt, MAX_RETRIES, tickers, exc,
                )
            if attempt < MAX_RETRIES:
                sleep_secs = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.info("Retrying in %.1f seconds …", sleep_secs)
                time.sleep(sleep_secs)

        logger.error("All %d download attempts failed for: %s", MAX_RETRIES, tickers)
        return None

    def _validate_and_clean(
        self,
        df: pd.DataFrame,
        tickers: list[str],
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Validate downloaded data, identify failed tickers, and clean the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw output from yf.download.
        tickers : list[str]
            Originally requested tickers.

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            (cleaned_df, list_of_failed_tickers)
        """
        failed: list[str] = []

        # Normalise single-ticker downloads (flat columns) into MultiIndex
        if not isinstance(df.columns, pd.MultiIndex) and len(tickers) == 1:
            df.columns = pd.MultiIndex.from_product([df.columns, tickers])

        # Identify tickers with entirely NaN close prices
        if isinstance(df.columns, pd.MultiIndex):
            for ticker in tickers:
                try:
                    col_data = df["Close"][ticker] if "Close" in df.columns.get_level_values(0) else None
                except (KeyError, TypeError):
                    col_data = None

                if col_data is None or col_data.isna().all():
                    logger.warning("Ticker '%s' returned all NaN — marking as failed.", ticker)
                    failed.append(ticker)

        # Drop rows where all values are NaN
        df.dropna(how="all", inplace=True)

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df.sort_index(inplace=True)
        return df, failed

    def _fetch_metadata(self, tickers: list[str]) -> dict:
        """
        Retrieve a curated subset of metadata for each ticker.

        Parameters
        ----------
        tickers : list[str]

        Returns
        -------
        dict
            {ticker: {shortName, sector, industry, currency, marketCap, …}}
        """
        METADATA_KEYS = [
            "shortName", "longName", "sector", "industry",
            "currency", "exchange", "quoteType",
            "marketCap", "trailingPE", "dividendYield",
            "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        ]
        metadata: dict = {}
        for ticker in tickers:
            raw_info = self.get_ticker_info(ticker)
            metadata[ticker] = {k: raw_info.get(k) for k in METADATA_KEYS}
        return metadata

    @staticmethod
    def _normalise_tickers(tickers: list[str]) -> list[str]:
        """
        Upper-case, strip whitespace, and deduplicate ticker list.

        Parameters
        ----------
        tickers : list[str]

        Returns
        -------
        list[str]
        """
        seen: set[str] = set()
        result: list[str] = []
        for t in tickers:
            normalised = t.strip().upper()
            if normalised and normalised not in seen:
                seen.add(normalised)
                result.append(normalised)
        return result


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def get_equity_data(
    tickers: list[str],
    start: str = DEFAULT_START,
    end: Optional[str] = None,
    **kwargs,
) -> DownloadResult:
    """
    Module-level shortcut: download equity data without instantiating a client.

    Parameters
    ----------
    tickers : list[str]
    start : str
    end : str | None
    **kwargs
        Passed to ``MarketDataClient.__init__``.

    Returns
    -------
    DownloadResult
    """
    return MarketDataClient(**kwargs).download_equities(tickers, start=start, end=end)


def get_index_data(
    tickers: list[str],
    start: str = DEFAULT_START,
    end: Optional[str] = None,
    **kwargs,
) -> DownloadResult:
    """Module-level shortcut for index data."""
    return MarketDataClient(**kwargs).download_indices(tickers, start=start, end=end)


def get_etf_data(
    tickers: list[str],
    start: str = DEFAULT_START,
    end: Optional[str] = None,
    **kwargs,
) -> DownloadResult:
    """Module-level shortcut for ETF data."""
    return MarketDataClient(**kwargs).download_etfs(tickers, start=start, end=end)


# ---------------------------------------------------------------------------
# Demo entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    QuantLab market_data.py — demonstration script.

    Exercises the full public API:
      1. Individual equity download
      2. Multi-ticker equity basket
      3. ETF basket (fixed income + sector)
      4. Benchmark indices
      5. Close-price extraction and return calculation
    """
    separator = "─" * 68
    client = MarketDataClient(auto_adjust=True, progress=False)

    # ── 1. Single equity ────────────────────────────────────────────────
    print(separator)
    print("  1. Single Equity: AAPL (5Y daily)")
    print(separator)
    aapl = client.download_equities(["AAPL"], start="2019-01-01")
    print(aapl)
    print("Close prices (last 5 rows):")
    print(aapl.prices("Close").tail())

    # ── 2. Multi-equity basket ──────────────────────────────────────────
    print(f"\n{separator}")
    print("  2. Equity Basket: FAANG (3Y daily)")
    print(separator)
    faang = client.download_equities(
        ["META", "AAPL", "AMZN", "NFLX", "GOOGL"],
        start="2022-01-01",
    )
    print(faang)
    px = faang.prices("Close")
    rets = faang.returns()
    print("\nClose prices (last 3 rows):")
    print(px.tail(3).to_string())
    print("\nDaily returns (last 3 rows):")
    print(rets.tail(3).round(4).to_string())

    # ── 3. ETF categories ───────────────────────────────────────────────
    print(f"\n{separator}")
    print("  3. ETF Universe (US Equity + Fixed Income)")
    print(separator)
    etf_results = client.download_etf_universe(
        categories=["us_equity", "fixed_income"],
        start="2020-01-01",
    )
    for cat, result in etf_results.items():
        print(f"\nCategory: {cat}")
        print(result)
        print(result.prices("Close").tail(2).round(2).to_string())

    # ── 4. Benchmark indices ─────────────────────────────────────────────
    print(f"\n{separator}")
    print("  4. Benchmark Indices")
    print(separator)
    benchmarks = client.download_benchmarks(
        keys=["SP500", "NASDAQ", "RUSSELL2000", "VIX"],
        start="2020-01-01",
    )
    print(benchmarks)
    print("\nClose prices (last 5 rows):")
    print(benchmarks.prices("Close").tail(5).round(2).to_string())

    # ── 5. Log-return summary ────────────────────────────────────────────
    print(f"\n{separator}")
    print("  5. Annualised Vol from Log-Returns (FAANG, 2Y)")
    print(separator)
    import numpy as np
    log_ret = faang.log_returns()
    ann_vol = log_ret.std() * np.sqrt(252)
    print(ann_vol.rename("Annualised Vol (σ)").round(4).to_string())

    print(f"\n{separator}")
    print("  Demo complete.")
    print(separator)


if __name__ == "__main__":
    main()
