"""Price data loading.

Pulls adjusted close prices from Yahoo Finance and caches them to disk so that
repeated runs do not re-hit the network. If the network is unavailable and no
cache exists, a synthetic generator produces correlated return series with
realistic fat tails so the rest of the pipeline remains testable offline.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def _cache_path(tickers: list[str], start: str, end: str) -> Path:
    key = f"{'-'.join(sorted(tickers))}_{start}_{end}.csv"
    return CACHE_DIR / key


def load_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices indexed by date.

    Falls back to cached data, then to synthetic data, so the pipeline runs
    in any environment.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(tickers, start, end)

    if use_cache and path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)

    try:
        import yfinance as yf

        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        prices = prices.dropna(how="all").ffill().dropna()
        if prices.empty:
            raise ValueError("no data returned")
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers[0])
        prices.to_csv(path)
        return prices
    except Exception as exc:  # network down, bad ticker, rate limit
        print(f"[data] live download failed ({exc}); using synthetic series")
        return synthetic_prices(tickers, start, end)


def synthetic_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate correlated, fat-tailed price paths for offline testing.

    Returns are drawn from a multivariate Student-t (df=4) so that tail risk
    is present and the normality assumption in parametric VaR is genuinely
    violated, which is the point the backtest is meant to expose.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    dates = pd.bdate_range(start, end)
    n, k = len(dates), len(tickers)
    rng = np.random.default_rng(seed)

    # Single market factor plus idiosyncratic noise -> realistic correlation.
    betas = rng.uniform(0.7, 1.3, size=k)
    df = 4
    market = rng.standard_t(df, size=n) * (0.011 / np.sqrt(df / (df - 2)))
    idio = rng.standard_t(df, size=(n, k)) * (0.009 / np.sqrt(df / (df - 2)))
    rets = market[:, None] * betas[None, :] + idio
    rets += 0.0003  # small positive drift

    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """Convert prices to simple (default) or log returns."""
    if log:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()
