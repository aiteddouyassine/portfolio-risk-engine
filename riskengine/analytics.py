"""Drawdown, volatility, and correlation diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(returns: pd.Series) -> pd.DataFrame:
    """Running peak, drawdown, and underwater duration."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1

    underwater = (dd < 0).astype(int)
    groups = (underwater.diff() != 0).cumsum()
    duration = underwater.groupby(groups).cumsum()

    return pd.DataFrame(
        {"wealth": wealth, "peak": peak, "drawdown": dd, "days_underwater": duration}
    )


def max_drawdown(returns: pd.Series) -> dict:
    """Worst peak-to-trough decline, with dates and recovery time."""
    dd = drawdown_series(returns)
    trough = dd["drawdown"].idxmin()
    peak = dd.loc[:trough, "wealth"].idxmax()

    after = dd.loc[trough:]
    recovered = after[after["wealth"] >= dd.loc[peak, "wealth"]]
    recovery = recovered.index[0] if len(recovered) else None

    return {
        "max_drawdown": dd["drawdown"].min(),
        "peak_date": peak,
        "trough_date": trough,
        "recovery_date": recovery,
        "days_to_trough": int((trough - peak).days),
        "days_to_recover": int((recovery - trough).days) if recovery is not None else None,
        "recovered": recovery is not None,
    }


def rolling_volatility(returns: pd.Series, window: int = 63, annualize: bool = True) -> pd.Series:
    """Trailing realized volatility. Default window is one quarter."""
    vol = returns.rolling(window).std()
    return vol * np.sqrt(252) if annualize else vol


def ewma_volatility(returns: pd.Series, lam: float = 0.94, annualize: bool = True) -> pd.Series:
    """RiskMetrics-style EWMA volatility.

    Responds faster to regime change than an equal-weighted window, which is
    why it is the standard choice for short-horizon risk.
    """
    var = returns.ewm(alpha=1 - lam).var()
    vol = np.sqrt(var)
    return vol * np.sqrt(252) if annualize else vol


def stress_correlation(returns: pd.DataFrame, drawdown_threshold: float = -0.10) -> dict:
    """Compare correlations in calm regimes against correlations in stressed ones.

    Diversification tends to fail when it is most needed: correlations rise
    together in a selloff. Quantifying that honestly requires care.

    The obvious approach -- take the worst decile of days and correlate them --
    is biased. Boyer, Gibson and Loretan (1999) showed that conditioning on one
    tail of a common factor truncates that factor's variance in the subsample,
    which mechanically depresses measured correlation even when the true
    correlation is unchanged. Naive implementations report *falling*
    correlations in a crisis for this reason.

    This implementation instead defines the stress regime by portfolio
    drawdown state, which is a persistent condition rather than a selection on
    the same daily returns being correlated. Days are labelled stressed when
    the portfolio sits more than `drawdown_threshold` below its running peak.
    """
    portfolio = returns.mean(axis=1)
    dd = drawdown_series(portfolio)["drawdown"]

    stress_mask = dd <= drawdown_threshold
    stressed = returns[stress_mask]
    calm = returns[~stress_mask]

    if len(stressed) < 30 or len(calm) < 30:
        raise ValueError(
            f"insufficient observations to compare regimes "
            f"(stress={len(stressed)}, calm={len(calm)}); "
            f"try a shallower drawdown_threshold"
        )

    def avg_offdiag(corr: pd.DataFrame) -> float:
        vals = corr.values
        mask = ~np.eye(len(vals), dtype=bool)
        return float(vals[mask].mean())

    stress_corr = stressed.corr()
    calm_corr = calm.corr()

    return {
        "calm_avg_correlation": avg_offdiag(calm_corr),
        "stress_avg_correlation": avg_offdiag(stress_corr),
        "correlation_increase": avg_offdiag(stress_corr) - avg_offdiag(calm_corr),
        "stress_days": len(stressed),
        "calm_days": len(calm),
        "drawdown_threshold": drawdown_threshold,
        "stress_matrix": stress_corr,
        "calm_matrix": calm_corr,
    }


def tail_statistics(returns: pd.Series) -> dict:
    """Distributional diagnostics that justify choosing one VaR model over another."""
    from scipy import stats

    jb_stat, jb_p = stats.jarque_bera(returns)
    return {
        "mean": returns.mean(),
        "std": returns.std(),
        "skewness": returns.skew(),
        "excess_kurtosis": returns.kurtosis(),
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_p,
        "normality_rejected": jb_p < 0.05,
        "worst_day": returns.min(),
        "best_day": returns.max(),
    }
