"""Portfolio construction and return aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd


class Portfolio:
    """A long-only portfolio defined by asset weights.

    Weights are fixed (rebalanced daily by construction) which keeps the risk
    calculations interpretable. A buy-and-hold variant is provided separately
    for drawdown work, where drift matters.
    """

    def __init__(self, returns: pd.DataFrame, weights: np.ndarray | dict | None = None):
        self.returns = returns
        self.assets = list(returns.columns)

        if weights is None:
            w = np.repeat(1 / len(self.assets), len(self.assets))
        elif isinstance(weights, dict):
            w = np.array([weights[a] for a in self.assets], dtype=float)
        else:
            w = np.asarray(weights, dtype=float)

        if len(w) != len(self.assets):
            raise ValueError("weight vector length does not match asset count")
        if not np.isclose(w.sum(), 1.0):
            raise ValueError(f"weights must sum to 1, got {w.sum():.4f}")

        self.weights = w

    @classmethod
    def inverse_volatility(cls, returns: pd.DataFrame, lookback: int | None = None):
        """Risk-parity-style weights: allocate inversely to realized volatility."""
        window = returns.tail(lookback) if lookback else returns
        vol = window.std()
        w = (1 / vol) / (1 / vol).sum()
        return cls(returns, w.values)

    @property
    def portfolio_returns(self) -> pd.Series:
        """Daily portfolio return series."""
        return pd.Series(
            self.returns.values @ self.weights,
            index=self.returns.index,
            name="portfolio",
        )

    def cumulative(self) -> pd.Series:
        return (1 + self.portfolio_returns).cumprod()

    def covariance(self, annualize: bool = False) -> pd.DataFrame:
        cov = self.returns.cov()
        return cov * 252 if annualize else cov

    def volatility(self, annualize: bool = True) -> float:
        var = self.weights @ self.covariance().values @ self.weights
        vol = np.sqrt(var)
        return vol * np.sqrt(252) if annualize else vol

    def marginal_contribution_to_risk(self) -> pd.Series:
        """Each asset's share of total portfolio volatility.

        Component contributions sum to 1. This is what tells you a portfolio
        that looks diversified by weight is not diversified by risk.
        """
        cov = self.covariance().values
        port_vol = np.sqrt(self.weights @ cov @ self.weights)
        marginal = cov @ self.weights / port_vol
        contribution = self.weights * marginal
        return pd.Series(contribution / contribution.sum(), index=self.assets)

    def summary(self) -> dict:
        r = self.portfolio_returns
        ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
        ann_vol = r.std() * np.sqrt(252)
        return {
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": ann_ret / ann_vol if ann_vol else np.nan,
            "skewness": r.skew(),
            "excess_kurtosis": r.kurtosis(),
            "observations": len(r),
        }
