"""Value-at-Risk and Expected Shortfall estimators.

Three VaR methods are implemented because they disagree, and the disagreement
is the interesting part. Parametric VaR assumes normality and therefore
understates tail loss for assets with excess kurtosis. Historical VaR makes no
distributional assumption but cannot produce a loss larger than the worst one
already observed. Monte Carlo sits in between and lets the distributional
assumption be stated explicitly.

Sign convention: VaR and ES are returned as POSITIVE numbers representing a
loss. A 1-day 99% VaR of 0.031 means "a 1% chance of losing more than 3.1%".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RiskEstimate:
    method: str
    confidence: float
    horizon_days: int
    var: float
    expected_shortfall: float

    def as_dollars(self, portfolio_value: float) -> dict:
        return {
            "VaR": self.var * portfolio_value,
            "ES": self.expected_shortfall * portfolio_value,
        }

    def __repr__(self) -> str:
        return (
            f"{self.method:>12} | {self.confidence:.0%} | {self.horizon_days}d | "
            f"VaR {self.var:7.4%} | ES {self.expected_shortfall:7.4%}"
        )


def _scale(value: float, horizon_days: int) -> float:
    """Square-root-of-time scaling.

    Valid only under i.i.d. returns. It understates risk when volatility
    clusters, which it does in practice; noted here rather than hidden.
    """
    return value * np.sqrt(horizon_days)


def historical_var(
    returns: pd.Series, confidence: float = 0.99, horizon_days: int = 1
) -> RiskEstimate:
    """Empirical quantile of the realized return distribution."""
    q = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= q]
    es = tail.mean() if len(tail) else q
    return RiskEstimate(
        "historical", confidence, horizon_days,
        _scale(-q, horizon_days), _scale(-es, horizon_days),
    )


def parametric_var(
    returns: pd.Series, confidence: float = 0.99, horizon_days: int = 1
) -> RiskEstimate:
    """Variance-covariance VaR under a normal distribution.

    Included partly to demonstrate where it fails: the backtest in
    backtest.py rejects this model on fat-tailed data.
    """
    mu, sigma = returns.mean(), returns.std()
    z = stats.norm.ppf(1 - confidence)
    var = -(mu + z * sigma)
    # Closed-form normal ES.
    es = -(mu - sigma * stats.norm.pdf(z) / (1 - confidence))
    return RiskEstimate(
        "parametric", confidence, horizon_days,
        _scale(var, horizon_days), _scale(es, horizon_days),
    )


def student_t_var(
    returns: pd.Series, confidence: float = 0.99, horizon_days: int = 1
) -> RiskEstimate:
    """Parametric VaR with a fitted Student-t, which admits fat tails."""
    df, loc, scale = stats.t.fit(returns)
    q = stats.t.ppf(1 - confidence, df, loc=loc, scale=scale)
    # ES for the t-distribution, integrated over the tail.
    x = stats.t.ppf(1 - confidence, df)
    es_std = -(stats.t.pdf(x, df) / (1 - confidence)) * ((df + x**2) / (df - 1))
    es = loc + scale * es_std
    return RiskEstimate(
        "student-t", confidence, horizon_days,
        _scale(-q, horizon_days), _scale(-es, horizon_days),
    )


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
    n_sims: int = 100_000,
    distribution: str = "t",
    seed: int = 42,
) -> RiskEstimate:
    """Simulate correlated asset returns and revalue the portfolio.

    Uses the Cholesky factor of the covariance matrix to preserve correlation
    structure. The t-distribution option matters: with normal draws the
    simulation inherits the same tail underestimation as parametric VaR.
    """
    rng = np.random.default_rng(seed)
    mu = returns.mean().values
    cov = returns.cov().values

    # Ensure positive definiteness before factorizing.
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() <= 0:
        cov = cov + np.eye(len(cov)) * (abs(eigvals.min()) + 1e-10)
    L = np.linalg.cholesky(cov)

    k = len(mu)
    if distribution == "normal":
        shocks = rng.standard_normal((n_sims, k))
    elif distribution == "t":
        df = 5
        shocks = rng.standard_t(df, size=(n_sims, k))
        shocks /= np.sqrt(df / (df - 2))  # rescale to unit variance
    else:
        raise ValueError("distribution must be 'normal' or 't'")

    sims = mu + shocks @ L.T
    port = sims @ weights

    q = np.percentile(port, (1 - confidence) * 100)
    es = port[port <= q].mean()
    return RiskEstimate(
        f"monte-carlo-{distribution}", confidence, horizon_days,
        _scale(-q, horizon_days), _scale(-es, horizon_days),
    )


def compare_methods(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
) -> pd.DataFrame:
    """Run every estimator side by side."""
    port = pd.Series(returns.values @ weights, index=returns.index)
    estimates = [
        historical_var(port, confidence, horizon_days),
        parametric_var(port, confidence, horizon_days),
        student_t_var(port, confidence, horizon_days),
        monte_carlo_var(returns, weights, confidence, horizon_days, distribution="normal"),
        monte_carlo_var(returns, weights, confidence, horizon_days, distribution="t"),
    ]
    return pd.DataFrame(
        [
            {
                "method": e.method,
                "confidence": e.confidence,
                "horizon_days": e.horizon_days,
                "VaR": e.var,
                "ES": e.expected_shortfall,
                "ES/VaR": e.expected_shortfall / e.var,
            }
            for e in estimates
        ]
    )
