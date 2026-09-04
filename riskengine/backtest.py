"""VaR model backtesting.

Computing a VaR number is easy. Establishing whether the model was any good is
the part that matters, and it is what regulators require under the Basel
framework. Two tests are implemented:

  Kupiec (1995) unconditional coverage
      Did the model breach at roughly the promised rate? A 99% VaR should be
      exceeded on about 1% of days. Too many breaches means the model
      understates risk; too few means it is wasting capital.

  Christoffersen (1998) independence
      Were the breaches spread out, or did they cluster? Clustered breaches
      mean the model fails exactly when it is needed, which is a worse defect
      than a slightly wrong average rate.

Both are likelihood-ratio tests, asymptotically chi-squared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BacktestResult:
    method: str
    confidence: float
    observations: int
    breaches: int
    expected_breaches: float
    breach_rate: float
    kupiec_stat: float
    kupiec_pvalue: float
    christoffersen_stat: float
    christoffersen_pvalue: float

    @property
    def verdict(self) -> str:
        if self.kupiec_pvalue < 0.05 and self.christoffersen_pvalue < 0.05:
            return "REJECTED (coverage and clustering)"
        if self.kupiec_pvalue < 0.05:
            return "REJECTED (wrong breach rate)"
        if self.christoffersen_pvalue < 0.05:
            return "REJECTED (breaches cluster)"
        return "not rejected"

    def __repr__(self) -> str:
        return (
            f"{self.method:>18} | breaches {self.breaches:3d} vs "
            f"{self.expected_breaches:5.1f} expected | "
            f"Kupiec p={self.kupiec_pvalue:.4f} | "
            f"Christoffersen p={self.christoffersen_pvalue:.4f} | {self.verdict}"
        )


def kupiec_test(breaches: int, n: int, confidence: float) -> tuple[float, float]:
    """Unconditional coverage likelihood-ratio test."""
    p = 1 - confidence
    if breaches == 0:
        stat = -2 * n * np.log(1 - p)
    elif breaches == n:
        stat = -2 * n * np.log(p)
    else:
        pi = breaches / n
        stat = -2 * (
            (n - breaches) * np.log(1 - p) + breaches * np.log(p)
            - ((n - breaches) * np.log(1 - pi) + breaches * np.log(pi))
        )
    return stat, 1 - stats.chi2.cdf(stat, df=1)


def christoffersen_test(breach_flags: np.ndarray) -> tuple[float, float]:
    """Independence test on the sequence of breach indicators."""
    f = np.asarray(breach_flags).astype(int)
    if len(f) < 2:
        return 0.0, 1.0

    # Transition counts: n_ij = moves from state i to state j.
    prev, curr = f[:-1], f[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())

    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return 0.0, 1.0

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    if pi in (0, 1) or pi01 in (0,) or pi11 in (0, 1):
        return 0.0, 1.0

    ll_null = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll_alt = (
        n00 * np.log(1 - pi01) + n01 * np.log(pi01)
        + n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    )
    stat = -2 * (ll_null - ll_alt)
    return stat, 1 - stats.chi2.cdf(stat, df=1)


def rolling_var_backtest(
    returns: pd.DataFrame,
    weights: np.ndarray,
    method: str = "historical",
    window: int = 500,
    confidence: float = 0.99,
) -> tuple[BacktestResult, pd.DataFrame]:
    """Walk forward through history, re-estimating VaR on a rolling window.

    At each step the model sees only data available at that time, so this is
    an out-of-sample test rather than an in-sample fit.
    """
    from .var import historical_var, parametric_var, student_t_var

    port = pd.Series(returns.values @ weights, index=returns.index)
    if len(port) <= window:
        raise ValueError(f"need more than {window} observations, got {len(port)}")

    estimators = {
        "historical": historical_var,
        "parametric": parametric_var,
        "student-t": student_t_var,
    }
    if method not in estimators:
        raise ValueError(f"method must be one of {list(estimators)}")
    estimator = estimators[method]

    records = []
    for i in range(window, len(port)):
        train = port.iloc[i - window : i]
        actual = port.iloc[i]
        var = estimator(train, confidence).var
        records.append(
            {
                "date": port.index[i],
                "actual_return": actual,
                "var_forecast": var,
                "breach": actual < -var,
            }
        )

    frame = pd.DataFrame(records).set_index("date")
    flags = frame["breach"].values
    n, breaches = len(frame), int(flags.sum())

    k_stat, k_p = kupiec_test(breaches, n, confidence)
    c_stat, c_p = christoffersen_test(flags)

    result = BacktestResult(
        method=method,
        confidence=confidence,
        observations=n,
        breaches=breaches,
        expected_breaches=n * (1 - confidence),
        breach_rate=breaches / n,
        kupiec_stat=k_stat,
        kupiec_pvalue=k_p,
        christoffersen_stat=c_stat,
        christoffersen_pvalue=c_p,
    )
    return result, frame


def compare_backtests(
    returns: pd.DataFrame,
    weights: np.ndarray,
    window: int = 500,
    confidence: float = 0.99,
) -> pd.DataFrame:
    """Backtest every estimator and rank them."""
    rows = []
    for method in ("historical", "parametric", "student-t"):
        result, _ = rolling_var_backtest(returns, weights, method, window, confidence)
        rows.append(
            {
                "method": result.method,
                "breaches": result.breaches,
                "expected": round(result.expected_breaches, 1),
                "breach_rate": result.breach_rate,
                "kupiec_p": result.kupiec_pvalue,
                "christoffersen_p": result.christoffersen_pvalue,
                "verdict": result.verdict,
            }
        )
    return pd.DataFrame(rows)
