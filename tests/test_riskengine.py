"""Tests for the risk engine.

The important tests here are the ones that check statistical correctness
against known closed-form answers, not just that the code runs.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from riskengine import (
    Portfolio,
    historical_var,
    kupiec_test,
    max_drawdown,
    monte_carlo_var,
    parametric_var,
    rolling_var_backtest,
    synthetic_prices,
    to_returns,
)


@pytest.fixture
def returns():
    prices = synthetic_prices(["A", "B", "C"], "2015-01-01", "2024-01-01")
    return to_returns(prices)


# --- Portfolio -----------------------------------------------------------


def test_weights_must_sum_to_one(returns):
    with pytest.raises(ValueError, match="sum to 1"):
        Portfolio(returns, [0.5, 0.3, 0.1])


def test_equal_weights_default(returns):
    p = Portfolio(returns)
    assert np.allclose(p.weights, 1 / 3)


def test_risk_contributions_sum_to_one(returns):
    p = Portfolio(returns)
    assert np.isclose(p.marginal_contribution_to_risk().sum(), 1.0)


def test_inverse_vol_underweights_volatile_asset(returns):
    p = Portfolio.inverse_volatility(returns)
    vols = returns.std()
    # The most volatile asset should carry the smallest weight.
    assert p.weights[vols.values.argmax()] == p.weights.min()


# --- VaR correctness -----------------------------------------------------


def test_parametric_var_matches_closed_form():
    """On exactly normal data, parametric VaR must equal mu + z*sigma."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.01, 200_000))
    est = parametric_var(r, confidence=0.99)
    expected = -(r.mean() + stats.norm.ppf(0.01) * r.std())
    assert est.var == pytest.approx(expected, rel=1e-9)


def test_historical_var_recovers_known_quantile():
    r = pd.Series(np.linspace(-0.10, 0.10, 10_001))
    est = historical_var(r, confidence=0.95)
    assert est.var == pytest.approx(0.09, abs=1e-3)


def test_es_always_exceeds_var(returns):
    """Expected Shortfall is a tail average beyond VaR, so it must be larger."""
    p = Portfolio(returns)
    for est in (
        historical_var(p.portfolio_returns),
        parametric_var(p.portfolio_returns),
        monte_carlo_var(returns, p.weights),
    ):
        assert est.expected_shortfall > est.var, est.method


def test_var_increases_with_confidence(returns):
    p = Portfolio(returns)
    v95 = historical_var(p.portfolio_returns, 0.95).var
    v99 = historical_var(p.portfolio_returns, 0.99).var
    assert v99 > v95


def test_horizon_scaling_is_sqrt_time(returns):
    p = Portfolio(returns)
    one = historical_var(p.portfolio_returns, 0.99, 1).var
    ten = historical_var(p.portfolio_returns, 0.99, 10).var
    assert ten == pytest.approx(one * np.sqrt(10))


def test_monte_carlo_converges_to_parametric_on_normal_draws():
    """With normal shocks, MC VaR should approach the closed-form answer."""
    rng = np.random.default_rng(3)
    n = 8000
    data = pd.DataFrame(rng.normal(0, 0.01, (n, 2)), columns=["A", "B"])
    w = np.array([0.5, 0.5])
    mc = monte_carlo_var(data, w, 0.99, n_sims=400_000, distribution="normal").var
    par = parametric_var(pd.Series(data.values @ w), 0.99).var
    assert mc == pytest.approx(par, rel=0.03)


def test_fat_tails_produce_higher_var_than_normal(returns):
    """The whole argument for not using parametric VaR."""
    p = Portfolio(returns)
    normal = monte_carlo_var(returns, p.weights, 0.99, distribution="normal").var
    fat = monte_carlo_var(returns, p.weights, 0.99, distribution="t").var
    assert fat > normal


# --- Backtesting ---------------------------------------------------------


def test_kupiec_accepts_correct_model():
    """Exactly the expected breach count should not be rejected."""
    stat, p = kupiec_test(breaches=10, n=1000, confidence=0.99)
    assert stat == pytest.approx(0.0, abs=1e-9)
    assert p > 0.99


def test_kupiec_rejects_badly_understated_risk():
    """50 breaches where 10 were promised is a broken model."""
    _, p = kupiec_test(breaches=50, n=1000, confidence=0.99)
    assert p < 0.001


def test_kupiec_handles_zero_breaches():
    stat, p = kupiec_test(breaches=0, n=1000, confidence=0.99)
    assert np.isfinite(stat) and 0 <= p <= 1


def test_backtest_breach_rate_is_plausible(returns):
    p = Portfolio(returns)
    result, frame = rolling_var_backtest(returns, p.weights, "historical", 500, 0.99)
    assert 0.0 < result.breach_rate < 0.06
    assert len(frame) == len(returns) - 500


def test_backtest_is_out_of_sample(returns):
    """The forecast for day t must not depend on day t."""
    p = Portfolio(returns)
    _, frame = rolling_var_backtest(returns, p.weights, "historical", 500, 0.99)
    # Perturbing only the final observation must not change earlier forecasts.
    modified = returns.copy()
    modified.iloc[-1] = modified.iloc[-1] - 0.5
    _, frame2 = rolling_var_backtest(modified, p.weights, "historical", 500, 0.99)
    assert np.allclose(frame["var_forecast"][:-1], frame2["var_forecast"][:-1])


def test_backtest_rejects_short_history(returns):
    p = Portfolio(returns)
    with pytest.raises(ValueError, match="need more than"):
        rolling_var_backtest(returns, p.weights, "historical", window=len(returns) + 10)


# --- Drawdown ------------------------------------------------------------


def test_max_drawdown_on_known_path():
    """Up 100%, then down 50% -> a clean 50% drawdown."""
    r = pd.Series([1.0, -0.5], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    assert max_drawdown(r)["max_drawdown"] == pytest.approx(-0.5)


def test_drawdown_is_never_positive(returns):
    p = Portfolio(returns)
    assert max_drawdown(p.portfolio_returns)["max_drawdown"] <= 0


def test_monotonic_series_has_no_drawdown():
    r = pd.Series([0.01] * 100, index=pd.bdate_range("2020-01-01", periods=100))
    assert max_drawdown(r)["max_drawdown"] == pytest.approx(0.0)
