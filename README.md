# Portfolio Risk Analytics Engine

Value-at-Risk and Expected Shortfall estimation for a multi-asset portfolio, with
out-of-sample backtesting to test whether the risk models actually hold up.

Computing a VaR number is straightforward. Establishing whether the model was
right is the harder problem, and it is the one this project is built around.

---

## The finding

Three VaR models were estimated on a rolling 500-day window and tested out of
sample against roughly 2,500 subsequent trading days at 99% confidence.

| Model | Breaches | Expected | Rate | Kupiec p | Verdict |
|---|---|---|---|---|---|
| Historical simulation | 34 | 25.5 | 1.34% | 0.105 | not rejected |
| Parametric (normal) | 36 | 25.5 | 1.41% | 0.048 | **rejected** |
| Parametric (Student-t) | 27 | 25.5 | 1.06% | 0.760 | not rejected |

The variance-covariance model — the one taught first and used most often —
is rejected at the 5% level. It breached 41% more often than it promised.

The reason is visible in the return distribution: excess kurtosis of 8.3 and a
Jarque-Bera test that rejects normality with a p-value indistinguishable from
zero. A normal distribution cannot price a tail it does not believe exists.
Refitting the same parametric approach with a Student-t, which admits fat tails,
produces a model that survives the test.

This matters beyond the academic point. Under Basel's traffic-light framework,
a model breaching this often draws a capital multiplier. The cost of assuming
normality is not conceptual — it is charged in capital.

![VaR backtest](outputs/var_backtest.png)

---

## What it does

**Four VaR estimators, because they disagree and the disagreement is informative**

- Historical simulation — no distributional assumption, but cannot forecast a
  loss worse than the worst already observed
- Parametric normal — closed form, fast, and wrong in the tail
- Parametric Student-t — fitted degrees of freedom, admits kurtosis
- Monte Carlo — Cholesky-factorized covariance preserving correlation
  structure, with selectable normal or t innovations

**Expected Shortfall** alongside every VaR figure. VaR states the threshold; ES
states the average loss once the threshold is crossed. The ES/VaR ratio is
reported because it is a direct read on tail thickness — 1.14 under the normal
model against 1.41 empirically, which is the same failure the backtest catches.

**Model validation**

- Kupiec (1995) unconditional coverage test — did the model breach at the
  promised rate?
- Christoffersen (1998) independence test — were breaches spread out or
  clustered? Clustered breaches mean the model fails precisely when it is
  needed, which is the worse defect.

**Portfolio diagnostics**

- Risk contribution decomposition — a portfolio equally weighted by capital is
  not equally weighted by risk
- Drawdown path, depth, duration, and recovery
- Realized and EWMA (λ=0.94, RiskMetrics) volatility
- Correlation by regime

---

## A note on the correlation analysis

The standard way to show that "correlations rise in a crisis" is to take the
worst decile of days and correlate them. That approach is biased.

Boyer, Gibson and Loretan (1999) showed that conditioning on one tail of a
common factor truncates that factor's variance within the subsample, which
mechanically depresses measured correlation even when the true correlation is
unchanged. A naive implementation can report correlations *falling* during a
selloff, which is an artifact rather than a finding.

This implementation defines the stress regime by drawdown state — days where the
portfolio sits more than 10% below its running peak — which is a persistent
condition rather than a selection on the same returns being correlated.

---

## Running it

```bash
git clone https://github.com/<your-username>/portfolio-risk-engine.git
cd portfolio-risk-engine
pip install -r requirements.txt
python run_analysis.py
```

Options:

```bash
python run_analysis.py --tickers SPY QQQ TLT GLD EFA --start 2010-01-01
python run_analysis.py --confidence 0.95 --value 5000000
python run_analysis.py --weighting inverse-vol --window 750
```

Prices are pulled from Yahoo Finance via `yfinance` and cached to `data/`. If
the network is unavailable, a synthetic generator produces correlated
Student-t returns so the pipeline remains runnable and testable offline.

Tests:

```bash
pytest tests/ -v
```

Twenty tests, including closed-form checks — parametric VaR is verified against
the analytical normal quantile, Monte Carlo is verified to converge to the
parametric result under normal draws, and the backtest is verified to be
genuinely out of sample by confirming that perturbing the final observation
leaves earlier forecasts unchanged.

---

## Structure

```
riskengine/
  data.py         price loading, caching, synthetic fallback
  portfolio.py    weighting schemes, risk contribution
  var.py          VaR and ES estimators
  backtest.py     Kupiec and Christoffersen tests
  analytics.py    drawdown, volatility, regime correlation
  plots.py        figures
run_analysis.py   CLI
tests/            test suite
```

---

## Known limitations

Stated plainly, because a risk model whose assumptions are not written down is
not a risk model.

- **Square-root-of-time scaling** for multi-day horizons assumes i.i.d.
  returns. Volatility clusters, so this understates risk at longer horizons.
- **No volatility model.** GARCH(1,1) would let the VaR forecast respond to
  conditional volatility instead of trailing realized volatility. This is the
  most valuable extension.
- **Static weights.** Daily rebalancing is assumed and transaction costs are
  ignored.
- **Equity ETFs only.** No options, so no delta-gamma approximation and no
  jump risk beyond what the underlying already shows.
- **The Christoffersen test has low power** in small samples. Failing to reject
  is weak evidence of independence, not proof of it.

---

## References

Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement
models. *Journal of Derivatives*.

Christoffersen, P. (1998). Evaluating interval forecasts. *International
Economic Review*.

Boyer, B., Gibson, M., and Loretan, M. (1999). Pitfalls in tests for changes in
correlations. *Federal Reserve International Finance Discussion Paper*.
