#!/usr/bin/env python3
"""Run the full risk analysis and write charts to outputs/.

Usage:
    python run_analysis.py
    python run_analysis.py --tickers SPY QQQ TLT GLD --start 2010-01-01
    python run_analysis.py --confidence 0.95 --value 5000000
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

from riskengine import (
    Portfolio,
    compare_backtests,
    compare_methods,
    load_prices,
    max_drawdown,
    rolling_var_backtest,
    stress_correlation,
    tail_statistics,
    to_returns,
)
from riskengine import plots


def header(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Portfolio risk analytics")
    ap.add_argument("--tickers", nargs="+",
                    default=["SPY", "QQQ", "TLT", "GLD", "EFA"])
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--confidence", type=float, default=0.99)
    ap.add_argument("--value", type=float, default=1_000_000,
                    help="portfolio notional in dollars")
    ap.add_argument("--window", type=int, default=500,
                    help="rolling estimation window for the backtest")
    ap.add_argument("--weighting", choices=["equal", "inverse-vol"], default="equal")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    header("DATA")
    prices = load_prices(args.tickers, args.start, args.end)
    returns = to_returns(prices)
    print(f"{len(returns):,} daily observations, "
          f"{returns.index[0].date()} to {returns.index[-1].date()}")
    print(f"Assets: {', '.join(returns.columns)}")

    portfolio = (
        Portfolio.inverse_volatility(returns)
        if args.weighting == "inverse-vol"
        else Portfolio(returns)
    )
    port_rets = portfolio.portfolio_returns

    header("PORTFOLIO")
    for k, v in portfolio.summary().items():
        print(f"  {k:<26} {v:>12,.4f}")
    print("\n  Weights:")
    for asset, w in zip(portfolio.assets, portfolio.weights):
        print(f"    {asset:<8} {w:>7.2%}")
    print("\n  Risk contribution (share of portfolio volatility):")
    for asset, c in portfolio.marginal_contribution_to_risk().items():
        print(f"    {asset:<8} {c:>7.2%}")

    header("DISTRIBUTION")
    for k, v in tail_statistics(port_rets).items():
        print(f"  {k:<26} {v!s:>12}" if isinstance(v, bool) else f"  {k:<26} {v:>12,.4f}")

    header(f"RISK ESTIMATES — {args.confidence:.0%}, 1-day")
    table = compare_methods(returns, portfolio.weights, args.confidence)
    print(table.to_string(index=False))
    print(f"\n  On ${args.value:,.0f} notional:")
    for _, row in table.iterrows():
        print(f"    {row['method']:<20} VaR ${row['VaR'] * args.value:>12,.0f}"
              f"   ES ${row['ES'] * args.value:>12,.0f}")

    header(f"BACKTEST — rolling {args.window}-day window, out of sample")
    bt = compare_backtests(returns, portfolio.weights, args.window, args.confidence)
    print(bt.to_string(index=False))

    header("DRAWDOWN")
    for k, v in max_drawdown(port_rets).items():
        print(f"  {k:<26} {v!s:>16}")

    header("CORRELATION REGIMES")
    try:
        stress = stress_correlation(returns)
        print(f"  Calm average correlation     {stress['calm_avg_correlation']:>8.3f}"
              f"   ({stress['calm_days']:,} days)")
        print(f"  Stressed average correlation {stress['stress_avg_correlation']:>8.3f}"
              f"   ({stress['stress_days']:,} days)")
        print(f"  Change                       {stress['correlation_increase']:>+8.3f}")
    except ValueError as exc:
        print(f"  skipped: {exc}")
        stress = None

    if not args.no_plots:
        header("CHARTS")
        result, frame = rolling_var_backtest(
            returns, portfolio.weights, "historical", args.window, args.confidence
        )
        paths = [
            plots.plot_drawdown(port_rets),
            plots.plot_var_backtest(frame, "historical", args.confidence),
            plots.plot_var_comparison(table),
            plots.plot_return_distribution(port_rets),
            plots.plot_rolling_vol(port_rets),
        ]
        if stress:
            paths.append(plots.plot_correlation_regimes(stress))
        for p in paths:
            print(f"  wrote {p.relative_to(p.parent.parent)}")

    print()


if __name__ == "__main__":
    main()
