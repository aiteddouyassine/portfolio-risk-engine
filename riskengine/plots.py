"""Chart generation. All figures are written to outputs/."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "outputs"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

NAVY = "#1a2f4b"
RED = "#c0392b"
GREY = "#7f8c8d"


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_drawdown(returns: pd.Series, name: str = "drawdown.png") -> Path:
    from .analytics import drawdown_series

    dd = drawdown_series(returns)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(dd.index, dd["wealth"], color=NAVY, lw=1.1, label="Portfolio")
    ax1.plot(dd.index, dd["peak"], color=GREY, lw=0.8, ls="--", label="Running peak")
    ax1.set_ylabel("Growth of $1")
    ax1.set_yscale("log")
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_title("Cumulative performance and drawdown")

    ax2.fill_between(dd.index, dd["drawdown"] * 100, 0, color=RED, alpha=0.6, lw=0)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("")
    return _save(fig, name)


def plot_var_backtest(frame: pd.DataFrame, method: str, confidence: float,
                      name: str = "var_backtest.png") -> Path:
    """Realized returns against the VaR forecast, with breaches marked."""
    fig, ax = plt.subplots(figsize=(9, 4.2))

    ax.plot(frame.index, frame["actual_return"] * 100, color=GREY, lw=0.5,
            alpha=0.8, label="Daily return")
    ax.plot(frame.index, -frame["var_forecast"] * 100, color=NAVY, lw=1.2,
            label=f"{confidence:.0%} VaR forecast")

    breaches = frame[frame["breach"]]
    ax.scatter(breaches.index, breaches["actual_return"] * 100, color=RED,
               s=16, zorder=5, label=f"Breaches ({len(breaches)})")

    ax.set_ylabel("Return (%)")
    ax.set_title(f"Out-of-sample VaR backtest — {method}")
    ax.legend(frameon=False, loc="lower left", ncol=3)
    return _save(fig, name)


def plot_var_comparison(table: pd.DataFrame, name: str = "var_methods.png") -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(table))
    w = 0.38

    ax.bar(x - w / 2, table["VaR"] * 100, w, label="VaR", color=NAVY)
    ax.bar(x + w / 2, table["ES"] * 100, w, label="Expected Shortfall", color=RED, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(table["method"], rotation=20, ha="right")
    ax.set_ylabel("Loss (%)")
    ax.set_title("1-day 99% risk estimates by method")
    ax.legend(frameon=False)
    return _save(fig, name)


def plot_return_distribution(returns: pd.Series, name: str = "distribution.png") -> Path:
    """Empirical returns against a fitted normal, to show the tail gap."""
    from scipy import stats

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(returns * 100, bins=120, density=True, color=NAVY, alpha=0.6, label="Empirical")
    xs = np.linspace(returns.min(), returns.max(), 400)
    ax1.plot(xs * 100, stats.norm.pdf(xs, returns.mean(), returns.std()),
             color=RED, lw=1.4, label="Normal fit")
    ax1.set_xlabel("Daily return (%)")
    ax1.set_ylabel("Density")
    ax1.set_title("Return distribution")
    ax1.legend(frameon=False)

    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor(NAVY)
    ax2.get_lines()[0].set_markeredgecolor(NAVY)
    ax2.get_lines()[0].set_markersize(2.5)
    ax2.get_lines()[1].set_color(RED)
    ax2.set_title("Q-Q plot vs normal")

    return _save(fig, name)


def plot_rolling_vol(returns: pd.Series, name: str = "volatility.png") -> Path:
    from .analytics import rolling_volatility, ewma_volatility

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(returns.index, rolling_volatility(returns, 63) * 100, color=NAVY,
            lw=1.0, label="63-day realized")
    ax.plot(returns.index, ewma_volatility(returns) * 100, color=RED,
            lw=1.0, alpha=0.8, label="EWMA (λ=0.94)")
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_title("Volatility clustering")
    ax.legend(frameon=False)
    return _save(fig, name)


def plot_correlation_regimes(stress: dict, name: str = "correlations.png") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, title in [
        (axes[0], "calm_matrix", "Calm regime"),
        (axes[1], "stress_matrix", f"Stressed (drawdown ≤ {stress['drawdown_threshold']:.0%})"),
    ]:
        m = stress[key]
        im = ax.imshow(m.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(m)), m.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(m)), m.index)
        ax.set_title(title)
        ax.grid(False)
        for i in range(len(m)):
            for j in range(len(m)):
                ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(m.values[i, j]) > 0.55 else "black")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Correlation")
    return _save(fig, name)
