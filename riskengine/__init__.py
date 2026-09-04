"""Portfolio risk analytics engine."""

from .data import load_prices, to_returns, synthetic_prices
from .portfolio import Portfolio
from .var import (
    historical_var,
    parametric_var,
    student_t_var,
    monte_carlo_var,
    compare_methods,
)
from .backtest import rolling_var_backtest, compare_backtests, kupiec_test
from .analytics import (
    max_drawdown,
    drawdown_series,
    rolling_volatility,
    ewma_volatility,
    stress_correlation,
    tail_statistics,
)

__version__ = "0.1.0"
