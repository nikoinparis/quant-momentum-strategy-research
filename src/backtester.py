# src/backtester.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    positions: pd.DataFrame         # positions used (aligned)
    strategy_log_returns: pd.Series # portfolio-level log returns
    equity_curve: pd.Series         # cumulative equity (starts at 1.0)


def backtest_positions(
    asset_log_returns: pd.DataFrame,
    positions: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
) -> BacktestResult:
    """
    Generic backtester.

    asset_log_returns: DataFrame of log returns (index=date, cols=assets)
    positions: DataFrame of desired positions in {-1,0,1} (same shape)

    We use positions_{t-1} * returns_t to avoid look-ahead.

    transaction_cost_bps: cost per unit turnover in basis points.
      Example: 5 bps = 0.0005 per 1.0 change in position.
    """

    # Align on common dates/assets
    positions = positions.reindex(asset_log_returns.index).fillna(0.0)
    positions = positions.reindex(columns=asset_log_returns.columns).fillna(0.0)

    # Use yesterday's position to earn today's return
    held = positions.shift(1).fillna(0.0)

    # Equal-weight across assets (for SPY-only this is just SPY)
    n_assets = len(asset_log_returns.columns)
    weights = held / max(n_assets, 1)

    # Portfolio log return = sum_i w_i * r_i
    strat_lr = (weights * asset_log_returns).sum(axis=1)

    # Transaction costs based on turnover: sum |pos_t - pos_{t-1}|
    if transaction_cost_bps > 0:
        turnover = positions.diff().abs().sum(axis=1)  # across assets
        cost = (transaction_cost_bps / 10_000.0) * turnover
        strat_lr = strat_lr - cost

    # Equity curve from log returns: equity_t = exp(cumsum(log_returns))
    equity = np.exp(strat_lr.cumsum())
    equity.iloc[0] = 1.0  # normalize start

    return BacktestResult(
        positions=positions,
        strategy_log_returns=strat_lr,
        equity_curve=equity,
    )