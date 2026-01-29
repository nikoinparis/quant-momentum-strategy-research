# src/metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def annualized_return_from_log_returns(log_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    If log returns are g_t, total log return over T periods is sum(g_t).
    Annualized log return = mean(g_t) * periods_per_year.
    Annualized simple return = exp(annualized_log_return) - 1.
    """
    lr = log_returns.dropna()
    if len(lr) == 0:
        return float("nan")

    ann_log = lr.mean() * periods_per_year
    return float(np.exp(ann_log) - 1.0)


def annualized_volatility_from_log_returns(log_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Annualized volatility = std(log_returns) * sqrt(periods_per_year)
    """
    lr = log_returns.dropna()
    if len(lr) == 0:
        return float("nan")

    return float(lr.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio_from_log_returns(
    log_returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Sharpe = (E[R] - Rf) / Vol

    Using log returns:
      - convert annual risk-free to per-period approx: rf_per_period = rf_annual / periods_per_year
      - excess per-period = lr - rf_per_period
      - annualized Sharpe = sqrt(periods_per_year) * mean(excess) / std(excess)
    """
    lr = log_returns.dropna()
    if len(lr) == 0:
        return float("nan")

    rf_per_period = risk_free_rate_annual / periods_per_year
    excess = lr - rf_per_period

    vol = excess.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return float("nan")

    return float(np.sqrt(periods_per_year) * excess.mean() / vol)


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Max drawdown = minimum over t of (equity_t / running_max_t - 1).
    Returns a negative number (e.g., -0.32 means -32% peak-to-trough).
    """
    eq = equity_curve.dropna()
    if len(eq) == 0:
        return float("nan")

    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    return float(dd.min())


def win_rate(log_returns: pd.Series) -> float:
    """
    Fraction of periods with positive strategy return.
    """
    lr = log_returns.dropna()
    if len(lr) == 0:
        return float("nan")
    return float((lr > 0).mean())


def summarize_strategy(
    name: str,
    strategy_log_returns: pd.Series,
    equity_curve: pd.Series,
    risk_free_rate_annual: float = 0.0,
) -> pd.Series:
    """
    Returns a one-row summary (as a Series) for easy DataFrame construction.
    """
    ann_ret = annualized_return_from_log_returns(strategy_log_returns)
    ann_vol = annualized_volatility_from_log_returns(strategy_log_returns)
    sharpe = sharpe_ratio_from_log_returns(strategy_log_returns, risk_free_rate_annual=risk_free_rate_annual)
    mdd = max_drawdown(equity_curve)
    wr = win_rate(strategy_log_returns)

    return pd.Series(
        {
            "Strategy": name,
            "Final Equity": float(equity_curve.iloc[-1]),
            "Annual Return": ann_ret,
            "Annual Vol": ann_vol,
            "Sharpe": sharpe,
            "Max Drawdown": mdd,
            "Win Rate": wr,
        }
    )