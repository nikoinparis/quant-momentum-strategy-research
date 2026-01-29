# scripts/vol_threshold_sensitivity.py
from __future__ import annotations

import pandas as pd

from src.config import TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, DATA_DIR_RAW
from src.data_loader import get_price_data
from src.strategies import momentum, vol_regime_filter
from src.backtester import backtest_positions
from src.metrics import (
    annualized_return_from_log_returns,
    annualized_volatility_from_log_returns,
    sharpe_ratio_from_log_returns,
    max_drawdown,
)

MOM_LOOKBACK = 60
VOL_LOOKBACK = 20
VOL_THRESHOLDS = [0.015, 0.020, 0.025]


def main():
    cache_path = f"{DATA_DIR_RAW}/prices_{'_'.join(TICKERS)}_{START_DATE}.csv"
    data = get_price_data(
        TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, cache_path=cache_path
    )

    prices = data.prices.dropna()
    rets = data.log_returns.dropna()

    idx = prices.index.intersection(rets.index)
    prices = prices.loc[idx]
    rets = rets.loc[idx]

    base_positions = momentum(prices, lookback=MOM_LOOKBACK)

    rows = []

    for vt in VOL_THRESHOLDS:
        gate = vol_regime_filter(rets, vol_lookback=VOL_LOOKBACK, vol_threshold=vt)
        positions = base_positions * gate
        res = backtest_positions(rets, positions)

        rows.append({
            "Vol Threshold": vt,
            "Final Equity": float(res.equity_curve.iloc[-1]),
            "Annual Return": annualized_return_from_log_returns(res.strategy_log_returns),
            "Annual Vol": annualized_volatility_from_log_returns(res.strategy_log_returns),
            "Sharpe": sharpe_ratio_from_log_returns(res.strategy_log_returns),
            "Max Drawdown": max_drawdown(res.equity_curve),
            "Avg Gate %": gate.mean().iloc[0],
        })

    df = pd.DataFrame(rows).set_index("Vol Threshold")
    pd.set_option("display.max_columns", 100)

    print("\nVolatility Threshold Sensitivity (Momentum 60d)\n")
    print(df)


if __name__ == "__main__":
    main()