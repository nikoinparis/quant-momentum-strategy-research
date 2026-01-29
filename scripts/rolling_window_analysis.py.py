# scripts/rolling_window_analysis.py
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src.config import TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, DATA_DIR_RAW
from src.data_loader import get_price_data
from src.strategies import momentum
from src.backtester import backtest_positions
from src.metrics import (
    annualized_return_from_log_returns,
    annualized_volatility_from_log_returns,
    sharpe_ratio_from_log_returns,
    max_drawdown,
)

LOOKBACK = 60

TRADING_DAYS_PER_YEAR = 252
WINDOW_YEARS = 3
WINDOW_LEN = WINDOW_YEARS * TRADING_DAYS_PER_YEAR

STEP_DAYS = 21  # ~1 month. Change to 63 for quarterly if you want.


def main():
    cache_path = f"{DATA_DIR_RAW}/prices_{'_'.join(TICKERS)}_{START_DATE}.csv"
    data = get_price_data(
        TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, cache_path=cache_path
    )

    prices = data.prices.dropna()
    log_returns = data.log_returns.dropna()

    # Align indices (returns starts after prices because of diff)
    common_idx = prices.index.intersection(log_returns.index)
    prices = prices.loc[common_idx]
    log_returns = log_returns.loc[common_idx]

    if len(prices) < WINDOW_LEN + LOOKBACK + 5:
        raise ValueError("Not enough data for the chosen window length / lookback.")

    rows = []
    start_i = 0
    end_i = start_i + WINDOW_LEN

    while end_i <= len(prices):
        w_prices = prices.iloc[start_i:end_i]
        w_rets = log_returns.loc[w_prices.index]

        # Strategy in this window
        w_pos = momentum(w_prices, lookback=LOOKBACK)

        # Backtest (includes position lag + transaction costs)
        res = backtest_positions(w_rets, w_pos, transaction_cost_bps=2.0)

        ann_ret = annualized_return_from_log_returns(res.strategy_log_returns)
        ann_vol = annualized_volatility_from_log_returns(res.strategy_log_returns)
        sharpe = sharpe_ratio_from_log_returns(res.strategy_log_returns)
        mdd = max_drawdown(res.equity_curve)
        final_eq = float(res.equity_curve.iloc[-1])

        rows.append(
            {
                "Window Start": w_prices.index[0],
                "Window End": w_prices.index[-1],
                "Sharpe": sharpe,
                "Annual Return": ann_ret,
                "Annual Vol": ann_vol,
                "Max Drawdown": mdd,
                "Final Equity": final_eq,
            }
        )

        start_i += STEP_DAYS
        end_i = start_i + WINDOW_LEN

    df = pd.DataFrame(rows)
    df = df.set_index("Window End")

    # Print summary table
    pd.set_option("display.max_columns", 100)
    print("\nRolling Window Results (Momentum 60d, 3-year windows)\n")
    print(df[["Sharpe", "Annual Return", "Annual Vol", "Max Drawdown", "Final Equity"]].head())
    print("\nSummary:\n")
    print(df[["Sharpe", "Annual Return", "Max Drawdown"]].describe())

    # Plot Sharpe over time
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Sharpe"].values)
    plt.axhline(0.0, linewidth=1)
    plt.title("Rolling 3-Year Sharpe (Momentum 60d)")
    plt.xlabel("Window End Date")
    plt.ylabel("Sharpe")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot rolling max drawdown over time
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Max Drawdown"].values)
    plt.axhline(0.0, linewidth=1)
    plt.title("Rolling 3-Year Max Drawdown (Momentum 60d)")
    plt.xlabel("Window End Date")
    plt.ylabel("Max Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()