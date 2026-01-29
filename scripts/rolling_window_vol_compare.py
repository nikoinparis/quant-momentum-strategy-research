# scripts/rolling_window_vol_compare.py
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src.config import TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, DATA_DIR_RAW
from src.data_loader import get_price_data
from src.strategies import momentum, vol_regime_filter
from src.backtester import backtest_positions
from src.metrics import sharpe_ratio_from_log_returns

# Strategy parameters
MOM_LOOKBACK = 60
VOL_LOOKBACK = 20
VOL_THRESHOLD = 0.02

# Rolling window setup
TRADING_DAYS = 252
WINDOW_YEARS = 3
WINDOW_LEN = WINDOW_YEARS * TRADING_DAYS
STEP = 21  # monthly


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

    rows = []
    i = 0

    while i + WINDOW_LEN < len(prices):
        w_prices = prices.iloc[i:i + WINDOW_LEN]
        w_rets = rets.loc[w_prices.index]

        # Base momentum
        pos_mom = momentum(w_prices, lookback=MOM_LOOKBACK)
        res_mom = backtest_positions(w_rets, pos_mom)

        # Vol-filtered momentum
        gate = vol_regime_filter(w_rets, VOL_LOOKBACK, VOL_THRESHOLD)
        pos_vf = pos_mom * gate
        res_vf = backtest_positions(w_rets, pos_vf)

        rows.append({
            "Window End": w_prices.index[-1],
            "Sharpe Momentum": sharpe_ratio_from_log_returns(res_mom.strategy_log_returns),
            "Sharpe Vol-Filtered": sharpe_ratio_from_log_returns(res_vf.strategy_log_returns),
        })

        i += STEP

    df = pd.DataFrame(rows).set_index("Window End")

    print("\nRolling Window Sharpe Summary\n")
    print(df.describe())

    # Plot comparison
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Sharpe Momentum"], label="Momentum (60d)")
    plt.plot(df.index, df["Sharpe Vol-Filtered"], label="Vol-Filtered Momentum")
    plt.axhline(0.0, linewidth=1)
    plt.title("Rolling 3-Year Sharpe Comparison")
    plt.ylabel("Sharpe")
    plt.xlabel("Window End Date")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()