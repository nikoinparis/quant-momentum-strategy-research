# scripts/grid_search_momentum.py
import pandas as pd

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

LOOKBACKS = [5, 10, 20, 40, 60, 120, 180]


def main():
    cache_path = f"{DATA_DIR_RAW}/prices_{'_'.join(TICKERS)}_{START_DATE}.csv"
    data = get_price_data(
        TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, cache_path=cache_path
    )

    rows = []

    for L in LOOKBACKS:
        positions = momentum(data.prices, lookback=L)
        res = backtest_positions(
            data.log_returns, positions, transaction_cost_bps=2.0
        )

        ann_ret = annualized_return_from_log_returns(res.strategy_log_returns)
        ann_vol = annualized_volatility_from_log_returns(res.strategy_log_returns)
        sharpe = sharpe_ratio_from_log_returns(res.strategy_log_returns)
        mdd = max_drawdown(res.equity_curve)

        rows.append(
            {
                "Lookback": L,
                "Annual Return": ann_ret,
                "Annual Vol": ann_vol,
                "Sharpe": sharpe,
                "Max Drawdown": mdd,
                "Final Equity": float(res.equity_curve.iloc[-1]),
            }
        )

    df = pd.DataFrame(rows).set_index("Lookback")
    pd.set_option("display.max_columns", 100)
    print(df.sort_index())


if __name__ == "__main__":
    main()