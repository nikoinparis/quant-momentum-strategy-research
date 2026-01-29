# scripts/run_backtests.py
from src.config import TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, DATA_DIR_RAW
from src.data_loader import get_price_data
from src.strategies import momentum, mean_reversion_zscore
from src.backtester import backtest_positions


def main():
    cache_path = f"{DATA_DIR_RAW}/prices_{'_'.join(TICKERS)}_{START_DATE}.csv"
    data = get_price_data(TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, cache_path=cache_path)

    mom_pos = momentum(data.prices, lookback=20)
    mr_pos = mean_reversion_zscore(data.prices, lookback=20, entry_z=1.0, exit_z=0.2)

    # basic cost example: 2 bps per unit turnover
    mom_res = backtest_positions(data.log_returns, mom_pos, transaction_cost_bps=2.0)
    mr_res = backtest_positions(data.log_returns, mr_pos, transaction_cost_bps=2.0)

    print("Momentum final equity:", float(mom_res.equity_curve.iloc[-1]))
    print("Mean reversion final equity:", float(mr_res.equity_curve.iloc[-1]))

    print("\nMomentum equity head:\n", mom_res.equity_curve.head())
    print("\nMean reversion equity head:\n", mr_res.equity_curve.head())


if __name__ == "__main__":
    main()