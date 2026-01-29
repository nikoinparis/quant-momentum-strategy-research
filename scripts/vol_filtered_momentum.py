# scripts/vol_filtered_momentum.py
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
VOL_THRESHOLD = 0.02  # daily vol threshold


def summarize(name: str, res):
    ann_ret = annualized_return_from_log_returns(res.strategy_log_returns)
    ann_vol = annualized_volatility_from_log_returns(res.strategy_log_returns)
    sharpe = sharpe_ratio_from_log_returns(res.strategy_log_returns)
    mdd = max_drawdown(res.equity_curve)
    final_eq = float(res.equity_curve.iloc[-1])
    print(
        f"{name:>22} | FinalEq={final_eq:.3f}  AnnRet={ann_ret:.3%}  "
        f"AnnVol={ann_vol:.3%}  Sharpe={sharpe:.3f}  MaxDD={mdd:.3%}"
    )


def main():
    cache_path = f"{DATA_DIR_RAW}/prices_{'_'.join(TICKERS)}_{START_DATE}.csv"
    data = get_price_data(
        TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, cache_path=cache_path
    )

    prices = data.prices.dropna()
    rets = data.log_returns.dropna()

    common_idx = prices.index.intersection(rets.index)
    prices = prices.loc[common_idx]
    rets = rets.loc[common_idx]

    # Base momentum positions
    base_pos = momentum(prices, lookback=MOM_LOOKBACK)

    # Vol regime gate (1 = trade, 0 = flat)
    gate = vol_regime_filter(rets, vol_lookback=VOL_LOOKBACK, vol_threshold=VOL_THRESHOLD)

    # Apply gate: when gate=0, positions go to 0
    filtered_pos = base_pos * gate

    # Backtest both
    base_res = backtest_positions(rets, base_pos, transaction_cost_bps=2.0)
    filt_res = backtest_positions(rets, filtered_pos, transaction_cost_bps=2.0)

    print("\nMomentum vs Vol-Filtered Momentum\n")
    summarize("Momentum (60d)", base_res)
    summarize("Vol-Filtered (60d)", filt_res)

    # How often are we “risk-off”?
    # (Average gate value is % of time we are allowed to trade)
    avg_gate = float(gate.mean().iloc[0])
    print(f"\nAvg gate (fraction trading): {avg_gate:.2%} of days\n")


if __name__ == "__main__":
    main()