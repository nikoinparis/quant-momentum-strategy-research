from src.config import TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, DATA_DIR_RAW
from src.data_loader import get_price_data
from src.strategies import (
    momentum_signal, mean_reversion_zscore_signal,
    sign_threshold_rule, zscore_entry_exit_rule,
)


def main():
    cache_path = f"{DATA_DIR_RAW}/prices_{'_'.join(TICKERS)}_{START_DATE}.csv"
    data = get_price_data(TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, cache_path=cache_path)

    mom_sig = momentum_signal(data.prices, lookback=20)
    mom_pos = sign_threshold_rule(mom_sig, threshold=0.0)

    mr_sig = mean_reversion_zscore_signal(data.prices, lookback=20)
    mr_pos = zscore_entry_exit_rule(mr_sig, entry_z=1.0, exit_z=0.2)

    print("\n=== Momentum SIGNAL (first 30 rows) ===\n", mom_sig.head(30))
    print("\n=== Momentum POSITIONS (first 30 rows) ===\n", mom_pos.head(30))

    print("\n=== Mean Reversion Z-SCORE SIGNAL (first 30 rows) ===\n", mr_sig.head(30))
    print("\n=== Mean Reversion POSITIONS (first 30 rows) ===\n", mr_pos.head(30))

    # Show first non-zero trade days for each strategy
    print("\nFirst momentum non-zero days:\n", mom_pos[mom_pos["SPY"] != 0].head(10))
    print("\nFirst mean-reversion non-zero days:\n", mr_pos[mr_pos["SPY"] != 0].head(10))


if __name__ == "__main__":
    main()