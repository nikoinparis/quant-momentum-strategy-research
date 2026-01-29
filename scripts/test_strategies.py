from src.config import TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, DATA_DIR_RAW
from src.data_loader import get_price_data
from src.strategies import momentum, mean_reversion_zscore

cache_path = f"{DATA_DIR_RAW}/prices_{'_'.join(TICKERS)}_{START_DATE}.csv"
data = get_price_data(TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, cache_path=cache_path)

mom_pos = momentum(data.prices, lookback=20)
mr_pos = mean_reversion_zscore(data.prices, lookback=20)

print("Momentum positions head:\n", mom_pos.head(30))
print("\nMean reversion positions head:\n", mr_pos.head(30))

print("\nMomentum unique values:", sorted(set(mom_pos.stack().dropna().unique())))
print("Mean reversion unique values:", sorted(set(mr_pos.stack().dropna().unique())))