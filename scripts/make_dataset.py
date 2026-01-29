from src.config import TICKERS, START_DATE, END_DATE, PRICE_FIELD, INTERVAL, DATA_DIR_RAW
from src.data_loader import get_price_data


def main():
    cache_path = f"{DATA_DIR_RAW}/prices_{'_'.join(TICKERS)}_{START_DATE}.csv"

    data = get_price_data(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        price_field=PRICE_FIELD,
        interval=INTERVAL,
        cache_path=cache_path,
        force_download=False,
    )

    print("Saved/Loaded:", cache_path)
    print("\nPrices head:\n", data.prices.head())
    print("\nReturns head:\n", data.log_returns.head())
    print("\nPrices shape:", data.prices.shape)
    print("Returns shape:", data.log_returns.shape)


if __name__ == "__main__":
    main()