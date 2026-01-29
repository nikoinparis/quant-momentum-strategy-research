# src/config.py

TICKERS = ["SPY"]          # start with one ticker for simplicity
START_DATE = "2015-01-01"
END_DATE = None            # None = up to today

PRICE_FIELD = "Adj Close"  # adjusted close accounts for splits/dividends
INTERVAL = "1d"

DATA_DIR_RAW = "data/raw"
DATA_DIR_PROCESSED = "data/processed"