from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class PriceData:
    prices: pd.DataFrame
    log_returns: pd.DataFrame


def download_prices_yfinance(
    tickers: List[str],
    start: str,
    end: Optional[str] = None,
    price_field: str = "Adj Close",
    interval: str = "1d",
) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if df.empty:
        raise ValueError("No data returned. Check tickers/dates/network.")

    # Multiple tickers => MultiIndex columns: (field, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        prices = df[price_field].copy()
    else:
        # Single ticker => flat columns
        prices = df[[price_field]].copy()
        prices.columns = tickers

    prices = prices.sort_index().dropna(how="all")
    prices.index = pd.to_datetime(prices.index)
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna(how="all")


def save_prices_csv(prices: pd.DataFrame, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    prices.to_csv(filepath)


def load_prices_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0, parse_dates=True).sort_index()
    return df


def get_price_data(
    tickers: List[str],
    start: str,
    end: Optional[str],
    price_field: str,
    interval: str,
    cache_path: Optional[str] = None,
    force_download: bool = False,
) -> PriceData:
    # What we’re doing: cache downloaded data to CSV
    # Why: reproducibility + faster reruns + no dependency on network every run
    if cache_path and (not force_download) and os.path.exists(cache_path):
        prices = load_prices_csv(cache_path)
    else:
        prices = download_prices_yfinance(tickers, start, end, price_field, interval)
        if cache_path:
            save_prices_csv(prices, cache_path)

    log_returns = compute_log_returns(prices)
    return PriceData(prices=prices, log_returns=log_returns)