# src/strategies.py
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------
# Signals (numbers)
# ----------------------------

def momentum_signal(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Momentum signal:
      s_t = P_t / P_{t-lookback} - 1

    Output: DataFrame of real-valued signals (same shape as prices).
    """
    return prices / prices.shift(lookback) - 1.0


def mean_reversion_zscore_signal(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Mean reversion signal via z-score:
      z_t = (P_t - MA_t) / SD_t

    Output: DataFrame of real-valued z-scores (same shape as prices).
    """
    ma = prices.rolling(lookback).mean()
    sd = prices.rolling(lookback).std(ddof=0)
    return (prices - ma) / sd


# ----------------------------
# Strategy rules (signal -> positions)
# ----------------------------

def sign_threshold_rule(signal: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Convert a signal into positions using sign + threshold:
      +1 if signal > threshold
      -1 if signal < -threshold
       0 otherwise
    """
    positions = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    positions = positions.mask(signal > threshold, 1.0)
    positions = positions.mask(signal < -threshold, -1.0)
    return positions


def zscore_entry_exit_rule(
    z: pd.DataFrame,
    entry_z: float = 1.0,
    exit_z: float = 0.2,
) -> pd.DataFrame:
    """
    Stateful entry/exit rule for z-scores:
      Enter short if z > +entry_z
      Enter long  if z < -entry_z
      Exit to flat if |z| < exit_z
      Otherwise hold previous position

    Output: positions in {-1,0,+1}
    """
    positions = pd.DataFrame(0.0, index=z.index, columns=z.columns)

    # We'll iterate safely with .iat to avoid pandas chained assignment warnings.
    for j, col in enumerate(z.columns):
        pos = 0.0
        for i in range(len(z.index)):
            zt = z.iat[i, j]
            if np.isnan(zt):
                positions.iat[i, j] = 0.0
                continue

            # Exit first
            if abs(zt) < exit_z:
                pos = 0.0
            else:
                if zt > entry_z:
                    pos = -1.0
                elif zt < -entry_z:
                    pos = 1.0
                # else: keep pos

            positions.iat[i, j] = pos

    return positions


def vol_regime_filter(
    log_returns: pd.DataFrame,
    vol_lookback: int = 20,
    vol_threshold: float = 0.02,
    ) -> pd.DataFrame:
    """
    Returns a DataFrame of 1/0 where 1 means "vol is low enough to trade",
    0 means "vol too high -> go flat".

    log_returns: daily log returns (same shape/index as prices columns)
    vol_lookback: rolling window (e.g., 20 trading days ~ 1 month)
    vol_threshold: daily vol threshold (e.g., 0.02 = 2% daily std)
    """
    # rolling daily volatility per asset
    rolling_vol = log_returns.rolling(vol_lookback).std()

    # gate: 1 if vol <= threshold else 0
    regime = (rolling_vol <= vol_threshold).astype(float)

    return regime

# ----------------------------
# Convenience wrappers (old behavior)
# ----------------------------

def momentum(prices: pd.DataFrame, lookback: int = 20, threshold: float = 0.0) -> pd.DataFrame:
    s = momentum_signal(prices, lookback=lookback)
    return sign_threshold_rule(s, threshold=threshold)


def mean_reversion_zscore(
    prices: pd.DataFrame,
    lookback: int = 20,
    entry_z: float = 1.0,
    exit_z: float = 0.2,
) -> pd.DataFrame:
    z = mean_reversion_zscore_signal(prices, lookback=lookback)
    return zscore_entry_exit_rule(z, entry_z=entry_z, exit_z=exit_z)


