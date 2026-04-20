from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def atr(frame: pd.DataFrame, length: int) -> pd.Series:
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift(1)).abs()
    low_close = (frame["low"] - frame["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(length).mean()


def bollinger_bands(series: pd.Series, length: int, stddev: float) -> pd.DataFrame:
    basis = series.rolling(length).mean()
    deviation = series.rolling(length).std(ddof=0)
    return pd.DataFrame(
        {
            "basis": basis,
            "upper": basis + deviation * stddev,
            "lower": basis - deviation * stddev,
            "deviation": deviation,
        }
    )


def donchian_channel(frame: pd.DataFrame, length: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "donchian_upper": frame["high"].rolling(length).max(),
            "donchian_lower": frame["low"].rolling(length).min(),
        }
    )


def average_volume(series: pd.Series, length: int = 20) -> pd.Series:
    return series.rolling(length).mean()


def slope(series: pd.Series, lookback: int = 3) -> pd.Series:
    return series.diff(lookback)