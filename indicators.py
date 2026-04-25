"""Technical indicators computed directly on pandas Series to avoid TA-Lib.

All functions take plain pandas Series/DataFrames and return Series aligned
to the input index. NaN-prefix is preserved so callers can `.iloc[-1]` safely
once enough history exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_ = 100 - (100 / (1 + rs))
    # If there are no losses, RSI saturates at 100 (all gains); if no gains, at 0.
    rsi_ = rsi_.where(avg_loss != 0, 100.0)
    rsi_ = rsi_.where(avg_gain != 0, 0.0)
    return rsi_


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / period, adjust=False
    ).mean() / atr_.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / period, adjust=False
    ).mean() / atr_.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def swing_low(low: pd.Series, lookback: int = 20) -> float:
    """Lowest low over the last `lookback` bars (exclusive of current)."""
    if len(low) < 2:
        return float("nan")
    window = low.iloc[-(lookback + 1) : -1]
    return float(window.min()) if len(window) else float("nan")


def is_uptrend_structure(close: pd.Series, lookback: int = 20) -> bool:
    """Higher highs AND higher lows over two consecutive lookback windows."""
    if len(close) < 2 * lookback:
        return False
    recent = close.iloc[-lookback:]
    prior = close.iloc[-2 * lookback : -lookback]
    return bool(recent.max() > prior.max() and recent.min() > prior.min())


def macd_bullish_cross(macd_line: pd.Series, signal_line: pd.Series, within: int = 3) -> bool:
    """True if MACD crossed above signal within the last `within` bars."""
    if len(macd_line) < within + 1:
        return False
    hist = macd_line - signal_line
    recent = hist.iloc[-(within + 1) :]
    # Transition from <=0 to >0 anywhere in the window.
    return bool(((recent.shift(1) <= 0) & (recent > 0)).any())


def macd_bearish_cross(macd_line: pd.Series, signal_line: pd.Series, within: int = 3) -> bool:
    if len(macd_line) < within + 1:
        return False
    hist = macd_line - signal_line
    recent = hist.iloc[-(within + 1) :]
    return bool(((recent.shift(1) >= 0) & (recent < 0)).any())


def volume_rising(volume: pd.Series, lookback: int = 5) -> bool:
    """Recent average volume greater than prior average."""
    if len(volume) < 2 * lookback:
        return False
    recent = volume.iloc[-lookback:].mean()
    prior = volume.iloc[-2 * lookback : -lookback].mean()
    return bool(recent > prior)
