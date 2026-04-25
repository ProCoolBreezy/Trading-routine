"""Entry / exit signal generation from the trading plan rules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

import indicators as ind


@dataclass
class EntrySignal:
    symbol: str
    price: float
    stop: float
    atr: float
    adx: float
    rsi: float
    ema50: float
    ema200: float
    reasons: list[str]


@dataclass
class ExitSignal:
    reason: str
    price: float


def check_daily_uptrend(daily: pd.DataFrame, cfg: dict[str, Any]) -> tuple[bool, dict]:
    """Daily filter: price > 50/200 EMA and higher highs/lows."""
    i = cfg["indicators"]
    close = daily["close"]
    ema50 = ind.ema(close, i["ema_fast"])
    ema200 = ind.ema(close, i["ema_slow"])
    price = float(close.iloc[-1])

    above_emas = price > float(ema50.iloc[-1]) and price > float(ema200.iloc[-1])
    structure = ind.is_uptrend_structure(close, lookback=20)

    return (above_emas and structure), {
        "price": price,
        "ema50": float(ema50.iloc[-1]),
        "ema200": float(ema200.iloc[-1]),
        "above_emas": above_emas,
        "structure_ok": structure,
    }


def check_4h_entry(
    four_h: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """4H setup: pullback to 50 EMA + MACD bullish cross + RSI>50 + ADX>25.

    Returns (should_enter, details). should_enter requires at least
    confluence.min_conditions to be true AND the ADX filter to pass
    (ADX is a non-negotiable range filter per the plan).
    """
    i = cfg["indicators"]
    c = cfg["confluence"]
    close = four_h["close"]
    high = four_h["high"]
    low = four_h["low"]
    vol = four_h["volume"]

    ema50 = ind.ema(close, i["ema_fast"])
    rsi_series = ind.rsi(close, i["rsi_period"])
    macd_line, signal_line, _ = ind.macd(
        close, i["macd_fast"], i["macd_slow"], i["macd_signal"]
    )
    adx_series = ind.adx(high, low, close, i["adx_period"])
    atr_series = ind.atr(high, low, close, i["atr_period"])

    price = float(close.iloc[-1])
    ema50_v = float(ema50.iloc[-1])
    rsi_v = float(rsi_series.iloc[-1])
    adx_v = float(adx_series.iloc[-1])
    atr_v = float(atr_series.iloc[-1])

    # Pullback: price within tolerance of the 50 EMA (from above).
    pullback = price >= ema50_v and (price - ema50_v) / ema50_v <= i["pullback_tolerance"]
    macd_cross = ind.macd_bullish_cross(macd_line, signal_line, within=3)
    rsi_ok = rsi_v > i["rsi_entry_min"]
    vol_ok = ind.volume_rising(vol, lookback=5)

    adx_ok = adx_v > i["adx_min"]  # Hard filter: no entry in ranges.

    conditions = {
        "pullback_to_50ema": pullback,
        "macd_bullish_cross": macd_cross,
        "rsi_gt_50": rsi_ok,
        "volume_rising": vol_ok,
    }
    confluence_count = sum(conditions.values())
    should_enter = adx_ok and confluence_count >= c["min_conditions"]

    reasons = [k for k, v in conditions.items() if v]
    return should_enter, {
        "price": price,
        "ema50": ema50_v,
        "rsi": rsi_v,
        "adx": adx_v,
        "atr": atr_v,
        "adx_ok": adx_ok,
        "confluence_count": confluence_count,
        "reasons": reasons,
        "swing_low": ind.swing_low(low, i.get("swing_low_lookback", 20)),
    }


def generate_entry_signal(
    symbol: str,
    daily: pd.DataFrame,
    four_h: pd.DataFrame,
    cfg: dict[str, Any],
) -> EntrySignal | None:
    daily_ok, daily_info = check_daily_uptrend(daily, cfg)
    if not daily_ok:
        return None

    entry_ok, h4 = check_4h_entry(four_h, cfg)
    if not entry_ok:
        return None

    price = h4["price"]
    atr_v = h4["atr"]
    swing = h4["swing_low"]

    # Stop = higher of (swing low, price - atr_mult * ATR).
    # Higher = tighter stop, smaller position — safer against flukes.
    atr_stop = price - cfg["risk"]["atr_stop_mult"] * atr_v
    stop = max(swing, atr_stop) if swing == swing else atr_stop  # swing NaN check
    if stop >= price:
        return None  # Invalid stop (below-entry guarantee failed).

    return EntrySignal(
        symbol=symbol,
        price=price,
        stop=stop,
        atr=atr_v,
        adx=h4["adx"],
        rsi=h4["rsi"],
        ema50=daily_info["ema50"],
        ema200=daily_info["ema200"],
        reasons=h4["reasons"],
    )


def generate_exit_signal(
    four_h: pd.DataFrame, cfg: dict[str, Any]
) -> ExitSignal | None:
    """Exit triggers on 4H: 50 EMA break, MACD bearish cross, or RSI < 40.

    Stop-hit and trailing-stop are handled by the broker's stop order and
    the manage loop's trail updates — not here.
    """
    i = cfg["indicators"]
    close = four_h["close"]
    high = four_h["high"]
    low = four_h["low"]

    price = float(close.iloc[-1])
    ema50_v = float(ind.ema(close, i["ema_fast"]).iloc[-1])
    macd_line, signal_line, _ = ind.macd(
        close, i["macd_fast"], i["macd_slow"], i["macd_signal"]
    )
    rsi_v = float(ind.rsi(close, i["rsi_period"]).iloc[-1])

    if price < ema50_v:
        return ExitSignal(reason="close_below_50ema", price=price)
    if ind.macd_bearish_cross(macd_line, signal_line, within=2):
        return ExitSignal(reason="macd_bearish_cross", price=price)
    if rsi_v < i["rsi_exit_max"]:
        return ExitSignal(reason="rsi_below_40", price=price)
    return None


def trail_stop(
    entry: float,
    current_price: float,
    current_stop: float,
    four_h: pd.DataFrame,
    cfg: dict[str, Any],
) -> float:
    """Return new stop — max of (current, 20 EMA, price - atr_trail_mult*ATR).

    Stops only ratchet up, never down. Uses tighter ATR multiple in strong
    trends (ADX > adx_strong).
    """
    i = cfg["indicators"]
    t = cfg["targets"]
    close = four_h["close"]
    high = four_h["high"]
    low = four_h["low"]

    atr_v = float(ind.atr(high, low, close, i["atr_period"]).iloc[-1])
    ema20_v = float(ind.ema(close, i["ema_trail"]).iloc[-1])
    adx_v = float(ind.adx(high, low, close, i["adx_period"]).iloc[-1])

    mult = t["atr_trail_mult_strong"] if adx_v > t["adx_strong"] else t["atr_trail_mult"]
    atr_trail = current_price - mult * atr_v
    candidate = max(ema20_v, atr_trail)
    return max(current_stop, candidate)
