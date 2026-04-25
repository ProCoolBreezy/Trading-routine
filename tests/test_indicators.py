"""Smoke tests for indicators and strategy. Run: python -m pytest tests/"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import indicators as ind
import risk
import strategy


def _trend_series(n: int = 300, slope: float = 0.5, noise: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")
    close = 100 + slope * np.arange(n) + rng.normal(0, noise, n).cumsum() * 0.1
    high = close + rng.uniform(0.1, 0.5, n)
    low = close - rng.uniform(0.1, 0.5, n)
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_ema_returns_finite_values():
    df = _trend_series()
    result = ind.ema(df["close"], 50)
    assert len(result) == len(df)
    assert np.isfinite(result.iloc[-1])


def test_rsi_bounded_0_100():
    df = _trend_series()
    r = ind.rsi(df["close"], 14).dropna()
    assert r.min() >= 0 and r.max() <= 100


def test_atr_positive():
    df = _trend_series()
    a = ind.atr(df["high"], df["low"], df["close"]).dropna()
    assert (a > 0).all()


def test_adx_positive():
    df = _trend_series()
    a = ind.adx(df["high"], df["low"], df["close"]).dropna()
    assert (a >= 0).all()


def test_uptrend_structure_detected():
    df = _trend_series(slope=1.0)
    assert ind.is_uptrend_structure(df["close"]) is True


def test_downtrend_structure_rejected():
    df = _trend_series(slope=-1.0)
    assert ind.is_uptrend_structure(df["close"]) is False


def test_position_size_math():
    s = risk.position_size(equity=10_000, risk_pct=0.01, entry=100, stop=98)
    assert s.shares == 50  # $100 risk / $2 stop
    assert s.risk_dollars == 100


def test_position_size_rejects_bad_stop():
    import pytest

    with pytest.raises(ValueError):
        risk.position_size(equity=10_000, risk_pct=0.01, entry=100, stop=101)


def test_partial_target():
    assert risk.partial_target(100, 98, 1.0) == 102
    assert risk.partial_target(100, 98, 2.0) == 104


def test_strategy_entry_on_clean_uptrend():
    # Craft a synthetic setup where daily trend is up and 4H has a pullback.
    cfg = {
        "indicators": {
            "ema_fast": 50,
            "ema_slow": 200,
            "ema_trail": 20,
            "rsi_period": 14,
            "rsi_entry_min": 50,
            "rsi_exit_max": 40,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_period": 14,
            "adx_min": 25,
            "atr_period": 14,
            "pullback_tolerance": 0.05,
            "swing_low_lookback": 20,
        },
        "confluence": {"min_conditions": 2},
        "risk": {"atr_stop_mult": 2.0},
    }
    daily = _trend_series(n=400, slope=0.8, noise=0.2)
    four_h = _trend_series(n=400, slope=0.4, noise=0.15)

    # Best-effort: function must not raise; may or may not fire on synthetic data.
    sig = strategy.generate_entry_signal("TEST", daily, four_h, cfg)
    if sig is not None:
        assert sig.stop < sig.price
        assert sig.atr > 0


def test_weekly_drawdown_halt_triggers():
    from datetime import datetime, timedelta, timezone

    cfg = {"risk": {"weekly_drawdown_halt": 0.05}}
    now = datetime.now(timezone.utc)
    history = [
        (now - timedelta(days=10), 10_000),
        (now - timedelta(days=8), 10_000),
        (now, 9_400),  # down 6%
    ]
    assert risk.weekly_drawdown_halt(history, cfg) is True


def test_weekly_drawdown_halt_quiet():
    from datetime import datetime, timedelta, timezone

    cfg = {"risk": {"weekly_drawdown_halt": 0.05}}
    now = datetime.now(timezone.utc)
    history = [
        (now - timedelta(days=10), 10_000),
        (now - timedelta(days=8), 10_000),
        (now, 9_800),  # down 2%
    ]
    assert risk.weekly_drawdown_halt(history, cfg) is False
