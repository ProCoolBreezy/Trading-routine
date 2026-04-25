"""Risk manager: position sizing, drawdown halt, concurrent-trade cap."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


@dataclass
class Sizing:
    shares: int
    risk_dollars: float
    notional: float
    stop_distance: float


def position_size(
    equity: float, risk_pct: float, entry: float, stop: float
) -> Sizing:
    """Risk-based position sizing. Returns whole shares only."""
    if stop >= entry:
        raise ValueError(f"Stop {stop} must be below entry {entry} for a long.")
    risk_dollars = equity * risk_pct
    stop_distance = entry - stop
    raw_shares = risk_dollars / stop_distance
    shares = int(math.floor(raw_shares))
    return Sizing(
        shares=shares,
        risk_dollars=risk_dollars,
        notional=shares * entry,
        stop_distance=stop_distance,
    )


def max_trades_exceeded(open_positions: int, cfg: dict[str, Any]) -> bool:
    return open_positions >= cfg["risk"]["max_open_trades"]


def weekly_drawdown_halt(
    equity_history: list[tuple[datetime, float]], cfg: dict[str, Any]
) -> bool:
    """True if equity is down more than the configured weekly threshold.

    equity_history: list of (timestamp, equity) tuples, ascending.
    """
    if not equity_history:
        return False
    threshold = cfg["risk"]["weekly_drawdown_halt"]
    now = equity_history[-1][0]
    cutoff = now - timedelta(days=7)

    past = [(t, e) for t, e in equity_history if t <= cutoff]
    if not past:
        return False
    baseline = past[-1][1]
    current = equity_history[-1][1]
    if baseline <= 0:
        return False
    return (baseline - current) / baseline >= threshold


def validate_entry(
    equity: float,
    open_positions: int,
    equity_history: list[tuple[datetime, float]],
    cfg: dict[str, Any],
) -> str | None:
    """Return None if entry is allowed, else a reason string."""
    if max_trades_exceeded(open_positions, cfg):
        return f"max_open_trades ({cfg['risk']['max_open_trades']}) reached"
    if weekly_drawdown_halt(equity_history, cfg):
        return "weekly drawdown halt active"
    if equity <= 0:
        return "non-positive equity"
    return None


def partial_target(entry: float, stop: float, rr: float) -> float:
    """Price at which to scale out (1:1 RR by default)."""
    return entry + rr * (entry - stop)
