"""SQLite trade journal + performance reporting."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "journal.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,                 -- 'long'
    qty INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    entry_time TEXT NOT NULL,
    stop_price REAL NOT NULL,
    partial_qty INTEGER DEFAULT 0,
    partial_price REAL,
    partial_time TEXT,
    exit_price REAL,
    exit_time TEXT,
    exit_reason TEXT,
    pnl REAL,
    entry_reasons TEXT,
    mode TEXT NOT NULL                  -- 'paper' or 'live'
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    details TEXT
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    mode TEXT NOT NULL
);
"""


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init() -> None:
    with conn() as c:
        c.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_signal(symbol: str, signal_type: str, details: str) -> None:
    with conn() as c:
        c.execute(
            "INSERT INTO signals (ts, symbol, signal_type, details) VALUES (?, ?, ?, ?)",
            (_now(), symbol, signal_type, details),
        )


def log_entry(
    symbol: str,
    qty: int,
    entry_price: float,
    stop_price: float,
    reasons: list[str],
    mode: str,
) -> int:
    with conn() as c:
        cur = c.execute(
            """INSERT INTO trades
               (symbol, side, qty, entry_price, entry_time, stop_price, entry_reasons, mode)
               VALUES (?, 'long', ?, ?, ?, ?, ?, ?)""",
            (symbol, qty, entry_price, _now(), stop_price, ",".join(reasons), mode),
        )
        return cur.lastrowid


def log_partial(trade_id: int, partial_qty: int, partial_price: float) -> None:
    with conn() as c:
        c.execute(
            """UPDATE trades
               SET partial_qty = ?, partial_price = ?, partial_time = ?
               WHERE id = ?""",
            (partial_qty, partial_price, _now(), trade_id),
        )


def log_exit(
    trade_id: int, exit_price: float, exit_reason: str, pnl: float
) -> None:
    with conn() as c:
        c.execute(
            """UPDATE trades
               SET exit_price = ?, exit_time = ?, exit_reason = ?, pnl = ?
               WHERE id = ?""",
            (exit_price, _now(), exit_reason, pnl, trade_id),
        )


def find_open_trade(symbol: str) -> dict | None:
    with conn() as c:
        row = c.execute(
            "SELECT * FROM trades WHERE symbol = ? AND exit_price IS NULL "
            "ORDER BY id DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        return dict(row) if row else None


def snapshot_equity(equity: float, mode: str) -> None:
    with conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO equity_snapshots (ts, equity, mode) VALUES (?, ?, ?)",
            (_now(), equity, mode),
        )


def equity_history() -> list[tuple[datetime, float]]:
    with conn() as c:
        rows = c.execute(
            "SELECT ts, equity FROM equity_snapshots ORDER BY ts ASC"
        ).fetchall()
    return [(datetime.fromisoformat(r["ts"]), float(r["equity"])) for r in rows]


@dataclass
class Stats:
    total: int
    wins: int
    losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    total_pnl: float


def stats() -> Stats:
    with conn() as c:
        rows = c.execute(
            "SELECT pnl FROM trades WHERE pnl IS NOT NULL"
        ).fetchall()
    pnls = [float(r["pnl"]) for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total = len(pnls)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return Stats(
        total=total,
        wins=len(wins),
        losses=len(losses),
        win_rate=(len(wins) / total) if total else 0.0,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=(avg_win / abs(avg_loss)) if avg_loss else 0.0,
        total_pnl=sum(pnls),
    )
