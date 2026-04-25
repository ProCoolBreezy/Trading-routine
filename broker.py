"""Alpaca broker wrapper. Defaults to paper trading."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import (
    ClosePositionRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopOrderRequest,
)


@dataclass
class Position:
    symbol: str
    qty: int
    avg_entry_price: float
    current_price: float
    unrealized_pl: float


class Broker:
    def __init__(self, api_key: str, secret_key: str, live: bool = False):
        self.live = live
        self.trading = TradingClient(api_key, secret_key, paper=not live)
        self.data = StockHistoricalDataClient(api_key, secret_key)

    # --- Account ---

    def equity(self) -> float:
        return float(self.trading.get_account().equity)

    def buying_power(self) -> float:
        return float(self.trading.get_account().buying_power)

    def is_market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)

    # --- Positions ---

    def positions(self) -> list[Position]:
        out = []
        for p in self.trading.get_all_positions():
            out.append(
                Position(
                    symbol=p.symbol,
                    qty=int(float(p.qty)),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    unrealized_pl=float(p.unrealized_pl),
                )
            )
        return out

    def get_position(self, symbol: str) -> Position | None:
        for p in self.positions():
            if p.symbol == symbol:
                return p
        return None

    # --- Market data ---

    def bars(
        self,
        symbol: str,
        timeframe: Literal["1Day", "4Hour", "1Hour"],
        lookback_days: int,
    ) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        tf = {
            "1Day": TimeFrame.Day,
            "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
            "1Hour": TimeFrame.Hour,
        }[timeframe]
        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=tf, start=start, end=end
        )
        resp = self.data.get_stock_bars(req)
        if symbol not in resp.data or not resp.data[symbol]:
            return pd.DataFrame()
        rows = [
            {
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in resp.data[symbol]
        ]
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df

    # --- Orders ---

    def submit_market_buy(self, symbol: str, qty: int) -> str:
        order = self.trading.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        return str(order.id)

    def submit_stop_loss(self, symbol: str, qty: int, stop_price: float) -> str:
        order = self.trading.submit_order(
            StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                stop_price=round(stop_price, 2),
            )
        )
        return str(order.id)

    def open_sell_stops(self, symbol: str) -> list:
        orders = self.trading.get_orders()
        return [
            o
            for o in orders
            if o.symbol == symbol
            and o.side == OrderSide.SELL
            and o.status in (OrderStatus.NEW, OrderStatus.ACCEPTED, OrderStatus.PENDING_NEW)
            and o.order_type.value == "stop"
        ]

    def cancel_order(self, order_id: str) -> None:
        self.trading.cancel_order_by_id(order_id)

    def replace_stop(self, symbol: str, qty: int, new_stop: float) -> str:
        """Cancel existing sell-stops for the symbol and place a fresh one."""
        for o in self.open_sell_stops(symbol):
            self.cancel_order(str(o.id))
        return self.submit_stop_loss(symbol, qty, new_stop)

    def close_position(self, symbol: str, qty: int | None = None) -> None:
        if qty is None:
            self.trading.close_position(symbol)
        else:
            self.trading.close_position(
                symbol, close_options=ClosePositionRequest(qty=str(qty))
            )
