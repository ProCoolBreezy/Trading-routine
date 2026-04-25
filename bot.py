"""Trading bot entrypoint.

Commands:
    python bot.py scan      # Scan watchlist and place entries for valid setups.
    python bot.py manage    # Update trailing stops and exit on signals.
    python bot.py report    # Print performance stats.
    python bot.py snapshot  # Record account equity (run daily via cron).

Recommended cron (market hours, paper or live):
    */60 9-16 * * 1-5  cd /path/to/Trading-routine && python bot.py manage
    5    15   * * 1-5  cd /path/to/Trading-routine && python bot.py scan
    30   16   * * 1-5  cd /path/to/Trading-routine && python bot.py snapshot

SAFETY: Paper trading by default. Set LIVE_TRADING=true in .env to trade real
money. The bot will print a warning banner when live.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import config as cfg_mod
import journal
import risk
import strategy
from broker import Broker

log = logging.getLogger("bot")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _warn_if_live(settings) -> None:
    if settings.live_trading:
        print("=" * 64)
        print("  LIVE TRADING MODE — real money. Ctrl-C within 5s to abort.")
        print("=" * 64)
        import time

        time.sleep(5)


def scan(settings, broker: Broker) -> None:
    """Scan watchlist for entries and place orders."""
    if not broker.is_market_open():
        log.info("market closed — scan aborted")
        return

    equity = broker.equity()
    open_positions = broker.positions()
    hist = journal.equity_history()

    block = risk.validate_entry(equity, len(open_positions), hist, settings.raw)
    if block:
        log.info("entry blocked: %s", block)
        return

    held_symbols = {p.symbol for p in open_positions}
    slots_left = settings["risk"]["max_open_trades"] - len(open_positions)
    if slots_left <= 0:
        log.info("no position slots available")
        return

    mode = "live" if settings.live_trading else "paper"
    candidates: list[strategy.EntrySignal] = []

    for symbol in settings["watchlist"]:
        if symbol in held_symbols:
            continue
        try:
            daily = broker.bars(symbol, "1Day", settings["data"]["daily_lookback_days"])
            four_h = broker.bars(
                symbol, "4Hour", settings["data"]["fourhour_lookback_days"]
            )
        except Exception as e:
            log.warning("data fetch failed for %s: %s", symbol, e)
            continue
        if daily.empty or four_h.empty:
            log.debug("%s: insufficient bars", symbol)
            continue

        signal = strategy.generate_entry_signal(symbol, daily, four_h, settings.raw)
        if signal:
            candidates.append(signal)
            journal.log_signal(
                symbol,
                "entry",
                json.dumps(
                    {
                        "price": signal.price,
                        "stop": signal.stop,
                        "atr": signal.atr,
                        "adx": signal.adx,
                        "rsi": signal.rsi,
                        "reasons": signal.reasons,
                    }
                ),
            )

    # Rank by ADX (trend strength) and take the best ones fitting the slot budget.
    candidates.sort(key=lambda s: s.adx, reverse=True)
    for signal in candidates[:slots_left]:
        sizing = risk.position_size(
            equity, settings["risk"]["per_trade_pct"], signal.price, signal.stop
        )
        if sizing.shares < 1:
            log.info(
                "%s: stop too wide for min-size (shares=%d, stop_dist=%.2f)",
                signal.symbol,
                sizing.shares,
                sizing.stop_distance,
            )
            continue
        if sizing.notional > broker.buying_power():
            log.info(
                "%s: insufficient buying power (need %.2f, have %.2f)",
                signal.symbol,
                sizing.notional,
                broker.buying_power(),
            )
            continue

        log.info(
            "ENTER %s shares=%d @ ~%.2f stop=%.2f risk=$%.2f reasons=%s",
            signal.symbol,
            sizing.shares,
            signal.price,
            signal.stop,
            sizing.risk_dollars,
            signal.reasons,
        )
        broker.submit_market_buy(signal.symbol, sizing.shares)
        broker.submit_stop_loss(signal.symbol, sizing.shares, signal.stop)
        journal.log_entry(
            symbol=signal.symbol,
            qty=sizing.shares,
            entry_price=signal.price,
            stop_price=signal.stop,
            reasons=signal.reasons,
            mode=mode,
        )


def manage(settings, broker: Broker) -> None:
    """Walk open positions: check exits, scale out at 1R, trail stop."""
    if not broker.is_market_open():
        log.info("market closed — manage aborted")
        return

    t = settings["targets"]
    for pos in broker.positions():
        trade = journal.find_open_trade(pos.symbol)
        if not trade:
            log.warning(
                "%s: position exists with no journal entry — managing without journal",
                pos.symbol,
            )

        try:
            four_h = broker.bars(
                pos.symbol, "4Hour", settings["data"]["fourhour_lookback_days"]
            )
        except Exception as e:
            log.warning("data fetch failed for %s: %s", pos.symbol, e)
            continue
        if four_h.empty:
            continue

        entry = pos.avg_entry_price
        current = pos.current_price

        # Exit signal? Close full position, cancel stops.
        exit_sig = strategy.generate_exit_signal(four_h, settings.raw)
        if exit_sig:
            log.info(
                "EXIT %s reason=%s price=%.2f", pos.symbol, exit_sig.reason, current
            )
            for o in broker.open_sell_stops(pos.symbol):
                broker.cancel_order(str(o.id))
            broker.close_position(pos.symbol)
            if trade:
                pnl = (current - entry) * pos.qty
                journal.log_exit(trade["id"], current, exit_sig.reason, pnl)
            continue

        # Partial scale-out at 1R if not yet done.
        if trade and not trade.get("partial_qty"):
            stop = float(trade["stop_price"])
            target = risk.partial_target(entry, stop, t["partial_rr"])
            if current >= target:
                partial_qty = max(1, int(pos.qty * t["partial_fraction"]))
                log.info(
                    "PARTIAL %s qty=%d @ %.2f (1R target %.2f)",
                    pos.symbol,
                    partial_qty,
                    current,
                    target,
                )
                broker.close_position(pos.symbol, qty=partial_qty)
                journal.log_partial(trade["id"], partial_qty, current)
                # After partial, move stop to breakeven on remainder.
                remaining_qty = pos.qty - partial_qty
                if remaining_qty > 0:
                    broker.replace_stop(pos.symbol, remaining_qty, entry)
                continue

        # Trail stop upward only.
        current_stop = (
            float(trade["stop_price"]) if trade else entry * 0.98
        )
        remaining_qty = pos.qty
        new_stop = strategy.trail_stop(
            entry, current, current_stop, four_h, settings.raw
        )
        if new_stop > current_stop and new_stop < current:
            log.info(
                "TRAIL %s stop %.2f -> %.2f", pos.symbol, current_stop, new_stop
            )
            broker.replace_stop(pos.symbol, remaining_qty, new_stop)
            if trade:
                with journal.conn() as c:
                    c.execute(
                        "UPDATE trades SET stop_price = ? WHERE id = ?",
                        (new_stop, trade["id"]),
                    )


def snapshot(settings, broker: Broker) -> None:
    equity = broker.equity()
    mode = "live" if settings.live_trading else "paper"
    journal.snapshot_equity(equity, mode)
    log.info("equity snapshot: $%.2f (%s)", equity, mode)


def report(settings, broker: Broker) -> None:
    s = journal.stats()
    eq = broker.equity()
    print(f"Mode:           {'LIVE' if settings.live_trading else 'paper'}")
    print(f"Account equity: ${eq:,.2f}")
    print(f"Trades:         {s.total}  (wins {s.wins} / losses {s.losses})")
    print(f"Win rate:       {s.win_rate:.1%}")
    print(f"Avg win:        ${s.avg_win:,.2f}")
    print(f"Avg loss:       ${s.avg_loss:,.2f}")
    print(f"Win/Loss ratio: {s.win_loss_ratio:.2f}")
    print(f"Total PnL:      ${s.total_pnl:,.2f}")


def main() -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(description="Trend-follow trading bot")
    parser.add_argument("command", choices=["scan", "manage", "report", "snapshot"])
    args = parser.parse_args()

    journal.init()
    settings = cfg_mod.load()
    _warn_if_live(settings)

    broker = Broker(settings.api_key, settings.secret_key, live=settings.live_trading)

    {
        "scan": scan,
        "manage": manage,
        "report": report,
        "snapshot": snapshot,
    }[args.command](settings, broker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
