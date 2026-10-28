"""Daily order runner for Alpaca (https://alpaca.markets).

Run once per day after the US close (e.g. 16:30 New York time). It
downloads fresh daily bars, applies the same rules as the backtest, and
queues market orders for the next open. Each new position gets a GTC
stop-loss attached (Alpaca "OTO" order), so the stop is held by the broker
even if this script never runs again.

Safety:
* Paper account unless --live AND the environment has LEVELBOT_LIVE=yes.
* Without --execute it only prints the plan.
* Kill switch: once equity falls --max-drawdown below its recorded peak,
  no new positions are opened (exits and stops still work) until you
  delete the state file after reviewing what went wrong.
* Refuses to trade on stale data.
"""
from __future__ import annotations

import json
import math
import os
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import data
from .strategy import Params, compute_indicators, market_ok

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
MAX_DATA_AGE_DAYS = 4


@dataclass
class Order:
    symbol: str
    side: str
    qty: int
    stop: float | None = None
    reason: str = ""


class Alpaca:
    def __init__(self, base_url: str, key: str, secret: str):
        self.base_url, self.key, self.secret = base_url, key, secret

    def _req(self, method: str, path: str, body: dict | None = None):
        req = urllib.request.Request(
            self.base_url + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret,
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
        return json.loads(raw) if raw else None

    def account(self) -> dict:
        return self._req("GET", "/v2/account")

    def positions(self) -> dict[str, int]:
        return {p["symbol"]: int(float(p["qty"])) for p in self._req("GET", "/v2/positions")}

    def open_orders(self) -> list[dict]:
        return self._req("GET", "/v2/orders?status=open&nested=true&limit=500")

    def cancel(self, order_id: str) -> None:
        self._req("DELETE", f"/v2/orders/{order_id}")

    def submit(self, o: Order) -> dict:
        body = {"symbol": o.symbol, "qty": str(o.qty), "side": o.side, "type": "market",
                "time_in_force": "gtc" if o.stop else "day"}
        if o.stop:
            body["order_class"] = "oto"
            body["stop_loss"] = {"stop_price": f"{o.stop:.2f}"}
        return self._req("POST", "/v2/orders", body)


def plan_orders(prices: dict[str, pd.DataFrame], bench: pd.DataFrame | None, p: Params,
                positions: dict[str, int], equity: float, cash: float,
                allow_entries: bool = True) -> list[Order]:
    """Same rules as the backtest, applied to the latest completed bar."""
    regime = market_ok(bench, p)
    risk_on = bool(regime.iloc[-1]) if regime is not None else True
    orders, candidates = [], []
    for t, df in prices.items():
        row = compute_indicators(df, p).iloc[-1]
        if positions.get(t, 0) > 0:
            if row["exit"]:
                orders.append(Order(t, "sell", positions[t], reason=f"close {df['Close'].iloc[-1]:.2f} < support {row['support']:.2f}"))
        elif row["entry"]:
            candidates.append((row["momentum"] if pd.notna(row["momentum"]) else -math.inf, t, row, df["Close"].iloc[-1]))

    if not (allow_entries and risk_on):
        return orders
    held_after = sum(1 for q in positions.values() if q > 0) - len(orders)
    slots = p.max_positions - held_after
    budget = cash
    for _, t, row, price in sorted(candidates, key=lambda c: c[0], reverse=True)[:max(slots, 0)]:
        stop_dist = p.stop_atr * row["atr"]
        est_price = price * 1.01  # headroom for an overnight gap up
        qty = math.floor(min(equity * p.risk_per_trade / stop_dist,
                             equity * p.max_position_pct / est_price,
                             budget / est_price))
        if qty <= 0 or price <= stop_dist:
            continue
        budget -= qty * est_price
        orders.append(Order(t, "buy", qty, stop=round(price - stop_dist, 2),
                            reason=f"close {price:.2f} > resistance {row['resistance']:.2f}"))
    return orders


def check_kill_switch(state_path: Path, equity: float, max_dd: float) -> bool:
    """Returns True when new entries are allowed."""
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("halted"):
        return False
    peak = max(state.get("peak_equity", equity), equity)
    state["peak_equity"] = peak
    if equity < peak * (1 - max_dd):
        state["halted"] = True
        print(f"!!! KILL SWITCH: equity {equity:,.0f} is more than {max_dd:.0%} below peak {peak:,.0f}. "
              f"No new entries until you review and delete {state_path}.")
    state_path.write_text(json.dumps(state))
    return not state.get("halted", False)


def run_live(args, p: Params) -> None:
    live = args.live and os.environ.get("LEVELBOT_LIVE") == "yes"
    if args.live and not live:
        raise SystemExit("--live also requires the environment variable LEVELBOT_LIVE=yes")
    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
    broker = Alpaca(LIVE_URL if live else PAPER_URL, key, secret)
    print(f"account: {'LIVE MONEY' if live else 'paper'}")

    tickers = sorted(f.stem.upper() for f in Path(args.data).glob("*.csv"))
    bench_t = args.benchmark.upper()
    start = (date.today() - timedelta(days=500)).isoformat()
    prices = data.download([t for t in tickers if t != bench_t] + [bench_t], start=start)
    bench = prices.pop(bench_t, None)
    if bench is None:
        raise SystemExit(f"could not download benchmark {bench_t}")
    last = bench.index[-1].date()
    if (date.today() - last).days > MAX_DATA_AGE_DAYS:
        raise SystemExit(f"latest bar is {last}; data looks stale, not trading")
    prices = {t: df for t, df in prices.items() if df.index[-1].date() == last}

    acct = broker.account()
    equity, cash = float(acct["equity"]), float(acct["cash"])
    positions = broker.positions()
    allow = check_kill_switch(Path(args.state), equity, args.max_drawdown)
    orders = plan_orders(prices, bench, p, positions, equity, cash, allow_entries=allow)

    print(f"bars as of {last}; equity {equity:,.2f}; cash {cash:,.2f}; {len(positions)} positions")
    for t, q in positions.items():
        if t not in prices:
            print(f"warning: holding {t} but it is not in the data folder; it is not managed")
    if not orders:
        print("no orders")
    for o in orders:
        stop = f" stop {o.stop:.2f}" if o.stop else ""
        print(f"  {o.side.upper():4} {o.qty:>6} {o.symbol:<6}{stop}   ({o.reason})")

    if not args.execute:
        print("\ndry run: nothing sent. Add --execute to place these orders.")
        return
    open_orders = broker.open_orders()
    for o in orders:
        if o.side == "sell":
            for oo in open_orders:  # cancel the resting stop before selling
                if oo["symbol"] == o.symbol:
                    broker.cancel(oo["id"])
        r = broker.submit(o)
        print(f"sent {o.side} {o.qty} {o.symbol}: {r.get('status')}")
