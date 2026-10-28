"""Daily portfolio backtester for the level strategy.

Order of events on each trading day:
  1. Fill orders decided at yesterday's close at today's OPEN (exits first).
  2. Check protective stops against today's range; a gap below the stop
     fills at the open, not at the stop price.
  3. Mark the portfolio to market at the close.
  4. Read today's signals and queue orders for tomorrow's open.

Costs: slippage in basis points on every fill plus a per-share commission.
No leverage and no shorting: cash can never go negative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from .strategy import Params, compute_indicators, market_ok


@dataclass(frozen=True)
class Costs:
    slippage_bps: float = 5.0
    commission_per_share: float = 0.005
    min_commission: float = 1.0

    def commission(self, shares: int) -> float:
        return max(self.min_commission, shares * self.commission_per_share)


@dataclass
class Position:
    shares: int
    entry_price: float
    entry_date: pd.Timestamp
    stop: float


@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame
    params: Params
    costs: Costs = field(default_factory=Costs)


def run_backtest(
    prices: dict[str, pd.DataFrame],
    params: Params = Params(),
    costs: Costs = Costs(),
    initial_cash: float = 100_000.0,
    benchmark: pd.DataFrame | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> Result:
    ind = {t: compute_indicators(df, params) for t, df in prices.items()}
    regime = market_ok(benchmark, params)

    dates = sorted(set().union(*(df.index for df in prices.values())))
    dates = pd.DatetimeIndex(dates)
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]

    slip = costs.slippage_bps / 10_000
    cash = initial_cash
    positions: dict[str, Position] = {}
    last_close: dict[str, float] = {}
    pending_exits: set[str] = set()
    pending_entries: list[str] = []
    prev_day: dict[str, pd.Timestamp] = {}  # signal date of each pending entry
    trades: list[dict] = []
    equity_curve: list[tuple[pd.Timestamp, float]] = []
    equity = initial_cash

    def bar(t: str, d: pd.Timestamp):
        df = prices[t]
        return df.loc[d] if d in df.index else None

    def close_position(t: str, d: pd.Timestamp, raw_price: float, reason: str) -> None:
        nonlocal cash
        pos = positions.pop(t)
        fill = raw_price * (1 - slip)
        cash += pos.shares * fill - costs.commission(pos.shares)
        trades.append(
            dict(ticker=t, entry_date=pos.entry_date, exit_date=d, shares=pos.shares,
                 entry_price=pos.entry_price, exit_price=fill, reason=reason,
                 pnl=pos.shares * (fill - pos.entry_price) - 2 * costs.commission(pos.shares),
                 ret=fill / pos.entry_price - 1)
        )

    for d in dates:
        # 1a. exits at the open
        for t in sorted(pending_exits):
            b = bar(t, d)
            if t in positions and b is not None:
                close_position(t, d, b["Open"], "support break")
        pending_exits = {t for t in pending_exits if t in positions}

        # 1b. entries at the open, sized off equity at the previous close
        for t in pending_entries:
            b = bar(t, d)
            if b is None or t in positions or len(positions) >= params.max_positions:
                continue
            row = ind[t].loc[prev_day[t]]
            stop_dist = params.stop_atr * row["atr"]
            fill = b["Open"] * (1 + slip)
            if not stop_dist > 0 or fill <= stop_dist:
                continue
            shares = math.floor(min(
                equity * params.risk_per_trade / stop_dist,
                equity * params.max_position_pct / fill,
                (cash - costs.min_commission) / fill,
            ))
            if shares <= 0:
                continue
            cash -= shares * fill + costs.commission(shares)
            positions[t] = Position(shares, fill, d, fill - stop_dist)
        pending_entries = []

        # 2. protective stops
        for t in list(positions):
            b = bar(t, d)
            if b is None:
                continue
            pos = positions[t]
            if b["Open"] <= pos.stop and pos.entry_date != d:
                close_position(t, d, b["Open"], "stop (gap)")
            elif b["Low"] <= pos.stop:
                close_position(t, d, pos.stop, "stop")

        # 3. mark to market
        for t in positions:
            b = bar(t, d)
            if b is not None:
                last_close[t] = b["Close"]
        equity = cash + sum(p.shares * last_close.get(t, p.entry_price) for t, p in positions.items())
        equity_curve.append((d, equity))

        # 4. signals for tomorrow
        risk_on = True
        if regime is not None:
            risk_on = bool(regime.get(d, False))
        candidates = []
        prev_day.clear()
        for t, df_ind in ind.items():
            if d not in df_ind.index:
                continue
            row = df_ind.loc[d]
            if t in positions:
                if row["exit"]:
                    pending_exits.add(t)
            elif risk_on and row["entry"]:
                candidates.append((row["momentum"] if pd.notna(row["momentum"]) else -math.inf, t))
                prev_day[t] = d
        slots = params.max_positions - (len(positions) - len(pending_exits))
        candidates.sort(reverse=True)
        pending_entries = [t for _, t in candidates[:max(slots, 0)]]

    return Result(
        equity=pd.Series(dict(equity_curve), name="equity"),
        trades=pd.DataFrame(trades),
        params=params,
        costs=costs,
    )
