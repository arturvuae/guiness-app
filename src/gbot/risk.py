"""Risk manager: the only hard constraint on the model. Pure functions, no I/O.

The model proposes (symbol, side, entry, stop, target, p_win). This module decides
whether the trade is allowed and how large it may be. The model cannot change
these limits; they come from config/settings.toml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from gbot.config import Settings


@dataclass(frozen=True)
class Position:
    symbol: str
    cluster: int | None
    risk_usd: float      # loss if the stop is hit, including costs
    notional: float


@dataclass
class AccountState:
    equity: float
    peak_equity: float
    day_start_equity: float
    week_start_equity: float
    positions: list[Position] = field(default_factory=list)
    halted_until: datetime | None = None


@dataclass(frozen=True)
class Proposal:
    symbol: str
    cluster: int | None
    entry: float
    stop: float
    target: float
    p_win: float | None = None    # calibrated model probability; None -> base risk


@dataclass(frozen=True)
class Decision:
    allowed: bool
    quantity: float = 0.0
    notional: float = 0.0
    risk_usd: float = 0.0
    reasons: tuple[str, ...] = ()


def kelly_fraction(p: float, reward_to_risk: float) -> float:
    """Full-Kelly fraction of equity to risk for a binary bet."""
    if reward_to_risk <= 0:
        return 0.0
    return p - (1 - p) / reward_to_risk


def halt_status(state: AccountState, s: Settings, now: datetime) -> tuple[str, datetime | None] | None:
    """Return (reason, resume_time) if trading must stop; resume_time None = human review."""
    r = s.risk
    if state.halted_until and now < state.halted_until:
        return "halted", state.halted_until
    if state.equity <= state.peak_equity * (1 - r.max_drawdown):
        return "max_drawdown", None
    if state.equity <= state.week_start_equity * (1 - r.weekly_loss_limit):
        days = (7 - now.weekday()) % 7 or 7
        monday = (now + timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
        return "weekly_loss_limit", monday
    if state.equity <= state.day_start_equity * (1 - r.daily_loss_limit):
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return "daily_loss_limit", tomorrow
    return None


def size_trade(p: Proposal, state: AccountState, s: Settings,
               now: datetime | None = None, in_news_blackout: bool = False) -> Decision:
    r, c = s.risk, s.costs
    now = now or datetime.now(timezone.utc)
    reasons: list[str] = []

    halt = halt_status(state, s, now)
    if halt:
        return Decision(False, reasons=(halt[0],))
    if in_news_blackout:
        return Decision(False, reasons=("news_blackout",))
    if len(state.positions) >= r.max_positions:
        return Decision(False, reasons=("max_positions",))

    stop_dist = abs(p.entry - p.stop) / p.entry
    reward = abs(p.target - p.entry) / p.entry
    if stop_dist <= 0 or reward <= 0:
        return Decision(False, reasons=("invalid_levels",))

    # Risk fraction: quarter Kelly from the model, capped; base risk without a model.
    if p.p_win is None:
        risk_frac = r.risk_per_trade
    else:
        rr_net = (reward - c.round_trip) / (stop_dist + c.round_trip)
        risk_frac = min(kelly_fraction(p.p_win, rr_net) / 4, r.max_risk_per_trade)
        if risk_frac <= 0:
            return Decision(False, reasons=("no_edge",))

    # Room left under portfolio and cluster caps.
    open_risk = sum(x.risk_usd for x in state.positions)
    cluster_risk = sum(x.risk_usd for x in state.positions
                       if p.cluster is not None and x.cluster == p.cluster)
    risk_usd = min(
        state.equity * risk_frac,
        state.equity * r.max_open_risk - open_risk,
        state.equity * r.max_cluster_risk - cluster_risk if p.cluster is not None else float("inf"),
    )
    if risk_usd <= 0:
        return Decision(False, reasons=("risk_budget_exhausted",))

    # Loss per unit of notional if stopped = distance + costs.
    loss_per_notional = stop_dist + c.round_trip
    notional = risk_usd / loss_per_notional
    open_notional = sum(x.notional for x in state.positions)
    max_notional = state.equity * r.max_leverage - open_notional
    if notional > max_notional:
        notional = max_notional
        reasons.append("capped_by_leverage")
    if notional < c.min_notional(p.symbol):
        return Decision(False, reasons=("below_min_notional",))

    cost_to_risk = c.round_trip / loss_per_notional
    if cost_to_risk > r.max_cost_to_risk:
        return Decision(False, reasons=("costs_too_high_for_stop",))

    risk_usd = notional * loss_per_notional
    return Decision(True, quantity=notional / p.entry, notional=notional,
                    risk_usd=risk_usd, reasons=tuple(reasons))
