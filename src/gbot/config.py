"""Typed settings loaded from config/settings.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.toml"


@dataclass(frozen=True)
class Account:
    capital: float = 1000.0
    exchange: str = "binance-usdm"


@dataclass(frozen=True)
class Risk:
    risk_per_trade: float = 0.005
    max_risk_per_trade: float = 0.01
    max_open_risk: float = 0.015
    max_cluster_risk: float = 0.01
    max_positions: int = 3
    max_leverage: float = 5.0
    daily_loss_limit: float = 0.02
    weekly_loss_limit: float = 0.05
    max_drawdown: float = 0.10
    max_cost_to_risk: float = 0.25
    news_blackout_minutes: int = 5


@dataclass(frozen=True)
class Costs:
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    assumed_entry: str = "maker"
    assumed_exit: str = "taker"
    exit_slippage: float = 0.0001
    default_min_notional: float = 5.0
    min_notional_overrides: dict[str, float] = field(default_factory=dict)

    def fee(self, side: str) -> float:
        return self.maker_fee if side == "maker" else self.taker_fee

    @property
    def round_trip(self) -> float:
        """Fees + exit slippage as a fraction of notional (excludes spread)."""
        return self.fee(self.assumed_entry) + self.fee(self.assumed_exit) + self.exit_slippage

    def min_notional(self, symbol: str) -> float:
        return self.min_notional_overrides.get(symbol, self.default_min_notional)


@dataclass(frozen=True)
class Universe:
    quote: str = "USDT"
    volume_lookback_month: str = ""
    min_daily_quote_volume_b: float = 30e6
    min_daily_quote_volume_a: float = 100e6
    min_history_days: int = 180
    intraday_days: int = 14
    depth_days: int = 3
    max_cost_ratio_a: float = 0.50
    max_cost_ratio_b: float = 0.75
    max_participation: float = 0.05
    min_depth_multiple: float = 20.0
    max_zero_volume_share: float = 0.02
    spread_floor_bps: float = 0.5


@dataclass(frozen=True)
class Settings:
    account: Account = field(default_factory=Account)
    risk: Risk = field(default_factory=Risk)
    costs: Costs = field(default_factory=Costs)
    universe: Universe = field(default_factory=Universe)


def _build(cls, raw: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys in [{cls.__name__.lower()}]: {sorted(unknown)}")
    return cls(**raw)


def load_settings(path: str | Path | None = None) -> Settings:
    with open(path or DEFAULT_PATH, "rb") as fh:
        raw = tomllib.load(fh)
    return Settings(
        account=_build(Account, raw.get("account", {})),
        risk=_build(Risk, raw.get("risk", {})),
        costs=_build(Costs, raw.get("costs", {})),
        universe=_build(Universe, raw.get("universe", {})),
    )
