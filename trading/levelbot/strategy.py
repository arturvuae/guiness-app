"""Price-level breakout strategy (Donchian-style support/resistance).

Rules, all evaluated on the daily close and acted on at the NEXT day's open:

* Resistance level = highest high of the previous `entry_lookback` days.
  Buy when the close breaks above it, the stock is above its long-term
  moving average, and (optionally) the market itself is in an uptrend.
* Support level = lowest low of the previous `exit_lookback` days.
  Sell when the close breaks below it.
* A protective stop sits `stop_atr` ATRs below the entry price and is
  checked intraday.

Every indicator uses only data available at that day's close: levels are
computed from bars *before* today (shift(1)), so there is no look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Params:
    entry_lookback: int = 55      # days defining the resistance level
    exit_lookback: int = 20       # days defining the support level
    atr_period: int = 20
    stop_atr: float = 2.0         # initial stop distance in ATRs
    trend_sma: int = 200          # only buy stocks above this SMA
    market_filter: bool = True    # only buy when the benchmark is above its SMA
    rank_lookback: int = 126      # rank competing signals by ~6-month momentum
    risk_per_trade: float = 0.01  # fraction of equity lost if the stop is hit
    max_positions: int = 10
    max_position_pct: float = 0.20


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def compute_indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["resistance"] = df["High"].rolling(p.entry_lookback).max().shift(1)
    out["support"] = df["Low"].rolling(p.exit_lookback).min().shift(1)
    out["sma"] = df["Close"].rolling(p.trend_sma).mean()
    out["atr"] = atr(df, p.atr_period)
    out["momentum"] = df["Close"].pct_change(p.rank_lookback)
    out["entry"] = (df["Close"] > out["resistance"]) & (df["Close"] > out["sma"]) & out["atr"].notna()
    out["exit"] = df["Close"] < out["support"]
    return out


def market_ok(benchmark: pd.DataFrame | None, p: Params) -> pd.Series | None:
    """True on days the benchmark closes above its trend SMA (risk-on regime)."""
    if benchmark is None or not p.market_filter:
        return None
    return benchmark["Close"] > benchmark["Close"].rolling(p.trend_sma).mean()
