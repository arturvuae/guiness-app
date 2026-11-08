"""Wyckoff / VSA features, causal.

Trading range at bar t is the high/low of the previous `range_window` bars
(t excluded). Springs and upthrusts are tested against that range, so a bar can
never define the level it breaks. Volume uses real exchange volume and the
taker-buy split from Binance klines (order-flow delta).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WyckoffParams:
    range_window: int = 48        # bars defining the trading range (4h on M5)
    max_range_atr: float = 8.0    # range narrower than this many ATR counts as a TR
    z_window: int = 200           # bars for volume / spread z-scores
    climax_vol_z: float = 3.0
    climax_range_z: float = 2.0
    delta_window: int = 20


def _z(x: pd.Series, w: int) -> pd.Series:
    m = x.rolling(w, min_periods=w // 2).mean()
    s = x.rolling(w, min_periods=w // 2).std()
    return (x - m) / s.replace(0, np.nan)


def wyckoff_features(df: pd.DataFrame, atr: np.ndarray, p: WyckoffParams = WyckoffParams()) -> pd.DataFrame:
    a = pd.Series(atr, index=df.index).replace(0, np.nan)
    hi = df["high"].shift(1).rolling(p.range_window).max()
    lo = df["low"].shift(1).rolling(p.range_window).min()
    width_atr = (hi - lo) / a
    in_range = width_atr <= p.max_range_atr
    # consecutive bars spent inside a valid range
    grp = (~in_range).cumsum()
    bars_in_range = in_range.groupby(grp).cumsum().astype(float)

    log_vol = np.log1p(df["volume"])
    spread = (df["high"] - df["low"]) / a
    vol_z = _z(log_vol, p.z_window)
    range_z = _z(spread, p.z_window)

    buy = df["taker_buy_volume"] if "taker_buy_volume" in df else df["volume"] / 2
    delta = (2 * buy - df["volume"]) / df["volume"].replace(0, np.nan)
    cum_delta = (2 * buy - df["volume"]).rolling(p.delta_window).sum() / \
        df["volume"].rolling(p.delta_window).sum().replace(0, np.nan)
    price_chg = (df["close"] - df["close"].shift(p.delta_window)) / a

    spring = in_range & (df["low"] < lo) & (df["close"] > lo)
    upthrust = in_range & (df["high"] > hi) & (df["close"] < hi)
    at_low = df["low"] <= df["low"].rolling(50).min()
    at_high = df["high"] >= df["high"].rolling(50).max()
    climax = (vol_z >= p.climax_vol_z) & (range_z >= p.climax_range_z)

    return pd.DataFrame({
        "tr_width_atr": width_atr,
        "tr_pos": (df["close"] - lo) / (hi - lo).replace(0, np.nan),
        "bars_in_range": bars_in_range,
        "vol_z": vol_z,
        "range_z": range_z,
        "effort_vs_result": vol_z - range_z,      # high = lots of volume, little movement
        "delta": delta,
        "cum_delta": cum_delta,
        # +1: price up > 1 ATR on net selling (bearish), -1: price down on net buying
        "delta_divergence": np.where((price_chg.abs() > 1) & (np.sign(price_chg) != np.sign(cum_delta)),
                                     np.sign(price_chg), 0.0),
        "spring": spring.fillna(False),
        "upthrust": upthrust.fillna(False),
        "selling_climax": (climax & at_low & (df["close"] > df["low"] + 0.5 * (df["high"] - df["low"]))).fillna(False),
        "buying_climax": (climax & at_high & (df["close"] < df["high"] - 0.5 * (df["high"] - df["low"]))).fillna(False),
        "spring_vol_z": vol_z.where(spring),
    }, index=df.index)
