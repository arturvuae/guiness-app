"""Context features: sessions, volatility regime, VWAP, higher-timeframe structure.

Higher-timeframe values reach a lower-timeframe bar only after the HTF bar has
closed: a 1h bar opening 10:00 is usable from the LTF bar that closes at or after
11:00. Alignment is done on close times with merge_asof (backward).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gbot.features.structure import StructureParams, detect_structure

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
       "quote_volume": "sum", "taker_buy_volume": "sum"}


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {k: v for k, v in AGG.items() if k in df}
    out = df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["close"])
    # drop a trailing bar that is not complete in the source data
    end = df.index[-1] + (df.index[1] - df.index[0])
    if len(out) and out.index[-1] + pd.Timedelta(rule) > end:
        out = out.iloc[:-1]
    return out


def align_htf(ltf_index: pd.DatetimeIndex, ltf_step: pd.Timedelta,
              htf: pd.DataFrame, htf_step: pd.Timedelta, prefix: str) -> pd.DataFrame:
    left = pd.DataFrame({"close_time": ltf_index + ltf_step})
    right = htf.add_prefix(prefix).copy()
    right["close_time"] = htf.index + htf_step
    merged = pd.merge_asof(left, right.sort_values("close_time"), on="close_time", direction="backward")
    merged.index = ltf_index
    return merged.drop(columns="close_time")


def session_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour + index.minute / 60
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow": index.dayofweek,
        "session": np.select([hour < 7, hour < 12, hour < 20], [0, 1, 2], 3),  # asia, london, ny, late
        "killzone": ((hour >= 7) & (hour < 10)) | ((hour >= 12) & (hour < 15)),
    }, index=index)


def vwap_features(df: pd.DataFrame, atr: np.ndarray) -> pd.DataFrame:
    day = df.index.normalize()
    pv = (df["quote_volume"]).groupby(day).cumsum()
    v = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    vwap = pv / v
    a = pd.Series(atr, index=df.index).replace(0, np.nan)
    regime = a / a.rolling(288, min_periods=50).median()
    return pd.DataFrame({"vwap_dev_atr": (df["close"] - vwap) / a, "atr_regime": regime,
                         "atr_bps": a / df["close"] * 1e4}, index=df.index)


def htf_features(ltf: pd.DataFrame, ltf_rule: str, base_1m: pd.DataFrame,
                 rules: tuple[str, ...] = ("15min", "1h", "4h")) -> pd.DataFrame:
    parts = []
    for rule in rules:
        htf = resample(base_1m, rule)
        st = detect_structure(htf, StructureParams(swing_n=2))
        cols = st[["trend", "dealing_range_pos", "dist_swing_high", "dist_swing_low"]]
        parts.append(align_htf(ltf.index, pd.Timedelta(ltf_rule), cols, pd.Timedelta(rule), f"{rule}_"))
    return pd.concat(parts, axis=1)
