"""Smart Money Concepts as causal detectors.

Every output at bar t uses only bars <= t. A swing point at bar i is confirmed
only at bar i + n (n bars to its right must exist), and it is recorded at the
confirmation bar, never at i. Tests in tests/test_no_lookahead.py enforce this.

Input: OHLCV DataFrame on one timeframe. Output: one row per bar with event
flags and state features. Distances are in ATR units of the same timeframe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StructureParams:
    swing_n: int = 3              # fractal: bars on each side
    atr_period: int = 14
    equal_tol_atr: float = 0.10   # two swing highs within this many ATR = equal highs
    fvg_min_atr: float = 0.30     # minimum gap size
    ob_lookback: int = 20         # bars searched back for the order block candle
    max_zones: int = 30           # active FVG/OB zones kept per side


def atr(df: pd.DataFrame, period: int) -> np.ndarray:
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()


def _swing_flags(h: np.ndarray, l: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean arrays: bar i is a swing high / low (needs bars i+1..i+n)."""
    size = len(h)
    sh = np.zeros(size, bool)
    sl = np.zeros(size, bool)
    for i in range(n, size - n):
        left_h, right_h = h[i - n:i], h[i + 1:i + n + 1]
        if h[i] > left_h.max() and h[i] >= right_h.max():
            sh[i] = True
        left_l, right_l = l[i - n:i], l[i + 1:i + n + 1]
        if l[i] < left_l.min() and l[i] <= right_l.min():
            sl[i] = True
    return sh, sl


def detect_structure(df: pd.DataFrame, p: StructureParams = StructureParams()) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = atr(df, p.atr_period)
    size = len(df)
    n = p.swing_n
    sh_flag, sl_flag = _swing_flags(h, l, n)

    out = {k: np.zeros(size, bool) for k in (
        "swing_high_confirmed", "swing_low_confirmed", "bos_up", "bos_down", "choch_up",
        "choch_down", "sweep_high", "sweep_low", "fvg_bull_new", "fvg_bear_new",
        "fvg_bull_retest", "fvg_bear_retest", "ob_bull_retest", "ob_bear_retest",
        "equal_highs", "equal_lows")}
    trend = np.zeros(size, np.int8)                     # +1 up, -1 down, 0 unknown
    feats = {k: np.full(size, np.nan) for k in (
        "dist_swing_high", "dist_swing_low", "dealing_range_pos",
        "dist_bull_fvg", "dist_bear_fvg", "dist_bull_ob", "dist_bear_ob",
        "dist_eq_highs", "dist_eq_lows", "bars_since_bos", "bars_since_sweep")}

    swing_highs: list[tuple[int, float]] = []          # (bar, price), confirmed only
    swing_lows: list[tuple[int, float]] = []
    sh_level = sl_level = np.nan                        # latest unbroken swing levels
    sh_bar = sl_bar = -1
    sh_swept = sl_swept = False
    state = 0
    last_bos = last_sweep = -10**9
    bull_fvg: list[list[float]] = []                    # [bottom, top, touched]
    bear_fvg: list[list[float]] = []
    bull_ob: list[list[float]] = []
    bear_ob: list[list[float]] = []
    eq_highs: list[float] = []                          # unswept equal-high pools
    eq_lows: list[float] = []

    for t in range(size):
        at = a[t] if a[t] > 0 else np.nan
        new_bull_ob = new_bear_ob = False

        # 1) swings confirmed at t (bar t - n)
        i = t - n
        if i >= 0 and sh_flag[i]:
            if swing_highs and abs(h[i] - swing_highs[-1][1]) <= p.equal_tol_atr * at:
                eq_highs.append(max(h[i], swing_highs[-1][1]))
                out["equal_highs"][t] = True
            swing_highs.append((i, h[i]))
            sh_level, sh_bar, sh_swept = h[i], i, False
            out["swing_high_confirmed"][t] = True
        if i >= 0 and sl_flag[i]:
            if swing_lows and abs(l[i] - swing_lows[-1][1]) <= p.equal_tol_atr * at:
                eq_lows.append(min(l[i], swing_lows[-1][1]))
                out["equal_lows"][t] = True
            swing_lows.append((i, l[i]))
            sl_level, sl_bar, sl_swept = l[i], i, False
            out["swing_low_confirmed"][t] = True

        # 2) structure breaks on close
        if not np.isnan(sh_level) and c[t] > sh_level:
            out["choch_up" if state == -1 else "bos_up"][t] = True
            state, last_bos = 1, t
            # order block: last bearish candle between the swing low and t
            for j in range(t - 1, max(t - p.ob_lookback, sl_bar, 0) - 1, -1):
                if c[j] < o[j]:
                    bull_ob.append([l[j], h[j], 0.0])
                    new_bull_ob = True
                    break
            sh_level = np.nan
        if not np.isnan(sl_level) and c[t] < sl_level:
            out["choch_down" if state == 1 else "bos_down"][t] = True
            state, last_bos = -1, t
            for j in range(t - 1, max(t - p.ob_lookback, sh_bar, 0) - 1, -1):
                if c[j] > o[j]:
                    bear_ob.append([l[j], h[j], 0.0])
                    new_bear_ob = True
                    break
            sl_level = np.nan

        # 3) liquidity sweeps: wick through, close back inside
        if not np.isnan(sh_level) and not sh_swept and h[t] > sh_level and c[t] <= sh_level:
            out["sweep_high"][t], sh_swept, last_sweep = True, True, t
        if not np.isnan(sl_level) and not sl_swept and l[t] < sl_level and c[t] >= sl_level:
            out["sweep_low"][t], sl_swept, last_sweep = True, True, t
        eq_highs = [x for x in eq_highs if h[t] <= x]
        eq_lows = [x for x in eq_lows if l[t] >= x]

        # 4) fair value gaps formed at t (bars t-2, t-1, t)
        if t >= 2 and not np.isnan(at):
            if l[t] > h[t - 2] and (l[t] - h[t - 2]) >= p.fvg_min_atr * at:
                bull_fvg.append([h[t - 2], l[t], 0.0])
                out["fvg_bull_new"][t] = True
            if h[t] < l[t - 2] and (l[t - 2] - h[t]) >= p.fvg_min_atr * at:
                bear_fvg.append([h[t], l[t - 2], 0.0])
                out["fvg_bear_new"][t] = True

        # 5) zone retests and invalidation (zones created this bar are skipped)
        def update(zones, bullish, key, created_now):
            keep = []
            for k, z in enumerate(zones):
                bottom, top, touched = z
                if created_now and k == len(zones) - 1:
                    keep.append(z)
                    continue
                if bullish and c[t] < bottom or (not bullish) and c[t] > top:
                    continue                                    # invalidated
                if not touched and (l[t] <= top if bullish else h[t] >= bottom):
                    out[key][t] = True
                    z[2] = 1.0
                keep.append(z)
            return keep[-p.max_zones:]

        bull_fvg = update(bull_fvg, True, "fvg_bull_retest", out["fvg_bull_new"][t])
        bear_fvg = update(bear_fvg, False, "fvg_bear_retest", out["fvg_bear_new"][t])
        bull_ob = update(bull_ob, True, "ob_bull_retest", new_bull_ob)
        bear_ob = update(bear_ob, False, "ob_bear_retest", new_bear_ob)

        # 6) state features
        trend[t] = state
        if not np.isnan(at):
            last_h = swing_highs[-1][1] if swing_highs else np.nan
            last_l = swing_lows[-1][1] if swing_lows else np.nan
            feats["dist_swing_high"][t] = (last_h - c[t]) / at
            feats["dist_swing_low"][t] = (c[t] - last_l) / at
            if last_h > last_l:
                feats["dealing_range_pos"][t] = (c[t] - last_l) / (last_h - last_l)
            below = [z[1] for z in bull_fvg if z[1] <= c[t]]
            above = [z[0] for z in bear_fvg if z[0] >= c[t]]
            feats["dist_bull_fvg"][t] = (c[t] - max(below)) / at if below else np.nan
            feats["dist_bear_fvg"][t] = (min(above) - c[t]) / at if above else np.nan
            below = [z[1] for z in bull_ob if z[1] <= c[t]]
            above = [z[0] for z in bear_ob if z[0] >= c[t]]
            feats["dist_bull_ob"][t] = (c[t] - max(below)) / at if below else np.nan
            feats["dist_bear_ob"][t] = (min(above) - c[t]) / at if above else np.nan
            up = [x for x in eq_highs if x >= c[t]]
            dn = [x for x in eq_lows if x <= c[t]]
            feats["dist_eq_highs"][t] = (min(up) - c[t]) / at if up else np.nan
            feats["dist_eq_lows"][t] = (c[t] - max(dn)) / at if dn else np.nan
        feats["bars_since_bos"][t] = t - last_bos if last_bos >= 0 else np.nan
        feats["bars_since_sweep"][t] = t - last_sweep if last_sweep >= 0 else np.nan

    res = pd.DataFrame(out, index=df.index)
    res["trend"] = trend
    for k, v in feats.items():
        res[k] = v
    res["atr"] = a
    return res
