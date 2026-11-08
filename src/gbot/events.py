"""Candidate events: bars where an SMC or Wyckoff trigger fires, with full context.

One row per (bar, direction). The model later decides which candidates have
positive expected value; this module only describes the situation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gbot.features.context import htf_features, resample, session_features, vwap_features
from gbot.features.structure import StructureParams, detect_structure
from gbot.features.wyckoff import WyckoffParams, wyckoff_features

LONG_TRIGGERS = ["sweep_low", "choch_up", "bos_up", "fvg_bull_retest", "ob_bull_retest",
                 "spring", "selling_climax"]
SHORT_TRIGGERS = ["sweep_high", "choch_down", "bos_down", "fvg_bear_retest", "ob_bear_retest",
                  "upthrust", "buying_climax"]
# names used for both directions after mirroring
GENERIC = ["sweep", "choch", "bos", "fvg_retest", "ob_retest", "spring", "climax"]


def build_features(k1m: pd.DataFrame, tf: str = "5min",
                   sp: StructureParams = StructureParams(),
                   wp: WyckoffParams = WyckoffParams()) -> pd.DataFrame:
    bars = k1m if tf == "1min" else resample(k1m, tf)
    st = detect_structure(bars, sp)
    wy = wyckoff_features(bars, st["atr"].to_numpy(), wp)
    ctx = pd.concat([session_features(bars.index), vwap_features(bars, st["atr"].to_numpy()),
                     htf_features(bars, tf, k1m)], axis=1)
    return pd.concat([bars[["open", "high", "low", "close", "volume"]], st, wy, ctx], axis=1)


def _mirror(f: pd.DataFrame, direction: int) -> pd.DataFrame:
    """Express direction-dependent features from the trade's side (+ = favourable)."""
    d = direction
    out = pd.DataFrame(index=f.index)
    out["trend_aligned"] = f["trend"] * d
    for rule in ("15min", "1h", "4h"):
        out[f"htf_{rule}_trend_aligned"] = f[f"{rule}_trend"] * d
        pos = f[f"{rule}_dealing_range_pos"]
        out[f"htf_{rule}_discount"] = (1 - pos) if d == 1 else pos   # 1 = deepest discount for longs
    out["discount"] = (1 - f["dealing_range_pos"]) if d == 1 else f["dealing_range_pos"]
    out["dist_target_liq"] = f["dist_eq_highs"] if d == 1 else f["dist_eq_lows"]
    out["dist_opposing_swing"] = f["dist_swing_high"] if d == 1 else f["dist_swing_low"]
    out["dist_support_fvg"] = f["dist_bull_fvg"] if d == 1 else f["dist_bear_fvg"]
    out["dist_support_ob"] = f["dist_bull_ob"] if d == 1 else f["dist_bear_ob"]
    out["delta_aligned"] = f["delta"] * d
    out["cum_delta_aligned"] = f["cum_delta"] * d
    out["delta_divergence_aligned"] = -f["delta_divergence"] * d
    out["tr_pos_aligned"] = (1 - f["tr_pos"]) if d == 1 else f["tr_pos"]
    out["vwap_dev_aligned"] = -f["vwap_dev_atr"] * d
    return out


SHARED = ["vol_z", "range_z", "effort_vs_result", "tr_width_atr", "bars_in_range",
          "bars_since_bos", "bars_since_sweep", "atr_bps", "atr_regime",
          "hour_sin", "hour_cos", "dow", "session", "killzone"]


def extract_events(f: pd.DataFrame, symbol: str) -> pd.DataFrame:
    frames = []
    for d, trig in ((1, LONG_TRIGGERS), (-1, SHORT_TRIGGERS)):
        flags = f[trig].astype(bool)
        mask = flags.any(axis=1)
        if not mask.any():
            continue
        ev = flags[mask].copy()
        ev.columns = [f"ev_{g}" for g in GENERIC]
        ev = ev.astype(np.int8)
        ev["n_triggers"] = ev.sum(axis=1)
        part = pd.concat([ev, _mirror(f[mask], d), f.loc[mask, SHARED],
                          f.loc[mask, ["close", "atr"]]], axis=1)
        part["direction"] = d
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    out.insert(0, "symbol", symbol)
    return out


def forward_returns(f: pd.DataFrame, events: pd.DataFrame, horizon: int, cost_bps: float) -> dict[str, np.ndarray]:
    """Exploratory only: direction-adjusted return after `horizon` bars, net of cost, in bps.

    Keys are event names plus baseline_long / baseline_short (every bar, one side).
    Real labels come from triple-barrier simulation in the next stage.
    """
    fwd = (f["close"].shift(-horizon) / f["close"] - 1) * 1e4
    res: dict[str, np.ndarray] = {}
    for g in GENERIC + ["any"]:
        sel = events if g == "any" else events[events[f"ev_{g}"] == 1]
        if sel.empty:
            continue
        r = fwd.reindex(sel.index).to_numpy() * sel["direction"].to_numpy() - cost_bps
        res[g] = r[~np.isnan(r)]
    base = fwd.dropna().to_numpy()
    res["baseline_long"] = base - cost_bps
    res["baseline_short"] = -base - cost_bps
    return res


def summarise(pooled: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    rows = []
    for k, parts in pooled.items():
        r = np.concatenate(parts) if parts else np.array([])
        if len(r) < 2:
            continue
        rows.append({"event": k, "n": len(r), "mean_net_bps": r.mean(), "median_net_bps": np.median(r),
                     "t_stat": r.mean() / (r.std(ddof=1) / np.sqrt(len(r))), "hit_rate": (r > 0).mean()})
    return pd.DataFrame(rows)
