"""Liquidity filter: which instruments are cheap enough to trade on M1/M5.

Main metric is the cost ratio: round-trip cost (spread + fees + slippage) divided by
the median 5-minute ATR. A setup with a 1xATR(5m) stop pays this share of its
risk in costs before it can win anything. Since the risk manager rejects trades whose
costs exceed `max_cost_to_risk` of the loss, cost_ratio / max_cost_to_risk is the
smallest stop allowed, in ATR(5m) units: 0.5 -> stop >= 2 ATR(5m).

Pipeline, from cheap to expensive:
1. one monthly 1d file per symbol -> average daily quote volume, prefilter
2. history check (a 1d file exists `min_history_days` back)
3. `intraday_days` of 1m klines -> ATR(5m), 1m volume, zero-volume share, spread estimate
4. `depth_days` of bookDepth -> notional resting within +-0.2% of mid
5. classify A / B / C, then cluster tradable symbols by return correlation
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from gbot.config import Settings
from gbot.data import binance_vision as bv


# ---------------------------------------------------------------- metrics

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "quote_volume": "sum"}
    return df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["close"])


def atr_bps(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-free simple ATR, expressed in basis points of close."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean() / df["close"] * 1e4


def abdi_ranaldo_spread_bps(df: pd.DataFrame, window: str = "1D") -> float:
    """Close-high-low effective spread estimator (Abdi & Ranaldo, 2017).

    Estimated per window from consecutive bars, then the median across windows.
    Upward-biased when volatility dominates, so it is a conservative proxy until
    live bookTicker data replaces it.
    """
    c = np.log(df["close"])
    eta = (np.log(df["high"]) + np.log(df["low"])) / 2
    prod = (c - eta) * (c - eta.shift(-1))
    per_window = prod.resample(window).mean()
    s = np.sqrt(np.clip(4 * per_window, 0, None)).dropna()
    return float(s.median() * 1e4) if len(s) else float("nan")


def zero_volume_share(df: pd.DataFrame) -> float:
    """Share of minutes with no trades, within the days present in the data.

    Whole days absent from the archive are an archive gap, not a market gap.
    """
    if df.empty:
        return 1.0
    expected = int(df.index.normalize().nunique()) * 1440
    zero = int((df["volume"] <= 0).sum()) + max(expected - len(df), 0)
    return zero / expected


def depth_02(depth: pd.DataFrame) -> float:
    """Median over snapshots of min(bid, ask) notional within 0.2% of mid."""
    d = depth[depth["percentage"].abs().round(2) == 0.20]
    per_snap = d.pivot_table(index="timestamp", columns="percentage", values="notional")
    return float(per_snap.min(axis=1).median()) if not per_snap.empty else float("nan")


@dataclass
class IntradayStats:
    atr5_bps: float
    median_1m_quote_volume: float
    zero_volume_share: float
    spread_bps: float
    returns_5m: pd.Series


def intraday_stats(k1m: pd.DataFrame, spread_floor_bps: float) -> IntradayStats:
    k5 = resample_ohlc(k1m, "5min")
    return IntradayStats(
        atr5_bps=float(atr_bps(k5).median()),
        median_1m_quote_volume=float(k1m["quote_volume"].median()),
        zero_volume_share=zero_volume_share(k1m),
        spread_bps=max(abdi_ranaldo_spread_bps(k1m), spread_floor_bps),
        returns_5m=np.log(k5["close"]).diff().dropna(),
    )


# ---------------------------------------------------------------- classification

def evaluate(symbol: str, adv: float, st: IntradayStats, depth: float, s: Settings) -> dict:
    u, c, r, a = s.universe, s.costs, s.risk, s.account
    cost_bps = st.spread_bps + c.round_trip * 1e4
    cost_ratio = cost_bps / st.atr5_bps if st.atr5_bps > 0 else np.inf

    # Reference trade: the tightest stop the risk manager accepts here, i.e. at least
    # 1x ATR(5m) and wide enough that costs stay within max_cost_to_risk of the loss.
    stop = max(st.atr5_bps, cost_bps / r.max_cost_to_risk) / 1e4
    notional = min(a.capital * r.risk_per_trade / stop, a.capital * r.max_leverage) if stop > 0 else 0.0
    participation = notional / st.median_1m_quote_volume if st.median_1m_quote_volume > 0 else np.inf
    depth_multiple = depth / notional if notional > 0 and not np.isnan(depth) else np.nan

    reasons = []
    if participation > u.max_participation:
        reasons.append("participation")
    if not np.isnan(depth_multiple) and depth_multiple < u.min_depth_multiple:
        reasons.append("depth")
    if st.zero_volume_share > u.max_zero_volume_share:
        reasons.append("gaps")
    if notional < c.min_notional(symbol):
        reasons.append("min_notional")

    if not reasons and adv >= u.min_daily_quote_volume_a and cost_ratio <= u.max_cost_ratio_a:
        tier = "A"
    elif not reasons and adv >= u.min_daily_quote_volume_b and cost_ratio <= u.max_cost_ratio_b:
        tier = "B"
    else:
        tier = "C"
        if cost_ratio > u.max_cost_ratio_b:
            reasons.append("cost_ratio")
    return {
        "symbol": symbol, "tier": tier, "adv_usdt": adv,
        "atr5_bps": st.atr5_bps, "spread_bps": st.spread_bps, "cost_bps": cost_bps,
        "cost_ratio": cost_ratio, "min_stop_bps": stop * 1e4,
        "min_stop_atr5": stop * 1e4 / st.atr5_bps if st.atr5_bps > 0 else np.inf, "ref_notional": notional, "participation": participation,
        "depth02_usdt": depth, "depth_multiple": depth_multiple,
        "zero_volume_share": st.zero_volume_share, "reasons": ",".join(reasons),
    }


def correlation_clusters(returns: dict[str, pd.Series], threshold: float = 0.7) -> dict[str, int]:
    """Union-find on pairs with 5m return correlation above threshold."""
    names = list(returns)
    if not names:
        return {}
    corr = pd.DataFrame(returns).corr(min_periods=500).fillna(0).to_numpy()
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if corr[i, j] >= threshold:
                parent[find(i)] = find(j)
    roots = {}
    return {n: roots.setdefault(find(i), len(roots)) for i, n in enumerate(names)}


# ---------------------------------------------------------------- pipeline

def _previous_month(today: date) -> str:
    first = today.replace(day=1)
    return (first - timedelta(days=1)).strftime("%Y-%m")


def _month_back(month: str, days: int) -> str:
    y, m = map(int, month.split("-"))
    return (date(y, m, 1) - timedelta(days=days)).strftime("%Y-%m")


def build_universe(s: Settings, symbols: list[str] | None = None, log=print) -> pd.DataFrame:
    u = s.universe
    month = u.volume_lookback_month or _previous_month(date.today())
    symbols = symbols or bv.list_symbols(u.quote)
    log(f"{len(symbols)} archived {u.quote} perpetuals; volume month {month}")

    def adv_of(sym):
        df = bv.monthly_klines(sym, "1d", month)
        return sym, (float(df["quote_volume"].mean()) if df is not None and len(df) >= 20 else 0.0)

    advs = dict(bv.parallel_map(adv_of, symbols))
    liquid = [x for x, v in advs.items() if v >= u.min_daily_quote_volume_b]
    log(f"{len(liquid)} pass volume floor {u.min_daily_quote_volume_b:,.0f} USDT/day")

    old_month = _month_back(month, u.min_history_days)
    aged = [x for x, ok in zip(liquid, bv.parallel_map(
        lambda x: bv.monthly_klines(x, "1d", old_month) is not None, liquid)) if ok]
    log(f"{len(aged)} have >= {u.min_history_days} days of history")

    days = bv.last_days(u.intraday_days)
    depth_days = days[-u.depth_days:]

    def measure(sym):
        k1m = bv.klines_range(sym, "1m", days)
        if k1m is None or len(k1m) < 1440:
            return None
        st = intraday_stats(k1m, u.spread_floor_bps)
        snaps = [d for day in depth_days if (d := bv.daily_book_depth(sym, day)) is not None]
        depth = depth_02(pd.concat(snaps)) if snaps else float("nan")
        return sym, st, depth

    rows, returns = [], {}
    for res in bv.parallel_map(measure, aged, workers=8):
        if res is None:
            continue
        sym, st, depth = res
        row = evaluate(sym, advs[sym], st, depth, s)
        rows.append(row)
        if row["tier"] in ("A", "B"):
            returns[sym] = st.returns_5m

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    clusters = correlation_clusters(returns)
    out["cluster"] = out["symbol"].map(clusters).astype("Int64")
    out = out.sort_values(["tier", "cost_ratio"]).reset_index(drop=True)
    log(f"tiers: {out['tier'].value_counts().to_dict()}")
    return out
