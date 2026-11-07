import numpy as np
import pandas as pd
import pytest

from gbot.config import load_settings
from gbot.universe import (abdi_ranaldo_spread_bps, atr_bps, correlation_clusters,
                           depth_02, evaluate, intraday_stats, resample_ohlc, zero_volume_share)

S = load_settings()


def synthetic_1m(n=1440 * 3, vol_bps=5.0, spread_bps=0.0, seed=0, qv=200_000.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-11-01", periods=n, freq="1min", tz="UTC")
    mid = 100 * np.exp(np.cumsum(rng.normal(0, vol_bps / 1e4, n)))
    half = spread_bps / 2e4
    # trades print at bid or ask at random
    close = mid * (1 + half * rng.choice([-1, 1], n))
    high = np.maximum(close, mid * (1 + half)) * (1 + np.abs(rng.normal(0, vol_bps / 2e4, n)))
    low = np.minimum(close, mid * (1 - half)) * (1 - np.abs(rng.normal(0, vol_bps / 2e4, n)))
    opn = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": opn, "high": high, "low": low, "close": close,
                         "volume": 1.0, "quote_volume": qv}, index=idx)


def test_resample_and_atr_positive():
    k5 = resample_ohlc(synthetic_1m(), "5min")
    assert len(k5) == 1440 * 3 // 5
    assert atr_bps(k5).dropna().gt(0).all()


def test_spread_estimator_detects_wider_spread():
    narrow = abdi_ranaldo_spread_bps(synthetic_1m(spread_bps=1, seed=1))
    wide = abdi_ranaldo_spread_bps(synthetic_1m(spread_bps=20, seed=1))
    assert wide > narrow
    assert 10 < wide < 40


def test_zero_volume_share_counts_missing_minutes():
    df = synthetic_1m(n=1440)
    df = df.drop(df.index[10:154])
    assert zero_volume_share(df) == pytest.approx(0.10)


def test_depth_02_takes_thinner_side():
    ts = pd.Timestamp("2026-11-05", tz="UTC")
    d = pd.DataFrame({"timestamp": [ts] * 4, "percentage": [-0.2, 0.2, -1.0, 1.0],
                      "notional": [300.0, 200.0, 900.0, 800.0]})
    assert depth_02(d) == 200.0


def test_evaluate_tiers():
    st = intraday_stats(synthetic_1m(vol_bps=12), S.universe.spread_floor_bps)
    row = evaluate("SOLUSDT", 500e6, st, depth=5e6, s=S)
    assert row["cost_ratio"] == pytest.approx(row["cost_bps"] / row["atr5_bps"])
    calm = intraday_stats(synthetic_1m(vol_bps=0.3), S.universe.spread_floor_bps)
    row2 = evaluate("SOLUSDT", 500e6, calm, depth=5e6, s=S)
    assert row2["tier"] == "C" and "cost_ratio" in row2["reasons"]


def test_clusters_group_correlated_series():
    rng = np.random.default_rng(3)
    base = pd.Series(rng.normal(size=2000))
    rets = {"A": base, "B": base + rng.normal(0, 0.1, 2000), "C": pd.Series(rng.normal(size=2000))}
    cl = correlation_clusters(rets)
    assert cl["A"] == cl["B"] != cl["C"]


def test_missing_archive_day_is_not_a_gap():
    df = synthetic_1m(n=1440 * 3)
    df = df[df.index.normalize() != df.index[1440].normalize()]
    assert zero_volume_share(df) == 0.0


def test_min_stop_respects_cost_to_risk():
    st = intraday_stats(synthetic_1m(vol_bps=3), S.universe.spread_floor_bps)
    row = evaluate("SOLUSDT", 500e6, st, depth=5e6, s=S)
    assert row["min_stop_bps"] >= row["cost_bps"] / S.risk.max_cost_to_risk - 1e-9
