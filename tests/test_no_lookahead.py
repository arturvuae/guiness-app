"""Features at bar t must not change when data after t is removed."""
import numpy as np
import pandas as pd
import pytest

from gbot.events import build_features, extract_events
from gbot.features.structure import detect_structure


def market(n=1440 * 2, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-11-01", periods=n, freq="1min", tz="UTC")
    # trending + mean-reverting pieces so swings, BOS and FVGs all occur
    drift = np.sin(np.arange(n) / 300) * 2e-4
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 8e-4, n)))
    opn = np.r_[close[0], close[:-1]]
    wig = np.abs(rng.normal(0, 6e-4, (2, n)))
    high = np.maximum(opn, close) * (1 + wig[0])
    low = np.minimum(opn, close) * (1 - wig[1])
    vol = rng.lognormal(3, 0.6, n)
    return pd.DataFrame({"open": opn, "high": high, "low": low, "close": close, "volume": vol,
                         "quote_volume": vol * close, "taker_buy_volume": vol * rng.uniform(0.3, 0.7, n)},
                        index=idx)


@pytest.mark.parametrize("tf", ["1min", "5min"])
def test_features_are_causal(tf):
    k = market()
    full = build_features(k, tf)
    for cut in (1500, 2222, 2700):
        part = build_features(k.iloc[:cut], tf)
        common = part.index[:-1]                     # last LTF bar may be incomplete after resample
        a = full.loc[common].astype(float)
        b = part.loc[common].astype(float)
        diff = (a - b).abs().where(~(a.isna() & b.isna()), 0)
        bad = diff.max()
        assert (bad.fillna(0) < 1e-9).all(), f"lookahead in {list(bad[bad > 1e-9].index)} at cut {cut}"


def test_detectors_fire():
    st = detect_structure(build_features(market(), "1min")[["open", "high", "low", "close", "volume"]])
    for col in ["bos_up", "bos_down", "sweep_high", "sweep_low", "fvg_bull_new", "fvg_bull_retest",
                "swing_high_confirmed"]:
        assert st[col].sum() > 0, col


def test_swing_recorded_at_confirmation():
    idx = pd.date_range("2026-11-01", periods=9, freq="1min", tz="UTC")
    h = np.array([1, 2, 3, 9, 3, 2, 1, 1, 1], float)
    df = pd.DataFrame({"open": h - .5, "high": h, "low": h - 1, "close": h - .2, "volume": 1.0}, index=idx)
    st = detect_structure(df)
    assert not st["swing_high_confirmed"].iloc[:6].any()
    assert st["swing_high_confirmed"].iloc[6]       # bar 3 + n(3)


def test_events_mirror_direction():
    f = build_features(market(), "5min")
    ev = extract_events(f, "TEST")
    assert set(ev["direction"]) <= {1, -1} and len(ev) > 0
    assert (ev.filter(regex="^ev_").sum(axis=1) == ev["n_triggers"]).all()
