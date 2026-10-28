import numpy as np
import pandas as pd
import pytest

from levelbot.backtest import Costs, run_backtest
from levelbot.strategy import Params, compute_indicators

FREE = Costs(slippage_bps=0, commission_per_share=0, min_commission=0)
P = Params(entry_lookback=5, exit_lookback=3, atr_period=3, stop_atr=2.0, trend_sma=5,
           market_filter=False, rank_lookback=3, risk_per_trade=0.01, max_positions=5)


def frame(closes, opens=None, lows=None, highs=None, start="2020-01-01"):
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    highs = np.maximum(opens, closes) + 0.5 if highs is None else np.asarray(highs, dtype=float)
    lows = np.minimum(opens, closes) - 0.5 if lows is None else np.asarray(lows, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 1e6}, index=idx)


def test_levels_use_only_past_bars():
    df = frame([10, 11, 12, 13, 14, 15, 30])
    ind = compute_indicators(df, P)
    # resistance on the last day is the max high of the 5 bars before it, not including the 30 spike
    assert ind["resistance"].iloc[-1] == pytest.approx(df["High"].iloc[1:6].max())
    assert bool(ind["entry"].iloc[-1])


def test_changing_future_does_not_change_past_signals():
    base = frame(list(range(10, 40)))
    altered = base.copy()
    altered.iloc[-5:, :4] = altered.iloc[-5:, :4] * 0.3
    a = compute_indicators(base, P).iloc[:-5]
    b = compute_indicators(altered, P).iloc[:-5]
    pd.testing.assert_frame_equal(a, b)


def test_entry_fills_at_next_open_not_signal_close():
    closes = [10] * 8 + [12, 13, 13, 8]
    opens = [10] * 8 + [10, 12.5, 13, 13]
    df = frame(closes, opens)
    res = run_backtest({"X": df}, P, FREE)
    # breakout on the close of bar 8; must be bought at bar 9's open (12.5)
    t = res.trades.iloc[0]
    assert t["entry_date"] == df.index[9]
    assert t["entry_price"] == pytest.approx(12.5)


def test_gap_through_stop_fills_at_open():
    closes = [10] * 8 + [12, 12, 12, 6]
    opens = [10] * 8 + [10, 12, 12, 6]
    res = run_backtest({"X": frame(closes, opens)}, P, FREE)
    t = res.trades.iloc[0]
    assert t["reason"] == "stop (gap)"
    assert t["exit_price"] == pytest.approx(6.0)


def test_cash_never_negative_and_risk_capped():
    rng = np.random.default_rng(0)
    prices = {}
    for i in range(20):
        c = 50 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 400)))
        prices[f"S{i}"] = frame(c)
    res = run_backtest(prices, P, Costs())
    assert (res.equity > 0).all()
    tr = res.trades
    stopped = tr[tr["reason"] == "stop"]
    # a stop-out should cost roughly the risk budget, never far more
    assert (stopped["pnl"] > -0.02 * 100_000 * 2).all()


def test_costs_reduce_returns():
    rng = np.random.default_rng(1)
    prices = {f"S{i}": frame(50 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 400)))) for i in range(10)}
    free = run_backtest(prices, P, FREE).equity.iloc[-1]
    costly = run_backtest(prices, P, Costs(slippage_bps=20)).equity.iloc[-1]
    assert costly < free
