import json

from levelbot.live import check_kill_switch, plan_orders
from levelbot.strategy import Params
from tests.test_backtest import frame

P = Params(entry_lookback=5, exit_lookback=3, atr_period=3, stop_atr=2.0, trend_sma=5,
           market_filter=False, rank_lookback=3, max_positions=2)


def test_plans_entry_with_stop_and_exit():
    breakout = frame([10] * 8 + [12])
    breakdown = frame([10] * 8 + [7])
    orders = plan_orders({"UP": breakout, "DOWN": breakdown}, None, P,
                         positions={"DOWN": 100}, equity=100_000, cash=50_000)
    by = {o.symbol: o for o in orders}
    assert by["DOWN"].side == "sell" and by["DOWN"].qty == 100
    assert by["UP"].side == "buy" and 0 < by["UP"].stop < 12


def test_respects_max_positions_and_cash():
    prices = {f"S{i}": frame([10] * 8 + [12 + i]) for i in range(5)}
    orders = plan_orders(prices, None, P, positions={}, equity=100_000, cash=1_000)
    assert len(orders) <= P.max_positions
    assert orders
    assert sum(o.qty * prices[o.symbol]["Close"].iloc[-1] for o in orders) <= 1_000


def test_no_entries_when_halted():
    orders = plan_orders({"UP": frame([10] * 8 + [12])}, None, P, {}, 100_000, 100_000, allow_entries=False)
    assert orders == []


def test_kill_switch_trips_and_stays_tripped(tmp_path):
    s = tmp_path / "state.json"
    assert check_kill_switch(s, 100_000, 0.25)
    assert check_kill_switch(s, 120_000, 0.25)
    assert not check_kill_switch(s, 89_000, 0.25)
    assert not check_kill_switch(s, 130_000, 0.25)  # stays halted until reviewed
    assert json.loads(s.read_text())["halted"]
