from datetime import datetime, timezone

import pytest

from gbot.config import load_settings
from gbot.risk import AccountState, Position, Proposal, halt_status, kelly_fraction, size_trade

S = load_settings()
NOW = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)  # Wednesday


def fresh(equity=1000.0, **kw):
    return AccountState(equity, equity, equity, equity, **kw)


def test_base_risk_is_half_percent_including_costs():
    d = size_trade(Proposal("SOLUSDT", 1, 100.0, 99.0, 102.0), fresh(), S, now=NOW)
    assert d.allowed
    assert d.risk_usd == pytest.approx(5.0)
    # 1% stop + 0.08% costs -> ~463 USDT notional
    assert d.notional == pytest.approx(5.0 / (0.01 + S.costs.round_trip))


def test_tight_stop_rejected_when_costs_dominate():
    # 0.15% stop: costs (0.08%) are > 25% of the loss
    d = size_trade(Proposal("SOLUSDT", 1, 100.0, 99.85, 100.5), fresh(), S, now=NOW)
    assert not d.allowed and "costs_too_high_for_stop" in d.reasons


def test_leverage_cap():
    d = size_trade(Proposal("SOLUSDT", 1, 100.0, 99.7, 101.0), fresh(), S, now=NOW)
    assert d.notional <= 1000 * S.risk.max_leverage + 1e-9


def test_cluster_and_portfolio_budget():
    pos = [Position("BTCUSDT", 0, 5.0, 500), Position("ETHUSDT", 0, 5.0, 500)]
    d = size_trade(Proposal("SOLUSDT", 0, 100.0, 99.0, 102.0), fresh(positions=pos), S, now=NOW)
    assert not d.allowed and "risk_budget_exhausted" in d.reasons
    d2 = size_trade(Proposal("XRPUSDT", 3, 1.0, 0.99, 1.02), fresh(positions=pos), S, now=NOW)
    assert d2.allowed and d2.risk_usd == pytest.approx(5.0)


def test_no_edge_means_no_trade():
    d = size_trade(Proposal("SOLUSDT", 1, 100.0, 99.0, 101.0, p_win=0.45), fresh(), S, now=NOW)
    assert not d.allowed and d.reasons == ("no_edge",)


def test_kelly_capped_at_max_risk():
    d = size_trade(Proposal("SOLUSDT", 1, 100.0, 99.0, 103.0, p_win=0.9), fresh(), S, now=NOW)
    assert d.risk_usd == pytest.approx(1000 * S.risk.max_cluster_risk)  # cluster cap 1% binds


def test_kelly_formula():
    assert kelly_fraction(0.5, 2.0) == pytest.approx(0.25)


@pytest.mark.parametrize("eq,day,week,peak,reason", [
    (979, 1000, 1000, 1000, "daily_loss_limit"),
    (949, 960, 1000, 1000, "weekly_loss_limit"),
    (899, 899, 899, 1000, "max_drawdown"),
])
def test_halts(eq, day, week, peak, reason):
    st = AccountState(eq, peak, day, week)
    assert halt_status(st, S, NOW)[0] == reason
    assert not size_trade(Proposal("SOLUSDT", 1, 100, 99, 102), st, S, now=NOW).allowed


def test_max_positions():
    pos = [Position(f"X{i}USDT", i, 1.0, 50) for i in range(3)]
    d = size_trade(Proposal("SOLUSDT", 9, 100, 99, 102), fresh(positions=pos), S, now=NOW)
    assert d.reasons == ("max_positions",)
