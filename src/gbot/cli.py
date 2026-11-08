"""Command line: `gbot universe` builds today's tradable universe report."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from gbot.config import load_settings
from gbot.data import binance_vision as bv
from gbot.universe import build_universe

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
EVENTS = ROOT / "data" / "events"
HORIZON = {"1min": 30, "5min": 12}


def _universe_symbols() -> list[str]:
    files = sorted(REPORTS.glob("universe-*.csv"))
    if not files:
        raise SystemExit("run `gbot universe` first or pass --symbols")
    df = pd.read_csv(files[-1])
    return df.loc[df["tier"].isin(["A", "B"]), "symbol"].tolist()


def cmd_events(args, s) -> None:
    from gbot.events import build_features, extract_events, forward_returns, summarise

    symbols = args.symbols or _universe_symbols()
    days = bv.last_days(args.days)
    cost_bps = s.universe.spread_floor_bps + s.costs.round_trip * 1e4
    pooled: dict[str, list] = {}
    out_dir = EVENTS / args.tf
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(sym):
        k1m = bv.klines_range(sym, "1m", days)
        if k1m is None or len(k1m) < 1440 * 3:
            return sym, None, None
        f = build_features(k1m, args.tf)
        ev = extract_events(f, sym)
        return sym, ev, forward_returns(f, ev, HORIZON[args.tf], cost_bps) if len(ev) else None

    total = 0
    for sym, ev, fr in bv.parallel_map(one, symbols, workers=4):
        if ev is None or fr is None:
            print(f"{sym}: no data")
            continue
        ev.to_parquet(out_dir / f"{sym}.parquet")
        total += len(ev)
        for k, v in fr.items():
            pooled.setdefault(k, []).append(v)
    summary = summarise(pooled)
    REPORTS.mkdir(exist_ok=True)
    summary.to_csv(REPORTS / f"event-preview-{args.tf}.csv", index=False, float_format="%.3f")
    print(f"{total} candidate events from {len(symbols)} symbols, {args.days} days, tf {args.tf}")
    print(f"forward {HORIZON[args.tf]} bars, net of {cost_bps:.1f} bps round-trip cost (exploratory)")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


def cmd_chart(args, s) -> None:
    from gbot.chart import plot
    from gbot.events import build_features

    day = pd.Timestamp(args.day, tz="UTC")
    lookback = bv.last_days(3, end=day.date())       # warm-up for indicators
    k1m = bv.klines_range(args.symbol, "1m", lookback)
    f = build_features(k1m, args.tf)
    start = day + pd.Timedelta(hours=args.start_hour)
    f = f.loc[start:start + pd.Timedelta(hours=args.hours)]
    path = plot(f, f"{args.symbol} {args.tf}  {start:%Y-%m-%d %H:%M} UTC +{args.hours}h",
                REPORTS / "charts" / f"{args.symbol}-{args.tf}-{start:%Y%m%d-%H}.png")
    print(path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gbot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    uni = sub.add_parser("universe", help="rank Binance USD-M perpetuals by tradability")
    uni.add_argument("--config", default=None)
    uni.add_argument("--symbols", nargs="*", help="limit to these symbols")
    ev = sub.add_parser("events", help="detect SMC/Wyckoff candidate events, save parquet + preview")
    ev.add_argument("--config", default=None)
    ev.add_argument("--symbols", nargs="*", help="default: tiers A and B of the latest universe")
    ev.add_argument("--tf", choices=["1min", "5min"], default="5min")
    ev.add_argument("--days", type=int, default=30)
    ch = sub.add_parser("chart", help="annotated chart of detections")
    ch.add_argument("symbol")
    ch.add_argument("--day", required=True, help="YYYY-MM-DD (UTC)")
    ch.add_argument("--tf", choices=["1min", "5min"], default="5min")
    ch.add_argument("--start-hour", type=int, default=0)
    ch.add_argument("--hours", type=int, default=24)
    ch.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "events":
        return cmd_events(args, load_settings(args.config))
    if args.cmd == "chart":
        return cmd_chart(args, load_settings(args.config))

    if args.cmd == "universe":
        s = load_settings(args.config)
        df = build_universe(s, symbols=args.symbols)
        REPORTS.mkdir(exist_ok=True)
        out = REPORTS / f"universe-{date.today():%Y-%m-%d}.csv"
        df.to_csv(out, index=False, float_format="%.4f")
        with pd.option_context("display.width", 200, "display.max_rows", 60):
            cols = ["symbol", "tier", "adv_usdt", "atr5_bps", "spread_bps",
                    "cost_bps", "cost_ratio", "depth_multiple", "cluster", "reasons"]
            print(df[cols].head(60).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
