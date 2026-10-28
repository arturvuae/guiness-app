"""Command line entry point.

  python -m levelbot download --tickers-file universe.txt --data data/
  python -m levelbot backtest --data data/ [--split 2015-01-01]
  python -m levelbot sweep    --data data/
  python -m levelbot trade    --data data/ [--execute]   (Alpaca, paper by default)
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace

import pandas as pd

from . import data
from .backtest import Costs, run_backtest
from .metrics import buy_and_hold, summarize
from .strategy import Params


def equal_weight_index(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stand-in market benchmark when no index ETF is in the data folder."""
    rets = pd.DataFrame({t: df["Close"].pct_change() for t, df in prices.items()}).mean(axis=1).fillna(0)
    close = (1 + rets).cumprod()
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 0})


def load(args) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    prices = data.load_csv_dir(args.data)
    bench = prices.pop(args.benchmark.upper(), None)
    if bench is None:
        print(f"note: {args.benchmark} not in {args.data}; using an equal-weight index of the universe as the market filter")
        bench = equal_weight_index(prices)
    return prices, bench


def params_from(args) -> Params:
    return Params(
        entry_lookback=args.entry, exit_lookback=args.exit, stop_atr=args.stop_atr,
        trend_sma=args.sma, risk_per_trade=args.risk, max_positions=args.max_positions,
        market_filter=not args.no_market_filter,
    )


def report(name: str, stats: dict) -> None:
    print(f"\n== {name}")
    for k, v in stats.items():
        print(f"  {k:<16} {v}")


def cmd_backtest(args) -> None:
    prices, bench = load(args)
    p = params_from(args)
    res = run_backtest(prices, p, Costs(slippage_bps=args.slippage), args.cash, bench)
    periods = [("full period", None, None)]
    if args.split:
        periods = [("in-sample", None, args.split), ("out-of-sample", args.split, None)]
    for name, s, e in periods:
        eq = res.equity.loc[s:e]
        tr = res.trades
        if len(tr):
            tr = tr[(tr["exit_date"] >= eq.index[0]) & (tr["exit_date"] <= eq.index[-1])]
        report(f"strategy, {name}", summarize(eq, tr))
        report(f"equal-weight buy & hold, {name}", summarize(buy_and_hold(prices, eq.index, eq.iloc[0])))
    if args.out:
        res.equity.to_csv(f"{args.out}_equity.csv")
        res.trades.to_csv(f"{args.out}_trades.csv", index=False)
        print(f"\nwrote {args.out}_equity.csv and {args.out}_trades.csv")


def cmd_sweep(args) -> None:
    """Run a grid of level lookbacks. A real edge is good across the grid,
    not only at one hand-picked setting."""
    prices, bench = load(args)
    base = params_from(args)
    rows = []
    for entry, exit_, stop in itertools.product([20, 40, 55, 100], [10, 20, 40], [1.5, 2.0, 3.0]):
        if exit_ >= entry:
            continue
        p = replace(base, entry_lookback=entry, exit_lookback=exit_, stop_atr=stop)
        res = run_backtest(prices, p, Costs(slippage_bps=args.slippage), args.cash, bench)
        s = summarize(res.equity, res.trades)
        rows.append(dict(entry=entry, exit=exit_, stop_atr=stop, CAGR=s["CAGR %"],
                         maxDD=s["max drawdown %"], sharpe=s["Sharpe"], trades=s.get("trades", 0)))
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print(df.to_string(index=False))
    print(f"\nmedian CAGR {df['CAGR'].median():.2f}%  median maxDD {df['maxDD'].median():.2f}%  "
          f"settings with positive CAGR: {(df['CAGR'] > 0).sum()}/{len(df)}")


def cmd_download(args) -> None:
    tickers = [t.strip().upper() for t in open(args.tickers_file) if t.strip() and not t.startswith("#")]
    if args.benchmark.upper() not in tickers:
        tickers.append(args.benchmark.upper())
    got = data.download(tickers, start=args.start, folder=args.data)
    print(f"saved {len(got)} tickers to {args.data}")


def cmd_trade(args) -> None:
    from .live import run_live
    run_live(args, params_from(args))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="levelbot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data", default="data")
        sp.add_argument("--benchmark", default="SPY")
        sp.add_argument("--entry", type=int, default=55, help="resistance lookback days")
        sp.add_argument("--exit", type=int, default=20, help="support lookback days")
        sp.add_argument("--stop-atr", type=float, default=2.0)
        sp.add_argument("--sma", type=int, default=200)
        sp.add_argument("--risk", type=float, default=0.01, help="equity risked per trade")
        sp.add_argument("--max-positions", type=int, default=10)
        sp.add_argument("--no-market-filter", action="store_true")
        sp.add_argument("--slippage", type=float, default=5.0, help="bps per fill")
        sp.add_argument("--cash", type=float, default=100_000.0)

    sp = sub.add_parser("download"); sp.add_argument("--tickers-file", required=True)
    sp.add_argument("--data", default="data"); sp.add_argument("--benchmark", default="SPY")
    sp.add_argument("--start", default="2000-01-01"); sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("backtest"); common(sp)
    sp.add_argument("--split", help="date separating in-sample from out-of-sample")
    sp.add_argument("--out", help="prefix for equity/trades CSV output"); sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("sweep"); common(sp); sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("trade"); common(sp)
    sp.add_argument("--execute", action="store_true", help="actually send orders (default: print plan only)")
    sp.add_argument("--live", action="store_true", help="real-money account; also requires LEVELBOT_LIVE=yes")
    sp.add_argument("--max-drawdown", type=float, default=0.25, help="kill switch: stop new entries past this drawdown")
    sp.add_argument("--state", default="levelbot_state.json")
    sp.set_defaults(func=cmd_trade)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
