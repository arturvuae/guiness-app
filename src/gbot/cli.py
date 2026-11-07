"""Command line: `gbot universe` builds today's tradable universe report."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from gbot.config import load_settings
from gbot.universe import build_universe

REPORTS = Path(__file__).resolve().parents[2] / "reports"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gbot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    uni = sub.add_parser("universe", help="rank Binance USD-M perpetuals by tradability")
    uni.add_argument("--config", default=None)
    uni.add_argument("--symbols", nargs="*", help="limit to these symbols")
    args = ap.parse_args(argv)

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
