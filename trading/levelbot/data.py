"""Load daily OHLCV data from a folder of CSVs, or download it with yfinance.

CSV format: Date,Open,High,Low,Close[,Adj Close],Volume. When an
"Adj Close" column is present, Open/High/Low/Close are scaled by
Adj Close / Close so splits and dividends don't create fake level breaks.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated()].sort_index()
    if "Adj Close" in df.columns:
        ratio = df["Adj Close"] / df["Close"]
        for c in ["Open", "High", "Low", "Close"]:
            df[c] = df[c] * ratio
    df = df[COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
    return df[(df["Close"] > 0) & (df["High"] >= df["Low"])]


def load_csv_dir(folder: str | Path) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(Path(folder).glob("*.csv")):
        df = pd.read_csv(path, index_col=0)
        df = clean(df)
        if len(df):
            out[path.stem.upper()] = df
    if not out:
        raise FileNotFoundError(f"no CSV files with data in {folder}")
    return out


def download(tickers: list[str], start: str = "2000-01-01", folder: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Download split/dividend-adjusted daily bars from Yahoo Finance."""
    import yfinance as yf

    out = {}
    for t in tickers:
        df = yf.download(t, start=start, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            print(f"warning: no data for {t}")
            continue
        df = clean(df)
        out[t.upper()] = df
        if folder is not None:
            Path(folder).mkdir(parents=True, exist_ok=True)
            df.to_csv(Path(folder) / f"{t.upper()}.csv", index_label="Date")
    return out
