"""Historical data from Binance's public archive (data.binance.vision), USD-M futures.

Files are cached on disk as the original zips, so re-runs are free and the raw
archive stays auditable. Delisted symbols remain in the archive, which keeps
research free of survivorship bias.
"""
from __future__ import annotations

import io
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um"
LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
CACHE = Path(__file__).resolve().parents[3] / "data" / "raw"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

_session = requests.Session()


def _get(url: str, retries: int = 4) -> bytes | None:
    """GET with backoff. Returns None for 404 (file not published)."""
    for attempt in range(retries):
        try:
            r = _session.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    return None


def list_symbols(quote: str = "USDT") -> list[str]:
    """All USD-M perpetual symbols ever archived (including delisted)."""
    prefix = "data/futures/um/monthly/klines/"
    symbols: list[str] = []
    marker = ""
    while True:
        params = {"delimiter": "/", "prefix": prefix}
        if marker:
            params["marker"] = marker
        r = _session.get(LISTING, params=params, timeout=60)
        r.raise_for_status()
        xml = r.text
        found = re.findall(rf"<Prefix>{re.escape(prefix)}([^/<]+)/</Prefix>", xml)
        symbols.extend(found)
        nxt = re.search(r"<NextMarker>([^<]+)</NextMarker>", xml)
        if "<IsTruncated>true</IsTruncated>" not in xml or not nxt:
            break
        marker = nxt.group(1)
    # Perpetuals only: dated delivery contracts carry a _YYMMDD suffix.
    return sorted(s for s in set(symbols) if s.endswith(quote) and "_" not in s)


def _cached_zip(rel: str) -> bytes | None:
    path = CACHE / rel
    if path.exists():
        return path.read_bytes()
    missing = path.with_suffix(path.suffix + ".missing")
    if missing.exists():
        return None
    blob = _get(f"{BASE}/{rel}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if blob is None:
        missing.touch()
        return None
    path.write_bytes(blob)
    return blob


def _read_csv_zip(blob: bytes, columns: list[str] | None) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    first = raw[:64].decode(errors="ignore")
    has_header = not first[:1].isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    if columns is not None and not has_header:
        df.columns = columns
    return df


def _normalise_klines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["ignore", "close_time"], errors="ignore")
    ts = pd.to_numeric(df["open_time"])
    # Archive switched some files to microseconds; normalise to ms.
    ts = ts.where(ts < 10**14, ts // 1000)
    df["open_time"] = pd.to_datetime(ts, unit="ms", utc=True)
    for c in df.columns:
        if c != "open_time":
            df[c] = pd.to_numeric(df[c])
    return df.set_index("open_time").sort_index()


def monthly_klines(symbol: str, interval: str, month: str) -> pd.DataFrame | None:
    """month as 'YYYY-MM'."""
    rel = f"monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    blob = _cached_zip(rel)
    return None if blob is None else _normalise_klines(_read_csv_zip(blob, KLINE_COLUMNS))


def daily_klines(symbol: str, interval: str, day: date) -> pd.DataFrame | None:
    rel = f"daily/klines/{symbol}/{interval}/{symbol}-{interval}-{day:%Y-%m-%d}.zip"
    blob = _cached_zip(rel)
    return None if blob is None else _normalise_klines(_read_csv_zip(blob, KLINE_COLUMNS))


def daily_book_depth(symbol: str, day: date) -> pd.DataFrame | None:
    """Snapshots of cumulative depth at +-0.2%, 1%..5% from mid (notional in USDT)."""
    rel = f"daily/bookDepth/{symbol}/{symbol}-bookDepth-{day:%Y-%m-%d}.zip"
    blob = _cached_zip(rel)
    if blob is None:
        return None
    df = _read_csv_zip(blob, None)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def last_days(n: int, end: date | None = None) -> list[date]:
    """The n most recent complete UTC days (archive publishes with ~1 day lag)."""
    end = end or (date.today() - timedelta(days=2))
    return [end - timedelta(days=i) for i in range(n)][::-1]


def klines_range(symbol: str, interval: str, days: list[date]) -> pd.DataFrame | None:
    frames = [f for d in days if (f := daily_klines(symbol, interval, d)) is not None]
    if not frames:
        return None
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index()


def parallel_map(fn, items, workers: int = 16):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))
