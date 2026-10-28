from __future__ import annotations

import math

import pandas as pd


def summarize(equity: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    equity = equity.dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0]
    daily = equity.pct_change().dropna()
    drawdown = equity / equity.cummax() - 1
    out = {
        "start": equity.index[0].date(),
        "end": equity.index[-1].date(),
        "years": round(years, 1),
        "CAGR %": round((total ** (1 / years) - 1) * 100, 2) if years > 0 else float("nan"),
        "max drawdown %": round(drawdown.min() * 100, 2),
        "Sharpe": round(daily.mean() / daily.std() * math.sqrt(252), 2) if daily.std() > 0 else float("nan"),
        "final equity": round(equity.iloc[-1], 2),
    }
    if trades is not None and len(trades):
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        out.update({
            "trades": len(trades),
            "win rate %": round(len(wins) / len(trades) * 100, 1),
            "avg win %": round(wins["ret"].mean() * 100, 2) if len(wins) else 0.0,
            "avg loss %": round(losses["ret"].mean() * 100, 2) if len(losses) else 0.0,
            "profit factor": round(wins["pnl"].sum() / -losses["pnl"].sum(), 2) if losses["pnl"].sum() < 0 else float("inf"),
        })
    return out


def buy_and_hold(prices: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, initial: float) -> pd.Series:
    """Equal-weight buy-and-hold of every ticker that has data on the first date."""
    closes = pd.DataFrame({t: df["Close"] for t, df in prices.items()}).reindex(dates).ffill()
    closes = closes.loc[:, closes.iloc[0].notna()]
    return (closes / closes.iloc[0]).mean(axis=1) * initial
