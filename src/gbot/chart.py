"""Annotated candlestick chart to check detections by eye."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

UP, DOWN, INK = "#0f8a6a", "#c8402f", "#14203a"
MARKS = {
    "sweep_low": ("^", UP, "sweep"), "sweep_high": ("v", DOWN, "sweep"),
    "choch_up": ("P", UP, "CHoCH"), "choch_down": ("X", DOWN, "CHoCH"),
    "bos_up": ("2", UP, "BOS"), "bos_down": ("1", DOWN, "BOS"),
    "spring": ("*", UP, "spring"), "upthrust": ("*", DOWN, "upthrust"),
    "fvg_bull_retest": ("o", UP, "FVG retest"), "fvg_bear_retest": ("o", DOWN, "FVG retest"),
    "ob_bull_retest": ("s", UP, "OB retest"), "ob_bear_retest": ("s", DOWN, "OB retest"),
}


def plot(f: pd.DataFrame, title: str, path: Path) -> Path:
    n = len(f)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(16, 7), dpi=110)
    up = f["close"] >= f["open"]
    colors = np.where(up, UP, DOWN)
    ax.vlines(x, f["low"], f["high"], color=colors, linewidth=0.7)
    body_lo = np.minimum(f["open"], f["close"])
    body_h = np.maximum((f["close"] - f["open"]).abs(), f["close"].abs() * 1e-5)
    ax.bar(x, body_h, bottom=body_lo, color=colors, width=0.7, linewidth=0)

    # FVG zones from formation until 40 bars later (visual only)
    for i in np.flatnonzero(f["fvg_bull_new"].to_numpy()):
        if i >= 2:
            ax.add_patch(plt.Rectangle((i - 2, f["high"].iloc[i - 2]), 40,
                                       f["low"].iloc[i] - f["high"].iloc[i - 2], color=UP, alpha=0.08))
    for i in np.flatnonzero(f["fvg_bear_new"].to_numpy()):
        if i >= 2:
            ax.add_patch(plt.Rectangle((i - 2, f["high"].iloc[i]), 40,
                                       f["low"].iloc[i - 2] - f["high"].iloc[i], color=DOWN, alpha=0.08))

    pad = (f["high"] - f["low"]).median() * 1.5
    seen = set()
    for col, (m, colr, label) in MARKS.items():
        idx = np.flatnonzero(f[col].to_numpy(bool))
        if not len(idx):
            continue
        bullish = colr == UP
        y = f["low"].iloc[idx] - pad if bullish else f["high"].iloc[idx] + pad
        key = (label, bullish)
        ax.scatter(idx, y, marker=m, color=colr, s=46, zorder=5,
                   label=f"{label} {'long' if bullish else 'short'}" if key not in seen else None)
        seen.add(key)

    ticks = np.linspace(0, n - 1, 10).astype(int)
    ax.set_xticks(ticks, [f.index[i].strftime("%m-%d %H:%M") for i in ticks], fontsize=8)
    ax.set_xlim(-1, n + 1)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", fontsize=8, ncol=4, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
