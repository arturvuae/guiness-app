# levelbot: price-level breakout trading for US stocks

A small, testable automated system: backtest → paper trade → (maybe) live.

## The strategy

Daily bars. Decisions on the close, orders at the next open.

| Rule | Default |
|---|---|
| **Buy** when the close breaks above the **resistance level** (highest high of the prior 55 days) | `--entry 55` |
| ...and the stock is above its 200-day average | `--sma 200` |
| ...and SPY is above its 200-day average (market filter) | `--no-market-filter` to disable |
| **Sell** when the close breaks below the **support level** (lowest low of the prior 20 days) | `--exit 20` |
| **Stop-loss** 2 × ATR below entry, held at the broker | `--stop-atr 2` |
| **Size**: lose at most 1% of equity if stopped; max 20% per stock; max 10 positions | `--risk 0.01 --max-positions 10` |

It loses on most trades (~70%) and makes its money on a few large trends.
Expect long flat stretches.

## What to expect

Measured here on 87 large US stocks, Sep 2012 – Sep 2017, including costs (the only data reachable from the build environment):

| | CAGR | Max drawdown |
|---|---|---|
| Strategy, 2012–2015 (includes 200-day warm-up) | 1.2% | -16.9% |
| Strategy, 2015–2017 | 8.1% | -9.4% |
| Equal-weight buy & hold, same stocks | 14–19% | -11 to -12% |
| Parameter sweep, 27 settings | median 4.6% | median -19.4% |

In a steady bull market, buy-and-hold beats this. Breakout systems earn their keep
by sitting in cash during crashes (2008, 2020, 2022), which this sample doesn't
include. **Run it yourself on 20+ years before believing anything.**

Nothing here makes 30%/year reliably. Results also suffer from survivorship bias
(`universe.txt` lists today's survivors), so real results are lower than the backtest.

## Usage

```bash
cd trading
pip install -r requirements.txt
python -m pytest -q                       # 10 tests: no look-ahead, gap stops, costs, kill switch

# 1. Data: 25 years of adjusted daily bars from Yahoo
python -m levelbot download --tickers-file universe.txt --data data --start 2000-01-01

# 2. Backtest with an out-of-sample check
python -m levelbot backtest --data data --split 2015-01-01 --out run1

# 3. Robustness: a real edge works across many settings, not one
python -m levelbot sweep --data data
```

Decide your parameters on the in-sample period only, then look at out-of-sample **once**.
Re-tuning after peeking turns the out-of-sample check into another backtest.

### Paper trading (Alpaca)

1. Create a free account at alpaca.markets and generate **paper** API keys.
2. After each US close (~16:30 New York time):

```bash
export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...
python -m levelbot trade --data data              # prints the order plan only
python -m levelbot trade --data data --execute    # sends orders to the paper account
```

Schedule it with cron on a machine that's always on, e.g. `35 16 * * 1-5` in New York time.

### Real money

Only after 3–6 months of paper trading that matches the backtest's behavior:

```bash
LEVELBOT_LIVE=yes python -m levelbot trade --data data --live --execute
```

Start with money you can afford to lose entirely.

## Safety features

- Levels use only past bars; tests verify that changing future prices can't change past signals.
- Stops gapped through fill at the open price, not at the stop.
- Slippage (5 bps) and commissions on every fill.
- No leverage, no shorting.
- Each entry carries a broker-side GTC stop, so positions stay protected if the script stops running.
- **Kill switch**: past a 25% drawdown from peak (`--max-drawdown`), no new entries until you review and delete `levelbot_state.json`.
- Refuses to trade if the latest data is more than 4 days old.
- Dry-run by default, paper by default; live money needs two separate opt-ins.

## Known limitations

- Entry stops are computed from the signal-day close; the actual fill is at the next open, so the real stop distance differs a little.
- Yahoo data is free and occasionally wrong. For real money, consider a paid data source.
- Positions you hold that aren't in the data folder are ignored.
- The live order path was unit-tested against the planner but hasn't yet been run against Alpaca's servers. Paper trade first.
- Not financial advice.
