# Veritas — Stock Price Predictor

A machine learning project that predicts the closing price **tomorrow, next
week, next month, next quarter, and next year** for:

- **S&P 500** (`^GSPC`) — an index of 500 big US companies
- **NVIDIA** (`NVDA`)
- **JPMorgan Chase** (`JPM`)

It downloads 10 years of real daily price data from Yahoo Finance, computes
"clues" (features) from the price history, trains two ML models, tests them
honestly on the most recent year, and prints a prediction for tomorrow.

## Quick start

You need [uv](https://docs.astral.sh/uv/) installed (it handles Python and
all packages for you). Then:

```bash
uv sync          # one-time setup: installs everything
uv run predict   # train + backtest + predict all 5 horizons, all 3 tickers
                 # (also saves a forecast chart per ticker to predictions/)
uv run predict NVDA   # or just one (works with any Yahoo ticker, e.g. AAPL)
uv run backtest  # replay the last year day-by-day and grade the models
uv run backtest 21    # same, but at the 1-month (21 trading day) horizon
uv run hindcast  # rewind 1 year, re-make the full forecast with only the
                 # data available then, and grade it against what happened
uv run hindcast 126   # rewind half a year instead (any number of days)
uv run pytest    # run the tests
```

A separate model is trained per horizon — "what's the return over the next
21 days?" is a different question from "over the next 1 day?". One trap to
know when reading long-horizon results: stocks drift upward, so a 1-year
model scoring ~100% direction accuracy mostly proves "markets usually go up
over a year," not that the model is clairvoyant. Also: long horizons mean
the next quarter/year prediction comes with a HUGE error bar — the printed
RMSE column is that error bar, so always quote it alongside the prediction.

## How it works (the 4 steps)

1. **Get data** ([data.py](src/veritas/data.py)) — download daily
   open/high/low/close/volume history from Yahoo Finance.

2. **Build features** ([features.py](src/veritas/features.py)) — for each
   day, compute things a trader might glance at: recent returns, volatility,
   distance from moving averages, RSI, unusual volume. Crucially, every
   feature for day *t* uses only information available **on or before** day
   *t* — letting the model peek at the future is the #1 way these projects
   go wrong, and there's a test that proves we don't.

3. **Train + test honestly** ([model.py](src/veritas/model.py)) — we train
   on the older data and test on the most recent year *in time order* (no
   shuffling!). Two models compete:
   - **Ridge regression** — draws the best straight-line relationship
   - **Random forest** — hundreds of small decision trees voting together

   ...and both must beat the **naive baseline**: "tomorrow's price = today's
   price." That baseline sounds dumb but is famously hard to beat.

4. **Predict** ([predict.py](src/veritas/predict.py)) — retrain the winning
   model on all the data and predict tomorrow's closing price.

## Why predict *returns* instead of prices?

A raw price chart drifts upward for years, so a model trained on prices
mostly learns "the number goes up." Instead we predict the daily **log
return** (≈ percent change), which looks the same in 2016 as in 2026.
Converting back is one line: `tomorrow = today × e^(predicted return)`.

## Reading the results

```
  model                             MSE     RMSE      MAE     R^2  direction  avg $ err
  naive (tomorrow = today)     6.05e-05  0.00778  0.00585  -0.009        n/a      39.50
  random_forest                5.92e-05  0.00770  0.00581   0.012      56.4%      39.25
```

- **MSE (mean squared error)** — the classic ML metric: average of the
  squared mistakes. Squaring punishes big misses extra hard. Lower is better.
- **RMSE** — the square root of MSE, which puts it back in readable units
  (0.0077 ≈ typically off by 0.77% per day).
- **MAE (mean absolute error)** — the average miss size, ignoring direction.
- **R²** — how much of the day-to-day wiggle the model explains. 1.0 is
  perfect, 0 is "no better than guessing the average," and negative means
  worse than that. Real daily-return R² values hover barely above 0 —
  anything like 0.3 would be suspicious (probably a data leak!).
- **direction** — how often the model called *up vs down* correctly.
  A coin flip gets 50%, so 56% on the S&P 500 is genuinely interesting;
  52% is basically noise.
- **avg $ err** — the same error translated into price units: on average,
  how many dollars (or index points) off the next-day price guess was.

## The backtest (the most convincing part)

`uv run backtest` replays the last trading year **one day at a time**: the
models are refit every 21 trading days using only data available up to that
morning, predict the next day, and the prediction is graded before moving
on. No peeking — there's a test that doubles future prices and proves
earlier predictions don't budge.

Besides the error metrics, it simulates a toy strategy — start with
$10,000, hold the asset only on days the model says UP — and compares it to
plain buy & hold. It also saves a chart per ticker to `backtest_results/`
showing the equity curves and a rolling direction-accuracy line vs the 50%
coin-flip mark. Watch out for one trap when reading direction accuracy:
markets drift up, so "always guess up" already scores ~55% in a good year.
The printed footer tells you that number so you can compare fairly.

## The honest part

Daily stock prices are *mostly random noise*. Our models beat the naive
baseline by only a hair — and for JPM, the naive baseline actually **won**.
That's not a failed project; it's a real result that matches what
economists call the **Efficient Market Hypothesis**: if tomorrow's price
were easy to predict, traders would have already traded on it today.

**This is a learning project. Do not trade real money with it.**

## Ideas to extend it

- Add more tickers, or features like day-of-month or gap-from-open
- Try predicting 5 days ahead instead of 1 (is it easier or harder?)
- Plot predictions vs reality with matplotlib
- Compare against an "always predict +0.04%" baseline (the market's average
  daily drift) — can anything beat *that*?
- Simulate a trading strategy: buy when the model says up, and see if you'd
  beat just holding the index (spoiler: probably not after trading fees!)

## Project layout

```
src/veritas/
├── config.py     # all the settings in one place
├── data.py       # downloads price history
├── features.py   # turns prices into model inputs
├── model.py      # trains, evaluates, picks the winner
└── predict.py    # ties it all together, prints the report
tests/            # proves the tricky parts are correct
```
