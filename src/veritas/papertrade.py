"""Paper trading: a live-style simulation over the last two weeks.

This is the honest cousin of the backtest. Instead of grading thousands of
past days at once, it walks the LAST ~10 trading days one at a time, exactly
as if we had been trading live:

    for each of the last 10 trading days:
        train the model on ONLY the data available before that day
        (retrained fresh every single day - zero peeking)
        predict tomorrow's price, and "trade" on it
        the next day, see what actually happened and mark the result

We track a pretend $10,000 portfolio that holds the stock on days the model
predicts UP and sits in cash otherwise, and compare it to simply buying and
holding. Because every prediction is made before its outcome exists, this
can't be secretly overfit - it's the closest thing to real forward testing
without waiting weeks of calendar time.

Usage:
    uv run papertrade            # all three tickers, last 2 weeks
    uv run papertrade NVDA       # just one ticker
    uv run papertrade NVDA 20    # one ticker, last 20 trading days (~1 month)
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from veritas.backtest import walk_forward_predictions
from veritas.config import TICKERS
from veritas.data import fetch_history
from veritas.features import build_features

CHART_DIR = Path("paper_trading")
STARTING_CASH = 10_000.0
DEFAULT_DAYS = 10  # ~2 trading weeks

# Chosen IN ADVANCE (not based on this window) - the random forest is our
# best short-horizon model across earlier backtests. Picking the model by
# how it does in this same window would be a form of cheating.
TRADED_MODEL = "random_forest"


@dataclass(frozen=True)
class PaperTradeDay:
    """One day of the simulated live trade."""

    date: pd.Timestamp
    prior_close: float
    predicted_close: float
    actual_close: float
    predicted_up: bool
    actual_up: bool
    invested: bool        # were we holding the stock this day?
    equity: float         # portfolio value after this day


@dataclass(frozen=True)
class PaperTradeResult:
    ticker: str
    days: list[PaperTradeDay]
    final_equity: float
    buy_hold_equity: float
    direction_hits: int
    mean_abs_error_pct: float          # error we actually got, live
    backtest_expected_error_pct: float  # error the backtest predicted we'd get


def _mean_abs_price_error_pct(pred_returns: np.ndarray, actual_returns: np.ndarray) -> float:
    """Average |predicted - actual| next-day price error, as a percent.

    Uses returns directly (prior close cancels out), so it's comparable
    whether it comes from the backtest or the live paper trade.
    """
    predicted_price = np.exp(pred_returns)
    actual_price = np.exp(actual_returns)
    return float(np.mean(np.abs(predicted_price - actual_price) / actual_price) * 100.0)


def _backtest_expected_error_pct(ohlcv: pd.DataFrame, exclude_last: int) -> float:
    """What average error did the backtest say to expect for the traded model?

    Runs the walk-forward backtest over the year ENDING just before the
    paper-trading window, so the two numbers are independent and comparable.
    """
    prior = ohlcv.iloc[: len(ohlcv) - exclude_last]
    table = build_features(prior, horizon=1)
    preds = walk_forward_predictions(table)  # default: last ~250 days, monthly refit
    return _mean_abs_price_error_pct(
        preds[TRADED_MODEL].to_numpy(), preds["actual"].to_numpy()
    )


def run_paper_trade(ticker: str, days: int = DEFAULT_DAYS) -> PaperTradeResult:
    """Simulate the last `days` trading days as if trading live."""
    ohlcv = fetch_history(ticker)
    close = ohlcv["Close"]
    table = build_features(ohlcv, horizon=1)

    # Retrain EVERY day (retrain_every=1) for maximum rigor over a short window.
    preds = walk_forward_predictions(table, test_days=days, retrain_every=1)

    equity = STARTING_CASH
    buy_hold = STARTING_CASH
    ledger: list[PaperTradeDay] = []
    abs_errors: list[float] = []
    hits = 0

    for pred_date in preds.index:
        pos = close.index.get_loc(pred_date)
        prior_close = float(close.iloc[pos])
        actual_close = float(close.iloc[pos + 1])
        actual_return = float(preds.loc[pred_date, "actual"])
        predicted_return = float(preds.loc[pred_date, TRADED_MODEL])

        predicted_close = prior_close * np.exp(predicted_return)
        predicted_up = predicted_return > 0
        actual_up = actual_return > 0
        if predicted_up == actual_up:
            hits += 1
        abs_errors.append(abs(predicted_close - actual_close) / actual_close * 100.0)

        # Trade: if we predict UP, we hold the stock for the next day and earn
        # its real return; otherwise we sit safely in cash.
        invested = predicted_up
        if invested:
            equity *= np.exp(actual_return)
        buy_hold *= np.exp(actual_return)

        ledger.append(
            PaperTradeDay(
                date=close.index[pos + 1],  # the day the outcome lands
                prior_close=prior_close,
                predicted_close=predicted_close,
                actual_close=actual_close,
                predicted_up=predicted_up,
                actual_up=actual_up,
                invested=invested,
                equity=equity,
            )
        )

    return PaperTradeResult(
        ticker=ticker,
        days=ledger,
        final_equity=equity,
        buy_hold_equity=buy_hold,
        direction_hits=hits,
        mean_abs_error_pct=float(np.mean(abs_errors)),
        backtest_expected_error_pct=_backtest_expected_error_pct(ohlcv, exclude_last=days),
    )


def print_result(label: str, result: PaperTradeResult) -> None:
    n = len(result.days)
    first, last = result.days[0].date.date(), result.days[-1].date.date()
    print(f"\n=== {label} ({result.ticker}) — paper trade {first} → {last} ===")
    print(f"Traded model: {TRADED_MODEL} (chosen in advance). "
          f"Retrained fresh every day.\n")

    print(f"  {'date':<12}{'prior':>10}{'predicted':>11}{'actual':>10}"
          f"{'call':>7}{'result':>9}{'portfolio':>12}")
    for d in result.days:
        call = "UP" if d.predicted_up else "cash"
        hit = "right" if d.predicted_up == d.actual_up else "WRONG"
        print(f"  {d.date.date().isoformat():<12}{d.prior_close:>10,.2f}"
              f"{d.predicted_close:>11,.2f}{d.actual_close:>10,.2f}"
              f"{call:>7}{hit:>9}{d.equity:>12,.2f}")

    hit_rate = result.direction_hits / n * 100.0
    model_pnl = result.final_equity - STARTING_CASH
    bh_pnl = result.buy_hold_equity - STARTING_CASH
    print(f"\n  direction accuracy:   {result.direction_hits}/{n} ({hit_rate:.0f}%)")
    print(f"  model portfolio:      ${result.final_equity:,.2f} "
          f"({model_pnl:+,.2f})")
    print(f"  buy & hold portfolio: ${result.buy_hold_equity:,.2f} "
          f"({bh_pnl:+,.2f})")
    winner = "model" if result.final_equity > result.buy_hold_equity else "buy & hold"
    print(f"  winner over {n} days:   {winner}")

    # The headline: does the live error match what the backtest promised?
    expected = result.backtest_expected_error_pct
    got = result.mean_abs_error_pct
    ratio = got / expected if expected else float("nan")
    print(f"\n  DOES PAPER TRADING MATCH THE BACKTEST?")
    print(f"    backtest said to expect avg error: {expected:.2f}%")
    print(f"    live paper-trading avg error:      {got:.2f}%")
    verdict = "consistent - the backtest was honest" if ratio < 1.5 \
        else "live error ran higher than promised"
    print(f"    → {got:.2f}% vs {expected:.2f}%  ({verdict})")

    if all(d.predicted_up for d in result.days):
        print("\n  Note: the model predicted UP every day (it learned the market's"
              "\n  upward drift), so it never sat in cash and its portfolio matched"
              "\n  buy & hold exactly. That's why price error, not P&L, is the real"
              "\n  signal here.")


def save_chart(label: str, result: PaperTradeResult) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates = [d.date for d in result.days]
    predicted = [d.predicted_close for d in result.days]
    actual = [d.actual_close for d in result.days]

    # Rebuild both equity curves for plotting.
    model_curve, bh_curve = [], []
    eq, bh = STARTING_CASH, STARTING_CASH
    for d in result.days:
        ret = np.log(d.actual_close / d.prior_close)
        if d.invested:
            eq *= np.exp(ret)
        bh *= np.exp(ret)
        model_curve.append(eq)
        bh_curve.append(bh)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

    ax1.plot(dates, actual, "o-", color="#1a1a2e", label="actual price")
    ax1.plot(dates, predicted, "s--", color="#e63946", label="predicted price")
    for d in result.days:
        color = "#2a9d8f" if d.predicted_up == d.actual_up else "#e76f51"
        ax1.scatter([d.date], [d.actual_close], color=color, zorder=5, s=40)
    ax1.set_title(f"{label} ({result.ticker}) — predicted vs actual "
                  f"(green dot = right call, red = wrong)", fontweight="bold")
    ax1.set_ylabel("price")
    ax1.legend()
    ax1.grid(alpha=0.25)

    ax2.plot(dates, model_curve, "o-", color="#e63946", label=f"{TRADED_MODEL} strategy")
    ax2.plot(dates, bh_curve, "s-", color="#457b9d", label="buy & hold")
    ax2.axhline(STARTING_CASH, color="gray", linestyle="--", alpha=0.6,
                label="starting $10,000")
    ax2.set_title("Pretend $10,000 portfolio over the paper-trading window",
                  fontweight="bold")
    ax2.set_ylabel("portfolio value ($)")
    ax2.legend()
    ax2.grid(alpha=0.25)

    CHART_DIR.mkdir(exist_ok=True)
    path = CHART_DIR / f"{result.ticker.replace('^', '')}_papertrade.png"
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    args = sys.argv[1:]
    numbers = [int(a) for a in args if a.isdigit()]
    days = numbers[0] if numbers else DEFAULT_DAYS
    requested = [a for a in args if not a.isdigit()] or list(TICKERS)

    for ticker in requested:
        label = TICKERS.get(ticker, ticker)
        try:
            result = run_paper_trade(ticker, days)
        except (ValueError, OSError) as exc:
            print(f"\n=== {label} ({ticker}) ===\nFAILED: {exc}", file=sys.stderr)
            continue
        print_result(label, result)
        chart = save_chart(label, result)
        print(f"  Chart saved to {chart}")

    print(
        "\nThis is a SIMULATED live trade over recent real days: each prediction"
        "\nwas made before its outcome existed, so it can't be overfit. But 10"
        "\ndays is a tiny sample - treat it as a demo, not proof. No fees or"
        "\ntaxes are included, and past results never guarantee future ones."
    )


if __name__ == "__main__":
    main()
