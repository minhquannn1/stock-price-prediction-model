"""Hindcast: backtest the WHOLE forecast by rewinding time.

The walk-forward backtest grades thousands of small predictions. This is
the complement: rewind to a date in the past, hand the models only the
data they would have had THAT day, let them draw the full 5-horizon
forecast - and then grade it against what actually happened, both in a
table and on the chart.

Usage:
    uv run hindcast            # all tickers, forecast made 1 year ago
    uv run hindcast NVDA       # just one ticker
    uv run hindcast 126        # rewind half a trading year instead
                               # (only horizons that fit get graded)
"""

import sys
from dataclasses import dataclass

import pandas as pd

from veritas.config import HORIZONS, TICKERS
from veritas.data import fetch_history
from veritas.features import build_features
from veritas.model import TickerReport, train_and_predict
from veritas.prediction_chart import save_prediction_chart

DEFAULT_DAYS_BACK = 252  # rewind a full trading year so every horizon is gradeable


@dataclass(frozen=True)
class HindcastRow:
    """One horizon's prediction next to what really happened."""

    name: str
    predicted: float
    actual: float
    error_pct: float        # (predicted - actual) / actual
    direction_hit: bool     # did we at least call up-vs-down right?


def run_hindcast(
    ticker: str, days_back: int = DEFAULT_DAYS_BACK
) -> tuple[list[TickerReport], list[str], list[HindcastRow], pd.DataFrame]:
    """Forecast from `days_back` trading days ago, then grade it.

    Returns (reports, horizon_names, graded rows, full price history).
    Horizons longer than `days_back` can't be graded yet and are skipped.
    """
    ohlcv = fetch_history(ticker)
    if days_back >= len(ohlcv) - 100:
        raise ValueError(f"Can't rewind {days_back} days with {len(ohlcv)} days of data")

    past = ohlcv.iloc[: len(ohlcv) - days_back]  # the world as of the rewind date
    cutoff_pos = len(past) - 1
    cutoff_close = float(past["Close"].iloc[-1])

    usable = {name: h for name, h in HORIZONS.items() if h <= days_back}
    reports, rows = [], []
    for name, horizon in usable.items():
        table = build_features(past, horizon=horizon)
        report = train_and_predict(ticker, table, past["Close"], horizon=horizon)
        reports.append(report)

        actual = float(ohlcv["Close"].iloc[cutoff_pos + horizon])
        predicted_up = report.predicted_price > cutoff_close
        actually_up = actual > cutoff_close
        rows.append(
            HindcastRow(
                name=name,
                predicted=report.predicted_price,
                actual=actual,
                error_pct=(report.predicted_price - actual) / actual * 100.0,
                direction_hit=predicted_up == actually_up,
            )
        )
    return reports, list(usable), rows, ohlcv


def print_hindcast(label: str, ticker: str, rows: list[HindcastRow], cutoff: pd.Timestamp) -> None:
    print(f"\n=== {label} ({ticker}) — forecast made {cutoff.date()}, graded today ===\n")
    print(f"  {'horizon':<14}{'predicted':>12}{'actual':>12}{'error':>9}{'direction':>11}")
    for r in rows:
        hit = "right" if r.direction_hit else "WRONG"
        print(
            f"  {r.name:<14}{r.predicted:>12,.2f}{r.actual:>12,.2f}"
            f"{r.error_pct:>+8.1f}%{hit:>11}"
        )


def main() -> None:
    args = sys.argv[1:]
    numbers = [int(a) for a in args if a.isdigit()]
    days_back = numbers[0] if numbers else DEFAULT_DAYS_BACK
    requested = [a for a in args if not a.isdigit()] or list(TICKERS)

    for ticker in requested:
        label = TICKERS.get(ticker, ticker)
        try:
            reports, names, rows, ohlcv = run_hindcast(ticker, days_back)
        except (ValueError, OSError) as exc:
            print(f"\n=== {label} ({ticker}) ===\nFAILED: {exc}", file=sys.stderr)
            continue

        cutoff = reports[0].last_date
        print_hindcast(label, ticker, rows, cutoff)
        chart = save_prediction_chart(
            reports, names, label,
            close=ohlcv["Close"].loc[:cutoff],
            actuals=ohlcv["Close"].loc[cutoff:],
            filename_suffix="hindcast",
        )
        print(f"  Chart saved to {chart}")

    print(
        "\nOne rewind date is one sample - a great forecast here can be luck,"
        "\nand a bad one can be an unlucky year. Try several rewind dates"
        "\n(e.g. uv run hindcast 126) before drawing conclusions."
    )


if __name__ == "__main__":
    main()
