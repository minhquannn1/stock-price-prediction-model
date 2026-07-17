"""Train, backtest, and predict prices at five horizons per ticker.

For each ticker we train a SEPARATE model per horizon (1 day, 1 week,
1 month, 1 quarter, 1 year) - each one learns to answer a different
question, e.g. "what will the total return be over the next 21 days?"

Every horizon is backtested on the held-out final year before predicting.

Usage:
    uv run predict              # all three tickers
    uv run predict NVDA         # just one
"""

import sys

import pandas as pd

from veritas.config import HORIZONS, TICKERS
from veritas.data import fetch_history
from veritas.features import build_features
from veritas.model import TickerReport, train_and_predict
from veritas.prediction_chart import save_prediction_chart


def run_ticker(ticker: str) -> tuple[list[TickerReport], pd.Series]:
    """One report per horizon, all from a single data download."""
    ohlcv = fetch_history(ticker)
    reports = []
    for horizon in HORIZONS.values():
        table = build_features(ohlcv, horizon=horizon)
        reports.append(train_and_predict(ticker, table, ohlcv["Close"], horizon=horizon))
    return reports, ohlcv["Close"]


def print_reports(reports: list[TickerReport], label: str) -> None:
    first = reports[0]
    print(f"\n=== {label} ({first.ticker}) ===")
    print(f"Last close ({first.last_date.date()}): {first.last_close:,.2f}\n")

    header = (
        f"  {'horizon':<14}{'predicted close':>16}{'change':>9}"
        f"{'best model':>16}{'direction':>11}{'RMSE vs naive':>16}"
    )
    print(header)
    for name, report in zip(HORIZONS, reports):
        pct = (report.predicted_price / report.last_close - 1.0) * 100.0
        best = next(r for r in report.results if r.name == report.best_model)
        naive = next(r for r in report.results if r.name.startswith("naive"))
        direction = (
            "n/a" if best.direction_accuracy != best.direction_accuracy
            else f"{best.direction_accuracy:.0%}"
        )
        model_short = report.best_model.split(" ")[0]
        print(
            f"  {name:<14}{report.predicted_price:>16,.2f}{pct:>+8.1f}%"
            f"{model_short:>16}{direction:>11}"
            f"{best.rmse_return:>9.4f}/{naive.rmse_return:.4f}"
        )

    print(
        "\n  direction = how often the winning model called up-vs-down correctly"
        "\n  on the held-out test year. RMSE vs naive: left number lower = the"
        "\n  model beat 'price stays where it is'."
    )


def main() -> None:
    requested = sys.argv[1:] or list(TICKERS)
    for ticker in requested:
        label = TICKERS.get(ticker, ticker)
        try:
            reports, close = run_ticker(ticker)
        except (ValueError, OSError) as exc:
            print(f"\n=== {label} ({ticker}) ===\nFAILED: {exc}", file=sys.stderr)
            continue
        print_reports(reports, label)
        chart = save_prediction_chart(reports, list(HORIZONS), label, close)
        print(f"  Chart saved to {chart}")

    print(
        "\nReality check: short-horizon prices are mostly noise, and long-horizon"
        "\npredictions mostly reflect the market's average upward drift. Treat"
        "\nevery number above as a rough estimate with a wide error bar."
    )


if __name__ == "__main__":
    main()
