"""Walk-forward backtest: how would the model REALLY have done last year?

A single train/test split tells you how a model trained once performs. A
walk-forward backtest is stricter and closer to real life:

    for each day in the test year:
        train only on data BEFORE that day (we refit once a month)
        predict the NEXT day's return
        write the prediction down before looking at the answer

Then we grade the predictions two ways:
  1. Accuracy  - error sizes and how often the up/down call was right
  2. Money     - simulate "invest $10,000, hold the index only on days the
                 model says UP" and compare against just buying and holding.

Usage:
    uv run backtest          # all three tickers, 1-day horizon
    uv run backtest NVDA     # just one ticker
    uv run backtest 21       # 1-month horizon (any number of trading days)
    uv run backtest NVDA 5   # one ticker, 1-week horizon
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from veritas.config import RETRAIN_EVERY, TEST_DAYS, TICKERS
from veritas.data import fetch_history
from veritas.features import TARGET_COLUMN, build_features, feature_columns
from veritas.model import evaluate, make_models

CHART_DIR = Path("backtest_results")


def walk_forward_predictions(
    table: pd.DataFrame,
    test_days: int = TEST_DAYS,
    retrain_every: int = RETRAIN_EVERY,
) -> pd.DataFrame:
    """Run every model through the test period one day at a time.

    Returns a DataFrame indexed by date with one column per model plus the
    'actual' next-day return. Each prediction only ever saw earlier data.
    """
    labeled = table.dropna(subset=[TARGET_COLUMN])
    if len(labeled) < test_days * 3:
        raise ValueError(f"Not enough history ({len(labeled)} rows)")
    cols = feature_columns(table)
    start = len(labeled) - test_days

    models = make_models()
    predictions: dict[str, list[float]] = {name: [] for name in models}
    for i in range(start, len(labeled)):
        if (i - start) % retrain_every == 0:
            # Monthly refit on everything known so far (rows 0..i-1).
            for model in models.values():
                model.fit(labeled[cols].iloc[:i], labeled[TARGET_COLUMN].iloc[:i].to_numpy())
        today = labeled[cols].iloc[[i]]
        for name, model in models.items():
            predictions[name].append(float(model.predict(today)[0]))

    result = pd.DataFrame(predictions, index=labeled.index[start:])
    result["actual"] = labeled[TARGET_COLUMN].iloc[start:].to_numpy()
    return result


def simulate_strategy(
    predicted: np.ndarray, actual: np.ndarray, horizon: int = 1
) -> np.ndarray:
    """Equity curve of: hold the asset when the model predicts UP, else cash.

    For multi-day horizons we only re-decide every `horizon` days - you
    can't place five overlapping one-week bets with the same money.
    Starts at 1.0 (a multiplier on your starting cash). Ignores trading
    fees and taxes, which would only make it worse.
    """
    sampled_pred = predicted[::horizon]
    sampled_actual = actual[::horizon]
    period_returns = np.where(sampled_pred > 0, sampled_actual, 0.0)
    return np.exp(np.cumsum(period_returns))


def backtest_ticker(ticker: str, horizon: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Run the walk-forward backtest. Returns (predictions, close prices)."""
    ohlcv = fetch_history(ticker)
    table = build_features(ohlcv, horizon=horizon)
    preds = walk_forward_predictions(table)
    return preds, ohlcv["Close"]


def print_summary(
    ticker: str, label: str, preds: pd.DataFrame, close: pd.Series, horizon: int = 1
) -> None:
    actual = preds["actual"].to_numpy()
    prices = close.loc[preds.index].to_numpy(dtype=float)
    model_names = [c for c in preds.columns if c != "actual"]

    first, last = preds.index[0].date(), preds.index[-1].date()
    print(f"\n=== {label} ({ticker}) — walk-forward backtest {first} → {last} ===")
    print(
        f"{len(preds)} predictions at a {horizon}-trading-day horizon,"
        f" retrained every {RETRAIN_EVERY} trading days\n"
    )

    print(f"  {'model':<26}{'RMSE':>9}{'MAE':>9}{'direction':>11}{'$10k grows to':>15}")
    buy_hold = simulate_strategy(np.ones_like(actual), actual, horizon)
    print(
        f"  {'buy & hold (no model)':<26}{'-':>9}{'-':>9}{'-':>11}"
        f"{10_000 * buy_hold[-1]:>15,.0f}"
    )
    for name in model_names:
        p = preds[name].to_numpy()
        scores = evaluate(name, p, actual, prices)
        equity = simulate_strategy(p, actual, horizon)
        print(
            f"  {name:<26}{scores.rmse_return:>9.5f}{scores.mae_return:>9.5f}"
            f"{scores.direction_accuracy:>10.1%}{10_000 * equity[-1]:>15,.0f}"
        )

    up_share = float(np.mean(actual > 0))
    print(f"\n  For reference: the market was UP over {up_share:.1%} of these periods,")
    print("  so 'always guess up' would score that as its direction accuracy.")
    if horizon > 1:
        print(
            f"  Note: consecutive {horizon}-day predictions overlap, so the error"
            "\n  metrics are less independent than they look. The $ strategy only"
            f"\n  trades every {horizon} days (no overlap)."
        )


def save_chart(ticker: str, preds: pd.DataFrame, horizon: int = 1) -> Path:
    """Save a 2-panel chart: equity curves + rolling direction accuracy."""
    import matplotlib

    matplotlib.use("Agg")  # no GUI needed, just write the file
    import matplotlib.pyplot as plt

    actual = preds["actual"].to_numpy()
    model_names = [c for c in preds.columns if c != "actual"]
    trade_dates = preds.index[::horizon]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(trade_dates, 10_000 * simulate_strategy(np.ones_like(actual), actual, horizon),
             label="buy & hold", linewidth=2, color="gray")
    for name in model_names:
        equity = simulate_strategy(preds[name].to_numpy(), actual, horizon)
        ax1.plot(trade_dates, 10_000 * equity, label=f"{name} strategy")
    ax1.set_ylabel("value of $10,000")
    ax1.set_title(
        f"{ticker}: model-following strategy vs buy & hold ({horizon}-day horizon)"
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    window = 50
    for name in model_names:
        correct = np.sign(preds[name]) == np.sign(preds["actual"])
        rolling = correct.rolling(window).mean()
        ax2.plot(preds.index, rolling, label=name)
    ax2.axhline(0.5, color="gray", linestyle="--", label="coin flip (50%)")
    ax2.set_ylabel(f"direction accuracy ({window}-day rolling)")
    ax2.set_title("How often the up/down call was right")
    ax2.legend()
    ax2.grid(alpha=0.3)

    CHART_DIR.mkdir(exist_ok=True)
    suffix = "" if horizon == 1 else f"_{horizon}d"
    path = CHART_DIR / f"{ticker.replace('^', '')}_backtest{suffix}.png"
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    # Plain numbers in the arguments set the horizon; everything else is a ticker.
    args = sys.argv[1:]
    horizons = [int(a) for a in args if a.isdigit()]
    horizon = horizons[0] if horizons else 1
    requested = [a for a in args if not a.isdigit()] or list(TICKERS)

    for ticker in requested:
        label = TICKERS.get(ticker, ticker)
        try:
            preds, close = backtest_ticker(ticker, horizon)
        except (ValueError, OSError) as exc:
            print(f"\n=== {label} ({ticker}) ===\nFAILED: {exc}", file=sys.stderr)
            continue
        print_summary(ticker, label, preds, close, horizon)
        chart = save_chart(ticker, preds, horizon)
        print(f"  Chart saved to {chart}")

    print(
        "\nRemember: no trading fees, taxes, or slippage are included here,"
        "\nand one good backtest year can easily be luck. Don't trade on this!"
    )


if __name__ == "__main__":
    main()
