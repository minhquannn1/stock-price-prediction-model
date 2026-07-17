"""Exploratory Data Analysis: summary statistics and charts for the dataset.

Run `uv run eda` to regenerate every statistic and figure used in the
EDA document. It prints a stats table per ticker plus cross-ticker
correlations, and saves charts to eda_results/.
"""

import sys

import numpy as np
import pandas as pd

from veritas.config import TICKERS
from veritas.data import fetch_history

from pathlib import Path

CHART_DIR = Path("eda_results")
TRADING_DAYS_PER_YEAR = 252


def daily_log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).dropna()


def summary_stats(close: pd.Series) -> dict[str, float]:
    """Headline statistics a reader would want about one ticker."""
    r = daily_log_returns(close)
    ann_return = float(r.mean() * TRADING_DAYS_PER_YEAR)
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    return {
        "days": int(len(close)),
        "start": close.index[0].date().isoformat(),
        "end": close.index[-1].date().isoformat(),
        "start_price": float(close.iloc[0]),
        "end_price": float(close.iloc[-1]),
        "total_return_pct": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
        "ann_return_pct": ann_return * 100,
        "ann_volatility_pct": ann_vol * 100,
        "sharpe_no_rf": ann_return / ann_vol if ann_vol else float("nan"),
        "best_day_pct": float(r.max() * 100),
        "worst_day_pct": float(r.min() * 100),
        "up_day_share_pct": float((r > 0).mean() * 100),
        "daily_skew": float(r.skew()),
        "daily_excess_kurtosis": float(r.kurtosis()),  # 0 = normal; >0 = fat tails
        "return_autocorr_lag1": float(r.autocorr(lag=1)),
        "abs_return_autocorr_lag1": float(r.abs().autocorr(lag=1)),
    }


def load_returns() -> pd.DataFrame:
    """Aligned daily-return table for all tickers (inner join on dates)."""
    series = {}
    for ticker in TICKERS:
        series[ticker] = daily_log_returns(fetch_history(ticker)["Close"])
    return pd.DataFrame(series).dropna()


def print_stats(stats: dict[str, dict[str, float]]) -> None:
    for ticker, s in stats.items():
        print(f"\n=== {TICKERS[ticker]} ({ticker}) ===")
        print(f"  {s['days']} trading days, {s['start']} → {s['end']}")
        print(f"  price: {s['start_price']:,.2f} → {s['end_price']:,.2f} "
              f"({s['total_return_pct']:+.1f}% total)")
        print(f"  annualized return:     {s['ann_return_pct']:+.1f}%")
        print(f"  annualized volatility: {s['ann_volatility_pct']:.1f}%")
        print(f"  return/volatility:     {s['sharpe_no_rf']:.2f}")
        print(f"  best / worst day:      {s['best_day_pct']:+.1f}% / {s['worst_day_pct']:+.1f}%")
        print(f"  up days:               {s['up_day_share_pct']:.1f}%")
        print(f"  skew:                  {s['daily_skew']:+.2f}")
        print(f"  excess kurtosis:       {s['daily_excess_kurtosis']:+.1f} "
              f"(0 = normal bell curve; higher = fatter tails)")
        print(f"  return autocorrelation (lag 1):     {s['return_autocorr_lag1']:+.3f} "
              "(near 0 = next move is unpredictable)")
        print(f"  |return| autocorrelation (lag 1):   {s['abs_return_autocorr_lag1']:+.3f} "
              "(high = calm/wild days cluster)")


def save_charts(stats: dict[str, dict[str, float]], returns: pd.DataFrame) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHART_DIR.mkdir(exist_ok=True)
    paths = []
    closes = {t: fetch_history(t)["Close"] for t in TICKERS}

    # 1. Normalized price history (everyone starts at 100 for a fair race).
    fig, ax = plt.subplots(figsize=(11, 6))
    for ticker, close in closes.items():
        norm = close / close.iloc[0] * 100
        ax.plot(norm.index, norm.to_numpy(), label=f"{TICKERS[ticker]} ({ticker})", linewidth=1.4)
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Growth of $100 invested 10 years ago", fontweight="bold")
    ax.set_ylabel("value (start = 100)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    p = CHART_DIR / "price_history.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    # 2. Return distributions vs a normal curve (shows fat tails).
    fig, axes = plt.subplots(1, len(TICKERS), figsize=(14, 4.2), sharey=True)
    for ax, ticker in zip(axes, TICKERS):
        r = returns[ticker] * 100
        ax.hist(r, bins=80, density=True, color="#457b9d", alpha=0.7)
        grid = np.linspace(r.min(), r.max(), 200)
        normal = np.exp(-0.5 * ((grid - r.mean()) / r.std()) ** 2) / (r.std() * np.sqrt(2 * np.pi))
        ax.plot(grid, normal, color="#e63946", linewidth=1.6, label="normal curve")
        ax.set_title(f"{ticker}\nexcess kurtosis {stats[ticker]['daily_excess_kurtosis']:+.1f}")
        ax.set_xlabel("daily return (%)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Daily returns have FATTER TAILS than a normal bell curve "
                 "(big moves happen more than 'normal' predicts)", fontweight="bold")
    fig.tight_layout()
    p = CHART_DIR / "return_distributions.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    # 3. Correlation heatmap of daily returns.
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.to_numpy(), cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontweight="bold")
    ax.set_title("How the three move together (daily return correlation)", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    p = CHART_DIR / "correlation.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    # 4. Rolling 21-day volatility (shows volatility clustering over time).
    fig, ax = plt.subplots(figsize=(11, 5))
    for ticker in TICKERS:
        roll = returns[ticker].rolling(21).std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        ax.plot(roll.index, roll.to_numpy(), label=ticker, linewidth=1.2)
    ax.set_title("Volatility comes in waves (21-day rolling, annualized)", fontweight="bold")
    ax.set_ylabel("annualized volatility (%)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    p = CHART_DIR / "rolling_volatility.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    return paths


def main() -> None:
    stats = {t: summary_stats(fetch_history(t)["Close"]) for t in TICKERS}
    print_stats(stats)

    returns = load_returns()
    print("\n=== Daily-return correlation matrix ===")
    print(returns.corr().round(3).to_string())

    paths = save_charts(stats, returns)
    print("\nCharts saved:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
