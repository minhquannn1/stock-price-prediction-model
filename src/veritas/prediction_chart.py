"""Draw the predictions as a PNG: price history + the forecast fan.

The chart shows the last year of real prices, then the model's predicted
price at each horizon (tomorrow, week, month, quarter, year), connected by
a dashed line. The shaded cone is the honest part: +/-1 backtest RMSE
around each prediction, i.e. "about 2 times out of 3, the truth landed
inside a band this wide during the test year."
"""

from pathlib import Path

import numpy as np
import pandas as pd

from veritas.model import TickerReport

CHART_DIR = Path("predictions")
HISTORY_DAYS_SHOWN = 250


def save_prediction_chart(
    reports: list[TickerReport],
    horizon_names: list[str],
    label: str,
    close: pd.Series,
    actuals: pd.Series | None = None,
    filename_suffix: str = "forecast",
) -> Path:
    """Render one ticker's forecast chart to predictions/<TICKER>_<suffix>.png.

    If `actuals` is given (prices from AFTER the forecast date, used by the
    hindcast), they're drawn over the forecast fan so you can see how the
    prediction actually played out.
    """
    import matplotlib

    matplotlib.use("Agg")  # write the file, no window needed
    import matplotlib.pyplot as plt

    first = reports[0]
    last_date = first.last_date
    last_close = first.last_close

    # Future calendar dates: N trading days ~ N business days ahead.
    future_dates = [
        pd.bdate_range(start=last_date, periods=r.horizon + 1)[-1] for r in reports
    ]
    predicted = [r.predicted_price for r in reports]
    # +/-1 RMSE of the winning model, translated from returns into prices.
    rmse = [
        next(res for res in r.results if res.name == r.best_model).rmse_return
        for r in reports
    ]
    upper = [last_close * np.exp(r.predicted_return + e) for r, e in zip(reports, rmse)]
    lower = [last_close * np.exp(r.predicted_return - e) for r, e in zip(reports, rmse)]

    fig, ax = plt.subplots(figsize=(11, 6))

    history = close.iloc[-HISTORY_DAYS_SHOWN:]
    ax.plot(history.index, history.to_numpy(), color="#1a1a2e", linewidth=1.4,
            label="actual closing price")

    # Dashed forecast line starting from today's close.
    fan_x = [last_date, *future_dates]
    ax.plot(fan_x, [last_close, *predicted], "o--", color="#e63946",
            linewidth=1.6, markersize=5, label="predicted price")
    ax.fill_between(fan_x, [last_close, *lower], [last_close, *upper],
                    color="#e63946", alpha=0.12, label="±1 RMSE uncertainty band")

    if actuals is not None:
        ax.plot(actuals.index, actuals.to_numpy(), color="#2a6f97", linewidth=1.4,
                label="what actually happened")

    point_label = "today" if actuals is None else "forecast\nmade here"
    ax.scatter([last_date], [last_close], color="#1a1a2e", zorder=5)
    ax.annotate(f"{point_label}\n{last_close:,.2f}", (last_date, last_close),
                textcoords="offset points", xytext=(-12, 14),
                ha="right", fontsize=9, fontweight="bold")

    # Label a point on the chart only when it has breathing room - the 1-day
    # and 1-week points sit almost on top of "today" at this zoom level.
    min_gap = pd.Timedelta(days=30)
    last_labeled = last_date
    for name, date, price in zip(horizon_names, future_dates, predicted):
        if date - last_labeled < min_gap:
            continue
        pct = (price / last_close - 1.0) * 100.0
        ax.annotate(f"{name}\n{price:,.2f} ({pct:+.1f}%)", (date, price),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=8.5, color="#b1242f")
        last_labeled = date

    # Every horizon's numbers, in one tidy box (incl. the cramped short ones).
    lines = [f"{'horizon':<13}{'price':>10}{'change':>8}"]
    for name, price in zip(horizon_names, predicted):
        pct = (price / last_close - 1.0) * 100.0
        lines.append(f"{name:<13}{price:>10,.2f}{pct:>+7.1f}%")
    ax.text(0.98, 0.03, "\n".join(lines), transform=ax.transAxes,
            fontsize=8.5, fontfamily="monospace", va="bottom", ha="right",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white",
                  "edgecolor": "#cccccc", "alpha": 0.9})

    ax.set_title(f"{label} ({first.ticker}) — price forecast from {last_date.date()}",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("price")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    ax.margins(y=0.15)

    CHART_DIR.mkdir(exist_ok=True)
    path = CHART_DIR / f"{first.ticker.replace('^', '')}_{filename_suffix}.png"
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
