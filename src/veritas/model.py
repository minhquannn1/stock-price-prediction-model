"""Train models, test them honestly, and pick the best one.

Two models, both from scikit-learn:

  * Ridge        - linear regression with a safety brake against overfitting
  * RandomForest - an "ensemble" of decision trees that vote together

And one baseline everyone must beat:

  * Naive - just predict "tomorrow's price = today's price" (zero return).
    This sounds dumb, but for daily stock prices it is famously hard to
    beat. If a model can't beat it, the model isn't actually predicting.

IMPORTANT: we split the data by TIME (train on the past, test on the most
recent year). Shuffling rows like a normal ML tutorial would let the model
peek into the future, which makes results look amazing and be worthless.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from veritas.config import TEST_DAYS
from veritas.features import TARGET_COLUMN, feature_columns


def make_models() -> dict[str, Pipeline]:
    """The candidate models. Each is a pipeline that first standardizes the
    features (mean 0, std 1) so no single feature dominates just because
    its numbers are bigger."""
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=300,
                max_depth=4,          # shallow trees = less overfitting
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }


@dataclass(frozen=True)
class EvalResult:
    """Test-set scores for one model. All "return" metrics are in daily
    log-return units (0.01 = roughly a 1% move)."""

    name: str
    mse_return: float            # mean squared error - the classic ML metric
    rmse_return: float           # sqrt of MSE, back in readable units
    mae_return: float            # mean absolute error - average miss size
    r2: float                    # 1 = perfect, 0 = no better than the mean
    direction_accuracy: float    # how often we got up-vs-down right (0..1)
    mae_price: float             # average miss in actual price units ($/points)


def evaluate(
    name: str, predicted: np.ndarray, actual: np.ndarray, prices: np.ndarray
) -> EvalResult:
    """Score one model's test-set predictions.

    `prices` is the closing price on each prediction day, used to translate
    the return error into a dollar error people can actually picture.
    """
    errors = predicted - actual
    mse = float(np.mean(errors**2))
    if np.all(predicted == 0):
        # The naive baseline never calls a direction, so accuracy is undefined.
        direction = float("nan")
    else:
        direction = float(np.mean(np.sign(predicted) == np.sign(actual)))
    # R^2: how much of the day-to-day variation we explained (can be negative!)
    variance = float(np.mean((actual - actual.mean()) ** 2))
    r2 = 1.0 - mse / variance if variance > 0 else float("nan")
    # Predicted vs actual NEXT-day price, averaged over the test year.
    price_errors = prices * np.exp(predicted) - prices * np.exp(actual)
    return EvalResult(
        name=name,
        mse_return=mse,
        rmse_return=float(np.sqrt(mse)),
        mae_return=float(np.mean(np.abs(errors))),
        r2=r2,
        direction_accuracy=direction,
        mae_price=float(np.mean(np.abs(price_errors))),
    )


@dataclass(frozen=True)
class TickerReport:
    """Everything we learned about one ticker at one horizon."""

    ticker: str
    horizon: int
    results: list[EvalResult]
    best_model: str
    last_close: float
    last_date: pd.Timestamp
    predicted_return: float
    predicted_price: float


def train_and_predict(
    ticker: str, table: pd.DataFrame, close: pd.Series, horizon: int = 1
) -> TickerReport:
    """Backtest every model on the held-out year, then retrain the winner on
    ALL labeled data and predict the closing price `horizon` days ahead.

    `close` is the full daily closing-price series (used to express errors
    in price units and to anchor the final prediction).
    """
    cols = feature_columns(table)
    last_close = float(close.iloc[-1])

    labeled = table.dropna(subset=[TARGET_COLUMN])
    if len(labeled) < TEST_DAYS * 3 + horizon:
        raise ValueError(f"Not enough history for {ticker!r} ({len(labeled)} rows)")

    # Past = training, most recent year = testing. No shuffling!
    # The `horizon`-day gap ("purge") between train and test matters for
    # long horizons: the last training rows' targets reach into the test
    # period, and without the gap the model gets a sneak peek at test data.
    train = labeled.iloc[: -(TEST_DAYS + horizon)]
    test = labeled.iloc[-TEST_DAYS:]
    x_train, y_train = train[cols], train[TARGET_COLUMN].to_numpy()
    x_test, y_test = test[cols], test[TARGET_COLUMN].to_numpy()
    test_prices = close.loc[test.index].to_numpy(dtype=float)

    # The baseline predicts a 0% return every single day.
    results = [
        evaluate("naive (tomorrow = today)", np.zeros_like(y_test), y_test, test_prices)
    ]

    models = make_models()
    for name, model in models.items():
        model.fit(x_train, y_train)
        results.append(evaluate(name, model.predict(x_test), y_test, test_prices))

    # Pick the model with the lowest RMSE on the test year (baseline can win!).
    best = min(results, key=lambda r: r.rmse_return)

    # Predict tomorrow using the most recent row (its target is still unknown).
    latest_row = table[cols].iloc[[-1]]
    if best.name in models:
        # Retrain the winning model on ALL labeled data so the final
        # prediction uses every bit of history we have.
        final_model = make_models()[best.name]
        final_model.fit(labeled[cols], labeled[TARGET_COLUMN].to_numpy())
        predicted_return = float(final_model.predict(latest_row)[0])
    else:
        predicted_return = 0.0  # the naive baseline won

    # Convert the predicted log return back into an actual price.
    predicted_price = last_close * float(np.exp(predicted_return))

    return TickerReport(
        ticker=ticker,
        horizon=horizon,
        results=results,
        best_model=best.name,
        last_close=last_close,
        last_date=table.index[-1],
        predicted_return=predicted_return,
        predicted_price=predicted_price,
    )
