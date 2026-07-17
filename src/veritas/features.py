"""Turn raw price history into a table a model can learn from.

The big idea
------------
A model can't learn from raw prices directly, so for each day we compute a
row of "clues" (features) using ONLY information known by that day's close:

  * recent returns  - how much the price moved over the last 1, 2, 3... days
  * volatility      - how shaky the price has been lately
  * moving averages - is the price above or below its recent average?
  * RSI             - a classic "overbought / oversold" momentum indicator
  * volume z-score  - is today's trading volume unusually high or low?

The thing we ask the model to predict (the "target") is tomorrow's return,
not tomorrow's price. Returns are roughly the same scale every day, which
makes them much easier to learn than raw prices that drift upward for years.
Once the model predicts a return, turning it back into a price is just:

    predicted price = today's close * e^(predicted log return)
"""

import numpy as np
import pandas as pd

from veritas.config import (
    RETURN_LAGS,
    RSI_WINDOW,
    SMA_WINDOWS,
    VOL_WINDOWS,
    VOLUME_Z_WINDOW,
)

TARGET_COLUMN = "target_next_log_return"


def compute_rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """RSI (Relative Strength Index), scaled to 0..1 instead of the usual 0..100.

    Near 1 means the stock has mostly gone UP lately ("overbought"),
    near 0 means it has mostly gone DOWN ("oversold").
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window).mean()
    # Equivalent to the textbook 100 - 100/(1+RS) formula, but doesn't
    # divide by zero when a stock has had no down days in the window.
    rsi = avg_gain / (avg_gain + avg_loss)
    return rsi.fillna(0.5)  # neutral until we have enough history


def build_features(ohlcv: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Build the features + target table from daily OHLCV price history.

    `horizon` is how many trading days ahead we're predicting: 1 = tomorrow,
    5 = next week, 21 = next month, and so on. The features are the same
    either way - only the target changes.

    The returned table keeps its most recent rows even though their targets
    are NaN (the future hasn't happened yet) - the LAST row is exactly what
    we feed the trained model to predict the next closing price.
    """
    close = ohlcv["Close"]
    # Log return: log(today / yesterday). Very close to % change for small
    # moves, but nicer mathematically (you can add them across days).
    log_return = np.log(close / close.shift(1))

    feats = pd.DataFrame(index=ohlcv.index)

    # How much did the price move over the last N days?
    for lag in RETURN_LAGS:
        feats[f"return_last_{lag}d"] = np.log(close / close.shift(lag))

    # How shaky (volatile) has the price been?
    for window in VOL_WINDOWS:
        feats[f"volatility_{window}d"] = log_return.rolling(window).std()

    # Is the price above (+) or below (-) its moving average, in % terms?
    for window in SMA_WINDOWS:
        feats[f"close_vs_avg_{window}d"] = close / close.rolling(window).mean() - 1.0

    feats["rsi"] = compute_rsi(close)

    # Is today's volume unusual compared to the last ~month? (z-score)
    volume = ohlcv["Volume"].astype(float)
    vol_mean = volume.rolling(VOLUME_Z_WINDOW).mean()
    vol_std = volume.rolling(VOLUME_Z_WINDOW).std()
    feats["volume_zscore"] = ((volume - vol_mean) / vol_std).clip(-5.0, 5.0)

    # Monday=0.0 ... Friday=1.0 (markets sometimes behave differently by weekday)
    feats["day_of_week"] = ohlcv.index.dayofweek / 4.0

    # TARGET: the log return over the next `horizon` trading days.
    # shift(-horizon) pulls that future value back onto today's row.
    feats[TARGET_COLUMN] = np.log(close.shift(-horizon) / close)

    # Drop early rows where rolling windows don't have enough history yet.
    feature_cols = [c for c in feats.columns if c != TARGET_COLUMN]
    return feats.dropna(subset=feature_cols)


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Names of the input columns (everything except the target)."""
    return [c for c in table.columns if c != TARGET_COLUMN]
