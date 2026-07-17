"""Settings for the whole project, kept in one place so they're easy to tweak."""

TICKERS = {
    "^GSPC": "S&P 500",
    "NVDA": "NVIDIA",
    "JPM": "JPMorgan Chase",
}

# How much daily history to download from Yahoo Finance.
HISTORY_PERIOD = "10y"

# We hold out the most recent year (~250 trading days) to test the model
# on data it has never seen. Everything before that is for training.
TEST_DAYS = 250

# During the walk-forward backtest, refit the models every ~1 trading month.
RETRAIN_EVERY = 21

# Prediction horizons, in TRADING days (markets are closed on weekends,
# so a week is 5 days, a month ~21, a quarter ~63, a year ~252).
HORIZONS = {
    "tomorrow": 1,
    "next week": 5,
    "next month": 21,
    "next quarter": 63,
    "next year": 252,
}

# Feature windows (in trading days).
RETURN_LAGS = (1, 2, 3, 5, 10)   # "how did the price move over the last N days?"
VOL_WINDOWS = (5, 21)            # "how shaky has the price been lately?"
SMA_WINDOWS = (10, 50)           # "is the price above or below its average?"
RSI_WINDOW = 14                  # classic momentum indicator
VOLUME_Z_WINDOW = 21             # "is trading volume unusually high today?"
