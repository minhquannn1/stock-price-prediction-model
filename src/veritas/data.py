"""Market data loading via yfinance."""

import pandas as pd
import yfinance as yf

from veritas.config import HISTORY_PERIOD

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def fetch_history(ticker: str, period: str = HISTORY_PERIOD) -> pd.DataFrame:
    """Fetch daily OHLCV history for one ticker, adjusted for splits/dividends."""
    df = yf.download(
        ticker, period=period, auto_adjust=True, progress=False, multi_level_index=False
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker!r}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker!r} is missing columns: {missing}")
    df = df[list(REQUIRED_COLUMNS)].dropna()
    if df.index.tz is not None:
        df = df.tz_localize(None)
    return df
