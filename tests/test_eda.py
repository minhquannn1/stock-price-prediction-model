"""Tests for the EDA statistics."""

import numpy as np
import pytest

from veritas.eda import daily_log_returns, summary_stats
from tests.test_features import make_fake_ohlcv


def test_daily_log_returns_match_definition():
    close = make_fake_ohlcv(days=50)["Close"]
    r = daily_log_returns(close)
    assert len(r) == len(close) - 1  # first day has no prior day
    expected = np.log(close.iloc[10] / close.iloc[9])
    assert r.iloc[9] == pytest.approx(expected)


def test_summary_stats_are_internally_consistent():
    close = make_fake_ohlcv(days=600)["Close"]
    s = summary_stats(close)

    assert s["days"] == 600
    assert s["start_price"] == pytest.approx(float(close.iloc[0]))
    assert s["end_price"] == pytest.approx(float(close.iloc[-1]))
    # Total return must agree with start/end prices.
    assert s["total_return_pct"] == pytest.approx(
        (close.iloc[-1] / close.iloc[0] - 1) * 100
    )
    # Up-day share is a percentage.
    assert 0.0 <= s["up_day_share_pct"] <= 100.0
    # Volatility is non-negative.
    assert s["ann_volatility_pct"] >= 0.0
