"""Tests for feature engineering - the part where bugs silently ruin a model."""

import numpy as np
import pandas as pd
import pytest

from veritas.features import TARGET_COLUMN, build_features, compute_rsi, feature_columns


def make_fake_ohlcv(days: int = 300, seed: int = 7) -> pd.DataFrame:
    """Generate a realistic-looking fake price history for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=days)
    returns = rng.normal(0.0005, 0.01, size=days)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.0, 0.01, size=days))
    low = close * (1 - rng.uniform(0.0, 0.01, size=days))
    volume = rng.integers(1_000_000, 5_000_000, size=days).astype(float)
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def test_target_is_tomorrows_return_not_todays():
    # This is THE classic bug in price prediction: accidentally letting the
    # model see the answer. Here we verify row t's target equals the actual
    # return from close t to close t+1.
    ohlcv = make_fake_ohlcv()
    table = build_features(ohlcv)

    some_day = table.index[100]
    next_day = ohlcv.index[ohlcv.index.get_loc(some_day) + 1]
    expected = np.log(ohlcv.loc[next_day, "Close"] / ohlcv.loc[some_day, "Close"])

    assert table.loc[some_day, TARGET_COLUMN] == pytest.approx(expected)


def test_multiday_target_spans_the_right_distance():
    # With horizon=5, row t's target must be the return from close t to t+5.
    ohlcv = make_fake_ohlcv()
    table = build_features(ohlcv, horizon=5)

    some_day = table.index[100]
    pos = ohlcv.index.get_loc(some_day)
    expected = np.log(ohlcv["Close"].iloc[pos + 5] / ohlcv["Close"].iloc[pos])

    assert table.loc[some_day, TARGET_COLUMN] == pytest.approx(expected)
    # And the last 5 rows can't have targets yet.
    assert table[TARGET_COLUMN].iloc[-5:].isna().all()


def test_last_row_has_no_target():
    # The most recent day has no "tomorrow" yet - that's the row we predict on.
    table = build_features(make_fake_ohlcv())
    assert np.isnan(table[TARGET_COLUMN].iloc[-1])
    assert not table[feature_columns(table)].iloc[-1].isna().any()


def test_no_missing_values_in_features():
    table = build_features(make_fake_ohlcv())
    assert not table[feature_columns(table)].isna().any().any()


def test_rsi_stays_between_0_and_1():
    close = make_fake_ohlcv()["Close"]
    rsi = compute_rsi(close)
    assert rsi.between(0.0, 1.0).all()


def test_rsi_is_high_when_price_only_goes_up():
    close = pd.Series(np.linspace(100, 200, 60))
    assert compute_rsi(close).iloc[-1] > 0.9


def test_features_only_use_past_data():
    # Changing the FUTURE must not change today's features (only its target).
    ohlcv = make_fake_ohlcv()
    table_before = build_features(ohlcv)

    tampered = ohlcv.copy()
    tampered.iloc[-50:, tampered.columns.get_loc("Close")] *= 2.0
    table_after = build_features(tampered)

    check_day = table_before.index[150]  # well before the tampered region
    cols = feature_columns(table_before)
    pd.testing.assert_series_equal(
        table_before.loc[check_day, cols], table_after.loc[check_day, cols]
    )
