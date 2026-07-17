"""Tests for model training and evaluation."""

import numpy as np
import pytest

from veritas.features import build_features
from veritas.model import evaluate, train_and_predict
from tests.test_features import make_fake_ohlcv


PRICES = np.array([100.0, 100.0, 100.0])


def test_evaluate_perfect_predictions_score_zero_error():
    actual = np.array([0.01, -0.02, 0.005])
    result = evaluate("perfect", actual.copy(), actual, PRICES)
    assert result.mse_return == 0.0
    assert result.rmse_return == 0.0
    assert result.mae_return == 0.0
    assert result.mae_price == 0.0
    assert result.r2 == pytest.approx(1.0)
    assert result.direction_accuracy == 1.0


def test_evaluate_naive_zero_predictions_have_no_direction():
    actual = np.array([0.01, -0.02, 0.005])
    result = evaluate("naive", np.zeros_like(actual), actual, PRICES)
    assert np.isnan(result.direction_accuracy)
    assert result.mae_return == pytest.approx(np.mean(np.abs(actual)))
    assert result.mse_return == pytest.approx(np.mean(actual**2))


def test_evaluate_price_error_uses_actual_prices():
    # Predicting +1% when the truth is 0% on a $100 stock ≈ a $1 miss.
    actual = np.zeros(3)
    predicted = np.full(3, 0.01)
    result = evaluate("test", predicted, actual, PRICES)
    assert result.mae_price == pytest.approx(100.0 * (np.exp(0.01) - 1.0))


def test_train_and_predict_produces_a_sane_price():
    ohlcv = make_fake_ohlcv(days=1200)
    table = build_features(ohlcv)
    last_close = float(ohlcv["Close"].iloc[-1])

    report = train_and_predict("FAKE", table, ohlcv["Close"])

    # Includes the naive baseline plus both real models.
    assert len(report.results) == 3
    assert report.last_close == pytest.approx(last_close)
    # A sane next-day prediction is within +/-20% of today's close.
    assert 0.8 * last_close < report.predicted_price < 1.2 * last_close


def test_train_and_predict_works_at_longer_horizons():
    ohlcv = make_fake_ohlcv(days=1500)
    table = build_features(ohlcv, horizon=21)
    last_close = float(ohlcv["Close"].iloc[-1])

    report = train_and_predict("FAKE", table, ohlcv["Close"], horizon=21)

    assert report.horizon == 21
    # A month ahead allows bigger moves, but +/-50% on fake data is plenty.
    assert 0.5 * last_close < report.predicted_price < 1.5 * last_close


def test_train_and_predict_rejects_short_history():
    ohlcv = make_fake_ohlcv(days=300)
    table = build_features(ohlcv)
    with pytest.raises(ValueError, match="Not enough history"):
        train_and_predict("FAKE", table, ohlcv["Close"])
