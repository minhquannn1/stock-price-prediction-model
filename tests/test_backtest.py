"""Tests for the walk-forward backtest."""

import numpy as np
import pytest

from veritas.backtest import simulate_strategy, walk_forward_predictions
from veritas.features import TARGET_COLUMN, build_features
from tests.test_features import make_fake_ohlcv


def test_simulate_strategy_only_earns_on_predicted_up_days():
    predicted = np.array([0.01, -0.01, 0.01])   # up, down, up
    actual = np.array([0.02, 0.05, -0.01])      # we miss day 2's big gain
    equity = simulate_strategy(predicted, actual)
    # Day 1: in the market (+0.02). Day 2: in cash (0). Day 3: in (-0.01).
    expected = np.exp(np.cumsum([0.02, 0.0, -0.01]))
    np.testing.assert_allclose(equity, expected)


def test_simulate_strategy_multiday_horizon_trades_without_overlap():
    # With a 2-day horizon we only act on every 2nd prediction.
    predicted = np.array([0.01, -0.5, 0.01, -0.5])  # days 1 and 3 say UP
    actual = np.array([0.02, 9.9, -0.01, 9.9])      # the 9.9s must be ignored
    equity = simulate_strategy(predicted, actual, horizon=2)
    expected = np.exp(np.cumsum([0.02, -0.01]))
    np.testing.assert_allclose(equity, expected)


def test_simulate_strategy_always_up_equals_buy_and_hold():
    actual = np.array([0.01, -0.02, 0.005])
    equity = simulate_strategy(np.ones_like(actual), actual)
    assert equity[-1] == pytest.approx(np.exp(actual.sum()))


def test_walk_forward_covers_test_period_with_real_predictions():
    table = build_features(make_fake_ohlcv(days=1200))
    preds = walk_forward_predictions(table, test_days=30, retrain_every=15)

    assert len(preds) == 30
    assert "actual" in preds.columns
    assert not preds.isna().any().any()
    # The predictions should line up with the last 30 labeled days.
    labeled = table.dropna(subset=[TARGET_COLUMN])
    assert (preds.index == labeled.index[-30:]).all()
    np.testing.assert_allclose(
        preds["actual"].to_numpy(), labeled[TARGET_COLUMN].iloc[-30:].to_numpy()
    )


def test_walk_forward_does_not_peek_at_the_future():
    # The acid test for any backtest: if we secretly change the FUTURE,
    # predictions made BEFORE that point must not move. Here we double the
    # last 5 days of prices; predictions from the first retrain block
    # (days 1-10 of a 20-day window) only ever saw data well before the
    # tampering, so they must be unchanged.
    #
    # (We allow ~1 part in 10^12 of wiggle: the forest predicts with
    # parallel threads, and summing 300 tree outputs in a different order
    # shifts the last decimal bit. A real leak moves values hugely more.)
    ohlcv = make_fake_ohlcv(days=1200)
    tampered = ohlcv.copy()
    tampered.iloc[-5:, tampered.columns.get_loc("Close")] *= 2.0

    original = walk_forward_predictions(build_features(ohlcv), test_days=20, retrain_every=10)
    altered = walk_forward_predictions(build_features(tampered), test_days=20, retrain_every=10)

    first_block = original.index[:10]
    for model_name in ("random_forest", "ridge"):
        np.testing.assert_allclose(
            original.loc[first_block, model_name].to_numpy(),
            altered.loc[first_block, model_name].to_numpy(),
            rtol=1e-12,
        )


def test_walk_forward_rejects_short_history():
    table = build_features(make_fake_ohlcv(days=200))
    with pytest.raises(ValueError, match="Not enough history"):
        walk_forward_predictions(table, test_days=100)
