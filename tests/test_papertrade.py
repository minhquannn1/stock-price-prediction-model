"""Tests for the paper-trading simulation."""

import numpy as np
import pytest

import veritas.papertrade as papertrade_module
from veritas.papertrade import STARTING_CASH, run_paper_trade
from tests.test_features import make_fake_ohlcv


@pytest.fixture
def fake_data(monkeypatch):
    ohlcv = make_fake_ohlcv(days=1000)
    monkeypatch.setattr(papertrade_module, "fetch_history", lambda ticker: ohlcv)
    return ohlcv


def test_paper_trade_produces_one_row_per_day(fake_data):
    result = run_paper_trade("FAKE", days=10)
    assert len(result.days) == 10
    assert 0 <= result.direction_hits <= 10


def test_paper_trade_grades_against_the_real_next_close(fake_data):
    result = run_paper_trade("FAKE", days=8)
    close = fake_data["Close"]
    for day in result.days:
        pos = close.index.get_loc(day.date)
        # The graded "actual" must be the true close on that date.
        assert day.actual_close == pytest.approx(float(close.iloc[pos]))
        # And prior_close is the day before.
        assert day.prior_close == pytest.approx(float(close.iloc[pos - 1]))


def test_cash_days_do_not_change_equity(fake_data):
    result = run_paper_trade("FAKE", days=10)
    prev = STARTING_CASH
    for day in result.days:
        if not day.invested:
            # Sitting in cash: portfolio must be unchanged from the day before.
            assert day.equity == pytest.approx(prev)
        prev = day.equity


def test_buy_and_hold_is_independent_of_predictions(fake_data):
    # Buy & hold compounds every day's real return regardless of the model.
    result = run_paper_trade("FAKE", days=10)
    close = fake_data["Close"]
    expected = STARTING_CASH
    for day in result.days:
        pos = close.index.get_loc(day.date)
        expected *= close.iloc[pos] / close.iloc[pos - 1]
    assert result.buy_hold_equity == pytest.approx(expected, rel=1e-9)
