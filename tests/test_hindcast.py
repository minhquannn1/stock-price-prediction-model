"""Tests for the hindcast (rewind-and-grade) backtest."""

import numpy as np
import pytest

import veritas.hindcast as hindcast_module
from veritas.hindcast import run_hindcast
from tests.test_features import make_fake_ohlcv


@pytest.fixture
def fake_data(monkeypatch):
    ohlcv = make_fake_ohlcv(days=1400)
    monkeypatch.setattr(hindcast_module, "fetch_history", lambda ticker: ohlcv)
    return ohlcv


def test_hindcast_grades_against_the_true_future_price(fake_data):
    reports, names, rows, ohlcv = run_hindcast("FAKE", days_back=63)

    # Only horizons that fit inside the rewind window get graded.
    assert names == ["tomorrow", "next week", "next month", "next quarter"]

    cutoff_pos = len(ohlcv) - 63 - 1
    for row, report in zip(rows, reports):
        expected_actual = float(ohlcv["Close"].iloc[cutoff_pos + report.horizon])
        assert row.actual == pytest.approx(expected_actual)
        # error_pct must be consistent with predicted and actual.
        assert row.error_pct == pytest.approx(
            (row.predicted - row.actual) / row.actual * 100.0
        )


def test_hindcast_models_never_see_the_rewound_future(fake_data):
    reports, _, _, ohlcv = run_hindcast("FAKE", days_back=63)
    cutoff = ohlcv.index[len(ohlcv) - 63 - 1]
    for report in reports:
        assert report.last_date == cutoff  # trained on data up to the cutoff only


def test_hindcast_rejects_rewinding_past_the_data(fake_data):
    with pytest.raises(ValueError, match="Can't rewind"):
        run_hindcast("FAKE", days_back=1395)
