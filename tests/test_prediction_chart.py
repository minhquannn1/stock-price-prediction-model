"""Test that the forecast PNG actually gets rendered."""

from veritas.features import build_features
from veritas.model import train_and_predict
from veritas.prediction_chart import save_prediction_chart
from tests.test_features import make_fake_ohlcv


def test_chart_file_is_created(tmp_path, monkeypatch):
    import veritas.prediction_chart as chart_module

    monkeypatch.setattr(chart_module, "CHART_DIR", tmp_path)

    ohlcv = make_fake_ohlcv(days=1200)
    reports = [
        train_and_predict("FAKE", build_features(ohlcv, horizon=h), ohlcv["Close"], horizon=h)
        for h in (1, 5)
    ]

    path = save_prediction_chart(reports, ["tomorrow", "next week"], "Fake Corp", ohlcv["Close"])

    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 10_000  # a real image, not an empty stub
