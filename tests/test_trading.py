import numpy as np
import pandas as pd
from app.trading import add_indicators, position_size, score_signal


def sample_df(n=260):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = np.linspace(100, 180, n) + np.sin(np.arange(n))*2
    return pd.DataFrame({"Open": close-1, "High": close+2, "Low": close-2,
                         "Close": close, "Volume": np.full(n, 100000)}, index=idx)


def test_ema_21_exists_and_is_populated():
    x = add_indicators(sample_df())
    assert "EMA_21" in x.columns
    assert x["EMA_21"].notna().sum() > 200


def test_position_size():
    assert position_size(100000, 1, 100, 95) == 200


def test_signal_is_bounded():
    s = score_signal(add_indicators(sample_df()).iloc[-1], "Ensemble")
    assert s.action in {"BUY", "SELL", "HOLD"}
    assert 0 <= s.confidence <= 100


def test_single_strategy_never_claims_100_percent():
    s = score_signal(add_indicators(sample_df()).iloc[-1], "EMA Crossover")
    assert s.confidence < 100
