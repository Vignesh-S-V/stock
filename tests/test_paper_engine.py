import pandas as pd
import numpy as np

from app.paper_engine import run_historical_2r, signal_accuracy


def synthetic_df(n=320):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.linspace(0, 40, n) + np.sin(np.arange(n))
    return pd.DataFrame({
        "Open": close,
        "High": close + 2,
        "Low": close - 2,
        "Close": close,
        "Volume": np.full(n, 100000),
    }, index=idx)


def test_2r_engine_returns_metrics():
    _, metrics, trades = run_historical_2r(synthetic_df(), "EMA Crossover", 100000, 1.0, 0.03, 2.0)
    assert set(["Trades", "Win Rate %", "Target Hit %", "Net P&L", "Max Drawdown %"]).issubset(metrics)
    assert metrics["Trades"] >= 0
    assert isinstance(trades, pd.DataFrame)


def test_accuracy_is_bounded():
    _, metrics = signal_accuracy(synthetic_df(), "Ensemble", horizon=5, reward_r=2.0)
    assert 0 <= metrics["Accuracy %"] <= 100
    assert 0 <= metrics["2R Target Hit %"] <= 100
