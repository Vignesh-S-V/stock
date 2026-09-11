from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit

FEATURES = [
    "RSI_14", "MACD", "MACD_HIST", "ATR_PCT", "BB_WIDTH", "VWAP_DIST",
    "ROC_12", "RET_1", "RET_5", "RET_20", "ADX_14", "STOCH_K", "STOCH_D",
    "CCI_20", "EMA9_21", "EMA21_50", "VOLUME_RATIO",
]
MODEL_TTL_SECONDS = 600
MODEL_THRESHOLD_DEFAULT = 95.0
MODEL_MIN_SAMPLES = 350
_MODEL_CACHE: dict[str, tuple[float, Any, dict[str, Any]]] = {}
_MODEL_LOCK = threading.Lock()


@dataclass
class MLSignal:
    action: str
    confidence: float
    trained: bool
    samples: int
    validation_accuracy: float | None
    threshold: float
    reason: str


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close = pd.to_numeric(x.get("Close"), errors="coerce")
    x["ATR_PCT"] = pd.to_numeric(x.get("ATR_14"), errors="coerce") / close.replace(0, np.nan) * 100
    x["VWAP_DIST"] = (close - pd.to_numeric(x.get("VWAP"), errors="coerce")) / close.replace(0, np.nan) * 100
    x["EMA9_21"] = (pd.to_numeric(x.get("EMA_9"), errors="coerce") / pd.to_numeric(x.get("EMA_21"), errors="coerce") - 1) * 100
    x["EMA21_50"] = (pd.to_numeric(x.get("EMA_21"), errors="coerce") / pd.to_numeric(x.get("EMA_50"), errors="coerce") - 1) * 100
    volume = pd.to_numeric(x.get("Volume"), errors="coerce")
    volume_sma = pd.to_numeric(x.get("VOLUME_SMA_20"), errors="coerce")
    x["VOLUME_RATIO"] = volume / volume_sma.replace(0, np.nan)
    return x.replace([np.inf, -np.inf], np.nan)


def _training_set(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = _prepare_features(df)
    close = pd.to_numeric(x["Close"], errors="coerce")
    # Predict a meaningful 5-minute move, scaled to the instrument's recent
    # volatility. This avoids labelling every tiny market fluctuation as a trade.
    future_return = close.shift(-5) / close - 1.0
    volatility = close.pct_change().rolling(30).std()
    adaptive = (volatility * 5.0 * 0.80).clip(lower=0.0008, upper=0.004)
    y = pd.Series(np.nan, index=x.index, dtype=float)
    y.loc[future_return > adaptive] = 1.0
    y.loc[future_return < -adaptive] = 0.0
    data = x[FEATURES].copy()
    mask = data.notna().all(axis=1) & y.notna()
    return data.loc[mask], y.loc[mask].astype(int)


def _calibrated_model(base: Any, X: pd.DataFrame, y: pd.Series):
    splitter = TimeSeriesSplit(n_splits=3)
    for train_idx, test_idx in splitter.split(X):
        if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
            return None
    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=splitter, ensemble=True)
    model.fit(X, y)
    return model


def _fit(df: pd.DataFrame) -> tuple[Any | None, dict[str, Any]]:
    X, y = _training_set(df)
    if len(X) < MODEL_MIN_SAMPLES or y.nunique() < 2:
        return None, {"samples": int(len(X)), "validation_accuracy": None}

    split = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return None, {"samples": int(len(X)), "validation_accuracy": None}

    rf = RandomForestClassifier(
        n_estimators=240, max_depth=10, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=42, n_jobs=1,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=180, learning_rate=0.04, max_leaf_nodes=15,
        l2_regularization=0.5, random_state=42,
    )
    base = VotingClassifier(
        estimators=[("rf", rf), ("hgb", hgb)], voting="soft", weights=[1.0, 1.2], n_jobs=1
    )
    try:
        validation_model = _calibrated_model(base, X_train, y_train)
        if validation_model is None:
            return None, {"samples": int(len(X)), "validation_accuracy": None}
        validation_accuracy = float((validation_model.predict(X_test) == y_test).mean())
        final_model = _calibrated_model(base, X, y)
        if final_model is None:
            return None, {"samples": int(len(X)), "validation_accuracy": validation_accuracy}
    except (ValueError, RuntimeError, TypeError):
        return None, {"samples": int(len(X)), "validation_accuracy": None}

    return final_model, {"samples": int(len(X)), "validation_accuracy": validation_accuracy}


def predict_ml_signal(df: pd.DataFrame, symbol: str, threshold: float = MODEL_THRESHOLD_DEFAULT) -> MLSignal:
    """Return a calibrated probability; never inflate confidence to hit 95%."""
    threshold = min(100.0, max(50.0, float(threshold)))
    now = time.time()
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(symbol)
        if cached and now - cached[0] < MODEL_TTL_SECONDS:
            model, meta = cached[1], cached[2]
        else:
            model, meta = _fit(df)
            _MODEL_CACHE[symbol] = (now, model, meta)

    if model is None:
        return MLSignal("HOLD", 0.0, False, int(meta.get("samples", 0)), meta.get("validation_accuracy"), threshold, "Model did not meet minimum data/calibration requirements.")

    latest = _prepare_features(df).iloc[[-1]][FEATURES]
    if latest.isna().any(axis=None):
        return MLSignal("HOLD", 0.0, True, int(meta["samples"]), meta.get("validation_accuracy"), threshold, "Latest feature row is incomplete.")

    probabilities = model.predict_proba(latest)[0]
    classes = list(model.classes_)
    p_down = float(probabilities[classes.index(0)]) if 0 in classes else 0.0
    p_up = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
    best = max(p_up, p_down)
    confidence = round(min(100.0, max(0.0, best * 100.0)), 1)

    if p_up >= threshold / 100.0 and p_up > p_down:
        action = "BUY"
        reason = f"Calibrated ML probability {confidence:.1f}% reached the {threshold:.1f}% trade threshold."
    elif p_down >= threshold / 100.0 and p_down > p_up:
        action = "SELL"
        reason = f"Calibrated ML probability {confidence:.1f}% reached the {threshold:.1f}% trade threshold."
    else:
        action = "HOLD"
        reason = f"Calibrated ML probability {confidence:.1f}% is below the {threshold:.1f}% trade threshold."
    return MLSignal(action, confidence, True, int(meta["samples"]), meta.get("validation_accuracy"), threshold, reason)
