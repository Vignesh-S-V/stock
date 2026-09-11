from __future__ import annotations

from dataclasses import asdict
from math import isfinite
from typing import Optional

import pandas as pd
import streamlit as st

from app.paper_engine import PaperPosition, _money, _position_from_state, _trade_cost, position_size
from app.trading import score_signal


def _finite(value) -> bool:
    try:
        return isfinite(float(value))
    except Exception:
        return False


def _ensure_state(capital: float) -> None:
    if "paper_cash" not in st.session_state:
        st.session_state.paper_cash = float(capital)
        st.session_state.paper_position = None
        st.session_state.paper_trades = []
        st.session_state.paper_equity = float(capital)
        st.session_state.paper_last_entry_bar = None
        st.session_state.paper_last_exit_bar = None


def _record(time_key: str, action: str, side: str, qty: int, price: float, pnl: float, reason: str) -> None:
    st.session_state.paper_trades.append(
        {
            "Time": time_key,
            "Action": action,
            "Side": side,
            "Qty": int(qty),
            "Price": float(price),
            "PnL": float(pnl),
            "Reason": reason,
        }
    )


def _close_position(price: float, time_key: str, brokerage_pct: float, reason: str) -> Optional[PaperPosition]:
    pos = _position_from_state(st.session_state.get("paper_position"))
    if pos is None:
        return None
    fee = _trade_cost(price, pos.qty, brokerage_pct)
    if pos.side == "LONG":
        gross = pos.qty * (price - pos.entry)
        pnl = gross - fee
        st.session_state.paper_cash += pos.qty * price - fee
        action = "SELL"
    else:
        gross = pos.qty * (pos.entry - price)
        pnl = gross - fee
        st.session_state.paper_cash += gross - fee
        action = "COVER"
    _record(time_key, action, pos.side, pos.qty, price, pnl, reason)
    st.session_state.paper_position = None
    st.session_state.paper_last_exit_bar = time_key
    return pos


def live_paper_step(
    x: pd.DataFrame,
    strategy: str,
    risk_pct: float,
    brokerage_pct: float,
    reward_r: float,
    threshold: float,
    strict: bool,
    auto: bool,
    force_action: Optional[str] = None,
) -> dict:
    """Execute one server-side live paper-trading step from the latest quote.

    This function is intentionally stateful: it can enter once per live bar/signal,
    then close immediately when the live price reaches stop/target or the model flips.
    It never sends a broker/real-money order.
    """
    if x is None or x.empty:
        return {"action": "HOLD", "qualified": False, "price": None, "position": None, "event": None}

    _ensure_state(float(st.session_state.get("paper_cash", 0) or 0))
    bar = x.iloc[-1]
    time_key = str(x.index[-1])
    price = float(bar["Close"])
    if not _finite(price):
        return {"action": "HOLD", "qualified": False, "price": None, "position": _position_from_state(st.session_state.get("paper_position")), "event": None}

    sig = score_signal(bar, strategy, reward_r=reward_r)
    qualified = sig.action != "HOLD" and (not strict or sig.confidence >= threshold)
    pos = _position_from_state(st.session_state.get("paper_position"))
    event = None

    # Target/stop are checked against the live quote on every fragment tick.
    if pos is not None:
        if pos.side == "LONG" and price <= pos.stop:
            _close_position(pos.stop, time_key, brokerage_pct, "STOP LOSS")
            event = "SELL · STOP LOSS"
            pos = None
        elif pos.side == "LONG" and price >= pos.target:
            _close_position(pos.target, time_key, brokerage_pct, "TARGET HIT")
            event = "SELL · TARGET HIT"
            pos = None
        elif pos.side == "SHORT" and price >= pos.stop:
            _close_position(pos.stop, time_key, brokerage_pct, "STOP LOSS")
            event = "COVER · STOP LOSS"
            pos = None
        elif pos.side == "SHORT" and price <= pos.target:
            _close_position(pos.target, time_key, brokerage_pct, "TARGET HIT")
            event = "COVER · TARGET HIT"
            pos = None
        elif (pos.side == "LONG" and sig.action == "SELL") or (pos.side == "SHORT" and sig.action == "BUY"):
            _close_position(price, time_key, brokerage_pct, "SIGNAL FLIP")
            event = "SELL/COVER · SIGNAL FLIP"
            pos = None

    # Auto entry happens from the current live signal. Manual buttons use force_action.
    requested = force_action if force_action in {"BUY", "SELL"} else (sig.action if auto and qualified else None)
    if pos is None and requested in {"BUY", "SELL"}:
        # Never re-enter repeatedly during the same live bar after an exit.
        if st.session_state.get("paper_last_entry_bar") != time_key:
            dist = abs(float(sig.entry) - float(sig.stop)) if _finite(sig.entry) and _finite(sig.stop) else 0.0
            if dist > 0:
                stop = price - dist if requested == "BUY" else price + dist
                qty = max(0, int(position_size(max(float(st.session_state.paper_cash), 0), risk_pct, price, stop)))
                if qty > 0:
                    fee = _trade_cost(price, qty, brokerage_pct)
                    if requested == "BUY":
                        total = qty * price + fee
                        if total <= float(st.session_state.paper_cash):
                            st.session_state.paper_cash -= total
                            pos = PaperPosition("LONG", qty, price, stop, price + reward_r * dist, time_key, float(sig.confidence))
                            _record(time_key, "BUY", "LONG", qty, price, -fee, "AUTO ENTRY" if auto else "MANUAL PAPER BUY")
                            event = "BUY · PAPER ORDER OPENED"
                    else:
                        st.session_state.paper_cash -= fee
                        pos = PaperPosition("SHORT", qty, price, stop, price - reward_r * dist, time_key, float(sig.confidence))
                        _record(time_key, "SHORT", "SHORT", qty, price, -fee, "AUTO ENTRY" if auto else "MANUAL PAPER SELL/SHORT")
                        event = "SELL/SHORT · PAPER ORDER OPENED"
                    if pos is not None:
                        st.session_state.paper_position = asdict(pos)
                        st.session_state.paper_last_entry_bar = time_key

    pos = _position_from_state(st.session_state.get("paper_position"))
    if pos is None:
        equity = float(st.session_state.paper_cash)
    elif pos.side == "LONG":
        equity = float(st.session_state.paper_cash) + pos.qty * price
    else:
        equity = float(st.session_state.paper_cash) + pos.qty * (pos.entry - price)
    st.session_state.paper_equity = equity

    return {
        "action": sig.action,
        "confidence": float(sig.confidence),
        "qualified": bool(qualified),
        "price": price,
        "entry": float(sig.entry) if _finite(sig.entry) else None,
        "stop": float(sig.stop) if _finite(sig.stop) else None,
        "target": float(sig.target) if _finite(sig.target) else None,
        "reasons": sig.reasons,
        "position": pos,
        "event": event,
        "trades": st.session_state.paper_trades[-5:],
    }
