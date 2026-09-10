from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from app.trading import INDEX_UNIVERSE, PERIODS, POPULAR_STOCKS, add_indicators, download_market_data, position_size, score_signal


@dataclass
class PaperPosition:
    side: str
    qty: int
    entry: float
    stop: float
    target: float
    opened_at: str
    confidence: float


def _finite(v):
    try:
        v = float(v)
        return math.isfinite(v)
    except Exception:
        return False


def reset_paper_account(capital: float):
    st.session_state.paper_cash = float(capital)
    st.session_state.paper_position = None
    st.session_state.paper_trades = []
    st.session_state.paper_equity = float(capital)
    st.session_state.paper_last_bar = None


def _init_state(capital: float):
    if "paper_cash" not in st.session_state:
        reset_paper_account(capital)


def _trade_cost(price: float, qty: int, brokerage_pct: float) -> float:
    return abs(price * qty) * brokerage_pct / 100.0


def process_latest_bar(x: pd.DataFrame, strategy: str, risk_pct: float, brokerage_pct: float, reward_r: float = 2.0):
    """Process only a new bar in the session-based paper account.

    Entry is at the bar open using the previous bar's signal to avoid look-ahead.
    Exit is checked against the current bar's high/low. If target and stop are both
    touched in the same bar, stop is assumed first (conservative).
    """
    if len(x) < 2:
        return
    prev = x.iloc[-2]
    bar = x.iloc[-1]
    idx = x.index[-1]
    bar_key = str(idx)
    if st.session_state.get("paper_last_bar") == bar_key:
        return

    pos = st.session_state.get("paper_position")
    cash = float(st.session_state.paper_cash)
    signal = score_signal(prev, strategy, reward_r=reward_r)
    o, h, l, c = map(float, [bar["Open"], bar["High"], bar["Low"], bar["Close"]])

    if pos is not None:
        p = PaperPosition(**pos)
        exit_price = None
        reason = None
        if p.side == "LONG":
            if l <= p.stop:
                exit_price, reason = p.stop, "STOP LOSS"
            elif h >= p.target:
                exit_price, reason = p.target, "2R TARGET"
            elif signal.action == "SELL":
                exit_price, reason = o, "SIGNAL EXIT"
            if exit_price is not None:
                gross = p.qty * (exit_price - p.entry)
                fee = _trade_cost(exit_price, p.qty, brokerage_pct)
                pnl = gross - fee
                cash += p.qty * exit_price - fee
                st.session_state.paper_trades.append({"Time": str(idx), "Action": "SELL", "Qty": p.qty, "Price": exit_price, "PnL": pnl, "Reason": reason})
                st.session_state.paper_position = None
        else:
            if h >= p.stop:
                exit_price, reason = p.stop, "STOP LOSS"
            elif l <= p.target:
                exit_price, reason = p.target, "2R TARGET"
            elif signal.action == "BUY":
                exit_price, reason = o, "SIGNAL EXIT"
            if exit_price is not None:
                gross = p.qty * (p.entry - exit_price)
                fee = _trade_cost(exit_price, p.qty, brokerage_pct)
                pnl = gross - fee
                cash += pnl
                st.session_state.paper_trades.append({"Time": str(idx), "Action": "COVER", "Qty": p.qty, "Price": exit_price, "PnL": pnl, "Reason": reason})
                st.session_state.paper_position = None

    if st.session_state.get("paper_position") is None and signal.action in {"BUY", "SELL"} and _finite(o):
        stop_distance = abs(float(signal.entry) - float(signal.stop))
        if stop_distance > 0:
            qty = position_size(max(cash, 0), risk_pct, o, o - stop_distance if signal.action == "BUY" else o + stop_distance)
            qty = max(0, int(qty))
            if qty > 0:
                fee = _trade_cost(o, qty, brokerage_pct)
                if signal.action == "BUY":
                    cash -= qty * o + fee
                    stop = o - stop_distance
                    target = o + reward_r * stop_distance
                    side = "LONG"
                    action = "BUY"
                else:
                    cash -= fee
                    stop = o + stop_distance
                    target = o - reward_r * stop_distance
                    side = "SHORT"
                    action = "SHORT"
                st.session_state.paper_position = asdict(PaperPosition(side, qty, o, stop, target, str(idx), signal.confidence))
                st.session_state.paper_trades.append({"Time": str(idx), "Action": action, "Qty": qty, "Price": o, "PnL": -fee, "Reason": "AUTO ENTRY"})

    pos = st.session_state.get("paper_position")
    if pos:
        p = PaperPosition(**pos)
        unrealized = p.qty * (c - p.entry) if p.side == "LONG" else p.qty * (p.entry - c)
        equity = cash + (p.qty * c if p.side == "LONG" else 0) + (unrealized if p.side == "SHORT" else 0)
    else:
        equity = cash
    st.session_state.paper_cash = cash
    st.session_state.paper_equity = float(equity)
    st.session_state.paper_last_bar = bar_key


def run_historical_2r(df: pd.DataFrame, strategy: str, initial_capital: float, risk_pct: float, brokerage_pct: float, reward_r: float = 2.0):
    """Walk-forward simulation: signal on t-1, entry on t open, then 1R/2R exits."""
    x = add_indicators(df).dropna(subset=["Open", "High", "Low", "Close"])
    cash = float(initial_capital)
    position = None
    trades = []
    equity = []

    for i in range(1, len(x)):
        prev, bar = x.iloc[i - 1], x.iloc[i]
        price = float(bar["Open"])
        high, low, close = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
        sig = score_signal(prev, strategy, reward_r=reward_r)

        if position is not None:
            p = position
            exit_price = None; reason = None
            if p["side"] == "LONG":
                if low <= p["stop"]: exit_price, reason = p["stop"], "STOP LOSS"
                elif high >= p["target"]: exit_price, reason = p["target"], "2R TARGET"
                elif sig.action == "SELL": exit_price, reason = price, "SIGNAL EXIT"
                if exit_price is not None:
                    fee = _trade_cost(exit_price, p["qty"], brokerage_pct)
                    pnl = p["qty"] * (exit_price - p["entry"]) - fee
                    cash += p["qty"] * exit_price - fee
                    trades.append({"Time": str(x.index[i]), "Side": "LONG", "Entry": p["entry"], "Exit": exit_price, "Qty": p["qty"], "PnL": pnl, "Reason": reason})
                    position = None
            else:
                if high >= p["stop"]: exit_price, reason = p["stop"], "STOP LOSS"
                elif low <= p["target"]: exit_price, reason = p["target"], "2R TARGET"
                elif sig.action == "BUY": exit_price, reason = price, "SIGNAL EXIT"
                if exit_price is not None:
                    fee = _trade_cost(exit_price, p["qty"], brokerage_pct)
                    pnl = p["qty"] * (p["entry"] - exit_price) - fee
                    cash += pnl
                    trades.append({"Time": str(x.index[i]), "Side": "SHORT", "Entry": p["entry"], "Exit": exit_price, "Qty": p["qty"], "PnL": pnl, "Reason": reason})
                    position = None

        if position is None and sig.action in {"BUY", "SELL"}:
            dist = abs(float(sig.entry) - float(sig.stop))
            if dist > 0:
                qty = position_size(max(cash, 0), risk_pct, price, price - dist if sig.action == "BUY" else price + dist)
                if qty > 0:
                    fee = _trade_cost(price, qty, brokerage_pct)
                    cash -= fee
                    if sig.action == "BUY":
                        position = {"side":"LONG", "qty":qty, "entry":price, "stop":price-dist, "target":price+reward_r*dist}
                    else:
                        position = {"side":"SHORT", "qty":qty, "entry":price, "stop":price+dist, "target":price-reward_r*dist}

        if position is None:
            eq = cash
        elif position["side"] == "LONG":
            eq = cash + position["qty"] * close
        else:
            eq = cash + position["qty"] * (position["entry"] - close)
        equity.append(eq)

    if position is not None:
        close = float(x["Close"].iloc[-1]); p = position
        if p["side"] == "LONG": pnl = p["qty"] * (close-p["entry"]); cash += p["qty"]*close + pnl*0
        else: pnl = p["qty"] * (p["entry"]-close); cash += pnl
        trades.append({"Time":str(x.index[-1]),"Side":p["side"],"Entry":p["entry"],"Exit":close,"Qty":p["qty"],"PnL":pnl,"Reason":"END OF TEST"})

    t = pd.DataFrame(trades)
    if t.empty:
        metrics = {"Trades":0,"Win Rate %":0.0,"Target Hit %":0.0,"Net P&L":cash-initial_capital,"Return %":(cash/initial_capital-1)*100,"Profit Factor":0.0,"Max Drawdown %":0.0}
    else:
        wins = t["PnL"] > 0
        gross_profit = t.loc[wins,"PnL"].sum(); gross_loss = abs(t.loc[~wins,"PnL"].sum())
        target_hits = (t["Reason"] == "2R TARGET").mean()*100
        eqs = pd.Series(equity) if equity else pd.Series([initial_capital])
        dd = (eqs-eqs.cummax())/eqs.cummax().replace(0,np.nan)
        metrics = {"Trades":len(t),"Win Rate %":float(wins.mean()*100),"Target Hit %":float(target_hits),"Net P&L":float(cash-initial_capital),"Return %":float((cash/initial_capital-1)*100),"Profit Factor":float(gross_profit/gross_loss) if gross_loss else float("inf"),"Max Drawdown %":float(dd.min()*100)}
    return pd.DataFrame({"Equity":equity}), metrics, t


def signal_accuracy(df: pd.DataFrame, strategy: str, horizon: int = 5, reward_r: float = 2.0):
    """Out-of-sample directional accuracy and 2R target/stop outcome rate."""
    x = add_indicators(df)
    records=[]
    for i in range(1, len(x)-horizon):
        sig = score_signal(x.iloc[i], strategy, reward_r=reward_r)
        if sig.action == "HOLD" or not _finite(sig.entry):
            continue
        future = x.iloc[i+1:i+1+horizon]
        entry = float(sig.entry); direction = 1 if sig.action == "BUY" else -1
        last_close = float(future["Close"].iloc[-1])
        directional = (last_close-entry)*direction > 0
        target = float(sig.target); stop = float(sig.stop)
        hit = "NEITHER"
        for _, b in future.iterrows():
            if direction == 1:
                if float(b["Low"]) <= stop: hit="STOP"; break
                if float(b["High"]) >= target: hit="TARGET"; break
            else:
                if float(b["High"]) >= stop: hit="STOP"; break
                if float(b["Low"]) <= target: hit="TARGET"; break
        records.append({"Time":str(x.index[i]),"Signal":sig.action,"Confidence":sig.confidence,"Directional Correct":directional,"2R Outcome":hit})
    r=pd.DataFrame(records)
    if r.empty:
        return r,{"Accuracy %":0.0,"2R Target Hit %":0.0,"Signals":0}
    return r,{"Accuracy %":float(r["Directional Correct"].mean()*100),"2R Target Hit %":float((r["2R Outcome"]=="TARGET").mean()*100),"Signals":len(r)}


def render_app():
    st.title("🚀 Advanced Algo Trading — Paper Lab")
    st.caption("Automatic BUY/SELL simulation only. No real broker orders are sent.")
    st.warning("🧪 PAPER TRADING MODE: The engine can automatically enter and exit trades, but this app never places real-money orders.")

    with st.sidebar:
        st.header("Market")
        universe = st.selectbox("Universe", ["Indices", "Popular Stocks", "Custom Ticker"])
        if universe == "Indices":
            label = st.selectbox("Instrument", list(INDEX_UNIVERSE))
            symbol = INDEX_UNIVERSE[label]
        elif universe == "Popular Stocks":
            label = st.selectbox("Instrument", list(POPULAR_STOCKS))
            symbol = POPULAR_STOCKS[label]
        else:
            label = st.text_input("Yahoo ticker", "RELIANCE.NS").strip().upper()
            symbol = label
        interval = st.selectbox("Interval", list(PERIODS))
        period = st.selectbox("History", PERIODS[interval], index=0 if interval != "1d" else 3)
        strategy = st.selectbox("Strategy", ["Ensemble", "Trend Momentum", "EMA Crossover", "RSI Reversion", "MACD", "Mean Reversion"])
        capital = st.number_input("Paper capital (₹)", min_value=10000.0, value=100000.0, step=10000.0)
        risk_pct = st.slider("Risk per trade (%)", 0.1, 3.0, 1.0, 0.1)
        brokerage = st.number_input("Brokerage/slippage assumption (%)", 0.0, 0.50, 0.03, 0.01)
        reward_r = st.number_input("Reward : Risk", 1.0, 5.0, 2.0, 0.5)
        auto = st.checkbox("Auto paper trading (60s refresh)", value=False)
        if st.button("Reset Paper Account", use_container_width=True):
            reset_paper_account(capital)

    df = download_market_data(symbol, period, interval)
    if df.empty:
        st.error(f"No market data returned for {symbol} ({interval}, {period}). Try 1d + 1mo first.")
        return
    x = add_indicators(df)
    sig = score_signal(x.iloc[-1], strategy, reward_r=reward_r)
    _init_state(capital)

    if auto:
        @st.fragment(run_every="60s")
        def live_fragment():
            process_latest_bar(x, strategy, risk_pct, brokerage, reward_r)
            _render_live(sig)
        live_fragment()
    else:
        if st.button("▶ Run Latest Paper Bar", type="primary"):
            process_latest_bar(x, strategy, risk_pct, brokerage, reward_r)
        _render_live(sig)

    tab1, tab2, tab3 = st.tabs(["📊 2R Backtest", "🎯 Accuracy", "📒 Paper Trade Log"])
    with tab1:
        eq, metrics, trades = run_historical_2r(df, strategy, capital, risk_pct, brokerage, reward_r)
        cols = st.columns(6)
        for col, (k, v) in zip(cols, [("Net P&L",metrics["Net P&L"]),("Return %",metrics["Return %"]),("Win Rate %",metrics["Win Rate %"]),("2R Target %",metrics["Target Hit %"]),("Max DD %",metrics["Max Drawdown %"]),("Trades",metrics["Trades"]) ]):
            col.metric(k, f"{v:,.2f}" if isinstance(v,float) else str(v))
        st.line_chart(eq)
        if not trades.empty: st.dataframe(trades.tail(100), use_container_width=True)
    with tab2:
        for horizon in [1, 3, 5, 10]:
            _, acc = signal_accuracy(df, strategy, horizon, reward_r)
            st.metric(f"Directional accuracy — next {horizon} bars", f"{acc['Accuracy %']:.1f}%", f"{acc['Signals']} signals")
        st.caption("Accuracy is historical out-of-sample directional correctness; it is not a probability of future profit.")
    with tab3:
        trades = pd.DataFrame(st.session_state.get("paper_trades", []))
        if trades.empty: st.info("No automatic paper trades yet. Click Run Latest Paper Bar or enable 60s auto mode.")
        else: st.dataframe(trades.tail(200), use_container_width=True)

    st.subheader("Price & indicators")
    st.line_chart(x.tail(250)[["Close","EMA_9","EMA_21","EMA_50","SMA_50","BB_UPPER","BB_LOWER","VWAP"]])


def _render_live(sig):
    pos = st.session_state.get("paper_position")
    a,b,c,d,e = st.columns(5)
    a.metric("Signal", sig.action)
    b.metric("Evidence", f"{sig.confidence:.1f}%")
    c.metric("Paper Equity", f"₹{st.session_state.paper_equity:,.2f}")
    d.metric("Cash", f"₹{st.session_state.paper_cash:,.2f}")
    e.metric("Position", f"{pos['side']} x {pos['qty']}" if pos else "FLAT")
    if pos:
        p=PaperPosition(**pos)
        st.info(f"AUTO POSITION: {p.side} | Entry ₹{p.entry:,.2f} | Stop ₹{p.stop:,.2f} | Target ₹{p.target:,.2f} | R:R {((abs(p.target-p.entry)/abs(p.entry-p.stop)) if p.entry!=p.stop else 0):.1f}:1")
    st.caption(f"Last processed bar: {st.session_state.get('paper_last_bar') or 'not started'}")
