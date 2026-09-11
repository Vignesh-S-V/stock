from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from app.trading import INDEX_UNIVERSE, PERIODS, POPULAR_STOCKS, add_indicators, backtest, position_size, regime_label, score_signal
from app.paper_engine import process_latest_bar, reset_paper_account, _money, _position_from_state

STRATEGIES = ["Ensemble", "Trend Momentum", "EMA Crossover", "RSI Reversion", "MACD", "Mean Reversion"]


def _finite(v):
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def _now():
    return datetime.now(timezone.utc).astimezone().strftime("%d %b %Y, %H:%M:%S")


def _live_data(symbol: str, history_period: str = "1d") -> pd.DataFrame:
    """Fresh Yahoo snapshot. Used only by the opt-in 1-second live display."""
    try:
        d = yf.download(symbol, period=history_period, interval="1m", auto_adjust=False, progress=False, threads=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        wanted = ["Open", "High", "Low", "Close", "Volume"]
        d = d[[c for c in wanted if c in d.columns]].dropna(subset=["Open", "High", "Low", "Close"])
        if not d.empty:
            d.index = pd.to_datetime(d.index)
        return d
    except Exception:
        return pd.DataFrame()


def _chart(x, sig, position=None):
    y = x.tail(120)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=y.index, open=y.Open, high=y.High, low=y.Low, close=y.Close, name="Market"))
    for c in ["EMA_9", "EMA_21", "EMA_50", "VWAP"]:
        if c in y:
            fig.add_trace(go.Scatter(x=y.index, y=y[c], mode="lines", name=c.replace("_", " ")))
    if sig.action != "HOLD":
        fig.add_hline(y=sig.entry, line_dash="dot", annotation_text=f"ENTRY {_money(sig.entry)}")
        fig.add_hline(y=sig.stop, line_dash="dash", annotation_text=f"STOP {_money(sig.stop)}")
        fig.add_hline(y=sig.target, line_dash="dash", annotation_text=f"TARGET {_money(sig.target)}")
    if position:
        fig.add_hline(y=position.entry, line_dash="dot", annotation_text=f"OPEN {position.side} {_money(position.entry)}")
        fig.add_hline(y=position.stop, line_dash="dash", annotation_text="LIVE STOP")
        fig.add_hline(y=position.target, line_dash="dash", annotation_text="LIVE TARGET")
    fig.update_layout(height=570, template="plotly_dark", paper_bgcolor="#080b10", plot_bgcolor="#080b10", margin=dict(l=8,r=8,t=25,b=8), xaxis_rangeslider_visible=False, hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
    return fig


def _accuracy(x, strategy, horizon=5, threshold=0):
    good = total = 0
    for i in range(1, len(x) - horizon):
        s = score_signal(x.iloc[i], strategy)
        if s.action == "HOLD" or s.confidence < threshold:
            continue
        future = float(x.Close.iloc[i + horizon])
        entry = float(s.entry)
        direction = 1 if s.action == "BUY" else -1
        good += ((future - entry) * direction > 0)
        total += 1
    return (100 * good / total if total else 0.0, total)


def _style():
    st.markdown("""
    <style>
    .stApp{background:#070a0f}.block-container{max-width:1580px;padding:12px 22px 50px}
    [data-testid="stSidebar"]{background:#0b1017;border-right:1px solid #202b37}
    .hero{background:linear-gradient(135deg,#101923,#0b1118);border:1px solid #263544;border-radius:18px;padding:18px 22px;margin-bottom:14px}
    .brand{font-size:31px;font-weight:900;letter-spacing:-1.2px}.sub{font-size:12px;color:#82909e;margin-top:3px}
    .live{display:inline-flex;align-items:center;gap:7px;border:1px solid #24533a;background:#0d2118;color:#6ee39a;border-radius:999px;padding:5px 9px;font-size:10px;font-weight:850}
    .dot{width:7px;height:7px;border-radius:50%;background:#48e58a;box-shadow:0 0 8px #48e58a}
    .section{font-size:19px;font-weight:850;margin:15px 0 8px}.signal{border:1px solid #2c3a48;border-radius:15px;padding:17px;background:#0d141c}.action{font-size:46px;font-weight:950;line-height:1}.buy{color:#58e39b}.sell{color:#ff7076}.hold{color:#e7c85b}
    .order{border:1px solid #334452;border-radius:13px;padding:13px;background:#0d151d}.small{font-size:11px;color:#81909f}.orderprice{font-size:23px;font-weight:850;margin:3px 0}
    div[data-testid="stMetric"]{background:#0e151d;border:1px solid #222e3a;border-radius:12px;padding:8px 10px}
    .stButton>button{border-radius:10px;font-weight:750}
    </style>
    """, unsafe_allow_html=True)


def _live_fragment(symbol, name, strategy, threshold, strict, reward, capital, risk, brokerage, auto):
    @st.fragment(run_every="1s")
    def live():
        fresh = _live_data(symbol, "1d")
        if fresh.empty:
            st.error("Live feed temporarily unavailable. Yahoo Finance may be throttling requests; the last successful snapshot is not fabricated.")
            return
        x = add_indicators(fresh)
        sig = score_signal(x.iloc[-1], strategy, reward_r=reward)
        qualified = sig.action != "HOLD" and sig.confidence >= threshold if strict else sig.action != "HOLD"
        qty = position_size(capital, risk, sig.entry, sig.stop) if qualified else 0
        last = float(x.Close.iloc[-1]); prev = float(x.Close.iloc[-2]) if len(x) > 1 else last
        change = (last / prev - 1) * 100 if prev else 0
        pos = _position_from_state(st.session_state.get("paper_position"))

        # Paper engine processes each newly completed Yahoo bar only. Mark-to-market is refreshed every second.
        if auto and qualified:
            process_latest_bar(x, strategy, risk, brokerage, reward)
            pos = _position_from_state(st.session_state.get("paper_position"))
        if pos and pos.side == "LONG":
            unrealized = pos.qty * (last - pos.entry)
        elif pos and pos.side == "SHORT":
            unrealized = pos.qty * (pos.entry - last)
        else:
            unrealized = 0.0
        cash = float(st.session_state.get("paper_cash", capital))
        equity = cash + (pos.qty * last if pos and pos.side == "LONG" else unrealized if pos else 0.0)

        st.markdown(f"<div class='hero'><span class='live'><span class='dot'></span>LIVE SNAPSHOT · 1 SECOND REFRESH · PAPER ONLY</span><div class='brand'>ALGO TRADING PRO</div><div class='sub'>{name} · {interval_label()} · Yahoo Finance snapshot · refreshed {_now()}</div></div>", unsafe_allow_html=True)
        a,b,c,d,e,f = st.columns(6)
        a.metric("LIVE PRICE", _money(last), f"{change:+.2f}%")
        b.metric("SIGNAL", sig.action, f"{sig.confidence:.1f}% evidence")
        c.metric("REGIME", regime_label(x.iloc[-1]))
        d.metric("PAPER EQUITY", _money(equity))
        e.metric("UNREALIZED P&L", _money(unrealized))
        f.metric("PAPER QTY", f"{qty:,}" if qty else "WAIT")

        left,right = st.columns([1.65,1])
        with left:
            st.markdown("<div class='section'>Live Market Chart · updates every second</div>", unsafe_allow_html=True)
            st.plotly_chart(_chart(x, sig, pos), use_container_width=True, config={"displaylogo":False,"scrollZoom":True})
        with right:
            st.markdown("<div class='section'>AI Trade Decision</div>", unsafe_allow_html=True)
            cls = sig.action.lower()
            st.markdown(f"<div class='signal'><div class='small'>CURRENT MODEL DECISION · {name}</div><div class='action {cls}'>{sig.action}</div><div class='small'>Evidence {sig.confidence:.1f}% · Risk/Reward 1:{reward:.1f}</div></div>", unsafe_allow_html=True)
            q1,q2=st.columns(2); q1.metric("ENTRY",_money(sig.entry)); q2.metric("QTY",f"{qty:,}" if qty else "—")
            q3,q4=st.columns(2); q3.metric("STOP",_money(sig.stop) if qualified else "WAIT"); q4.metric("TARGET",_money(sig.target) if qualified else "WAIT")
            if qualified:
                st.success(f"{'🟢 BUY' if sig.action=='BUY' else '🔴 SELL/SHORT'} · PAPER ORDER READY")
            else:
                st.warning(f"NO ORDER · requires ≥ {threshold}% evidence")
            st.markdown("#### Signal reasoning")
            for r in sig.reasons[:7]: st.write("• " + r)

        st.markdown("<div class='section'>Automatic Paper Execution</div>", unsafe_allow_html=True)
        if pos:
            p1,p2,p3,p4,p5=st.columns(5); p1.metric("POSITION",pos.side); p2.metric("ENTRY",_money(pos.entry)); p3.metric("LIVE PRICE",_money(last)); p4.metric("STOP",_money(pos.stop)); p5.metric("TARGET",_money(pos.target))
            st.markdown(f"<div class='order'><div class='small'>OPEN PAPER POSITION · {pos.opened_at}</div><div class='orderprice'>{pos.side} × {pos.qty:,} · Entry {_money(pos.entry)}</div><div class='small'>Live P&L {_money(unrealized)} · Evidence {pos.confidence:.1f}%</div></div>", unsafe_allow_html=True)
        else:
            st.info("FLAT · When a qualifying signal is processed, BUY/SELL, entry, stop, target and quantity will appear here automatically.")
        st.caption("1-second display refresh is enabled. Yahoo Finance does not provide a guaranteed exchange-grade tick feed; displayed prices update when Yahoo returns a newer snapshot. Paper execution is bar-based, not fabricated tick execution.")
    live()


def interval_label():
    return st.session_state.get("interval", "1m")


def render_app():
    _style()
    with st.sidebar:
        st.markdown("## ⚡ ALGO TRADING PRO")
        universe = st.selectbox("Market", ["Indices", "NSE Stocks", "Custom Ticker"])
        if universe == "Indices":
            name = st.selectbox("Instrument", list(INDEX_UNIVERSE)); symbol = INDEX_UNIVERSE[name]
        elif universe == "NSE Stocks":
            name = st.selectbox("Instrument", list(POPULAR_STOCKS)); symbol = POPULAR_STOCKS[name]
        else:
            symbol = st.text_input("Yahoo ticker", "RELIANCE.NS").strip().upper(); name = symbol
        interval = st.selectbox("Chart timeframe", list(PERIODS), index=min(4, len(PERIODS)-1), key="interval")
        period = st.selectbox("History", PERIODS[interval], index=0 if interval != "1d" else min(3,len(PERIODS[interval])-1))
        strategy = st.selectbox("Strategy", STRATEGIES)
        st.divider(); st.markdown("### 🎯 Precision Gate")
        threshold = st.slider("Minimum evidence", 50, 95, 90, 1)
        strict = st.checkbox("Trade only above gate", True)
        reward = st.slider("Target / Risk", 1.0, 5.0, 2.0, 0.5)
        st.divider(); st.markdown("### 🛡 Risk Engine")
        capital = st.number_input("Paper capital (₹)", 10000.0, 100000000.0, 100000.0, 10000.0)
        risk = st.slider("Risk / trade %", 0.1, 3.0, 1.0, 0.1)
        brokerage = st.number_input("Costs %", 0.0, 0.50, 0.03, 0.01, format="%.2f")
        st.divider(); st.markdown("### ⚡ Live Engine")
        auto = st.checkbox("AUTO PAPER TRADING", False, help="Runs the live display every second. Actual Yahoo data may be delayed or rate-limited.")
        if st.button("↻ Refresh now", use_container_width=True): st.rerun()
        if st.button("Reset paper account", use_container_width=True): reset_paper_account(capital); st.rerun()

    # For auto mode, the entire market/paper view is inside the 1-second fragment.
    if auto:
        _live_fragment(symbol, name, strategy, threshold, strict, reward, capital, risk, brokerage, auto)
        return

    df = __import__("app.trading", fromlist=["download_market_data"]).download_market_data(symbol, period, interval)
    if df.empty:
        st.error(f"No market data for {symbol}. Try 1d + 1mo."); return
    x = add_indicators(df); sig = score_signal(x.iloc[-1], strategy, reward_r=reward)
    qualified = sig.action != "HOLD" and sig.confidence >= threshold if strict else sig.action != "HOLD"
    qty = position_size(capital, risk, sig.entry, sig.stop) if qualified else 0
    last = float(x.Close.iloc[-1]); prev = float(x.Close.iloc[-2]) if len(x)>1 else last
    st.markdown(f"<div class='hero'><span class='live'><span class='dot'></span>MARKET CONNECTED · PAPER ONLY</span><div class='brand'>ALGO TRADING PRO</div><div class='sub'>{name} · {interval} · {period} · Last update {_now()}</div></div>", unsafe_allow_html=True)
    a,b,c,d,e = st.columns(5); a.metric("PRICE",_money(last),f"{((last/prev)-1)*100:+.2f}%" if prev else "0%"); b.metric("SIGNAL",sig.action,f"{sig.confidence:.1f}% evidence"); c.metric("REGIME",regime_label(x.iloc[-1])); d.metric("PAPER QTY",f"{qty:,}" if qty else "WAIT"); e.metric("R:R",f"1:{reward:.1f}")
    left,right=st.columns([1.65,1])
    with left:
        st.markdown("<div class='section'>Market Chart</div>",unsafe_allow_html=True); st.plotly_chart(_chart(x,sig),use_container_width=True,config={"displaylogo":False,"scrollZoom":True})
    with right:
        st.markdown("<div class='section'>AI Trade Decision</div>",unsafe_allow_html=True); st.markdown(f"<div class='signal'><div class='small'>CURRENT DECISION</div><div class='action {sig.action.lower()}'>{sig.action}</div><div class='small'>Evidence {sig.confidence:.1f}%</div></div>",unsafe_allow_html=True)
        q1,q2=st.columns(2); q1.metric("ENTRY",_money(sig.entry)); q2.metric("QTY",f"{qty:,}" if qty else "—")
        q3,q4=st.columns(2); q3.metric("STOP",_money(sig.stop) if qualified else "WAIT"); q4.metric("TARGET",_money(sig.target) if qualified else "WAIT")
        st.success("🟢 PAPER BUY READY" if qualified and sig.action=="BUY" else "🔴 PAPER SELL READY" if qualified else "🟡 NO ORDER")
        for r in sig.reasons[:7]: st.write("• "+r)
    st.markdown("### 🧪 Backtest Lab")
    eq,metrics = backtest(df,strategy,capital,risk,brokerage)
    m=st.columns(6)
    for col,label,key in zip(m,["Return","Net P&L","Max DD","Win Rate","Trades","Sharpe"],["Return %","Net P&L","Max Drawdown %","Win Rate %","Trades","Sharpe"]): col.metric(label, f"{metrics[key]:.2f}")
    fig=go.Figure(go.Scatter(x=eq.index,y=eq.Equity,mode="lines",name="Equity")); fig.update_layout(height=300,template="plotly_dark",paper_bgcolor="#080b10",plot_bgcolor="#080b10"); st.plotly_chart(fig,use_container_width=True)
    st.caption("Historical accuracy is diagnostic evidence, not a guaranteed future win rate. A 95% gate filters trades; it cannot create 95% future accuracy.")


if __name__ == "__main__":
    render_app()
