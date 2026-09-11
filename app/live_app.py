from __future__ import annotations

from datetime import datetime, timezone

import plotly.graph_objects as go
import streamlit as st

from app.live_feed import fetch_live_1m
from app.paper_engine import _money, _position_from_state, _init_state, process_latest_bar, reset_paper_account
from app.trading import INDEX_UNIVERSE, PERIODS, POPULAR_STOCKS, add_indicators, backtest, position_size, regime_label, score_signal

STRATEGIES = ["Ensemble", "Trend Momentum", "EMA Crossover", "RSI Reversion", "MACD", "Mean Reversion"]


def _now():
    return datetime.now(timezone.utc).astimezone().strftime("%d %b %Y, %H:%M:%S")


def _data(symbol, period, interval):
    try:
        d = fetch_live_1m(symbol, period if interval in {"1m", "5m"} else "1d") if interval in {"1m", "5m"} else None
        if d is not None and not d.empty:
            return d
        import yfinance as yf
        d = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
        if hasattr(d.columns, "levels"):
            d.columns = d.columns.get_level_values(0)
        return d[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in d.columns]].dropna(subset=["Open", "High", "Low", "Close"])
    except Exception:
        return None


def _chart(x, sig, pos=None):
    y = x.tail(160)
    fig = go.Figure(go.Candlestick(x=y.index, open=y.Open, high=y.High, low=y.Low, close=y.Close, name="Market"))
    for c in ["EMA_9", "EMA_21", "EMA_50", "VWAP"]:
        if c in y:
            fig.add_trace(go.Scatter(x=y.index, y=y[c], mode="lines", name=c.replace("_", " ")))
    if sig.action != "HOLD":
        for val, text, dash in [(sig.entry, "ENTRY", "dot"), (sig.stop, "STOP", "dash"), (sig.target, "TARGET", "dash")]:
            fig.add_hline(y=val, line_dash=dash, annotation_text=f"{text} {_money(val)}")
    if pos:
        for val, text in [(pos.entry, f"OPEN {pos.side}"), (pos.stop, "LIVE STOP"), (pos.target, "LIVE TARGET")]:
            fig.add_hline(y=val, line_dash="dot" if text.startswith("OPEN") else "dash", annotation_text=f"{text} {_money(val)}")
    fig.update_layout(height=500, template="plotly_dark", paper_bgcolor="#080b10", plot_bgcolor="#080b10", margin=dict(l=8, r=8, t=25, b=8), xaxis_rangeslider_visible=False, hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
    return fig


def _style():
    st.markdown("""
    <style>
    .stApp{background:#070a0f}.block-container{max-width:1580px;padding:12px 22px 50px}
    [data-testid="stSidebar"]{background:#0b1017;border-right:1px solid #202b37}
    .hero{background:linear-gradient(135deg,#101923,#0b1118);border:1px solid #263544;border-radius:18px;padding:18px 22px;margin-bottom:14px}
    .brand{font-size:31px;font-weight:900}.sub{font-size:12px;color:#82909e;margin-top:3px}.section{font-size:19px;font-weight:850;margin:15px 0 8px}
    .signal{border:1px solid #2c3a48;border-radius:15px;padding:17px;background:#0d141c}.action{font-size:46px;font-weight:950;line-height:1}.buy{color:#58e39b}.sell{color:#ff7076}.hold{color:#e7c85b}
    div[data-testid="stMetric"]{background:#0e151d;border:1px solid #222e3a;border-radius:12px;padding:8px 10px}
    .livebar{font-size:10px;color:#6ee39a;font-weight:850;letter-spacing:.5px;margin:2px 0 6px}.up{color:#58e39b}.down{color:#ff7076}
    </style>
    """, unsafe_allow_html=True)


@st.fragment(run_every="1s")
def _live_market_fragment(index_snapshot: list[tuple[str, str, float]], symbol: str, fallback_price: float, position_state):
    """Only this block reruns every second. The rest of the dashboard stays untouched."""
    rows = []
    for name, ticker, fallback in index_snapshot:
        live = fetch_live_1m(ticker, "1d")
        price = float(live.Close.iloc[-1]) if live is not None and not live.empty else fallback
        rows.append((name, ticker, price))

    selected = fetch_live_1m(symbol, "1d")
    selected_price = float(selected.Close.iloc[-1]) if selected is not None and not selected.empty else fallback_price

    st.markdown("<div class='livebar'>● LIVE QUOTES · SERVER-SIDE 1 SECOND POLLING · ONLY THIS SECTION UPDATES</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (name, ticker, price) in zip(cols, rows):
        with col:
            st.metric(name, f"₹{price:,.2f}", help=f"{ticker} · refreshed {_now()}")

    if position_state:
        pos = _position_from_state(position_state)
        if pos:
            if pos.side == "LONG":
                pnl = (selected_price - pos.entry) * pos.qty
            else:
                pnl = (pos.entry - selected_price) * pos.qty
            p1, p2, p3 = st.columns(3)
            p1.metric("PAPER LIVE PRICE", f"₹{selected_price:,.2f}")
            p2.metric("ENTRY", _money(pos.entry))
            p3.metric("UNREALISED P/L", f"{'+' if pnl >= 0 else '-'}₹{abs(pnl):,.2f}")

    return selected_price


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
        interval = st.selectbox("Chart timeframe", list(PERIODS), index=min(4, len(PERIODS)-1))
        period = st.selectbox("History", PERIODS[interval], index=0 if interval != "1d" else min(3, len(PERIODS[interval])-1))
        strategy = st.selectbox("Strategy", STRATEGIES)
        st.divider(); st.markdown("### 🎯 Precision Gate")
        threshold = st.slider("Minimum evidence", 50, 95, 90)
        strict = st.checkbox("Trade only above gate", True)
        reward = st.slider("Target / Risk", 1.0, 5.0, 2.0, 0.5)
        st.divider(); st.markdown("### 🛡 Risk Engine")
        capital = st.number_input("Paper capital (₹)", 10000.0, 100000000.0, 100000.0, 10000.0)
        risk = st.slider("Risk / trade %", 0.1, 3.0, 1.0, 0.1)
        brokerage = st.number_input("Costs %", 0.0, 0.50, 0.03, 0.01, format="%.2f")
        st.divider(); st.markdown("### ⚡ Live Engine")
        auto = st.checkbox("AUTO PAPER TRADING", True, help="When enabled, a qualifying signal opens a paper position. No real-money order is sent.")
        if st.button("↻ Refresh full dashboard", use_container_width=True): st.rerun()
        if st.button("Reset paper account", use_container_width=True): reset_paper_account(capital); st.rerun()

    _init_state(capital)

    st.markdown(f"<div class='hero'><div class='brand'>ALGO TRADING PRO</div><div class='sub'>Indian Market Command Center · NIFTY 50 · SENSEX · BANK NIFTY · loaded {_now()}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='section'>🇮🇳 Indian Market · Live Overview</div>", unsafe_allow_html=True)

    index_snapshot = []
    for index_name, index_symbol in INDEX_UNIVERSE.items():
        d = fetch_live_1m(index_symbol, "1d")
        if d is not None and not d.empty:
            index_snapshot.append((index_name, index_symbol, float(d.Close.iloc[-1])))
    if len(index_snapshot) == 3:
        # This is deliberately a Streamlit fragment, not a whole-page rerun.
        _live_market_fragment(index_snapshot, symbol, float(index_snapshot[0][2]), st.session_state.get("paper_position"))
    else:
        st.error("Live index feed did not return all three indexes. NIFTY 50, SENSEX and BANK NIFTY must all be available before rendering the live cards.")

    df = _data(symbol, period, interval)
    if df is None or df.empty:
        st.error(f"No market data for {symbol}. Try 1d + 1mo."); return
    x = add_indicators(df)
    sig = score_signal(x.iloc[-1], strategy, reward_r=reward)
    qualified = sig.action != "HOLD" and sig.confidence >= threshold if strict else sig.action != "HOLD"
    qty = position_size(capital, risk, sig.entry, sig.stop) if qualified else 0
    last = float(x.Close.iloc[-1])

    # The paper engine was previously called without initializing its session state.
    # That made AUTO PAPER TRADING unable to open a position reliably.
    if auto and qualified:
        process_latest_bar(x, strategy, risk, brokerage, reward)
    pos = _position_from_state(st.session_state.get("paper_position"))

    st.markdown("<div class='section'>⚡ Automatic Paper Trading</div>", unsafe_allow_html=True)
    if pos:
        st.success(f"PAPER {pos.side} OPEN · Qty {pos.qty:,} · Entry {_money(pos.entry)} · live price/P&L is updated above every second")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("POSITION", pos.side); p2.metric("QTY", f"{pos.qty:,}"); p3.metric("STOP", _money(pos.stop)); p4.metric("TARGET", _money(pos.target))
    elif qualified:
        st.info(f"{'🟢 BUY' if sig.action == 'BUY' else '🔴 SELL/SHORT'} signal ready · Qty {qty:,} · Entry {_money(sig.entry)} · AUTO PAPER TRADING {'ON' if auto else 'OFF'}")
    else:
        st.warning(f"NO PAPER ORDER · {sig.action} signal / evidence {sig.confidence:.1f}% / gate {threshold}%")

    b, c, d, e = st.columns(4)
    b.metric("SIGNAL", sig.action, f"{sig.confidence:.1f}% evidence")
    c.metric("REGIME", regime_label(x.iloc[-1]))
    d.metric("PAPER QTY", f"{qty:,}" if qty else "WAIT")
    e.metric("R:R", f"1:{reward:.1f}")

    left, right = st.columns([1.65, 1])
    with left:
        st.markdown(f"<div class='section'>{name} · Market Chart</div>", unsafe_allow_html=True)
        st.plotly_chart(_chart(x, sig, pos), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
    with right:
        st.markdown("<div class='section'>AI Trade Decision</div>", unsafe_allow_html=True)
        cls = sig.action.lower()
        st.markdown(f"<div class='signal'><div class='sub'>CURRENT MODEL DECISION · {name}</div><div class='action {cls}'>{sig.action}</div><div class='sub'>Evidence {sig.confidence:.1f}% · Risk/Reward 1:{reward:.1f}</div></div>", unsafe_allow_html=True)
        q1, q2 = st.columns(2); q1.metric("ENTRY", _money(sig.entry)); q2.metric("QTY", f"{qty:,}" if qty else "—")
        q3, q4 = st.columns(2); q3.metric("STOP", _money(sig.stop) if qualified else "WAIT"); q4.metric("TARGET", _money(sig.target) if qualified else "WAIT")
        st.markdown("#### Signal reasoning")
        for r in sig.reasons[:7]: st.write("• " + r)

    st.caption("Only the live quote/paper P&L fragment reruns every second. The main page and chart stay static. Yahoo data is delayed/throttled and is not a guaranteed exchange tick feed.")

    st.markdown("### 🧪 Backtest Lab")
    eq, metrics = backtest(df, strategy, capital, risk, brokerage)
    m = st.columns(6)
    for col, label, key in zip(m, ["Return", "Net P&L", "Max DD", "Win Rate", "Trades", "Sharpe"], ["Return %", "Net P&L", "Max Drawdown %", "Win Rate %", "Trades", "Sharpe"]):
        col.metric(label, f"{metrics[key]:.2f}")
    fig = go.Figure(go.Scatter(x=eq.index, y=eq.Equity, mode="lines", name="Equity")); fig.update_layout(height=300, template="plotly_dark", paper_bgcolor="#080b10", plot_bgcolor="#080b10"); st.plotly_chart(fig, use_container_width=True)
