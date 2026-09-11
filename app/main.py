import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app import live_terminal
from app.live_feed import fetch_live_1m
from app.paper_engine import reset_paper_account
from app.trading import INDEX_UNIVERSE, PERIODS, POPULAR_STOCKS

# Use the explicit cache-busting Yahoo chart request instead of yfinance's
# download helper for the live terminal.
live_terminal._live_data = fetch_live_1m

st.set_page_config(
    page_title="Algo Trading Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

live_terminal._style()

with st.sidebar:
    st.markdown("## ⚡ ALGO TRADING PRO")
    universe = st.selectbox("Market", ["Indices", "NSE Stocks", "Custom Ticker"])
    if universe == "Indices":
        name = st.selectbox("Instrument", list(INDEX_UNIVERSE))
        symbol = INDEX_UNIVERSE[name]
    elif universe == "NSE Stocks":
        name = st.selectbox("Instrument", list(POPULAR_STOCKS))
        symbol = POPULAR_STOCKS[name]
    else:
        symbol = st.text_input("Yahoo ticker", "RELIANCE.NS").strip().upper()
        name = symbol

    interval = st.selectbox("Chart timeframe", list(PERIODS), index=min(4, len(PERIODS) - 1), key="interval")
    strategy = st.selectbox("Strategy", live_terminal.STRATEGIES)

    st.divider()
    st.markdown("### 🎯 Precision Gate")
    threshold = st.slider("Minimum evidence", 50, 95, 90, 1)
    strict = st.checkbox("Trade only above gate", True)
    reward = st.slider("Target / Risk", 1.0, 5.0, 2.0, 0.5)

    st.divider()
    st.markdown("### 🛡 Risk Engine")
    capital = st.number_input("Paper capital (₹)", 10000.0, 100000000.0, 100000.0, 10000.0)
    risk = st.slider("Risk / trade %", 0.1, 3.0, 1.0, 0.1)
    brokerage = st.number_input("Costs %", 0.0, 0.50, 0.03, 0.01, format="%.2f")

    st.divider()
    st.markdown("### ⚡ Live Engine")
    st.success("LIVE REFRESH: 1 SECOND")
    auto = st.checkbox(
        "AUTO PAPER TRADING",
        False,
        help="Only controls automatic paper order execution. The market display refreshes every second even when this is OFF.",
    )
    if st.button("↻ Refresh now", use_container_width=True):
        st.rerun()
    if st.button("Reset paper account", use_container_width=True):
        reset_paper_account(capital)
        st.rerun()

# IMPORTANT: the live fragment is always rendered. AUTO PAPER TRADING only
# controls order execution; it no longer controls whether prices refresh.
live_terminal._live_fragment(
    symbol,
    name,
    strategy,
    threshold,
    strict,
    reward,
    capital,
    risk,
    brokerage,
    auto,
)
