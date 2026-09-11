import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app import live_terminal
from app.live_feed import fetch_live_1m

# The live terminal used yfinance's download helper, which can return the same
# cached 1-minute snapshot repeatedly. Replace that fetcher with the explicit
# cache-busting Yahoo chart request so the 1-second fragment gets fresh data.
live_terminal._live_data = fetch_live_1m

st.set_page_config(
    page_title="Algo Trading Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

live_terminal.render_app()
