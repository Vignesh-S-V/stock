import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.live_terminal import render_app

st.set_page_config(
    page_title="Algo Trading Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_app()
