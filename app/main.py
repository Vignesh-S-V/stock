import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.trading import render_app

st.set_page_config(page_title="Stock Algo Trading", page_icon="📈", layout="wide")
render_app()
