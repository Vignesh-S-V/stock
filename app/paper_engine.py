from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.trading import (
    INDEX_UNIVERSE,
    PERIODS,
    POPULAR_STOCKS,
    add_indicators,
    download_market_data,
    position_size,
    score_signal,
)


# -----------------------------------------------------------------------------
# Professional paper-trading dashboard
# -----------------------------------------------------------------------------

@dataclass
class PaperPosition:
    side: str
    qty: int
    entry: float
    stop: float
    target: float
    opened_at: str
    confidence: float


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _money(value: float) -> str:
    if not _finite(value):
        return "—"
    return f"₹{float(value):,.2f}"


def _pct(value: float) -> str:
    if not _finite(value):
        return "—"
    return f"{float(value):+.2f}%"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%d %b %Y, %H:%M:%S")


def reset_paper_account(capital: float) -> None:
    st.session_state.paper_cash = float(capital)
    st.session_state.paper_position = None
    st.session_state.paper_trades = []
    st.session_state.paper_equity = float(capital)
    st.session_state.paper_last_bar = None
    st.session_state.paper_started_at = _now()


def _init_state(capital: float) -> None:
    if "paper_cash" not in st.session_state:
        reset_paper_account(capital)


def _trade_cost(price: float, qty: int, brokerage_pct: float) -> float:
    return abs(float(price) * int(qty)) * float(brokerage_pct) / 100.0


def _position_from_state(value):
    return PaperPosition(**value) if value else None


def process_latest_bar(
    x: pd.DataFrame,
    strategy: str,
    risk_pct: float,
    brokerage_pct: float,
    reward_r: float = 2.0,
) -> None:
    """Process one unseen bar using previous-bar signals to avoid look-ahead."""
    if len(x) < 2:
        return
    prev = x.iloc[-2]
    bar = x.iloc[-1]
    bar_key = str(x.index[-1])
    if st.session_state.get("paper_last_bar") == bar_key:
        return

    cash = float(st.session_state.paper_cash)
    pos = _position_from_state(st.session_state.get("paper_position"))
    signal = score_signal(prev, strategy, reward_r=reward_r)
    o, h, l, c = [float(bar[col]) for col in ["Open", "High", "Low", "Close"]]

    # Exit first. If both stop and target are touched in one candle, stop wins.
    if pos is not None:
        exit_price = None
        reason = None
        if pos.side == "LONG":
            if l <= pos.stop:
                exit_price, reason = pos.stop, "STOP LOSS"
            elif h >= pos.target:
                exit_price, reason = pos.target, "2R TARGET"
            elif signal.action == "SELL":
                exit_price, reason = o, "SIGNAL EXIT"
            if exit_price is not None:
                gross = pos.qty * (exit_price - pos.entry)
                fee = _trade_cost(exit_price, pos.qty, brokerage_pct)
                pnl = gross - fee
                cash += pos.qty * exit_price - fee
                st.session_state.paper_trades.append(
                    {"Time": str(x.index[-1]), "Action": "SELL", "Side": "LONG", "Qty": pos.qty,
                     "Price": exit_price, "PnL": pnl, "Reason": reason}
                )
                pos = None
        else:
            if h >= pos.stop:
                exit_price, reason = pos.stop, "STOP LOSS"
            elif l <= pos.target:
                exit_price, reason = pos.target, "2R TARGET"
            elif signal.action == "BUY":
                exit_price, reason = o, "SIGNAL EXIT"
            if exit_price is not None:
                gross = pos.qty * (pos.entry - exit_price)
                fee = _trade_cost(exit_price, pos.qty, brokerage_pct)
                pnl = gross - fee
                cash += pnl
                st.session_state.paper_trades.append(
                    {"Time": str(x.index[-1]), "Action": "COVER", "Side": "SHORT", "Qty": pos.qty,
                     "Price": exit_price, "PnL": pnl, "Reason": reason}
                )
                pos = None

    # New entry at current bar open from previous bar's signal.
    if pos is None and signal.action in {"BUY", "SELL"} and _finite(o):
        stop_distance = abs(float(signal.entry) - float(signal.stop))
        if stop_distance > 0:
            if signal.action == "BUY":
                proposed_stop = o - stop_distance
            else:
                proposed_stop = o + stop_distance
            qty = max(0, int(position_size(max(cash, 0), risk_pct, o, proposed_stop)))
            if qty > 0:
                fee = _trade_cost(o, qty, brokerage_pct)
                if signal.action == "BUY":
                    # Long position reserves the notional from paper cash.
                    cash -= qty * o + fee
                    pos = PaperPosition("LONG", qty, o, proposed_stop, o + reward_r * stop_distance, str(x.index[-1]), signal.confidence)
                    action = "BUY"
                else:
                    # Short simulation tracks P&L without pretending to borrow cash.
                    cash -= fee
                    pos = PaperPosition("SHORT", qty, o, proposed_stop, o - reward_r * stop_distance, str(x.index[-1]), signal.confidence)
                    action = "SHORT"
                st.session_state.paper_trades.append(
                    {"Time": str(x.index[-1]), "Action": action, "Side": pos.side, "Qty": qty,
                     "Price": o, "PnL": -fee, "Reason": "AUTO ENTRY"}
                )

    if pos is None:
        equity = cash
    elif pos.side == "LONG":
        equity = cash + pos.qty * c
    else:
        equity = cash + pos.qty * (pos.entry - c)

    st.session_state.paper_cash = cash
    st.session_state.paper_position = asdict(pos) if pos else None
    st.session_state.paper_equity = float(equity)
    st.session_state.paper_last_bar = bar_key


def run_historical_2r(
    df: pd.DataFrame,
    strategy: str,
    initial_capital: float,
    risk_pct: float,
    brokerage_pct: float,
    reward_r: float = 2.0,
):
    """Walk-forward simulation: signal on t-1, entry at t open, stop/2R target."""
    x = add_indicators(df).dropna(subset=["Open", "High", "Low", "Close"])
    cash = float(initial_capital)
    position = None
    trades = []
    equity = []

    for i in range(1, len(x)):
        prev, bar = x.iloc[i - 1], x.iloc[i]
        price, high, low, close = [float(bar[c]) for c in ["Open", "High", "Low", "Close"]]
        sig = score_signal(prev, strategy, reward_r=reward_r)

        if position is not None:
            p = position
            exit_price = None
            reason = None
            if p["side"] == "LONG":
                if low <= p["stop"]:
                    exit_price, reason = p["stop"], "STOP LOSS"
                elif high >= p["target"]:
                    exit_price, reason = p["target"], "2R TARGET"
                elif sig.action == "SELL":
                    exit_price, reason = price, "SIGNAL EXIT"
                if exit_price is not None:
                    fee = _trade_cost(exit_price, p["qty"], brokerage_pct)
                    pnl = p["qty"] * (exit_price - p["entry"]) - fee
                    cash += p["qty"] * exit_price - fee
                    trades.append({"Time": str(x.index[i]), "Side": "LONG", "Entry": p["entry"], "Exit": exit_price,
                                   "Qty": p["qty"], "PnL": pnl, "Reason": reason})
                    position = None
            else:
                if high >= p["stop"]:
                    exit_price, reason = p["stop"], "STOP LOSS"
                elif low <= p["target"]:
                    exit_price, reason = p["target"], "2R TARGET"
                elif sig.action == "BUY":
                    exit_price, reason = price, "SIGNAL EXIT"
                if exit_price is not None:
                    fee = _trade_cost(exit_price, p["qty"], brokerage_pct)
                    pnl = p["qty"] * (p["entry"] - exit_price) - fee
                    cash += pnl
                    trades.append({"Time": str(x.index[i]), "Side": "SHORT", "Entry": p["entry"], "Exit": exit_price,
                                   "Qty": p["qty"], "PnL": pnl, "Reason": reason})
                    position = None

        if position is None and sig.action in {"BUY", "SELL"}:
            dist = abs(float(sig.entry) - float(sig.stop))
            if dist > 0:
                proposed_stop = price - dist if sig.action == "BUY" else price + dist
                qty = max(0, int(position_size(max(cash, 0), risk_pct, price, proposed_stop)))
                if qty > 0:
                    cash -= _trade_cost(price, qty, brokerage_pct)
                    if sig.action == "BUY":
                        position = {"side": "LONG", "qty": qty, "entry": price, "stop": price - dist, "target": price + reward_r * dist}
                    else:
                        position = {"side": "SHORT", "qty": qty, "entry": price, "stop": price + dist, "target": price - reward_r * dist}

        if position is None:
            eq = cash
        elif position["side"] == "LONG":
            eq = cash + position["qty"] * close
        else:
            eq = cash + position["qty"] * (position["entry"] - close)
        equity.append(eq)

    if position is not None:
        close = float(x["Close"].iloc[-1])
        p = position
        pnl = p["qty"] * (close - p["entry"]) if p["side"] == "LONG" else p["qty"] * (p["entry"] - close)
        if p["side"] == "LONG":
            cash += p["qty"] * close
        else:
            cash += pnl
        trades.append({"Time": str(x.index[-1]), "Side": p["side"], "Entry": p["entry"], "Exit": close,
                       "Qty": p["qty"], "PnL": pnl, "Reason": "END OF TEST"})

    t = pd.DataFrame(trades)
    eqs = pd.Series(equity) if equity else pd.Series([initial_capital])
    dd = (eqs - eqs.cummax()) / eqs.cummax().replace(0, np.nan)
    if t.empty:
        metrics = {"Trades": 0, "Win Rate %": 0.0, "Target Hit %": 0.0, "Net P&L": cash - initial_capital,
                   "Return %": (cash / initial_capital - 1) * 100, "Profit Factor": 0.0, "Max Drawdown %": dd.min() * 100}
    else:
        wins = t["PnL"] > 0
        gross_profit = float(t.loc[wins, "PnL"].sum())
        gross_loss = float(abs(t.loc[~wins, "PnL"].sum()))
        metrics = {
            "Trades": int(len(t)),
            "Win Rate %": float(wins.mean() * 100),
            "Target Hit %": float((t["Reason"] == "2R TARGET").mean() * 100),
            "Net P&L": float(cash - initial_capital),
            "Return %": float((cash / initial_capital - 1) * 100),
            "Profit Factor": float(gross_profit / gross_loss) if gross_loss else float("inf"),
            "Max Drawdown %": float(dd.min() * 100) if len(dd) else 0.0,
        }
    return pd.DataFrame({"Equity": equity}), metrics, t


def signal_accuracy(df: pd.DataFrame, strategy: str, horizon: int = 5, reward_r: float = 2.0):
    """Out-of-sample directional accuracy and first-touch 2R/stop outcome."""
    x = add_indicators(df)
    records = []
    for i in range(1, len(x) - horizon):
        sig = score_signal(x.iloc[i], strategy, reward_r=reward_r)
        if sig.action == "HOLD" or not _finite(sig.entry):
            continue
        future = x.iloc[i + 1:i + 1 + horizon]
        entry = float(sig.entry)
        direction = 1 if sig.action == "BUY" else -1
        directional = (float(future["Close"].iloc[-1]) - entry) * direction > 0
        hit = "NEITHER"
        for _, bar in future.iterrows():
            if direction == 1:
                if float(bar["Low"]) <= float(sig.stop): hit = "STOP"; break
                if float(bar["High"]) >= float(sig.target): hit = "TARGET"; break
            else:
                if float(bar["High"]) >= float(sig.stop): hit = "STOP"; break
                if float(bar["Low"]) <= float(sig.target): hit = "TARGET"; break
        records.append({"Time": str(x.index[i]), "Signal": sig.action, "Confidence": sig.confidence,
                        "Directional Correct": directional, "2R Outcome": hit})
    result = pd.DataFrame(records)
    if result.empty:
        return result, {"Accuracy %": 0.0, "2R Target Hit %": 0.0, "Signals": 0}
    return result, {"Accuracy %": float(result["Directional Correct"].mean() * 100),
                    "2R Target Hit %": float((result["2R Outcome"] == "TARGET").mean() * 100),
                    "Signals": int(len(result))}


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0b0f14; }
        [data-testid="stHeader"] { background: rgba(11,15,20,.92); }
        [data-testid="stSidebar"] { background: #0f141b; border-right: 1px solid #202832; }
        .block-container { max-width: 1500px; padding-top: 1.2rem; padding-bottom: 3rem; }
        .hero { padding: 20px 24px; border: 1px solid #26313d; border-radius: 16px; background: linear-gradient(135deg,#121a23,#0d131a); margin-bottom: 18px; }
        .hero-title { font-size: 30px; font-weight: 750; letter-spacing: -.7px; margin: 0; }
        .hero-sub { color: #9aa7b5; margin-top: 5px; font-size: 14px; }
        .status { display:inline-flex; align-items:center; gap:7px; padding:6px 10px; border-radius:999px; background:#15231d; border:1px solid #294535; color:#8fe0ad; font-size:12px; font-weight:650; }
        .dot { width:7px; height:7px; border-radius:50%; background:#45d483; display:inline-block; }
        .card { background:#111820; border:1px solid #25303b; border-radius:14px; padding:15px 17px; min-height:95px; }
        .label { color:#8e9aa8; font-size:12px; text-transform:uppercase; letter-spacing:.7px; }
        .value { color:#edf2f7; font-size:24px; font-weight:700; margin-top:5px; }
        .muted { color:#8793a0; font-size:12px; }
        .buy { color:#55d991; font-weight:800; }
        .sell { color:#ff7070; font-weight:800; }
        .hold { color:#e6c65a; font-weight:800; }
        .trade-box { border:1px solid #293541; border-radius:14px; padding:16px; background:#10171f; }
        .section-title { font-size:18px; font-weight:700; margin: 8px 0 12px; }
        div[data-testid="stMetric"] { background:#111820; border:1px solid #25303b; padding:10px 12px; border-radius:12px; }
        button[kind="primary"] { border-radius:10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_signal_card(sig, symbol: str) -> None:
    action_class = sig.action.lower()
    st.markdown(
        f"""
        <div class="trade-box">
          <div class="label">Current model decision · {symbol}</div>
          <div style="font-size:32px;margin-top:4px" class="{action_class}">{sig.action}</div>
          <div class="muted">Evidence strength: <b>{sig.confidence:.1f}%</b> · Risk/Reward: <b>1:{sig.risk_reward if _finite(sig.risk_reward) else 0:.1f}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_live(sig, symbol: str) -> None:
    pos = _position_from_state(st.session_state.get("paper_position"))
    cash = float(st.session_state.get("paper_cash", 0))
    equity = float(st.session_state.get("paper_equity", cash))
    trades = st.session_state.get("paper_trades", [])
    realized = sum(float(t.get("PnL", 0)) for t in trades if t.get("Reason") != "AUTO ENTRY")

    _render_signal_card(sig, symbol)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Paper Equity", _money(equity))
    c2.metric("Available Cash", _money(cash))
    c3.metric("Realized P&L", _money(realized))
    c4.metric("Open Position", f"{pos.side} × {pos.qty}" if pos else "FLAT")
    c5.metric("Trades", len(trades))

    if pos:
        st.markdown("#### Open position")
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Side", pos.side)
        p2.metric("Entry", _money(pos.entry))
        p3.metric("Stop", _money(pos.stop))
        p4.metric("2R Target", _money(pos.target))
        p5.metric("Quantity", f"{pos.qty:,}")
    else:
        st.info("Paper account is flat. Enable Auto Paper Trading or run the latest bar to simulate the next eligible trade.")


def _price_chart(x: pd.DataFrame, sig) -> go.Figure:
    view = x.tail(min(180, len(x))).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"], name="Price"))
    for col, name in [("EMA_9", "EMA 9"), ("EMA_21", "EMA 21"), ("EMA_50", "EMA 50")]:
        if col in view:
            fig.add_trace(go.Scatter(x=view.index, y=view[col], name=name, mode="lines", line={"width":1.5}))
    if _finite(sig.entry):
        fig.add_hline(y=sig.entry, line_dash="dot", annotation_text=f"Entry {_money(sig.entry)}")
    if _finite(sig.stop):
        fig.add_hline(y=sig.stop, line_dash="dash", annotation_text=f"Stop {_money(sig.stop)}")
    if _finite(sig.target):
        fig.add_hline(y=sig.target, line_dash="dash", annotation_text=f"2R {_money(sig.target)}")
    fig.update_layout(
        height=500, template="plotly_dark", paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14",
        margin=dict(l=10, r=10, t=20, b=10), xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0), hovermode="x unified",
    )
    return fig


def render_app() -> None:
    _inject_css()

    st.markdown(
        """
        <div class="hero">
          <div class="status"><span class="dot"></span> PAPER ENGINE ONLINE · NO LIVE ORDERS</div>
          <div class="hero-title">Advanced Algo Trading Terminal</div>
          <div class="hero-sub">Signal intelligence · 1:2 risk/reward execution · walk-forward validation · paper portfolio analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Terminal Controls")
        universe = st.selectbox("Universe", ["Indices", "Popular Stocks", "Custom Ticker"], key="universe")
        if universe == "Indices":
            label = st.selectbox("Instrument", list(INDEX_UNIVERSE), key="instrument_index")
            symbol = INDEX_UNIVERSE[label]
        elif universe == "Popular Stocks":
            label = st.selectbox("Instrument", list(POPULAR_STOCKS), key="instrument_stock")
            symbol = POPULAR_STOCKS[label]
        else:
            label = st.text_input("Yahoo ticker", "RELIANCE.NS", key="custom_ticker").strip().upper()
            symbol = label

        interval = st.selectbox("Bar interval", list(PERIODS), key="interval")
        period = st.selectbox("History", PERIODS[interval], index=3 if interval == "1d" else 0, key="period")
        strategy = st.selectbox("Strategy", ["Ensemble", "Trend Momentum", "EMA Crossover", "RSI Reversion", "MACD", "Mean Reversion"], key="strategy")
        st.divider()
        st.markdown("**Risk engine**")
        capital = st.number_input("Paper capital (₹)", min_value=10000.0, value=100000.0, step=10000.0, key="capital")
        risk_pct = st.slider("Risk per trade", 0.1, 3.0, 1.0, 0.1, format="%.1f%%", key="risk_pct")
        brokerage = st.number_input("Brokerage + slippage (%)", 0.0, 0.50, 0.03, 0.01, format="%.2f", key="brokerage")
        reward_r = st.number_input("Reward / Risk", 2.0, 5.0, 2.0, 0.5, key="reward_r")
        st.caption("Default execution model: 1R stop · 2R target. Intrabar stop wins if both levels are touched.")
        auto = st.checkbox("Auto paper trading · refresh 60s", value=False, key="auto")
        b1, b2 = st.columns(2)
        if b1.button("Refresh", use_container_width=True):
            download_market_data.clear()
            st.rerun()
        if b2.button("Reset", use_container_width=True):
            reset_paper_account(capital)
            st.rerun()

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
            st.caption(f"Last engine cycle: {_now()} · new bars only · source: Yahoo Finance")
            _render_live(sig, label)
        live_fragment()
    else:
        st.caption(f"Market data: {len(df):,} bars · Last bar: {str(df.index[-1])} · source: Yahoo Finance")
        if st.button("▶ Run Next Paper Bar", type="primary", use_container_width=True):
            process_latest_bar(x, strategy, risk_pct, brokerage, reward_r)
            st.rerun()
        _render_live(sig, label)

    st.markdown("<div class='section-title'>Market & model</div>", unsafe_allow_html=True)
    st.plotly_chart(_price_chart(x, sig), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

    with st.expander("Why the model selected this signal", expanded=True):
        for reason in sig.reasons:
            st.write("• " + reason)

    tab1, tab2, tab3 = st.tabs(["Backtest & validation", "Paper ledger", "Risk & execution"])

    with tab1:
        eq, metrics, trades = run_historical_2r(df, strategy, capital, risk_pct, brokerage, reward_r)
        st.markdown("#### Walk-forward 1:2 validation")
        st.caption("Signal is generated from the previous bar and executed at the next bar open. This reduces look-ahead bias.")
        m = st.columns(7)
        m[0].metric("Trades", metrics["Trades"])
        m[1].metric("Win rate", f"{metrics['Win Rate %']:.1f}%")
        m[2].metric("2R hit", f"{metrics['Target Hit %']:.1f}%")
        m[3].metric("Return", _pct(metrics["Return %"]))
        m[4].metric("Net P&L", _money(metrics["Net P&L"]))
        pf = metrics["Profit Factor"]
        m[5].metric("Profit factor", "∞" if not _finite(pf) else f"{pf:.2f}")
        m[6].metric("Max DD", f"{metrics['Max Drawdown %']:.1f}%")
        if not eq.empty:
            fig = go.Figure(go.Scatter(x=eq.index, y=eq["Equity"], mode="lines", name="Equity"))
            fig.update_layout(height=300, template="plotly_dark", paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14", margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        if not trades.empty:
            st.dataframe(trades.tail(100), use_container_width=True, hide_index=True)

        acc, am = signal_accuracy(df, strategy, horizon=5, reward_r=reward_r)
        st.markdown("#### Signal accuracy lab")
        a1, a2, a3 = st.columns(3)
        a1.metric("5-bar directional accuracy", f"{am['Accuracy %']:.1f}%")
        a2.metric("2R target hit", f"{am['2R Target Hit %']:.1f}%")
        a3.metric("Signals evaluated", am["Signals"])
        if not acc.empty:
            st.dataframe(acc.tail(100), use_container_width=True, hide_index=True)

    with tab2:
        trades = pd.DataFrame(st.session_state.get("paper_trades", []))
        st.markdown("#### Live-session paper ledger")
        if trades.empty:
            st.info("No paper trades yet. Run a paper bar or enable automatic mode.")
        else:
            st.dataframe(trades.iloc[::-1], use_container_width=True, hide_index=True)
            csv = trades.to_csv(index=False).encode("utf-8")
            st.download_button("Download ledger CSV", csv, "paper_trade_ledger.csv", "text/csv")

    with tab3:
        st.markdown("#### Execution policy")
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Position sizing**")
            st.write(f"Risk per trade: **{risk_pct:.1f}%**")
            st.write(f"Reward/Risk: **1:{reward_r:.1f}**")
            st.write(f"Estimated cost: **{brokerage:.2f}%**")
            st.write("Quantity is sized from cash risk ÷ stop distance.")
        with r2:
            st.markdown("**Controls**")
            st.write("• Previous-bar signal → next-bar open entry")
            st.write("• Stop loss checked before target")
            st.write("• Opposite signal can close an open position")
            st.write("• No live broker/API order is ever sent")
        st.warning("Research mode only. Historical results and paper fills do not guarantee live-market performance.")
