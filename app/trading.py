from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=300, show_spinner=False)
def download_market_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False,
                     progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={str(c): str(c).title() for c in df.columns})
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()
    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    # EMA_21 is explicitly calculated; never default a missing EMA to zero.
    for p in [5, 9, 20, 21, 50, 100, 200]:
        x[f"EMA_{p}"] = x["Close"].ewm(span=p, adjust=False).mean()
    x["SMA_20"] = x["Close"].rolling(20).mean()
    x["SMA_50"] = x["Close"].rolling(50).mean()

    delta = x["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    al = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = ag / al.replace(0, np.nan)
    x["RSI_14"] = 100 - 100 / (1 + rs)

    e12 = x["Close"].ewm(span=12, adjust=False).mean()
    e26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = e12 - e26
    x["MACD_SIGNAL"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["MACD_HIST"] = x["MACD"] - x["MACD_SIGNAL"]

    tr = pd.concat([x["High"]-x["Low"],
                    (x["High"]-x["Close"].shift()).abs(),
                    (x["Low"]-x["Close"].shift()).abs()], axis=1).max(axis=1)
    x["ATR_14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    mid = x["Close"].rolling(20).mean()
    sd = x["Close"].rolling(20).std()
    x["BB_MID"] = mid
    x["BB_UPPER"] = mid + 2*sd
    x["BB_LOWER"] = mid - 2*sd

    typical = (x["High"] + x["Low"] + x["Close"]) / 3
    x["VWAP"] = (typical*x["Volume"]).cumsum() / x["Volume"].replace(0, np.nan).cumsum()
    x["ROC_12"] = x["Close"].pct_change(12)*100
    x["VOLUME_SPIKE"] = x["Volume"] > x["Volume"].rolling(20).mean()*1.5

    up = x["High"].diff()
    down = -x["Low"].diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=x.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=x.index)
    atr = x["ATR_14"].replace(0, np.nan)
    pdi = 100*plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()/atr
    mdi = 100*minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()/atr
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    x["ADX_14"] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return x


def _num(v, default=np.nan):
    try:
        v = float(v)
        return v if math.isfinite(v) else default
    except Exception:
        return default


@dataclass
class Signal:
    action: str
    confidence: float
    reasons: list[str]
    entry: float
    stop: float
    target: float
    risk_reward: float


def score_signal(row: pd.Series, strategy: str = "Ensemble", atr_mult: float = 1.5,
                 reward_r: float = 2.0) -> Signal:
    close = _num(row.get("Close"))
    atr = _num(row.get("ATR_14"))
    if not math.isfinite(close):
        return Signal("HOLD", 0.0, ["No valid close price"], np.nan, np.nan, np.nan, np.nan)
    if not math.isfinite(atr) or atr <= 0:
        atr = max(close*0.01, 0.01)

    ema9, ema21 = _num(row.get("EMA_9")), _num(row.get("EMA_21"))
    macd, macd_sig = _num(row.get("MACD")), _num(row.get("MACD_SIGNAL"))
    rsi, vwap = _num(row.get("RSI_14")), _num(row.get("VWAP"))
    votes, reasons = [], []

    if math.isfinite(ema9) and math.isfinite(ema21):
        votes.append(1 if ema9 > ema21 else -1)
        reasons.append(f"EMA 9 {'above' if ema9 > ema21 else 'below'} EMA 21")
    if math.isfinite(macd) and math.isfinite(macd_sig):
        votes.append(1 if macd > macd_sig else -1)
        reasons.append(f"MACD {'bullish' if macd > macd_sig else 'bearish'}")
    if math.isfinite(rsi):
        if rsi < 35:
            votes.append(1); reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 65:
            votes.append(-1); reasons.append(f"RSI overbought ({rsi:.1f})")
        else:
            reasons.append(f"RSI neutral ({rsi:.1f})")
    if math.isfinite(vwap):
        votes.append(1 if close > vwap else -1)
        reasons.append(f"Price {'above' if close > vwap else 'below'} VWAP")

    if strategy == "EMA Crossover":
        raw = 1 if math.isfinite(ema9) and math.isfinite(ema21) and ema9 > ema21 else -1
        confidence = 65.0
    elif strategy == "RSI Reversion":
        raw = 1 if math.isfinite(rsi) and rsi < 35 else (-1 if math.isfinite(rsi) and rsi > 65 else 0)
        confidence = 65.0 if raw else 50.0
    elif strategy == "MACD":
        raw = 1 if math.isfinite(macd) and math.isfinite(macd_sig) and macd > macd_sig else -1
        confidence = 65.0
    else:
        raw = int(np.sign(sum(votes))) if votes else 0
        agreement = abs(sum(votes))/len(votes) if votes else 0
        confidence = 50.0 + 45.0*agreement

    action = "BUY" if raw > 0 else ("SELL" if raw < 0 else "HOLD")
    if action == "BUY":
        stop = close - atr_mult*atr
        target = close + reward_r*(close-stop)
    elif action == "SELL":
        stop = close + atr_mult*atr
        target = close - reward_r*(stop-close)
    else:
        stop = target = np.nan
    rr = reward_r if action != "HOLD" else np.nan
    reasons.append(f"{action} selected from {strategy} evidence." if action != "HOLD" else "Mixed evidence: no trade.")
    return Signal(action, round(float(min(100, max(0, confidence))), 1), reasons,
                  close, stop, target, rr)


def position_size(capital: float, risk_pct: float, entry: float, stop: float) -> int:
    risk_cash = max(0.0, capital*risk_pct/100)
    per_share = abs(entry-stop)
    return int(risk_cash//per_share) if per_share > 0 else 0


def backtest(df: pd.DataFrame, strategy: str, initial_capital: float, risk_pct: float):
    x = add_indicators(df)
    cash, position, entry, stop = float(initial_capital), 0, 0.0, 0.0
    equity, trades = [], []
    for idx, row in x.iterrows():
        price = _num(row["Close"])
        sig = score_signal(row, strategy)
        if position == 0 and sig.action == "BUY":
            qty = position_size(cash, risk_pct, price, sig.stop)
            if qty > 0:
                position, entry, stop = qty, price, sig.stop
                cash -= qty*price
                trades.append({"Date": idx, "Side": "BUY", "Price": price, "Qty": qty, "PnL": 0.0})
        elif position > 0 and (price <= stop or sig.action == "SELL"):
            pnl = position*(price-entry)
            cash += position*price
            trades.append({"Date": idx, "Side": "SELL", "Price": price, "Qty": position, "PnL": pnl})
            position = 0
        equity.append(cash + position*price)
    if position > 0:
        price = float(x["Close"].iloc[-1])
        pnl = position*(price-entry)
        cash += position*price
        trades.append({"Date": x.index[-1], "Side": "SELL", "Price": price, "Qty": position, "PnL": pnl})
        equity[-1] = cash
    eq = pd.Series(equity, index=x.index)
    dd = (eq-eq.cummax())/eq.cummax().replace(0, np.nan)
    t = pd.DataFrame(trades)
    sells = t[t["Side"] == "SELL"] if not t.empty else pd.DataFrame()
    win = float((sells["PnL"] > 0).mean()*100) if not sells.empty else 0.0
    metrics = {"Net P&L": cash-initial_capital, "Return %": (cash/initial_capital-1)*100,
               "Max Drawdown %": float(dd.min()*100) if len(dd) else 0.0,
               "Trades": len(sells), "Win Rate %": win, "Final Equity": cash, "Trade Log": t}
    return pd.DataFrame({"Equity": eq}), metrics


def render_app():
    st.title("📈 Stock Algo Trading Platform")
    st.caption("Research + paper-trading analytics. Signals are decision support, not guaranteed returns.")
    with st.sidebar:
        st.header("Market")
        symbol = st.text_input("Ticker", "RELIANCE.NS").strip().upper()
        interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"])
        periods = {"1d":["1mo","3mo","6mo","1y","5y","10y","max"],
                   "1h":["1mo","3mo","6mo","1y"], "30m":["1mo","3mo","6mo"],
                   "15m":["1mo","3mo"], "5m":["5d","1mo"]}
        period = st.selectbox("History", periods[interval], index=min(3, len(periods[interval])-1))
        strategy = st.selectbox("Strategy", ["Ensemble","EMA Crossover","RSI Reversion","MACD"])
        capital = st.number_input("Paper capital (₹)", 1000.0, value=100000.0, step=5000.0)
        risk_pct = st.number_input("Risk / trade (%)", 0.1, 5.0, 1.0, 0.1)
        if st.button("Load / Refresh", type="primary", use_container_width=True):
            download_market_data.clear()

    df = download_market_data(symbol, period, interval)
    if df.empty:
        st.error("No market data returned. Check ticker and interval/history compatibility.")
        st.info("Intraday Yahoo Finance data has shorter history limits; try a shorter History value.")
        return

    x = add_indicators(df)
    sig = score_signal(x.iloc[-1], strategy)
    qty = position_size(capital, risk_pct, sig.entry, sig.stop) if sig.action == "BUY" else 0
    a,b,c,d = st.columns(4)
    a.metric("Last Price", f"₹{sig.entry:,.2f}")
    b.metric("Signal", sig.action, f"{sig.confidence:.1f}% evidence")
    c.metric("Stop Loss", f"₹{sig.stop:,.2f}" if sig.action != "HOLD" else "—")
    d.metric("Target", f"₹{sig.target:,.2f}" if sig.action != "HOLD" else "—")
    if sig.action == "BUY":
        st.success(f"🟢 BUY / LONG — suggested paper quantity: **{qty:,} shares**")
    elif sig.action == "SELL":
        st.error("🔴 SELL / EXIT — bearish evidence detected")
    else:
        st.warning("🟡 HOLD / NO TRADE — evidence is mixed")
    with st.expander("Signal reasoning", expanded=True):
        for r in sig.reasons: st.write("• " + r)
        st.caption("Confidence is evidence strength, NOT probability of profit.")
    st.subheader("Price & indicators")
    st.line_chart(x.tail(250)[["Close","EMA_9","EMA_21","EMA_50","BB_UPPER","BB_LOWER","VWAP"]])

    t1,t2,t3 = st.tabs(["Backtest","Latest data","Risk"])
    with t1:
        equity, m = backtest(df, strategy, capital, risk_pct)
        cols = st.columns(5)
        cols[0].metric("Net P&L", f"₹{m['Net P&L']:,.0f}")
        cols[1].metric("Return", f"{m['Return %']:.2f}%")
        cols[2].metric("Max DD", f"{m['Max Drawdown %']:.2f}%")
        cols[3].metric("Trades", m["Trades"])
        cols[4].metric("Win rate", f"{m['Win Rate %']:.1f}%")
        st.line_chart(equity)
        if not m["Trade Log"].empty: st.dataframe(m["Trade Log"], use_container_width=True)
    with t2:
        st.dataframe(x.tail(100).sort_index(ascending=False), use_container_width=True)
    with t3:
        st.write(f"Capital: ₹{capital:,.0f}")
        st.write(f"Risk budget: ₹{capital*risk_pct/100:,.0f}")
        st.write(f"ATR(14): ₹{_num(x.iloc[-1].get('ATR_14')):,.2f}")
        st.write(f"Risk-based paper quantity: {qty:,}")
''