from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


INDEX_UNIVERSE = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
}

POPULAR_STOCKS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "AXIS BANK": "AXISBANK.NS",
}

PERIODS = {
    "1d": ["1mo", "3mo", "6mo", "1y", "5y", "10y", "max"],
    "1h": ["1mo", "3mo", "6mo", "1y"],
    "30m": ["1mo", "3mo", "6mo"],
    "15m": ["1mo", "3mo"],
    "5m": ["5d", "1mo"],
}


@st.cache_data(ttl=300, show_spinner=False)
def download_market_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Download OHLCV from Yahoo with a resilient fallback and normalization."""
    symbol = symbol.strip().upper()
    if not symbol:
        return pd.DataFrame()

    df = pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False, timeout=30)
    except Exception:
        pass
    if df is None or df.empty:
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, actions=False, raise_errors=False)
        except Exception:
            df = pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        fields = {"Open", "High", "Low", "Close", "Volume"}
        first = [str(c[0]) for c in df.columns]
        df.columns = [str(c[0]) for c in df.columns] if fields.intersection(first) else [str(c[-1]) for c in df.columns]
    df = df.rename(columns={str(c): str(c).title() for c in df.columns})
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()
    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Open", "High", "Low", "Close"]).loc[lambda z: ~z.index.duplicated(keep="last")].sort_index()


def _num(value, default=np.nan) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Broad technical feature set for signal generation and research."""
    x = df.copy()
    close, high, low, volume = x["Close"], x["High"], x["Low"], x["Volume"]
    for p in [5, 9, 20, 21, 50, 100, 200]:
        x[f"EMA_{p}"] = close.ewm(span=p, adjust=False).mean()
    for p in [20, 50, 100, 200]:
        x[f"SMA_{p}"] = close.rolling(p).mean()

    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    al = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    x["RSI_14"] = 100 - 100 / (1 + ag / al.replace(0, np.nan))

    e12, e26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    x["MACD"] = e12 - e26
    x["MACD_SIGNAL"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["MACD_HIST"] = x["MACD"] - x["MACD_SIGNAL"]

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    x["ATR_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    mid, sd = close.rolling(20).mean(), close.rolling(20).std()
    x["BB_MID"], x["BB_UPPER"], x["BB_LOWER"] = mid, mid + 2 * sd, mid - 2 * sd
    x["BB_WIDTH"] = (x["BB_UPPER"] - x["BB_LOWER"]) / mid.replace(0, np.nan)

    typical = (high + low + close) / 3
    x["VWAP"] = (typical * volume).cumsum() / volume.replace(0, np.nan).cumsum()
    x["ROC_12"] = close.pct_change(12) * 100
    x["VOLUME_SMA_20"] = volume.rolling(20).mean()
    x["VOLUME_SPIKE"] = volume > x["VOLUME_SMA_20"] * 1.5

    up, down = high.diff(), -low.diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=x.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=x.index)
    atr = x["ATR_14"].replace(0, np.nan)
    pdi = 100 * plus.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr
    mdi = 100 * minus.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    x["ADX_14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    x["STOCH_K"] = 100 * (close - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min()).replace(0, np.nan)
    x["STOCH_D"] = x["STOCH_K"].rolling(3).mean()
    x["CCI_20"] = (typical - typical.rolling(20).mean()) / (0.015 * typical.rolling(20).std()).replace(0, np.nan)
    x["OBV"] = (np.sign(close.diff()).fillna(0) * volume.fillna(0)).cumsum()
    x["RET_1"], x["RET_5"], x["RET_20"] = close.pct_change() * 100, close.pct_change(5) * 100, close.pct_change(20) * 100
    return x


@dataclass
class Signal:
    action: str
    confidence: float
    reasons: list[str]
    entry: float
    stop: float
    target: float
    risk_reward: float


def score_signal(row: pd.Series, strategy: str = "Ensemble", atr_mult: float = 1.5, reward_r: float = 2.0) -> Signal:
    close, atr = _num(row.get("Close")), _num(row.get("ATR_14"))
    if not math.isfinite(close):
        return Signal("HOLD", 0.0, ["No valid close price"], np.nan, np.nan, np.nan, np.nan)
    if not math.isfinite(atr) or atr <= 0:
        atr = max(close * 0.01, 0.01)

    e9, e21, e50 = _num(row.get("EMA_9")), _num(row.get("EMA_21")), _num(row.get("EMA_50"))
    macd, macd_sig = _num(row.get("MACD")), _num(row.get("MACD_SIGNAL"))
    rsi, vwap, adx = _num(row.get("RSI_14")), _num(row.get("VWAP")), _num(row.get("ADX_14"))
    stoch_k, stoch_d = _num(row.get("STOCH_K")), _num(row.get("STOCH_D"))
    votes, reasons = [], []

    if math.isfinite(e9) and math.isfinite(e21):
        votes.append(1 if e9 > e21 else -1); reasons.append(f"EMA 9 {'above' if e9 > e21 else 'below'} EMA 21")
    if math.isfinite(e21) and math.isfinite(e50):
        votes.append(1 if e21 > e50 else -1); reasons.append(f"EMA 21 {'above' if e21 > e50 else 'below'} EMA 50")
    if math.isfinite(macd) and math.isfinite(macd_sig):
        votes.append(1 if macd > macd_sig else -1); reasons.append(f"MACD {'bullish' if macd > macd_sig else 'bearish'}")
    if math.isfinite(rsi):
        if rsi < 35: votes.append(1); reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 65: votes.append(-1); reasons.append(f"RSI overbought ({rsi:.1f})")
        else: reasons.append(f"RSI neutral ({rsi:.1f})")
    if math.isfinite(vwap):
        votes.append(1 if close > vwap else -1); reasons.append(f"Price {'above' if close > vwap else 'below'} VWAP")
    if math.isfinite(adx): reasons.append(f"ADX {adx:.1f} ({'strong trend' if adx >= 25 else 'range/weak trend'})")
    if math.isfinite(stoch_k) and math.isfinite(stoch_d): reasons.append(f"Stochastic {'bullish' if stoch_k > stoch_d else 'bearish'}")

    if strategy == "EMA Crossover":
        raw = 1 if math.isfinite(e9) and math.isfinite(e21) and e9 > e21 else -1
        confidence = 65.0 + (10.0 if math.isfinite(e21) and math.isfinite(e50) and ((e9 > e21) == (e21 > e50)) else 0)
    elif strategy == "RSI Reversion":
        raw = 1 if math.isfinite(rsi) and rsi < 35 else (-1 if math.isfinite(rsi) and rsi > 65 else 0)
        confidence = 68.0 if raw else 50.0
    elif strategy == "MACD":
        raw = 1 if math.isfinite(macd) and math.isfinite(macd_sig) and macd > macd_sig else -1
        confidence = 65.0
    elif strategy == "Trend Momentum":
        bull = sum([math.isfinite(e21) and math.isfinite(e50) and e21 > e50, math.isfinite(close) and math.isfinite(e21) and close > e21, math.isfinite(macd) and math.isfinite(macd_sig) and macd > macd_sig, math.isfinite(adx) and adx >= 20, math.isfinite(vwap) and close > vwap])
        bear = sum([math.isfinite(e21) and math.isfinite(e50) and e21 < e50, math.isfinite(close) and math.isfinite(e21) and close < e21, math.isfinite(macd) and math.isfinite(macd_sig) and macd < macd_sig, math.isfinite(adx) and adx >= 20, math.isfinite(vwap) and close < vwap])
        raw = 1 if bull > bear else (-1 if bear > bull else 0); confidence = 50 + 45 * abs(bull - bear) / 5
    elif strategy == "Mean Reversion":
        raw = 1 if math.isfinite(rsi) and rsi < 35 and math.isfinite(vwap) and close < vwap else (-1 if math.isfinite(rsi) and rsi > 65 and math.isfinite(vwap) and close > vwap else 0)
        confidence = 72.0 if raw else 50.0
    else:
        raw = int(np.sign(sum(votes))) if votes else 0
        agreement = abs(sum(votes)) / len(votes) if votes else 0
        confidence = 50.0 + 45.0 * agreement

    action = "BUY" if raw > 0 else ("SELL" if raw < 0 else "HOLD")
    if action == "BUY":
        stop, target = close - atr_mult * atr, close + reward_r * atr_mult * atr
    elif action == "SELL":
        stop, target = close + atr_mult * atr, close - reward_r * atr_mult * atr
    else:
        stop = target = np.nan
    rr = reward_r if action != "HOLD" else np.nan
    reasons.append(f"{action} selected from {strategy} evidence." if action != "HOLD" else "Mixed evidence: no trade.")
    return Signal(action, round(float(min(100, max(0, confidence))), 1), reasons, close, stop, target, rr)


def position_size(capital: float, risk_pct: float, entry: float, stop: float) -> int:
    risk_cash = max(0.0, float(capital) * float(risk_pct) / 100)
    per_unit = abs(float(entry) - float(stop))
    return int(risk_cash // per_unit) if per_unit > 0 else 0


def backtest(df: pd.DataFrame, strategy: str, initial_capital: float, risk_pct: float, brokerage_pct: float = 0.03):
    """Long/short paper backtest with risk sizing and simple transaction costs."""
    x = add_indicators(df); cash = float(initial_capital); position = 0; entry = stop = 0.0; equity = []; trades = []
    for idx, row in x.iterrows():
        price = _num(row["Close"])
        if not math.isfinite(price): equity.append(cash); continue
        sig = score_signal(row, strategy)
        if position == 0 and sig.action in {"BUY", "SELL"}:
            qty = position_size(cash, risk_pct, price, sig.stop)
            if qty > 0:
                fee = qty * price * brokerage_pct / 100; cash -= fee; position = qty if sig.action == "BUY" else -qty; entry, stop = price, sig.stop
                trades.append({"Date": idx, "Side": sig.action, "Price": price, "Qty": qty, "PnL": -fee})
        elif position > 0 and (price <= stop or sig.action == "SELL"):
            fee = position * price * brokerage_pct / 100; pnl = position * (price - entry) - fee; cash += position * price - fee; trades.append({"Date": idx, "Side": "EXIT", "Price": price, "Qty": position, "PnL": pnl}); position = 0
        elif position < 0 and (price >= stop or sig.action == "BUY"):
            qty = abs(position); fee = qty * price * brokerage_pct / 100; pnl = qty * (entry - price) - fee; cash += pnl; trades.append({"Date": idx, "Side": "EXIT", "Price": price, "Qty": qty, "PnL": pnl}); position = 0
        unrealized = position * (price - entry) if position > 0 else (abs(position) * (entry - price) if position < 0 else 0)
        equity.append(cash + unrealized)
    if position != 0 and len(x):
        price = float(x["Close"].iloc[-1]); qty = abs(position); fee = qty * price * brokerage_pct / 100; pnl = (qty * (price - entry) if position > 0 else qty * (entry - price)) - fee; cash += pnl; trades.append({"Date": x.index[-1], "Side": "FORCED EXIT", "Price": price, "Qty": qty, "PnL": pnl}); equity[-1] = cash
    eq = pd.Series(equity, index=x.index, dtype=float); dd = (eq - eq.cummax()) / eq.cummax().replace(0, np.nan); t = pd.DataFrame(trades)
    exits = t[t["Side"].isin(["EXIT", "FORCED EXIT"])] if not t.empty else pd.DataFrame()
    win = float((exits["PnL"] > 0).mean() * 100) if not exits.empty else 0.0
    gross_profit = float(exits.loc[exits["PnL"] > 0, "PnL"].sum()) if not exits.empty else 0.0; gross_loss = float(-exits.loc[exits["PnL"] < 0, "PnL"].sum()) if not exits.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna(); sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0.0
    metrics = {"Net P&L": cash - initial_capital, "Return %": (cash / initial_capital - 1) * 100 if initial_capital else 0.0, "Max Drawdown %": float(dd.min() * 100) if len(dd) else 0.0, "Trades": len(exits), "Win Rate %": win, "Profit Factor": profit_factor, "Sharpe": sharpe, "Final Equity": cash, "Trade Log": t}
    return pd.DataFrame({"Equity": eq}), metrics


def signal_accuracy(df: pd.DataFrame, strategy: str, horizon: int = 5) -> dict:
    """Historical directional hit-rate; never present it as future probability."""
    x = add_indicators(df); outcomes = []
    for i in range(max(0, len(x) - horizon)):
        sig = score_signal(x.iloc[i], strategy)
        if sig.action == "HOLD": continue
        future, current = _num(x["Close"].iloc[i + horizon]), _num(x["Close"].iloc[i])
        if math.isfinite(future) and math.isfinite(current): outcomes.append((sig.action == "BUY" and future > current) or (sig.action == "SELL" and future < current))
    return {"Accuracy %": 100 * sum(outcomes) / len(outcomes) if outcomes else 0.0, "Signals": len(outcomes), "Horizon": horizon}


def regime_label(row: pd.Series) -> str:
    adx, e21, e50 = _num(row.get("ADX_14")), _num(row.get("EMA_21")), _num(row.get("EMA_50"))
    if math.isfinite(adx) and adx >= 25:
        return "Bull Trend" if math.isfinite(e21) and math.isfinite(e50) and e21 > e50 else "Bear Trend"
    return "Sideways / Range"


def build_signal_table(x: pd.DataFrame, strategy: str) -> pd.DataFrame:
    rows = []
    for idx, row in x.tail(120).iterrows():
        s = score_signal(row, strategy)
        rows.append({"Date": idx, "Signal": s.action, "Evidence %": s.confidence, "Close": _num(row["Close"]), "RSI": _num(row.get("RSI_14")), "ADX": _num(row.get("ADX_14")), "EMA 9/21": f"{_num(row.get('EMA_9')):.2f} / {_num(row.get('EMA_21')):.2f}"})
    return pd.DataFrame(rows)


def price_chart(x: pd.DataFrame, symbol: str) -> go.Figure:
    y = x.tail(300); fig = go.Figure()
    fig.add_trace(go.Candlestick(x=y.index, open=y["Open"], high=y["High"], low=y["Low"], close=y["Close"], name="Price"))
    for col in ["EMA_9", "EMA_21", "EMA_50", "VWAP", "BB_UPPER", "BB_LOWER"]:
        if col in y: fig.add_trace(go.Scatter(x=y.index, y=y[col], mode="lines", name=col))
    fig.update_layout(title=f"{symbol} — Price & Indicators", height=560, xaxis_rangeslider_visible=False); return fig


def indicator_chart(x: pd.DataFrame, indicator: str) -> go.Figure:
    y = x.tail(300); fig = go.Figure(go.Scatter(x=y.index, y=y[indicator], mode="lines", name=indicator))
    if indicator == "RSI_14": fig.add_hline(y=70, line_dash="dash"); fig.add_hline(y=30, line_dash="dash")
    if indicator == "ADX_14": fig.add_hline(y=25, line_dash="dash")
    fig.update_layout(title=indicator, height=320); return fig


@st.cache_data(ttl=300, show_spinner=False)
def index_overview() -> list[dict]:
    overview = []
    for name, ticker in INDEX_UNIVERSE.items():
        data = download_market_data(ticker, "1mo", "1d")
        if data.empty:
            overview.append({"Index": name, "Price": np.nan, "Change %": np.nan, "Signal": "N/A"}); continue
        enriched = add_indicators(data); latest = enriched.iloc[-1]; previous = _num(enriched["Close"].iloc[-2]) if len(enriched) > 1 else np.nan
        change = ((latest["Close"] / previous) - 1) * 100 if math.isfinite(previous) and previous else np.nan
        overview.append({"Index": name, "Price": _num(latest["Close"]), "Change %": change, "Signal": score_signal(latest, "Ensemble").action})
    return overview


def render_app():
    st.title("📈 Algo Trading Pro — Indian Markets")
    st.caption("NIFTY 50 • SENSEX • BANK NIFTY • NSE stocks | Research + paper-trading platform. Historical accuracy is backtest evidence, not a guarantee of future profit.")
    with st.sidebar:
        st.header("🎯 Market")
        market_type = st.radio("Universe", ["Indices", "Stocks", "Custom"])
        if market_type == "Indices":
            selected = st.selectbox("Index", list(INDEX_UNIVERSE)); symbol, display_name = INDEX_UNIVERSE[selected], selected
        elif market_type == "Stocks":
            selected = st.selectbox("Stock", list(POPULAR_STOCKS)); symbol, display_name = POPULAR_STOCKS[selected], selected
        else:
            symbol = st.text_input("Yahoo ticker", "RELIANCE.NS").strip().upper(); display_name = symbol
        interval = st.selectbox("Timeframe", list(PERIODS), index=0)
        period = st.selectbox("History", PERIODS[interval], index=min(3, len(PERIODS[interval]) - 1))
        strategy = st.selectbox("Strategy", ["Ensemble", "Trend Momentum", "EMA Crossover", "RSI Reversion", "MACD", "Mean Reversion"])
        st.divider(); st.header("🛡️ Risk")
        capital = st.number_input("Paper capital (₹)", min_value=1000.0, value=100000.0, step=5000.0)
        risk_pct = st.slider("Risk / trade (%)", 0.1, 5.0, 1.0, 0.1)
        atr_mult = st.slider("Stop ATR multiplier", 0.5, 4.0, 1.5, 0.1)
        reward_r = st.slider("Target (R)", 1.0, 5.0, 2.0, 0.5)
        brokerage = st.number_input("Brokerage + costs (%)", 0.0, 0.5, 0.03, 0.01)
        if st.button("🔄 Refresh market data", type="primary", use_container_width=True): download_market_data.clear(); index_overview.clear(); st.rerun()

    df = download_market_data(symbol, period, interval)
    if df.empty:
        st.error(f"No market data returned for {symbol} ({interval}, {period}).")
        st.info("For first validation use an index + 1d + 1mo. Intraday Yahoo Finance history is limited and can be throttled.")
        return

    st.subheader("🇮🇳 India Market Overview")
    overview = index_overview(); ov_cols = st.columns(3)
    for col, item in zip(ov_cols, overview):
        with col:
            price_text = f"₹{item['Price']:,.2f}" if math.isfinite(item["Price"]) else "N/A"; change_text = f"{item['Change %']:+.2f}%" if math.isfinite(item["Change %"]) else "N/A"
            st.metric(item["Index"], price_text, change_text); st.caption(f"Ensemble: {item['Signal']}")

    x = add_indicators(df); sig = score_signal(x.iloc[-1], strategy, atr_mult=atr_mult, reward_r=reward_r)
    qty = position_size(capital, risk_pct, sig.entry, sig.stop) if sig.action in {"BUY", "SELL"} else 0
    acc = signal_accuracy(df, strategy, horizon=5); regime = regime_label(x.iloc[-1])
    a, b, c, d, e = st.columns(5)
    a.metric("Last Price", f"₹{sig.entry:,.2f}"); b.metric("Signal", sig.action, f"{sig.confidence:.1f}% evidence"); c.metric("Historical Accuracy", f"{acc['Accuracy %']:.1f}%", f"{acc['Signals']} signals / {acc['Horizon']} bars"); d.metric("Regime", regime); e.metric("Risk Qty", f"{qty:,}" if qty else "—")
    if sig.action == "BUY": st.success(f"🟢 BUY / LONG — paper quantity **{qty:,}** | Stop ₹{sig.stop:,.2f} | Target ₹{sig.target:,.2f}")
    elif sig.action == "SELL": st.error(f"🔴 SELL / SHORT — paper quantity **{qty:,}** | Stop ₹{sig.stop:,.2f} | Target ₹{sig.target:,.2f}")
    else: st.warning("🟡 HOLD / NO TRADE — current evidence is mixed.")

    tabs = st.tabs(["📊 Dashboard", "🧪 Backtest", "🎯 Accuracy", "📈 Indicators", "🛡️ Risk", "📋 Signals", "ℹ️ Data"])
    with tabs[0]:
        st.plotly_chart(price_chart(x, display_name), use_container_width=True)
        with st.expander("Signal reasoning", expanded=True):
            for reason in sig.reasons: st.write("• " + reason)
            st.caption("Evidence % measures indicator agreement/strength. It is NOT probability of profit.")
    with tabs[1]:
        bt_equity, metrics = backtest(df, strategy, capital, risk_pct, brokerage)
        m1, m2, m3, m4 = st.columns(4); m1.metric("Net P&L", f"₹{metrics['Net P&L']:,.0f}"); m2.metric("Return", f"{metrics['Return %']:.2f}%"); m3.metric("Max Drawdown", f"{metrics['Max Drawdown %']:.2f}%"); m4.metric("Win Rate", f"{metrics['Win Rate %']:.1f}%")
        m5, m6, m7 = st.columns(3); m5.metric("Trades", metrics["Trades"]); m6.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}" if math.isfinite(metrics["Profit Factor"]) else "∞"); m7.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
        fig = go.Figure(go.Scatter(x=bt_equity.index, y=bt_equity["Equity"], mode="lines", name="Equity")); fig.update_layout(title="Backtest Equity Curve", height=400); st.plotly_chart(fig, use_container_width=True)
        if not metrics["Trade Log"].empty: st.dataframe(metrics["Trade Log"], use_container_width=True, hide_index=True)
        st.subheader("Strategy comparison")
        comparison = []
        for candidate in ["Ensemble", "Trend Momentum", "EMA Crossover", "RSI Reversion", "MACD", "Mean Reversion"]:
            _, cm = backtest(df, candidate, capital, risk_pct, brokerage)
            comparison.append({"Strategy": candidate, "Return %": round(cm["Return %"], 2), "Win Rate %": round(cm["Win Rate %"], 2), "Max DD %": round(cm["Max Drawdown %"], 2), "Profit Factor": round(cm["Profit Factor"], 2) if math.isfinite(cm["Profit Factor"]) else np.inf, "Trades": cm["Trades"], "Sharpe": round(cm["Sharpe"], 2)})
        st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
    with tabs[2]:
        st.subheader("Historical signal accuracy")
        st.info("Accuracy = percentage of non-HOLD signals whose direction matched the close move after the selected horizon. It is a historical diagnostic, not a forecast probability.")
        rows = []
        for h in [1, 3, 5, 10, 20]:
            result = signal_accuracy(df, strategy, h); rows.append({"Horizon (bars)": h, "Accuracy %": round(result["Accuracy %"], 2), "Signals": result["Signals"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with tabs[3]:
        indicator = st.selectbox("Indicator", ["RSI_14", "MACD_HIST", "ADX_14", "STOCH_K", "CCI_20", "ROC_12", "BB_WIDTH"]); st.plotly_chart(indicator_chart(x, indicator), use_container_width=True)
        latest = x.iloc[-1]; cols = st.columns(4)
        for i, name in enumerate(["RSI_14", "ADX_14", "MACD_HIST", "ATR_14"]): cols[i].metric(name, f"{_num(latest.get(name)):.2f}")
    with tabs[4]:
        st.write(f"**Capital at risk:** ₹{capital * risk_pct / 100:,.2f}"); st.write(f"**Suggested paper quantity:** {qty:,}"); st.write(f"**Entry:** ₹{sig.entry:,.2f}"); st.write(f"**Stop:** ₹{sig.stop:,.2f}" if sig.action != "HOLD" else "**Stop:** —"); st.write(f"**Target:** ₹{sig.target:,.2f}" if sig.action != "HOLD" else "**Target:** —"); st.warning("Use this for research/paper trading unless you separately integrate and authorize a regulated broker.")
    with tabs[5]: st.dataframe(build_signal_table(x, strategy), use_container_width=True, hide_index=True)
    with tabs[6]:
        st.write({"Ticker": symbol, "Rows": len(df), "First bar": str(df.index.min()), "Last bar": str(df.index.max()), "Interval": interval, "History": period, "Source": "Yahoo Finance via yfinance"})
        st.caption("Market-data availability depends on Yahoo Finance limits, ticker validity, and network/rate limits.")


if __name__ == "__main__":
    render_app()
