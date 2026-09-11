from __future__ import annotations

import math
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
    regime_label,
    backtest,
)
from app.paper_engine import process_latest_bar, reset_paper_account, _position_from_state, _money


STRATEGIES = ["Ensemble", "Trend Momentum", "EMA Crossover", "RSI Reversion", "MACD", "Mean Reversion"]


def _finite(v):
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def _pct(v):
    return "—" if not _finite(v) else f"{float(v):+.2f}%"


def _now():
    return datetime.now(timezone.utc).astimezone().strftime("%d %b %Y, %H:%M:%S")


def _style():
    st.markdown("""
    <style>
    .stApp{background:#080b10}.block-container{max-width:1540px;padding:1rem 2rem 4rem}
    [data-testid="stSidebar"]{background:#0c1118;border-right:1px solid #202a36}
    .topbar,.panel{background:linear-gradient(145deg,#111821,#0d131a);border:1px solid #24303d;border-radius:16px}
    .topbar{padding:18px 22px;margin-bottom:14px}.brand{font-size:30px;font-weight:800;letter-spacing:-1px}.sub{color:#8795a5;font-size:13px;margin-top:3px}
    .live{display:inline-flex;gap:8px;align-items:center;padding:6px 10px;border-radius:999px;border:1px solid #245139;background:#102219;color:#71e39a;font-size:11px;font-weight:800}
    .pulse{width:7px;height:7px;border-radius:50%;background:#49e58a}
    .panel{padding:16px;margin-bottom:14px}.panel-title{font-size:13px;text-transform:uppercase;letter-spacing:1px;color:#7e8c9c;font-weight:750;margin-bottom:9px}
    .price{font-size:34px;font-weight:850;letter-spacing:-1px}.small{font-size:12px;color:#7f8d9c}.buy{color:#55e29a;font-weight:850}.sell{color:#ff6e73;font-weight:850}.hold{color:#e7c85c;font-weight:850}
    .signal{border:1px solid #2a3947;border-radius:14px;padding:18px;background:#0d151d}.signal-action{font-size:42px;font-weight:900;line-height:1}.tag{font-size:11px;padding:5px 8px;border-radius:7px;background:#18222d;color:#aab7c4}
    .order{border:1px solid #314252;border-radius:14px;padding:14px;background:#0d141b}.order-head{font-size:12px;color:#8190a0}.order-price{font-size:24px;font-weight:800}
    div[data-testid="stMetric"]{background:#101720;border:1px solid #222e3a;border-radius:12px;padding:9px 11px}
    .stButton>button{border-radius:10px;font-weight:700}.section{font-size:19px;font-weight:800;margin:16px 0 9px}
    </style>""", unsafe_allow_html=True)


def _chart(x, sig, show_volume=True):
    y=x.tail(min(260,len(x)))
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=y.index,open=y.Open,high=y.High,low=y.Low,close=y.Close,name="Price"))
    for c in ["EMA_9","EMA_21","EMA_50"]:
        if c in y: fig.add_trace(go.Scatter(x=y.index,y=y[c],mode="lines",name=c.replace("_"," ")))
    if "VWAP" in y: fig.add_trace(go.Scatter(x=y.index,y=y.VWAP,mode="lines",name="VWAP"))
    if _finite(sig.entry): fig.add_hline(y=sig.entry,line_dash="dot",annotation_text=f"SIGNAL ENTRY  {_money(sig.entry)}")
    if _finite(sig.stop): fig.add_hline(y=sig.stop,line_dash="dash",annotation_text=f"STOP  {_money(sig.stop)}")
    if _finite(sig.target): fig.add_hline(y=sig.target,line_dash="dash",annotation_text=f"TARGET  {_money(sig.target)}")
    fig.update_layout(height=520,template="plotly_dark",paper_bgcolor="#080b10",plot_bgcolor="#080b10",margin=dict(l=8,r=8,t=25,b=8),xaxis_rangeslider_visible=False,hovermode="x unified",legend=dict(orientation="h",y=1.02,x=0))
    return fig


def _signal_accuracy(x, strategy, horizon=5, threshold=0):
    rows=[]
    for i in range(1,max(1,len(x)-horizon)):
        s=score_signal(x.iloc[i],strategy)
        if s.action=="HOLD" or s.confidence<threshold: continue
        future=x.iloc[i+1:i+1+horizon]
        if future.empty: continue
        direction=1 if s.action=="BUY" else -1
        correct=(float(future.Close.iloc[-1])-float(s.entry))*direction>0
        rows.append(correct)
    return (100*sum(rows)/len(rows) if rows else 0.0,len(rows))


def _scanner():
    rows=[]
    for name,ticker in POPULAR_STOCKS.items():
        d=download_market_data(ticker,"1mo","1d")
        if d.empty: continue
        x=add_indicators(d); s=score_signal(x.iloc[-1],"Ensemble")
        last=float(x.Close.iloc[-1]); prev=float(x.Close.iloc[-2]) if len(x)>1 else last
        rows.append({"Symbol":name,"Price":last,"Change %":(last/prev-1)*100 if prev else 0,"Signal":s.action,"Evidence %":s.confidence,"RSI":float(x.RSI_14.iloc[-1]) if _finite(x.RSI_14.iloc[-1]) else np.nan,"ADX":float(x.ADX_14.iloc[-1]) if _finite(x.ADX_14.iloc[-1]) else np.nan})
    return pd.DataFrame(rows).sort_values("Evidence %",ascending=False) if rows else pd.DataFrame()


def render_app():
    _style()
    with st.sidebar:
        st.markdown("## ⚡ Algo Trading Pro")
        universe=st.selectbox("Market",["Indices","NSE Stocks","Custom Ticker"])
        if universe=="Indices":
            name=st.selectbox("Instrument",list(INDEX_UNIVERSE)); symbol=INDEX_UNIVERSE[name]
        elif universe=="NSE Stocks":
            name=st.selectbox("Instrument",list(POPULAR_STOCKS)); symbol=POPULAR_STOCKS[name]
        else:
            symbol=st.text_input("Yahoo ticker","RELIANCE.NS").strip().upper(); name=symbol
        interval=st.selectbox("Timeframe",list(PERIODS),index=4)
        period=st.selectbox("History",PERIODS[interval],index=0 if interval!="1d" else 3)
        strategy=st.selectbox("Strategy",STRATEGIES,index=0)
        st.divider(); st.markdown("### Precision engine")
        threshold=st.slider("Minimum signal evidence",50,95,90,1)
        strict=st.checkbox("Precision mode: trade only ≥ threshold",True)
        reward=st.slider("Target / Risk (R)",1.0,5.0,2.0,0.5)
        st.divider(); st.markdown("### Risk engine")
        capital=st.number_input("Paper capital (₹)",10000.0,100000000.0,100000.0,10000.0)
        risk=st.slider("Risk per trade %",0.1,3.0,1.0,0.1)
        brokerage=st.number_input("Costs %",0.0,0.50,0.03,0.01,format="%.2f")
        st.divider(); st.markdown("### Execution")
        auto=st.checkbox("AUTO PAPER TRADING",False)
        if st.button("↻ Refresh market",use_container_width=True): download_market_data.clear(); st.rerun()
        if st.button("Reset paper account",use_container_width=True): reset_paper_account(capital); st.rerun()

    df=download_market_data(symbol,period,interval)
    if df.empty:
        st.error(f"No market data for {symbol}. Try 1d + 1mo."); return
    x=add_indicators(df); sig=score_signal(x.iloc[-1],strategy,reward_r=reward)
    qualified=(sig.action!="HOLD" and sig.confidence>=threshold) if strict else sig.action!="HOLD"
    qty=position_size(capital,risk,sig.entry,sig.stop) if qualified else 0
    last=float(x.Close.iloc[-1]); prev=float(x.Close.iloc[-2]) if len(x)>1 else last; change=(last/prev-1)*100 if prev else 0
    regime=regime_label(x.iloc[-1])
    acc,signals=_signal_accuracy(x,strategy,5,threshold if strict else 0)

    st.markdown(f"<div class='topbar'><span class='live'><span class='pulse'></span>MARKET DATA CONNECTED · PAPER ONLY</span><div class='brand'>ALGO TRADING PRO</div><div class='sub'>{name} · {interval} · {period} · Last update {_now()} · Yahoo Finance data</div></div>",unsafe_allow_html=True)

    a,b,c,d,e=st.columns(5)
    a.metric("LAST PRICE",_money(last),_pct(change))
    b.metric("MODEL SIGNAL",sig.action,f"{sig.confidence:.1f}% evidence")
    c.metric("MARKET REGIME",regime)
    d.metric("5-BAR HIST. HIT RATE",f"{acc:.1f}%",f"{signals} qualifying signals")
    e.metric("PAPER QTY",f"{qty:,}" if qty else "WAIT")

    left,right=st.columns([1.55,1])
    with left:
        st.markdown("<div class='section'>Live Market Chart</div>",unsafe_allow_html=True)
        st.plotly_chart(_chart(x,sig),use_container_width=True,config={"displaylogo":False,"scrollZoom":True})
    with right:
        st.markdown("<div class='section'>🎯 AI Trade Decision</div>",unsafe_allow_html=True)
        cls=sig.action.lower()
        st.markdown(f"<div class='signal'><div class='small'>CURRENT DECISION · {name}</div><div class='signal-action {cls}'>{sig.action}</div><div style='margin-top:9px'><span class='tag'>Evidence {sig.confidence:.1f}%</span> <span class='tag'>R:R 1:{reward:.1f}</span></div></div>",unsafe_allow_html=True)
        st.markdown("#### Exact order levels")
        q1,q2=st.columns(2); q1.metric("Entry",_money(sig.entry)); q2.metric("Quantity",f"{qty:,}" if qty else "—")
        q3,q4=st.columns(2); q3.metric("Stop Loss",_money(sig.stop) if qualified else "WAIT"); q4.metric("Target",_money(sig.target) if qualified else "WAIT")
        if qualified:
            if sig.action=="BUY": st.success(f"🟢 AUTO ORDER READY · BUY {qty:,} shares @ {_money(sig.entry)}")
            else: st.error(f"🔴 AUTO ORDER READY · SHORT {qty:,} shares @ {_money(sig.entry)}")
        else:
            st.warning(f"🟡 NO ORDER · precision filter requires ≥ {threshold}% evidence")

        st.markdown("#### Why this trade?")
        for r in sig.reasons[:8]: st.write("• "+r)

    st.markdown("<div class='section'>Paper Execution Monitor</div>",unsafe_allow_html=True)
    pos=_position_from_state(st.session_state.get("paper_position")); trades=st.session_state.get("paper_trades",[])
    if auto and qualified:
        @st.fragment(run_every="60s")
        def runner():
            process_latest_bar(x,strategy,risk,brokerage,reward)
            st.caption(f"AUTO ENGINE · last cycle {_now()} · new bars only")
        runner()
    else:
        st.caption("Auto execution is gated by the precision filter. Enable AUTO PAPER TRADING to process each new bar.")
    if pos:
        p1,p2,p3,p4,p5=st.columns(5); p1.metric("POSITION",pos.side); p2.metric("ENTRY",_money(pos.entry)); p3.metric("LIVE STOP",_money(pos.stop)); p4.metric("LIVE TARGET",_money(pos.target)); p5.metric("QTY",f"{pos.qty:,}")
        st.markdown(f"<div class='order'><div class='order-head'>OPEN PAPER ORDER · {pos.opened_at}</div><div class='order-price'>{pos.side} × {pos.qty:,} · Entry {_money(pos.entry)}</div><div class='small'>Stop {_money(pos.stop)} · Target {_money(pos.target)} · Evidence {pos.confidence:.1f}%</div></div>",unsafe_allow_html=True)
    else:
        st.info("FLAT · No automatic paper position is open. A qualifying BUY/SELL will appear above and be recorded in the ledger.")

    tabs=st.tabs(["📊 Dashboard","🔎 Signal Scanner","🧪 Backtest Lab","📒 Paper Ledger","🛡 Risk","⚙ Model"])
    with tabs[0]:
        st.markdown("### Signal confidence ladder")
        levels=[50,60,70,80,90,95]
        rows=[]
        for level in levels:
            hit,n=_signal_accuracy(x,strategy,5,level); rows.append({"Minimum evidence %":level,"Historical hit rate %":round(hit,2),"Signals":n})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.caption("Historical hit rate is a diagnostic, not a promise of future accuracy.")
    with tabs[1]:
        if st.button("Run NSE signal scanner",type="primary"):
            with st.spinner("Scanning selected NSE watchlist..."): scan=_scanner()
            st.session_state.scan=scan
        scan=st.session_state.get("scan",pd.DataFrame())
        if not scan.empty: st.dataframe(scan,use_container_width=True,hide_index=True)
        else: st.info("Click the scanner button to rank the watchlist by model evidence.")
    with tabs[2]:
        eq,metrics=backtest(df,strategy,capital,risk,brokerage)
        m=st.columns(7)
        for col,label,key in zip(m,["Return","Net P&L","Max DD","Win Rate","Trades","Profit Factor","Sharpe"],["Return %","Net P&L","Max Drawdown %","Win Rate %","Trades","Profit Factor","Sharpe"]):
            v=metrics[key]; col.metric(label,f"{v:.2f}" if isinstance(v,(int,float,np.floating)) else str(v))
        fig=go.Figure(go.Scatter(x=eq.index,y=eq.Equity,mode="lines",name="Equity")); fig.update_layout(height=330,template="plotly_dark",paper_bgcolor="#080b10",plot_bgcolor="#080b10",margin=dict(l=5,r=5,t=10,b=5)); st.plotly_chart(fig,use_container_width=True)
        if not metrics["Trade Log"].empty: st.dataframe(metrics["Trade Log"].tail(100),use_container_width=True,hide_index=True)
    with tabs[3]:
        t=pd.DataFrame(trades)
        if t.empty: st.info("No paper orders yet.")
        else:
            st.dataframe(t.iloc[::-1],use_container_width=True,hide_index=True)
            st.download_button("Download CSV",t.to_csv(index=False).encode(),"paper_ledger.csv","text/csv")
    with tabs[4]:
        r1,r2,r3=st.columns(3); r1.metric("Capital at risk",_money(capital*risk/100)); r2.metric("Risk / trade",f"{risk:.1f}%"); r3.metric("Target R",f"{reward:.1f}R")
        st.write(f"Position size = risk capital ÷ stop distance → **{qty:,}** units when the precision gate passes.")
        st.warning("Paper/research mode only. No real broker order is sent by this application.")
    with tabs[5]:
        st.write(f"**Strategy:** {strategy}")
        st.write(f"**Precision gate:** {threshold}% evidence")
        st.write(f"**Qualified now:** {'YES' if qualified else 'NO'}")
        st.write("**Signal meaning:** evidence strength from indicator confluence; it is not a probability of profit.")
        st.write("**Execution:** previous-bar signal → next-bar open; stop is checked before target when both occur in one candle.")
        st.write("**Data:** Yahoo Finance via yfinance; intraday history and refresh availability are provider-dependent.")
