from __future__ import annotations
import asyncio
import math
import time
from dataclasses import asdict, dataclass
from typing import Any
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.live_feed import fetch_live_1m
from app.ml_model import model_is_ready, predict_ml_signal, prewarm_ml_model
from app.news import get_news, news_confirmation
from app.option_chain import build_option_recommendation, fetch_bse_sensex_chain, fetch_nse_chain
from app.trading import INDEX_UNIVERSE, POPULAR_STOCKS, add_indicators, position_size, score_signal

app = FastAPI(title="Algo Trading Pro API", version="1.5.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
INDEX_OPTION_SYMBOLS = {"NIFTY 50": "NIFTY", "BANK NIFTY": "BANKNIFTY", "SENSEX": "SENSEX"}
INDEX_DISPLAY_ORDER = ("NIFTY 50", "BANK NIFTY", "SENSEX")
_option_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
_market_cache: dict[str, tuple[float, pd.DataFrame, float]] = {}
_news_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_index_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_index_refreshing: set[str] = set()
_index_refresh_lock = asyncio.Lock()
MARKET_CACHE_TTL = 5.0
NEWS_CACHE_TTL = 180.0
INDEX_CACHE_TTL = 15.0
OPTION_CACHE_TTL = 30.0

@dataclass
class PaperPosition:
    side: str; qty: int; entry: float; stop: float; target: float; opened_at: float; confidence: float

class PaperAccount:
    def __init__(self) -> None:
        self.cash=100000.0; self.position=None; self.realized=0.0; self.last_event=""; self.last_action_key=""
    def reset(self, capital: float) -> None:
        self.cash=float(capital); self.position=None; self.realized=0.0; self.last_event=""; self.last_action_key=""
accounts: dict[str, PaperAccount] = {}
def account_for(client_id: str, capital: float=100000.0) -> PaperAccount:
    if client_id not in accounts: accounts[client_id]=PaperAccount(); accounts[client_id].reset(capital)
    return accounts[client_id]

def _json_safe(value: Any) -> Any:
    if isinstance(value,dict): return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_json_safe(v) for v in value]
    if hasattr(value,"item") and not isinstance(value,(str,bytes)):
        try: return _json_safe(value.item())
        except (ValueError,TypeError): pass
    if isinstance(value,float): return value if math.isfinite(value) else None
    return value

def market_snapshot(symbol: str):
    now=time.time(); cached=_market_cache.get(symbol)
    if cached and now-cached[0]<MARKET_CACHE_TTL: return cached[1],cached[2]
    d=fetch_live_1m(symbol,"5d")
    if d is None or d.empty: return None,None
    enriched=add_indicators(d); price=float(d.Close.iloc[-1]); _market_cache[symbol]=(now,enriched,price); return enriched,price

def option_chain_cached(name: str):
    now=time.time(); cached=_option_cache.get(name)
    if cached and now-cached[0]<OPTION_CACHE_TTL: return cached[1]
    chain=fetch_bse_sensex_chain() if name=="SENSEX" else fetch_nse_chain(name); _option_cache[name]=(now,chain); return chain

def news_cached(symbol: str):
    now=time.time(); cached=_news_cache.get(symbol)
    if cached and now-cached[0]<NEWS_CACHE_TTL: return cached[1]
    value=get_news(symbol); _news_cache[symbol]=(now,value); return value

def model_signal(x: pd.DataFrame, symbol: str, strategy: str, threshold: float, reward_r: float = 2.0):
    technical=score_signal(x.iloc[-1],strategy,reward_r=reward_r)
    if strategy!="Ensemble": return technical,None
    ml=predict_ml_signal(x,symbol,threshold=threshold,train_if_missing=False)
    if not ml.trained:
        technical.reasons.append(f"Calibrated ML unavailable: {ml.reason}")
        technical.reasons.append(f"Technical evidence confidence: {technical.confidence:.1f}%")
        return technical,ml
    technical.action=ml.action; technical.confidence=ml.confidence
    if ml.action in {"BUY","SELL"}:
        atr=float(x.iloc[-1].get("ATR_14",0.0))
        if not math.isfinite(atr) or atr<=0: atr=technical.entry*0.01
        risk=1.5*atr
        if ml.action=="BUY": technical.stop,technical.target=technical.entry-risk,technical.entry+reward_r*risk
        else: technical.stop,technical.target=technical.entry+risk,technical.entry-reward_r*risk
        technical.risk_reward=reward_r
    else: technical.stop=technical.target=float("nan"); technical.risk_reward=float("nan")
    technical.reasons.append(ml.reason)
    if ml.validation_accuracy is not None: technical.reasons.append(f"Training samples: {ml.samples:,} · validation accuracy: {ml.validation_accuracy*100:.1f}%")
    else: technical.reasons.append(f"Training samples: {ml.samples:,}")
    return technical,ml

def sell_price_for(sig):
    if sig.action == "BUY": return sig.target
    if sig.action == "SELL": return sig.entry
    return float("nan")

def index_decision(name: str,ticker: str,threshold: float,reward_r: float=2.0):
    x,spot=market_snapshot(ticker)
    if x is None or spot is None: return {"name":name,"symbol":ticker,"available":False,"reason":"Live index data unavailable"}
    sig,ml=model_signal(x,ticker,"Ensemble",threshold,reward_r)
    option=build_option_recommendation(spot,sig.action,sig.target,option_chain_cached(INDEX_OPTION_SYMBOLS[name]))
    return {"name":name,"symbol":ticker,"available":True,"price":spot,"decision":sig.action,"confidence":sig.confidence,"entry":sig.entry,"stop":sig.stop,"target":sig.target,"sell_price":sell_price_for(sig),"risk_reward":sig.risk_reward,"reasons":sig.reasons,"model":asdict(ml) if ml else None,"option":option,"timestamp":time.time()}

async def refresh_index(name: str, ticker: str, threshold: float, reward_r: float=2.0) -> None:
    async with _index_refresh_lock:
        if name in _index_refreshing: return
        _index_refreshing.add(name)
    try:
        value=await asyncio.to_thread(index_decision,name,ticker,threshold,reward_r)
        _index_cache[name]=(time.time(),value)
    finally:
        async with _index_refresh_lock:
            _index_refreshing.discard(name)

async def refresh_stale_indices(threshold: float, reward_r: float=2.0) -> None:
    now=time.time(); jobs=[]
    for name in INDEX_DISPLAY_ORDER:
        cached=_index_cache.get(name)
        if cached is None or now-cached[0]>=INDEX_CACHE_TTL:
            jobs.append(asyncio.create_task(refresh_index(name,INDEX_UNIVERSE[name],threshold,reward_r)))
    if jobs: await asyncio.gather(*jobs,return_exceptions=True)

def current_indices() -> dict[str,dict[str,Any]]:
    return {name:data for name in INDEX_DISPLAY_ORDER if (cached:=_index_cache.get(name)) and (data:=cached[1])}

def execute_paper(account: PaperAccount,x: pd.DataFrame, symbol: str,strategy: str,risk_pct: float,brokerage_pct: float,reward_r: float,threshold: float,strict: bool,auto: bool,news: dict[str, Any]):
    price=float(x.iloc[-1]["Close"]); sig,ml=model_signal(x,symbol,strategy,threshold,reward_r)
    news_ok,news_reason=news_confirmation(sig.action,news)
    sig.reasons.append(news_reason)
    qualified=sig.action!="HOLD" and sig.confidence>=threshold and news_ok if strict else sig.action!="HOLD" and news_ok
    event=""
    if account.position:
        p=account.position; pnl=(price-p.entry)*p.qty if p.side=="LONG" else (p.entry-price)*p.qty; exit_reason=None
        if p.side=="LONG" and price<=p.stop: exit_reason="STOP"
        elif p.side=="LONG" and price>=p.target: exit_reason="TARGET"
        elif p.side=="SHORT" and price>=p.stop: exit_reason="STOP"
        elif p.side=="SHORT" and price<=p.target: exit_reason="TARGET"
        elif (p.side=="LONG" and sig.action=="SELL") or (p.side=="SHORT" and sig.action=="BUY"): exit_reason="SIGNAL FLIP"
        if exit_reason:
            fee=abs(p.qty*price)*brokerage_pct/100; account.realized+=pnl-fee; account.cash+=pnl-fee; event=f"{p.side} CLOSED · {exit_reason} · P/L ₹{pnl-fee:,.2f}"; account.position=None
    if account.position is None and auto and qualified and sig.action in {"BUY","SELL"}:
        qty=position_size(account.cash,risk_pct,price,sig.stop)
        if qty>0:
            side="LONG" if sig.action=="BUY" else "SHORT"; fee=abs(qty*price)*brokerage_pct/100; account.cash-=fee; account.position=PaperPosition(side,qty,price,sig.stop,sig.target,time.time(),sig.confidence); event=f"PAPER {sig.action} OPEN · Qty {qty:,} · Entry ₹{price:,.2f} · News {news.get('bias','NEUTRAL')}";
    p=account.position; live_pnl=(price-p.entry)*p.qty if p and p.side=="LONG" else ((p.entry-price)*p.qty if p else 0.0)
    signal_data=asdict(sig); signal_data["sell_price"]=sell_price_for(sig)
    return {"price":price,"signal":signal_data,"model":asdict(ml) if ml else None,"qualified":qualified,"position":asdict(p) if p else None,"live_pnl":live_pnl,"cash":account.cash,"realized_pnl":account.realized,"event":event,"timestamp":time.time()}

@app.get("/")
def root(): return {"service":"Algo Trading Pro API","status":"ok","websocket":"/ws","version":"1.5.1"}
@app.get("/health")
def health(): return {"status":"ok"}
@app.get("/universe")
def universe(): return {"indices":INDEX_UNIVERSE,"stocks":POPULAR_STOCKS}
@app.get("/news")
def news(symbol: str = "RELIANCE.NS"): return _json_safe(news_cached(symbol.upper()))

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept(); client_id=str(id(ws)); account=account_for(client_id)
    config={"symbol":"RELIANCE.NS","strategy":"Ensemble","risk_pct":1.0,"brokerage_pct":0.03,"reward_r":2.0,"threshold":95.0,"strict":True,"auto":True,"capital":100000.0}
    try:
        while True:
            try:
                while True:
                    message=await asyncio.wait_for(ws.receive_json(),timeout=0.01)
                    if isinstance(message,dict):
                        config.update({k:message[k] for k in config if k in message})
                        if "capital" in message and account.position is None: account.reset(float(message["capital"]))
            except asyncio.TimeoutError: pass
            x,price=await asyncio.to_thread(market_snapshot,config["symbol"])
            if x is None or price is None:
                payload={"type":"error","message":"Live market data unavailable","timestamp":time.time()}
            else:
                if config["strategy"]=="Ensemble" and not model_is_ready(config["symbol"]): asyncio.create_task(asyncio.to_thread(prewarm_ml_model,x,config["symbol"]))
                news_data=await asyncio.to_thread(news_cached,config["symbol"])
                result=await asyncio.to_thread(execute_paper,account,x,config["symbol"],config["strategy"],float(config["risk_pct"]),float(config["brokerage_pct"]),float(config["reward_r"]),float(config["threshold"]),bool(config["strict"]),bool(config["auto"]),news_data)
                result["type"]="tick"; result["symbol"]=config["symbol"]; result["news"]=news_data
                await ws.send_json(_json_safe(result))
                if not _index_cache or any(time.time()-v[0]>=INDEX_CACHE_TTL for v in _index_cache.values()): asyncio.create_task(refresh_stale_indices(float(config["threshold"]),float(config["reward_r"])))
                cached_indices=current_indices(); result["indices"]=cached_indices if cached_indices else {}; result["timestamp"]=time.time()
                await ws.send_json(_json_safe(result)); await asyncio.sleep(1.0); continue
            await ws.send_json(_json_safe(payload)); await asyncio.sleep(1.0)
    except (WebSocketDisconnect,RuntimeError): accounts.pop(client_id,None)
