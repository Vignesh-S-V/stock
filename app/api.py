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
from app.option_chain import build_option_recommendation, fetch_bse_sensex_chain, fetch_nse_chain
from app.trading import INDEX_UNIVERSE, POPULAR_STOCKS, add_indicators, position_size, score_signal

app = FastAPI(title="Algo Trading Pro API", version="1.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
INDEX_OPTION_SYMBOLS = {"NIFTY 50": "NIFTY", "BANK NIFTY": "BANKNIFTY", "SENSEX": "SENSEX"}
INDEX_DISPLAY_ORDER = ("NIFTY 50", "BANK NIFTY", "SENSEX")
_option_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
_market_cache: dict[str, tuple[float, pd.DataFrame, float]] = {}
MARKET_CACHE_TTL = 5.0

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
    if cached and now-cached[0]<10: return cached[1]
    chain=fetch_bse_sensex_chain() if name=="SENSEX" else fetch_nse_chain(name); _option_cache[name]=(now,chain); return chain

def model_signal(x: pd.DataFrame, symbol: str, strategy: str, threshold: float):
    technical=score_signal(x.iloc[-1],strategy,reward_r=2.0)
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
        if ml.action=="BUY": technical.stop,technical.target=technical.entry-risk,technical.entry+2.0*risk
        else: technical.stop,technical.target=technical.entry+risk,technical.entry-2.0*risk
    else: technical.stop=technical.target=float("nan")
    technical.reasons.append(ml.reason)
    if ml.validation_accuracy is not None: technical.reasons.append(f"Training samples: {ml.samples:,} · validation accuracy: {ml.validation_accuracy*100:.1f}%")
    else: technical.reasons.append(f"Training samples: {ml.samples:,}")
    return technical,ml

def index_decision(name: str,ticker: str,threshold: float):
    x,spot=market_snapshot(ticker)
    if x is None or spot is None: return {"name":name,"symbol":ticker,"available":False,"reason":"Live index data unavailable"}
    sig,ml=model_signal(x,ticker,"Ensemble",threshold)
    option=build_option_recommendation(spot,sig.action,sig.target,option_chain_cached(INDEX_OPTION_SYMBOLS[name]))
    return {"name":name,"symbol":ticker,"available":True,"price":spot,"decision":sig.action,"confidence":sig.confidence,"entry":sig.entry,"stop":sig.stop,"target":sig.target,"risk_reward":sig.risk_reward,"reasons":sig.reasons,"model":asdict(ml) if ml else None,"option":option,"timestamp":time.time()}

def execute_paper(account: PaperAccount,x: pd.DataFrame,symbol: str,strategy: str,risk_pct: float,brokerage_pct: float,reward_r: float,threshold: float,strict: bool,auto: bool):
    price=float(x.iloc[-1]["Close"]); sig,ml=model_signal(x,symbol,strategy,threshold)
    qualified=sig.action!="HOLD" and sig.confidence>=threshold if strict else sig.action!="HOLD"; event=""
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
            side="LONG" if sig.action=="BUY" else "SHORT"; fee=abs(qty*price)*brokerage_pct/100; account.cash-=fee; account.position=PaperPosition(side,qty,price,sig.stop,sig.target,time.time(),sig.confidence); event=f"PAPER {sig.action} OPEN · Qty {qty:,} · Entry ₹{price:,.2f}"
    p=account.position; live_pnl=(price-p.entry)*p.qty if p and p.side=="LONG" else ((p.entry-price)*p.qty if p else 0.0)
    return {"price":price,"signal":asdict(sig),"model":asdict(ml) if ml else None,"qualified":qualified,"position":asdict(p) if p else None,"live_pnl":live_pnl,"cash":account.cash,"realized_pnl":account.realized,"event":event,"timestamp":time.time()}

@app.get("/")
def root(): return {"service":"Algo Trading Pro API","status":"ok","websocket":"/ws","version":"1.3.0"}
@app.get("/health")
def health(): return {"status":"ok"}
@app.get("/universe")
def universe(): return {"indices":INDEX_UNIVERSE,"stocks":POPULAR_STOCKS}

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
                # Never block the first dashboard tick on a multi-fold ML training job.
                if config["strategy"]=="Ensemble" and not model_is_ready(config["symbol"]):
                    asyncio.create_task(asyncio.to_thread(prewarm_ml_model,x,config["symbol"]))
                result=await asyncio.to_thread(execute_paper,account,x,config["symbol"],config["strategy"],float(config["risk_pct"]),float(config["brokerage_pct"]),float(config["reward_r"]),float(config["threshold"]),bool(config["strict"]),bool(config["auto"]))
                result["type"]="tick"; result["symbol"]=config["symbol"]
                # Send selected-symbol data immediately; index models/options are allowed to follow.
                result["indices"]={}; await ws.send_json(_json_safe(result))
                jobs=[]
                for name in INDEX_DISPLAY_ORDER: jobs.append(asyncio.to_thread(index_decision,name,INDEX_UNIVERSE[name],float(config["threshold"])))
                index_results=await asyncio.gather(*jobs)
                result["indices"]=dict(zip(INDEX_DISPLAY_ORDER,index_results)); result["timestamp"]=time.time()
                await ws.send_json(_json_safe(result))
                await asyncio.sleep(1.0); continue
            await ws.send_json(_json_safe(payload)); await asyncio.sleep(1.0)
    except (WebSocketDisconnect,RuntimeError): accounts.pop(client_id,None)
