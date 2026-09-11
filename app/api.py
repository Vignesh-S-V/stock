from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.live_feed import fetch_live_1m
from app.trading import INDEX_UNIVERSE, POPULAR_STOCKS, add_indicators, position_size, score_signal

app = FastAPI(title="Algo Trading Pro API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class PaperPosition:
    side: str
    qty: int
    entry: float
    stop: float
    target: float
    opened_at: float
    confidence: float

class PaperAccount:
    def __init__(self) -> None:
        self.cash = 100000.0
        self.position: PaperPosition | None = None
        self.realized = 0.0
        self.last_event = ""
        self.last_action_key = ""

    def reset(self, capital: float) -> None:
        self.cash = float(capital)
        self.position = None
        self.realized = 0.0
        self.last_event = ""
        self.last_action_key = ""

accounts: dict[str, PaperAccount] = {}


def account_for(client_id: str, capital: float = 100000.0) -> PaperAccount:
    if client_id not in accounts:
        accounts[client_id] = PaperAccount()
        accounts[client_id].reset(capital)
    return accounts[client_id]


def yahoo_last(symbol: str) -> float | None:
    d = fetch_live_1m(symbol, "1d")
    if d is None or d.empty:
        return None
    return float(d.Close.iloc[-1])


def market_snapshot(symbol: str) -> tuple[pd.DataFrame | None, float | None]:
    d = fetch_live_1m(symbol, "1d")
    if d is None or d.empty:
        return None, None
    return add_indicators(d), float(d.Close.iloc[-1])


def execute_paper(
    account: PaperAccount,
    x: pd.DataFrame,
    strategy: str,
    risk_pct: float,
    brokerage_pct: float,
    reward_r: float,
    threshold: float,
    strict: bool,
    auto: bool,
    force_action: str | None = None,
) -> dict[str, Any]:
    row = x.iloc[-1]
    price = float(row["Close"])
    sig = score_signal(row, strategy, reward_r=reward_r)
    qualified = sig.action != "HOLD" and sig.confidence >= threshold if strict else sig.action != "HOLD"
    event = ""

    if account.position:
        p = account.position
        pnl = (price - p.entry) * p.qty if p.side == "LONG" else (p.entry - price) * p.qty
        exit_reason = None
        if p.side == "LONG" and price <= p.stop:
            exit_reason = "STOP"
        elif p.side == "LONG" and price >= p.target:
            exit_reason = "TARGET"
        elif p.side == "SHORT" and price >= p.stop:
            exit_reason = "STOP"
        elif p.side == "SHORT" and price <= p.target:
            exit_reason = "TARGET"
        elif (p.side == "LONG" and sig.action == "SELL") or (p.side == "SHORT" and sig.action == "BUY"):
            exit_reason = "SIGNAL FLIP"
        if exit_reason:
            fee = abs(p.qty * price) * brokerage_pct / 100
            account.realized += pnl - fee
            account.cash += pnl - fee
            event = f"{p.side} CLOSED · {exit_reason} · P/L ₹{pnl - fee:,.2f}"
            account.position = None

    if account.position is None and (auto and qualified or force_action in {"BUY", "SELL"}):
        action = force_action or sig.action
        if action in {"BUY", "SELL"}:
            qty = position_size(account.cash, risk_pct, price, sig.stop)
            if qty > 0:
                side = "LONG" if action == "BUY" else "SHORT"
                fee = abs(qty * price) * brokerage_pct / 100
                account.cash -= fee
                account.position = PaperPosition(side, qty, price, sig.stop, sig.target, time.time(), sig.confidence)
                event = f"PAPER {action} OPEN · Qty {qty:,} · Entry ₹{price:,.2f}"

    p = account.position
    live_pnl = 0.0
    if p:
        live_pnl = (price - p.entry) * p.qty if p.side == "LONG" else (p.entry - price) * p.qty

    return {
        "price": price,
        "signal": asdict(sig),
        "qualified": qualified,
        "position": asdict(p) if p else None,
        "live_pnl": live_pnl,
        "cash": account.cash,
        "realized_pnl": account.realized,
        "event": event,
        "timestamp": time.time(),
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Algo Trading Pro API", "status": "ok", "websocket": "/ws"}

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/universe")
def universe() -> dict[str, dict[str, str]]:
    return {"indices": INDEX_UNIVERSE, "stocks": POPULAR_STOCKS}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    client_id = f"{id(ws)}"
    account = account_for(client_id)
    config = {
        "symbol": "RELIANCE.NS",
        "strategy": "Ensemble",
        "risk_pct": 1.0,
        "brokerage_pct": 0.03,
        "reward_r": 2.0,
        "threshold": 90.0,
        "strict": True,
        "auto": True,
        "capital": 100000.0,
    }
    try:
        while True:
            try:
                while True:
                    message = await asyncio.wait_for(ws.receive_json(), timeout=0.01)
                    if isinstance(message, dict):
                        config.update({k: message[k] for k in config if k in message})
                        if "capital" in message and account.position is None:
                            account.reset(float(message["capital"]))
            except asyncio.TimeoutError:
                pass

            x, price = await asyncio.to_thread(market_snapshot, config["symbol"])
            if x is None or price is None:
                await ws.send_json({"type": "error", "message": "Live market data unavailable", "timestamp": time.time()})
            else:
                result = await asyncio.to_thread(
                    execute_paper,
                    account,
                    x,
                    config["strategy"],
                    float(config["risk_pct"]),
                    float(config["brokerage_pct"]),
                    float(config["reward_r"]),
                    float(config["threshold"]),
                    bool(config["strict"]),
                    bool(config["auto"]),
                )
                indices = {}
                for name, ticker in INDEX_UNIVERSE.items():
                    value = await asyncio.to_thread(yahoo_last, ticker)
                    if value is not None:
                        indices[name] = {"symbol": ticker, "price": value}
                result["type"] = "tick"
                result["symbol"] = config["symbol"]
                result["indices"] = indices
                await ws.send_json(result)
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, RuntimeError):
        accounts.pop(client_id, None)
