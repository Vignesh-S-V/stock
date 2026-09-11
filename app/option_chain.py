from __future__ import annotations

import io
import math
import os
import re
import time
from typing import Any

import pandas as pd
import requests

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}
BSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://m.bseindia.com/derivatives.aspx",
}
DHAN_INDEX_IDS = {"NIFTY 50": 13, "BANK NIFTY": 25, "SENSEX": 51}


def _num(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _session() -> requests.Session:
    s = requests.Session(); s.headers.update(NSE_HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=6)
    except requests.RequestException:
        pass
    return s


def fetch_dhan_chain(name: str) -> dict[str, Any] | None:
    """Use Dhan's authenticated option-chain API when configured."""
    token, client_id = os.getenv("DHAN_ACCESS_TOKEN"), os.getenv("DHAN_CLIENT_ID")
    security_id = DHAN_INDEX_IDS.get(name)
    if not token or not client_id or security_id is None:
        return None
    headers = {"Content-Type": "application/json", "access-token": token, "client-id": client_id}
    try:
        expiry_resp = requests.post(
            "https://api.dhan.co/v2/optionchain/expirylist",
            headers=headers,
            json={"UnderlyingScrip": security_id, "UnderlyingSeg": "IDX_I"},
            timeout=8,
        )
        expiry_resp.raise_for_status()
        expiry_data = expiry_resp.json().get("data") or []
        if not expiry_data:
            return None
        expiry = expiry_data[0]
        chain_resp = requests.post(
            "https://api.dhan.co/v2/optionchain",
            headers=headers,
            json={"UnderlyingScrip": security_id, "UnderlyingSeg": "IDX_I", "Expiry": expiry},
            timeout=10,
        )
        chain_resp.raise_for_status()
        data = chain_resp.json().get("data") or {}
        oc = data.get("oc") or {}
        rows: list[dict[str, Any]] = []
        for strike_text, legs in oc.items():
            strike = _num(strike_text)
            if strike is None:
                continue
            item: dict[str, Any] = {"strike": strike, "expiry": expiry}
            for side in ("ce", "pe"):
                leg = legs.get(side) or {}
                greeks = leg.get("greeks") or {}
                item[side.upper()] = {
                    "ltp": _num(leg.get("last_price")),
                    "bid": _num(leg.get("top_bid_price")),
                    "ask": _num(leg.get("top_ask_price")),
                    "iv": _num(leg.get("implied_volatility")),
                    "delta": _num(greeks.get("delta")),
                    "theta": _num(greeks.get("theta")),
                    "gamma": _num(greeks.get("gamma")),
                    "vega": _num(greeks.get("vega")),
                    "oi": _num(leg.get("oi")),
                    "volume": _num(leg.get("volume")),
                    "security_id": leg.get("security_id"),
                }
            rows.append(item)
        return {"source": "Dhan", "symbol": name, "expiry": expiry, "rows": rows} if rows else None
    except (requests.RequestException, ValueError, TypeError):
        return None


def fetch_nse_chain(symbol: str) -> dict[str, Any] | None:
    """Prefer authorized Dhan data; public NSE is only a best-effort fallback."""
    dhan_name = {"NIFTY": "NIFTY 50", "BANKNIFTY": "BANK NIFTY"}.get(symbol)
    if dhan_name:
        dhan = fetch_dhan_chain(dhan_name)
        if dhan:
            return dhan
    try:
        s = _session(); url = "https://www.nseindia.com/api/option-chain-indices"; payload = None
        for _ in range(2):
            r = s.get(url, params={"symbol": symbol}, timeout=8); r.raise_for_status()
            if "json" not in r.headers.get("content-type", "").lower():
                time.sleep(0.4); continue
            payload = r.json(); break
        if not isinstance(payload, dict): return None
        records = payload.get("records", {}); rows = records.get("data", []); expiries = records.get("expiryDates", [])
        if not rows or not expiries: return None
        expiry = expiries[0]; parsed: list[dict[str, Any]] = []
        for row in rows:
            if row.get("expiryDate") != expiry: continue
            strike = _num(row.get("strikePrice"))
            if strike is None: continue
            item: dict[str, Any] = {"strike": strike, "expiry": expiry}
            for side in ("CE", "PE"):
                leg = row.get(side) or {}
                item[side] = {"ltp": _num(leg.get("lastPrice")), "bid": _num(leg.get("bidprice")), "ask": _num(leg.get("askPrice")), "iv": _num(leg.get("impliedVolatility")), "oi": _num(leg.get("openInterest")), "volume": _num(leg.get("totalTradedVolume"))}
            parsed.append(item)
        return {"source": "NSE", "symbol": symbol, "expiry": expiry, "rows": parsed} if parsed else None
    except (requests.RequestException, ValueError, TypeError): return None


def fetch_bse_sensex_chain() -> dict[str, Any] | None:
    """Prefer authorized Dhan SENSEX data; BSE page is a fallback."""
    dhan = fetch_dhan_chain("SENSEX")
    if dhan: return dhan
    try:
        r = requests.get("https://m.bseindia.com/derivatives.aspx", headers=BSE_HEADERS, timeout=8); r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text)); rows: list[dict[str, Any]] = []
        pattern = re.compile(r"^SENSEX(?P<expiry>\d{2}(?:[A-Z]\d{2}|[A-Z]{3}))(?P<strike>\d+)(?P<type>CE|PE)$")
        for table in tables:
            if table.shape[1] < 2: continue
            headers = " ".join(str(c) for c in table.columns)
            if "Series Code" not in headers or "LTP" not in headers: continue
            code_col = next((c for c in table.columns if "Series Code" in str(c)), table.columns[0]); ltp_col = next((c for c in table.columns if str(c).strip() == "LTP"), table.columns[1])
            for _, row in table.iterrows():
                code = str(row.get(code_col, "")).strip(); match = pattern.match(code)
                if not match: continue
                ltp = _num(row.get(ltp_col))
                if ltp is None or ltp <= 0: continue
                option_type = match.group("type")
                rows.append({"strike": float(match.group("strike")), "expiry": match.group("expiry"), "contract_code": code, "CE": {"ltp": ltp} if option_type == "CE" else {}, "PE": {"ltp": ltp} if option_type == "PE" else {}})
        if not rows: return None
        return {"source": "BSE", "symbol": "SENSEX", "expiry": rows[0]["expiry"], "rows": rows}
    except (requests.RequestException, ValueError, ImportError, TypeError): return None


def _best_strike(rows: list[dict[str, Any]], spot: float, side: str, target: float) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        leg = row.get(side) or {}; ltp = leg.get("ltp")
        if ltp is None or ltp <= 0: continue
        strike = row["strike"]; liquidity = math.log1p((leg.get("volume") or 0) + (leg.get("oi") or 0))
        score = abs(strike - target) + abs(strike - spot) * 0.15 - liquidity * 5.0
        candidates.append((score, row))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def _atm_snapshot(rows: list[dict[str, Any]], spot: float) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        strike = _num(row.get("strike"))
        if strike is None: continue
        ce, pe = row.get("CE") or {}, row.get("PE") or {}
        if (ce.get("ltp") or 0) <= 0 and (pe.get("ltp") or 0) <= 0: continue
        distance = abs(strike - spot)
        liquidity = math.log1p((ce.get("volume") or 0) + (pe.get("volume") or 0) + (ce.get("oi") or 0) + (pe.get("oi") or 0))
        candidates.append((distance - liquidity * 0.1, row))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def build_option_recommendation(spot: float, action: str, target: float, chain: dict[str, Any] | None) -> dict[str, Any]:
    if chain is None:
        if action in {"BUY", "SELL"}:
            return {"available": False, "reason": "Live option feed not connected. Add an authorized Dhan Data API token."}
        return {"available": False, "reason": "Live option-chain feed not connected."}
    if action not in {"BUY", "SELL"}:
        row = _atm_snapshot(chain["rows"], spot)
        if row is None: return {"available": False, "reason": "No live ATM option contracts available."}
        ce, pe = row.get("CE") or {}, row.get("PE") or {}
        return {"available": True, "mode": "snapshot", "source": chain["source"], "expiry": chain["expiry"], "strike": row["strike"], "contract": f"ATM {int(row['strike'])}", "ce_premium": ce.get("ltp"), "pe_premium": pe.get("ltp"), "ce_iv": ce.get("iv"), "pe_iv": pe.get("iv"), "ce_delta": ce.get("delta"), "pe_delta": pe.get("delta"), "ce_oi": ce.get("oi"), "pe_oi": pe.get("oi"), "ce_volume": ce.get("volume"), "pe_volume": pe.get("volume"), "updated_at": time.time()}

    option_type = "CE" if action == "BUY" else "PE"; row = _best_strike(chain["rows"], spot, option_type, target)
    if row is None: return {"available": False, "reason": "No liquid option contract found in the live chain"}
    leg = row[option_type]; premium = leg.get("ltp")
    if premium is None or premium <= 0: return {"available": False, "reason": "Live option premium unavailable for the selected contract"}

    underlying_move = abs(target - spot)
    delta = abs(_num(leg.get("delta")) or 0.50)
    estimated_premium_move = max(0.05, underlying_move * delta)
    return {"available": True, "mode": "trade", "source": chain["source"], "expiry": chain["expiry"], "strike": row["strike"], "type": option_type, "contract": row.get("contract_code") or f"{int(row['strike'])} {option_type}", "premium": premium, "buy_price": premium, "target_price": round(premium + estimated_premium_move, 2), "stop_price": round(max(0.05, premium - estimated_premium_move * 0.60), 2), "bid": leg.get("bid"), "ask": leg.get("ask"), "iv": leg.get("iv"), "delta": leg.get("delta"), "oi": leg.get("oi"), "volume": leg.get("volume"), "security_id": leg.get("security_id"), "underlying_target": target, "updated_at": time.time()}
