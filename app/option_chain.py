from __future__ import annotations

import io
import math
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


def _num(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=6)
    except requests.RequestException:
        pass
    return s


def fetch_nse_chain(symbol: str) -> dict[str, Any] | None:
    """Fetch the first active NSE expiry without inventing option prices."""
    try:
        s = _session()
        url = "https://www.nseindia.com/api/option-chain-indices"
        payload = None
        for _ in range(2):
            r = s.get(url, params={"symbol": symbol}, timeout=8)
            r.raise_for_status()
            if "json" not in r.headers.get("content-type", "").lower():
                time.sleep(0.4)
                continue
            payload = r.json()
            break
        if not isinstance(payload, dict):
            return None
        records = payload.get("records", {})
        rows = records.get("data", [])
        expiries = records.get("expiryDates", [])
        if not rows or not expiries:
            return None
        expiry = expiries[0]
        parsed: list[dict[str, Any]] = []
        for row in rows:
            if row.get("expiryDate") != expiry:
                continue
            strike = _num(row.get("strikePrice"))
            if strike is None:
                continue
            item: dict[str, Any] = {"strike": strike, "expiry": expiry}
            for side in ("CE", "PE"):
                leg = row.get(side) or {}
                item[side] = {
                    "ltp": _num(leg.get("lastPrice")),
                    "bid": _num(leg.get("bidprice")),
                    "ask": _num(leg.get("askPrice")),
                    "iv": _num(leg.get("impliedVolatility")),
                    "oi": _num(leg.get("openInterest")),
                    "volume": _num(leg.get("totalTradedVolume")),
                }
            parsed.append(item)
        return {"source": "NSE", "symbol": symbol, "expiry": expiry, "rows": parsed} if parsed else None
    except (requests.RequestException, ValueError, TypeError):
        return None


def fetch_bse_sensex_chain() -> dict[str, Any] | None:
    """Best-effort BSE SENSEX derivatives market-watch feed."""
    try:
        r = requests.get("https://m.bseindia.com/derivatives.aspx", headers=BSE_HEADERS, timeout=8)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        rows: list[dict[str, Any]] = []
        # BSE currently exposes codes such as SENSEX25N0684000CE and
        # SENSEX25SEP81800PE. The expiry portion is variable-length.
        pattern = re.compile(r"^SENSEX(?P<expiry>\d{2}(?:[A-Z]\d{2}|[A-Z]{3}))(?P<strike>\d+)(?P<type>CE|PE)$")
        for table in tables:
            text = table.to_string(index=False)
            if "Series Code" not in text or "LTP" not in text:
                continue
            for _, row in table.iterrows():
                code = str(row.iloc[0]).strip()
                match = pattern.match(code)
                if not match:
                    continue
                ltp = _num(row.iloc[1])
                if ltp is None or ltp <= 0:
                    continue
                option_type = match.group("type")
                rows.append({
                    "strike": float(match.group("strike")),
                    "expiry": match.group("expiry"),
                    "contract_code": code,
                    "CE": {"ltp": ltp} if option_type == "CE" else {},
                    "PE": {"ltp": ltp} if option_type == "PE" else {},
                })
        if not rows:
            return None
        return {"source": "BSE", "symbol": "SENSEX", "expiry": rows[0]["expiry"], "rows": rows}
    except (requests.RequestException, ValueError, ImportError, TypeError):
        return None


def _best_strike(rows: list[dict[str, Any]], spot: float, side: str, target: float) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        leg = row.get(side) or {}
        ltp = leg.get("ltp")
        if ltp is None or ltp <= 0:
            continue
        strike = row["strike"]
        liquidity = math.log1p((leg.get("volume") or 0) + (leg.get("oi") or 0))
        score = abs(strike - target) + abs(strike - spot) * 0.15 - liquidity * 5.0
        candidates.append((score, row))
    return min(candidates, key=lambda x: x[0])[1] if candidates else None


def build_option_recommendation(spot: float, action: str, target: float, chain: dict[str, Any] | None) -> dict[str, Any]:
    if action not in {"BUY", "SELL"} or chain is None:
        return {"available": False, "reason": "Live option-chain data unavailable"}
    option_type = "CE" if action == "BUY" else "PE"
    row = _best_strike(chain["rows"], spot, option_type, target)
    if row is None:
        return {"available": False, "reason": "No liquid option contract found"}
    leg = row[option_type]
    premium = leg.get("ltp")
    if premium is None or premium <= 0:
        return {"available": False, "reason": "Option premium unavailable"}
    underlying_move = abs(target - spot)
    estimated_premium_move = underlying_move * 0.50
    return {
        "available": True,
        "source": chain["source"],
        "expiry": chain["expiry"],
        "strike": row["strike"],
        "type": option_type,
        "contract": row.get("contract_code") or f"{int(row['strike'])} {option_type}",
        "premium": premium,
        "buy_price": premium,
        "target_price": round(premium + estimated_premium_move, 2),
        "stop_price": round(max(0.05, premium - estimated_premium_move * 0.60), 2),
        "bid": leg.get("bid"),
        "ask": leg.get("ask"),
        "iv": leg.get("iv"),
        "oi": leg.get("oi"),
        "volume": leg.get("volume"),
        "underlying_target": target,
        "updated_at": time.time(),
    }
