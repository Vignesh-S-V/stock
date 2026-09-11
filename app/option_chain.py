from __future__ import annotations

import math
import time
from typing import Any

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/153 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.nseindia.com/option-chain",
    "Cache-Control": "no-cache",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=5)
    except requests.RequestException:
        pass
    return s


def _num(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def fetch_nse_chain(symbol: str) -> dict[str, Any] | None:
    """Fetch live NSE option-chain data for NIFTY/BANKNIFTY.

    Exchange data is used for premium/bid/ask/IV/OI. If the public endpoint
    cannot be reached, return None rather than inventing option prices.
    """
    try:
        s = _session()
        r = s.get(
            "https://www.nseindia.com/api/option-chain-indices",
            params={"symbol": symbol},
            timeout=8,
        )
        r.raise_for_status()
        payload = r.json()
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

    # The model's underlying target is converted into a conservative option
    # premium planning range. Live LTP is always the actual reference entry.
    underlying_move = abs(target - spot)
    estimated_premium_move = underlying_move * 0.50
    target_price = premium + estimated_premium_move
    stop_price = max(0.05, premium - estimated_premium_move * 0.60)
    return {
        "available": True,
        "source": chain["source"],
        "expiry": chain["expiry"],
        "strike": row["strike"],
        "type": option_type,
        "contract": f"{int(row['strike'])} {option_type}",
        "premium": premium,
        "buy_price": premium,
        "target_price": round(target_price, 2),
        "stop_price": round(stop_price, 2),
        "bid": leg.get("bid"),
        "ask": leg.get("ask"),
        "iv": leg.get("iv"),
        "oi": leg.get("oi"),
        "volume": leg.get("volume"),
        "underlying_target": target,
        "updated_at": time.time(),
    }
