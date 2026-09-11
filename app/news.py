from __future__ import annotations

import html
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

NEWS_TTL_SECONDS = 180.0
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}

POSITIVE = {
    "beat", "beats", "growth", "surge", "surges", "gain", "gains", "rally", "rallies",
    "bullish", "upgrade", "upgraded", "profit", "profits", "strong", "record", "outperform",
    "buyback", "approval", "approved", "deal", "wins", "win", "expands", "expansion", "recovery",
}
NEGATIVE = {
    "fall", "falls", "drop", "drops", "selloff", "sell-off", "bearish", "downgrade", "downgraded",
    "loss", "losses", "weak", "miss", "misses", "warning", "probe", "investigation", "fraud",
    "default", "debt", "resigns", "resignation", "cuts", "cut", "slump", "crash", "risk", "risks",
}


def _query_for(symbol: str) -> str:
    s = symbol.upper().strip()
    names = {
        "RELIANCE.NS": "Reliance Industries India stock",
        "TCS.NS": "TCS India stock",
        "INFY.NS": "Infosys India stock",
        "HDFCBANK.NS": "HDFC Bank India stock",
        "ICICIBANK.NS": "ICICI Bank India stock",
        "SBIN.NS": "SBI India stock",
        "ITC.NS": "ITC India stock",
        "LT.NS": "Larsen Toubro India stock",
        "BHARTIARTL.NS": "Bharti Airtel India stock",
        "AXISBANK.NS": "Axis Bank India stock",
        "^NSEI": "Nifty 50 India stock market",
        "^NSEBANK": "Nifty Bank India stock market",
        "^BSESN": "Sensex India stock market",
    }
    return names.get(s, f"{s.replace('.NS', '')} India stock")


def _sentiment(title: str) -> float:
    words = re.findall(r"[a-z][a-z-]+", title.lower())
    if not words:
        return 0.0
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    return max(-1.0, min(1.0, (pos - neg) / max(3.0, len(words) ** 0.5)))


def _parse_rss(raw: bytes) -> list[dict[str, Any]]:
    root = ET.fromstring(raw)
    items: list[dict[str, Any]] = []
    for item in root.findall(".//item")[:10]:
        title = html.unescape((item.findtext("title") or "").strip())
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        source = (item.findtext("source") or "Google News").strip()
        if title:
            items.append({"title": title, "link": link, "published": pub, "source": source, "sentiment": round(_sentiment(title), 2)})
    return items


def get_news(symbol: str) -> dict[str, Any]:
    now = time.time()
    key = symbol.upper().strip()
    cached = _CACHE.get(key)
    if cached and now - cached[0] < NEWS_TTL_SECONDS:
        return cached[1]
    query = _query_for(key)
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AlgoTradingPro/1.0"})
        with urllib.request.urlopen(req, timeout=8) as response:
            items = _parse_rss(response.read())
        scores = [float(i["sentiment"]) for i in items]
        score = round(sum(scores) / len(scores), 2) if scores else 0.0
        positive = sum(s > 0.15 for s in scores)
        negative = sum(s < -0.15 for s in scores)
        if score > 0.18:
            bias = "BULLISH"
        elif score < -0.18:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"
        result = {"available": True, "query": query, "score": score, "bias": bias, "positive": positive, "negative": negative, "items": items, "updated_at": now}
    except Exception as exc:
        result = {"available": False, "query": query, "score": 0.0, "bias": "NEUTRAL", "positive": 0, "negative": 0, "items": [], "reason": f"News feed unavailable: {type(exc).__name__}."}
    _CACHE[key] = (now, result)
    return result


def news_confirmation(action: str, news: dict[str, Any]) -> tuple[bool, str]:
    if action not in {"BUY", "SELL"} or not news.get("available"):
        return True, "News confirmation unavailable; technical/ML signal retained."
    score = float(news.get("score", 0.0))
    if action == "BUY" and score <= -0.45:
        return False, f"News conflict: headline sentiment is {score:+.2f} ({news.get('bias')}). Trade blocked."
    if action == "SELL" and score >= 0.45:
        return False, f"News conflict: headline sentiment is {score:+.2f} ({news.get('bias')}). Trade blocked."
    return True, f"News confirmation: {news.get('bias', 'NEUTRAL')} ({score:+.2f})."
