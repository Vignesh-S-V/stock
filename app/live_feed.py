from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests


def fetch_live_1m(symbol: str, period: str = "1d") -> pd.DataFrame:
    """Fetch a fresh Yahoo chart snapshot without Streamlit/yfinance caching.

    Yahoo's 1-minute endpoint only supports a short recent window. We use up to
    8 days so the live model gets materially more observations for calibration.
    This is still not a true exchange tick/WebSocket feed.
    """
    now = int(time.time())
    ranges = {"1d": 86400, "5d": 5 * 86400, "8d": 8 * 86400, "1mo": 31 * 86400}
    seconds = ranges.get(period, 86400)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": now - seconds,
        "period2": now + 5,
        "interval": "1m",
        "includePrePost": "true",
        "events": "div,splits",
        "_": str(now),
    }
    try:
        r = requests.get(
            url,
            params=params,
            headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        r.raise_for_status()
        payload = r.json().get("chart", {}).get("result", [])
        if not payload:
            return pd.DataFrame()
        result = payload[0]
        timestamps = result.get("timestamp", [])
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        if not timestamps:
            return pd.DataFrame()
        d = pd.DataFrame(
            {
                "Open": quote.get("open", []),
                "High": quote.get("high", []),
                "Low": quote.get("low", []),
                "Close": quote.get("close", []),
                "Volume": quote.get("volume", []),
            },
            index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
        )
        return d.dropna(subset=["Open", "High", "Low", "Close"])
    except Exception:
        return pd.DataFrame()


def live_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%d %b %Y, %H:%M:%S")
