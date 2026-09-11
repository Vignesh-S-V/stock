from app.api import app, health, root
from app.option_chain import build_option_recommendation


def test_api_health():
    assert health() == {"status": "ok"}


def test_api_root_advertises_websocket():
    payload = root()
    assert payload["status"] == "ok"
    assert payload["websocket"] == "/ws"
    assert any(getattr(route, "path", None) == "/ws" for route in app.routes)


def test_option_recommendation_uses_live_premium_as_entry():
    chain = {
        "source": "NSE",
        "expiry": "10-Sep-2026",
        "rows": [
            {"strike": 23400, "CE": {"ltp": 110, "bid": 109.9, "ask": 110.1, "iv": 12, "oi": 100000, "volume": 50000}, "PE": {}},
        ],
    }
    result = build_option_recommendation(23320, "BUY", 23400, chain)
    assert result["available"] is True
    assert result["contract"] == "23400 CE"
    assert result["buy_price"] == 110
    assert result["target_price"] > 110
    assert result["stop_price"] < 110
