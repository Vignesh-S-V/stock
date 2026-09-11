from app.api import app, health, root


def test_api_health():
    assert health() == {"status": "ok"}


def test_api_root_advertises_websocket():
    payload = root()
    assert payload["status"] == "ok"
    assert payload["websocket"] == "/ws"
    assert any(getattr(route, "path", None) == "/ws" for route in app.routes)
