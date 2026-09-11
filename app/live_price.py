from __future__ import annotations

import json
from urllib.parse import quote_plus

import streamlit.components.v1 as components


def render_live_price(symbol: str, initial_price: float, height: int = 72) -> None:
    """Render a browser-side price ticker that updates without rerunning Streamlit."""
    ticker = quote_plus(symbol)
    initial = json.dumps(float(initial_price))
    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html,body{{margin:0;background:transparent;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow:hidden}}
.card{{height:{height-4}px;box-sizing:border-box;background:#0e151d;border:1px solid #222e3a;border-radius:12px;padding:8px 12px;display:flex;align-items:center;justify-content:space-between;gap:12px}}
.label{{font-size:10px;font-weight:800;letter-spacing:.5px;color:#81909f}}
.price{{font-size:22px;font-weight:900;color:#f4f7fa;line-height:1.1;transition:transform .12s ease}}
.meta{{font-size:10px;color:#6ee39a;text-align:right;white-space:nowrap}}
.flash{{transform:scale(1.025)}}
</style>
</head>
<body>
<div class="card">
  <div><div class="label">LIVE PRICE</div><div id="price" class="price">₹{float(initial_price):,.2f}</div></div>
  <div id="meta" class="meta">LIVE · 1s</div>
</div>
<script>
const symbol = {json.dumps(ticker)};
let last = {initial};
const priceEl = document.getElementById('price');
const metaEl = document.getElementById('meta');

function setPrice(value, source='LIVE') {{
  if (!Number.isFinite(value) || value <= 0) return;
  const changed = Math.abs(value - last) > 0.0000001;
  last = value;
  priceEl.textContent = '₹' + value.toLocaleString('en-IN', {{minimumFractionDigits:2, maximumFractionDigits:2}});
  if (changed) {{
    priceEl.classList.remove('flash');
    void priceEl.offsetWidth;
    priceEl.classList.add('flash');
    setTimeout(() => priceEl.classList.remove('flash'), 180);
  }}
  metaEl.textContent = source + ' · ' + new Date().toLocaleTimeString('en-IN', {{hour12:false}});
}}

async function poll() {{
  try {{
    const url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + symbol + '?range=1d&interval=1m&includePrePost=true&_=' + Date.now();
    const response = await fetch(url, {{cache:'no-store'}});
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const data = await response.json();
    const result = data.chart && data.chart.result && data.chart.result[0];
    const closes = result && result.indicators && result.indicators.quote && result.indicators.quote[0] && result.indicators.quote[0].close;
    if (closes && closes.length) {{
      const value = Number(closes[closes.length - 1]);
      setPrice(value, 'LIVE');
    }}
  }} catch (err) {{
    metaEl.textContent = 'WAITING FOR FEED · 1s';
  }}
}}

poll();
setInterval(poll, 1000);
</script>
</body>
</html>
"""
    components.html(html, height=height, scrolling=False)
