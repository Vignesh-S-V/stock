from __future__ import annotations

import json
from urllib.parse import quote_plus

import streamlit.components.v1 as components


def _quote(symbol: str) -> str:
    return quote_plus(symbol)


def _format_js(value: str) -> str:
    return json.dumps(value)


def render_live_price(symbol: str, initial_price: float, height: int = 72) -> None:
    """Single browser-side ticker. Streamlit does not rerun."""
    ticker = _quote(symbol)
    initial = json.dumps(float(initial_price))
    html = f'''<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:transparent;font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;overflow:hidden}}
.card{{height:{height-4}px;box-sizing:border-box;background:#0e151d;border:1px solid #222e3a;border-radius:12px;padding:8px 12px;display:flex;align-items:center;justify-content:space-between;gap:12px}}
.label{{font-size:10px;font-weight:800;letter-spacing:.5px;color:#81909f}}
.price{{font-size:22px;font-weight:900;color:#f4f7fa;line-height:1.1;transition:transform .12s ease}}
.meta{{font-size:10px;color:#6ee39a;text-align:right;white-space:nowrap}}
.flash{{transform:scale(1.025)}}
</style></head><body>
<div class="card"><div><div class="label">LIVE PRICE</div><div id="price" class="price">₹{float(initial_price):,.2f}</div></div><div id="meta" class="meta">LIVE · 1s</div></div>
<script>
const symbol={json.dumps(ticker)};
let last={initial};
const p=document.getElementById('price');
const m=document.getElementById('meta');
function setPrice(v){{
  if(!Number.isFinite(v)||v<=0)return;
  const changed=Math.abs(v-last)>1e-7;
  last=v;
  p.textContent='₹'+v.toLocaleString('en-IN',{{minimumFractionDigits:2,maximumFractionDigits:2}});
  if(changed){{p.classList.remove('flash');void p.offsetWidth;p.classList.add('flash');setTimeout(()=>p.classList.remove('flash'),180);}}
  m.textContent='LIVE · '+new Date().toLocaleTimeString('en-IN',{{hour12:false}});
}}
async function poll(){{
  try{{
    const r=await fetch('https://query1.finance.yahoo.com/v8/finance/chart/'+symbol+'?range=1d&interval=1m&includePrePost=true&_='+Date.now(),{{cache:'no-store'}});
    if(!r.ok)throw 0;
    const d=await r.json();
    const q=d.chart?.result?.[0]?.indicators?.quote?.[0]?.close||[];
    if(q.length)setPrice(Number(q[q.length-1]));
  }}catch(e){{m.textContent='WAITING FOR FEED · 1s';}}
}}
poll();
setInterval(poll,1000);
</script></body></html>'''
    components.html(html, height=height, scrolling=False)


def render_market_overview(items: list[tuple[str, str, float]], height: int = 112) -> None:
    """Render NIFTY, SENSEX and BANK NIFTY with browser-only live price updates."""
    payload = json.dumps([{"name": n, "symbol": s, "price": float(p)} for n, s, p in items])
    cards = ''.join(
        f'<div class="market"><div class="top"><span class="name">{n}</span><span class="dot"></span></div>'
        f'<div id="p-{i}" class="big">₹{float(p):,.2f}</div><div id="c-{i}" class="meta">LIVE · starting</div></div>'
        for i, (n, s, p) in enumerate(items)
    )
    html = f'''<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:transparent;font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;overflow:hidden}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.market{{background:linear-gradient(145deg,#101923,#0c131b);border:1px solid #263544;border-radius:15px;padding:13px 15px;height:{height-8}px;box-sizing:border-box}}
.top{{display:flex;justify-content:space-between;align-items:center}}
.name{{font-size:13px;font-weight:900;color:#dfe7ef}}
.dot{{width:7px;height:7px;border-radius:50%;background:#48e58a;box-shadow:0 0 8px #48e58a}}
.big{{font-size:27px;font-weight:950;color:#f5f7fa;margin-top:8px;transition:transform .12s ease}}
.meta{{font-size:10px;color:#6ee39a;margin-top:3px}}
.flash{{transform:scale(1.025)}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><div class="grid">{cards}</div>
<script>
const items={payload};
function put(i,v){{
  if(!Number.isFinite(v)||v<=0)return;
  const p=document.getElementById('p-'+i),m=document.getElementById('c-'+i);
  const old=Number(p.dataset.value||items[i].price);
  const changed=Math.abs(v-old)>1e-7;
  p.dataset.value=v;
  p.textContent='₹'+v.toLocaleString('en-IN',{{minimumFractionDigits:2,maximumFractionDigits:2}});
  if(changed){{p.classList.remove('flash');void p.offsetWidth;p.classList.add('flash');setTimeout(()=>p.classList.remove('flash'),180);}}
  m.textContent='LIVE · '+new Date().toLocaleTimeString('en-IN',{{hour12:false}});
}}
async function poll(i){{
  try{{
    const s=encodeURIComponent(items[i].symbol);
    const r=await fetch('https://query1.finance.yahoo.com/v8/finance/chart/'+s+'?range=1d&interval=1m&includePrePost=true&_='+Date.now(),{{cache:'no-store'}});
    if(!r.ok)throw 0;
    const d=await r.json();
    const q=d.chart?.result?.[0]?.indicators?.quote?.[0]?.close||[];
    if(q.length)put(i,Number(q[q.length-1]));
  }}catch(e){{document.getElementById('c-'+i).textContent='WAITING FOR FEED · 1s';}}
}}
items.forEach((item,i)=>{{put(i,item.price);poll(i);setInterval(()=>poll(i),1000);}});
</script></body></html>'''
    components.html(html, height=height, scrolling=False)


def render_paper_live(symbol: str, initial_price: float, side: str, entry: float, qty: int, height: int = 112) -> None:
    """Live paper position card: live price and unrealised P/L update in browser only."""
    payload = json.dumps({"symbol": symbol, "price": float(initial_price), "side": side, "entry": float(entry), "qty": int(qty)})
    html = f'''<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:transparent;font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;overflow:hidden}}
.box{{height:{height-4}px;box-sizing:border-box;background:#0c151e;border:1px solid #2c4657;border-radius:15px;padding:12px 15px}}
.row{{display:flex;justify-content:space-between;align-items:center}}
.tag{{font-size:10px;font-weight:900;letter-spacing:.6px;color:#79e6a4}}
.side{{font-size:12px;font-weight:950;padding:4px 8px;border-radius:999px;background:#10291d;color:#69e69a}}
.values{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:9px}}
.label{{font-size:9px;color:#7f909f;text-transform:uppercase}}
.value{{font-size:17px;font-weight:900;color:#edf3f8;margin-top:2px}}
.pnl{{font-size:20px;font-weight:950;color:#6ee39a}}
.loss{{color:#ff7076!important}}
.flash{{transform:scale(1.025)}}
</style></head><body>
<div class="box"><div class="row"><span class="tag">⚡ AUTO PAPER TRADE · LIVE P/L</span><span class="side">{side}</span></div>
<div class="values"><div><div class="label">Entry</div><div class="value">₹{float(entry):,.2f}</div></div>
<div><div class="label">Live Price</div><div id="live" class="value">₹{float(initial_price):,.2f}</div></div>
<div><div class="label">Unrealised P/L</div><div id="pnl" class="pnl">₹0.00</div></div></div></div>
<script>
const x={payload};
const live=document.getElementById('live'),pnl=document.getElementById('pnl');
let old=x.price;
function update(v){{
  if(!Number.isFinite(v)||v<=0)return;
  const pl=x.side==='LONG'?(v-x.entry)*x.qty:(x.entry-v)*x.qty;
  const changed=Math.abs(v-old)>1e-7;
  old=v;
  live.textContent='₹'+v.toLocaleString('en-IN',{{minimumFractionDigits:2,maximumFractionDigits:2}});
  pnl.textContent=(pl>=0?'+₹':'-₹')+Math.abs(pl).toLocaleString('en-IN',{{minimumFractionDigits:2,maximumFractionDigits:2}});
  pnl.className='pnl '+(pl<0?'loss':'');
  if(changed){{live.classList.remove('flash');void live.offsetWidth;live.classList.add('flash');setTimeout(()=>live.classList.remove('flash'),180);}}
}}
async function poll(){{
  try{{
    const r=await fetch('https://query1.finance.yahoo.com/v8/finance/chart/'+encodeURIComponent(x.symbol)+'?range=1d&interval=1m&includePrePost=true&_='+Date.now(),{{cache:'no-store'}});
    if(!r.ok)throw 0;
    const d=await r.json();
    const q=d.chart?.result?.[0]?.indicators?.quote?.[0]?.close||[];
    if(q.length)update(Number(q[q.length-1]));
  }}catch(e){{}}
}}
update(x.price);poll();setInterval(poll,1000);
</script></body></html>'''
    components.html(html, height=height, scrolling=False)
