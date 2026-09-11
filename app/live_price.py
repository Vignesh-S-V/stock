from __future__ import annotations

import json
from urllib.parse import quote_plus
import streamlit.components.v1 as components


def _poll_script(symbol_expr: str, update_expr: str) -> str:
    return f'''<script>
const symbol={symbol_expr};
async function poll() {{
  try {{
    const r=await fetch('https://query1.finance.yahoo.com/v8/finance/chart/'+encodeURIComponent(symbol)+'?range=1d&interval=1m&includePrePost=true&_='+Date.now(),{{cache:'no-store'}});
    if(!r.ok) throw 0;
    const d=await r.json(), q=d.chart?.result?.[0]?.indicators?.quote?.[0]?.close||[];
    if(q.length) {{ const v=Number(q[q.length-1]); if(Number.isFinite(v)&&v>0) {{ {update_expr} }} }}
  }} catch(e) {{}}
}}
poll(); setInterval(poll,1000);
</script>'''


def render_live_price(symbol: str, initial_price: float, height: int = 72) -> None:
    initial=float(initial_price); s=json.dumps(quote_plus(symbol))
    update="""const changed=Math.abs(v-last)>1e-7; last=v; p.textContent='₹'+v.toLocaleString('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2}); if(changed){p.classList.add('flash');setTimeout(()=>p.classList.remove('flash'),180)} m.textContent='LIVE · '+new Date().toLocaleTimeString('en-IN',{hour12:false});"""
    html=f'''<!doctype html><style>html,body{{margin:0;background:transparent;font-family:Inter,system-ui,sans-serif;overflow:hidden}}.card{{height:{height-4}px;box-sizing:border-box;background:#0e151d;border:1px solid #222e3a;border-radius:12px;padding:8px 12px;display:flex;align-items:center;justify-content:space-between}}.label{{font-size:10px;font-weight:800;color:#81909f}}.price{{font-size:22px;font-weight:900;color:#f4f7fa;transition:.12s}}.meta{{font-size:10px;color:#6ee39a}}.flash{{transform:scale(1.025)}}</style><div class="card"><div><div class="label">LIVE PRICE</div><div id="p" class="price">₹{initial:,.2f}</div></div><div id="m" class="meta">LIVE · 1s</div></div><script>let last={json.dumps(initial)},p=document.getElementById('p'),m=document.getElementById('m');</script>{_poll_script(s,update)}'''
    components.html(html,height=height,scrolling=False)


def render_market_overview(items: list[tuple[str,str,float]], height: int = 112) -> None:
    payload=json.dumps([{'name':n,'symbol':s,'price':float(p)} for n,s,p in items])
    cards=''.join(f'<div class="market"><div class="top"><b>{n}</b><i></i></div><div id="p{i}" class="big">₹{float(p):,.2f}</div><small id="m{i}">LIVE · 1s</small></div>' for i,(n,s,p) in enumerate(items))
    html=f'''<!doctype html><style>html,body{{margin:0;background:transparent;font-family:Inter,system-ui,sans-serif;overflow:hidden}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.market{{background:linear-gradient(145deg,#101923,#0c131b);border:1px solid #263544;border-radius:15px;padding:13px 15px;height:{height-8}px;box-sizing:border-box}}.top{{display:flex;justify-content:space-between;color:#dfe7ef;font-size:13px}}i{{width:7px;height:7px;border-radius:50%;background:#48e58a;box-shadow:0 0 8px #48e58a}}.big{{font-size:27px;font-weight:950;color:#f5f7fa;margin-top:8px;transition:.12s}}small{{font-size:10px;color:#6ee39a}}.flash{{transform:scale(1.025)}}</style><div class="grid">{cards}</div><script>
const items={payload};
function put(i,v){{const p=document.getElementById('p'+i),m=document.getElementById('m'+i),old=Number(p.dataset.v||items[i].price),changed=Math.abs(v-old)>1e-7;p.dataset.v=v;p.textContent='₹'+v.toLocaleString('en-IN',{{minimumFractionDigits:2,maximumFractionDigits:2}});if(changed){{p.classList.add('flash');setTimeout(()=>p.classList.remove('flash'),180)}}m.textContent='LIVE · '+new Date().toLocaleTimeString('en-IN',{{hour12:false}})}}
async function poll(i){{try{{const r=await fetch('https://query1.finance.yahoo.com/v8/finance/chart/'+encodeURIComponent(items[i].symbol)+'?range=1d&interval=1m&includePrePost=true&_='+Date.now(),{{cache:'no-store'}});if(!r.ok)throw 0;const d=await r.json(),q=d.chart?.result?.[0]?.indicators?.quote?.[0]?.close||[];if(q.length)put(i,Number(q[q.length-1]))}}catch(e){{}}}}
items.forEach((x,i)=>{{put(i,x.price);poll(i);setInterval(()=>poll(i),1000)}});
</script>'''
    components.html(html,height=height,scrolling=False)


def render_paper_live(symbol: str, initial_price: float, side: str, entry: float, qty: int, height: int = 112) -> None:
    payload=json.dumps({'symbol':symbol,'price':float(initial_price),'side':side,'entry':float(entry),'qty':int(qty)})
    html=f'''<!doctype html><style>html,body{{margin:0;background:transparent;font-family:Inter,system-ui,sans-serif;overflow:hidden}}.box{{height:{height-4}px;box-sizing:border-box;background:#0c151e;border:1px solid #2c4657;border-radius:15px;padding:12px 15px}}.row{{display:flex;justify-content:space-between}}.tag{{font-size:10px;font-weight:900;color:#79e6a4}}.side{{font-size:11px;font-weight:950;color:#69e69a}}.values{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:9px}}.label{{font-size:9px;color:#7f909f}}.value{{font-size:17px;font-weight:900;color:#edf3f8}}.pnl{{font-size:20px;font-weight:950;color:#6ee39a}}.loss{{color:#ff7076!important}}.flash{{transform:scale(1.025)}}</style><div class="box"><div class="row"><span class="tag">⚡ AUTO PAPER TRADE · LIVE P/L</span><span class="side">{side}</span></div><div class="values"><div><div class="label">ENTRY</div><div class="value">₹{float(entry):,.2f}</div></div><div><div class="label">LIVE PRICE</div><div id="live" class="value">₹{float(initial_price):,.2f}</div></div><div><div class="label">UNREALISED P/L</div><div id="pnl" class="pnl">₹0.00</div></div></div></div><script>
const x={payload},live=document.getElementById('live'),pnl=document.getElementById('pnl');let old=x.price;
function update(v){{const pl=x.side==='LONG'?(v-x.entry)*x.qty:(x.entry-v)*x.qty,changed=Math.abs(v-old)>1e-7;old=v;live.textContent='₹'+v.toLocaleString('en-IN',{{minimumFractionDigits:2,maximumFractionDigits:2}});pnl.textContent=(pl>=0?'+₹':'-₹')+Math.abs(pl).toLocaleString('en-IN',{{minimumFractionDigits:2,maximumFractionDigits:2}});pnl.className='pnl '+(pl<0?'loss':'');if(changed){{live.classList.add('flash');setTimeout(()=>live.classList.remove('flash'),180)}}}}
</script>{_poll_script(json.dumps(symbol), 'update(v);')}'''
    components.html(html,height=height,scrolling=False)
