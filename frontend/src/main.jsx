import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './styles.css';

const API = import.meta.env.VITE_API_URL || 'https://stock-algo-trading.onrender.com';
const WS = API.replace(/^http/, 'ws') + '/ws';

function App(){
  const socket=useRef(null); const [data,setData]=useState(null); const [status,setStatus]=useState('CONNECTING');
  const [cfg,setCfg]=useState({symbol:'RELIANCE.NS',strategy:'Ensemble',risk_pct:1,brokerage_pct:.03,reward_r:2,threshold:90,strict:true,auto:true,capital:100000});
  useEffect(()=>{ let retry; const connect=()=>{setStatus('CONNECTING'); const ws=new WebSocket(WS); socket.current=ws;
    ws.onopen=()=>{setStatus('LIVE'); ws.send(JSON.stringify(cfg));}; ws.onmessage=e=>{try{setData(JSON.parse(e.data))}catch{}};
    ws.onclose=()=>{setStatus('RECONNECTING'); retry=setTimeout(connect,2000)}; ws.onerror=()=>setStatus('ERROR');}; connect(); return()=>{clearTimeout(retry);socket.current?.close()};},[]);
  useEffect(()=>{if(socket.current?.readyState===1)socket.current.send(JSON.stringify(cfg))},[cfg]);
  const change=(k,v)=>setCfg(c=>({...c,[k]:v}));
  const sig=data?.signal; const pos=data?.position;
  return <main><header><div><h1>⚡ ALGO TRADING PRO</h1><p>Live Indian market command center · WebSocket · no page refresh</p></div><span className={'status '+status.toLowerCase()}>● {status}</span></header>
    <section className="controls"><label>Instrument<select value={cfg.symbol} onChange={e=>change('symbol',e.target.value)}><option>RELIANCE.NS</option><option>TCS.NS</option><option>INFY.NS</option><option>HDFCBANK.NS</option><option>ICICIBANK.NS</option><option>SBIN.NS</option><option>ITC.NS</option><option>LT.NS</option><option>BHARTIARTL.NS</option><option>AXISBANK.NS</option><option>^NSEI</option><option>^BSESN</option><option>^NSEBANK</option></select></label>
    <label>Strategy<select value={cfg.strategy} onChange={e=>change('strategy',e.target.value)}>{['Ensemble','Trend Momentum','EMA Crossover','RSI Reversion','MACD','Mean Reversion'].map(x=><option key={x}>{x}</option>)}</select></label>
    <label>Mode<select value={cfg.auto?'auto':'suggest'} onChange={e=>change('auto',e.target.value==='auto')}><option value="auto">Auto Paper</option><option value="suggest">Suggestion Only</option></select></label>
    <label>Gate<input type="number" min="50" max="95" value={cfg.threshold} onChange={e=>change('threshold',+e.target.value)}/></label></section>
    <section className="indices">{Object.entries(data?.indices||{}).map(([n,v])=><article key={n}><small>{n}</small><strong>₹{v.price.toLocaleString('en-IN',{maximumFractionDigits:2})}</strong><em>LIVE</em></article>)}</section>
    <section className="grid"><article className="signal"><small>MODEL DECISION</small><div className={'action '+(sig?.action||'HOLD').toLowerCase()}>{sig?.action||'WAIT'}</div><p>Evidence <b>{sig?.confidence??'—'}%</b> · R:R 1:{cfg.reward_r}</p><div className="levels"><span>ENTRY ₹{fmt(sig?.entry)}</span><span>STOP ₹{fmt(sig?.stop)}</span><span>TARGET ₹{fmt(sig?.target)}</span></div></article>
    <article className="paper"><small>PAPER TRADING</small>{pos?<><h2>{pos.side} OPEN</h2><p>Qty <b>{pos.qty.toLocaleString()}</b> · Entry ₹{fmt(pos.entry)}</p><div className="levels"><span>STOP ₹{fmt(pos.stop)}</span><span>TARGET ₹{fmt(pos.target)}</span></div><div className="pnl">LIVE P/L {data.live_pnl>=0?'+':''}₹{fmt(data.live_pnl)}</div></>:<h2>NO OPEN POSITION</h2>}<p>{data?.event||'Waiting for a qualifying signal.'}</p></article></section>
    <section className="reasons"><h2>Signal reasoning</h2>{(sig?.reasons||[]).slice(0,7).map((r,i)=><div key={i}>• {r}</div>)}</section>
    <footer>Last server update: {data?new Date(data.timestamp*1000).toLocaleTimeString():'—'} · Yahoo 1-minute snapshot fallback; no real-money order is sent.</footer>
  </main>
}
function fmt(x){return Number.isFinite(+x)?(+x).toLocaleString('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2}):'—'}
createRoot(document.getElementById('root')).render(<App/>);
