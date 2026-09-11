import React, {useEffect, useMemo, useState} from 'react'
import {createRoot} from 'react-dom/client'
import './styles.css'

const API = import.meta.env.VITE_API_URL || 'https://stock-algo-trading-api.onrender.com'

function IndexCard({data}){
  const decision = data?.action || 'HOLD'
  const o = data?.option
  return <article className="index-card">
    <div className="index-head"><div><small>{data?.name}</small><strong>₹{fmt(data?.price)}</strong></div><span className={'decision '+decision.toLowerCase()}>{decision}</span></div>
    <div className="index-meta"><span>MODEL {data?.confidence ?? '—'}%</span><span>TARGET ₹{fmt(data?.target)}</span></div>
    {o?.available ? <div className="option-box">
      <div className="option-title"><b>{o.contract}</b><span>{o.expiry}</span></div>
      <div className="option-grid"><div><small>PREMIUM</small><b>₹{fmt(o.premium)}</b></div><div><small>BUY</small><b>₹{fmt(o.buy_price)}</b></div><div><small>SELL / TARGET</small><b>₹{fmt(o.target_price)}</b></div><div><small>STOP</small><b>₹{fmt(o.stop_price)}</b></div></div>
      <div className="option-foot"><span>IV {fmt(o.iv)}%</span><span>OI {fmtInt(o.oi)}</span><span>VOL {fmtInt(o.volume)}</span><span>{o.source}</span></div>
    </div> : <div className="option-unavailable">{decision==='HOLD'?'No option trade — model is HOLD.':(o?.reason||'Live option premium unavailable.')}</div>}
  </article>
}

function fmt(x){return Number.isFinite(+x) ? (+x).toLocaleString('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2}) : '—'}
function fmtInt(x){return Number.isFinite(+x) ? Math.round(+x).toLocaleString('en-IN') : '—'}

function App(){
  const [state,setState]=useState({})
  const [connected,setConnected]=useState(false)
  const [symbol,setSymbol]=useState('RELIANCE.NS')
  const [qty,setQty]=useState(1)
  const [error,setError]=useState('')

  useEffect(()=>{
    let ws
    let stopped=false
    const connect=()=>{
      const url=API.replace(/^http/,'ws')+'/ws?symbol='+encodeURIComponent(symbol)+'&qty='+qty
      ws=new WebSocket(url)
      ws.onopen=()=>{if(!stopped)setConnected(true)}
      ws.onmessage=(e)=>{try{setState(JSON.parse(e.data));setError('')}catch(err){setError('Invalid server response')}}
      ws.onerror=()=>{if(!stopped)setError('WebSocket connection error')}
      ws.onclose=()=>{if(!stopped){setConnected(false);setTimeout(connect,3000)}}
    }
    connect()
    return ()=>{stopped=true;ws?.close()}
  },[symbol,qty])

  const price=state?.price
  const signal=state?.signal
  const pnl=state?.paper?.unrealized_pnl
  const indices=useMemo(()=>state?.index_decisions||[],[state])

  return <main className="app">
    <header><div><h1>Algorithmic Trading Pro</h1><p>Live model signals, index options and paper execution</p></div><span className={'status '+(connected?'ok':'bad')}>{connected?'LIVE':'OFFLINE'}</span></header>
    <section className="controls"><label>Symbol <input value={symbol} onChange={e=>setSymbol(e.target.value.toUpperCase())}/></label><label>Qty <input type="number" min="1" value={qty} onChange={e=>setQty(Math.max(1,Number(e.target.value)||1))}/></label></section>
    <section className="hero"><div><small>SELECTED SYMBOL</small><h2>{symbol}</h2><strong>₹{fmt(price)}</strong></div><div><small>SIGNAL</small><b className={'signal '+String(signal?.action||'HOLD').toLowerCase()}>{signal?.action||'HOLD'}</b></div><div><small>CONFIDENCE</small><b>{signal?.confidence ?? '—'}%</b></div><div><small>PAPER P&amp;L</small><b>₹{fmt(pnl)}</b></div></section>
    {error && <div className="error">{error}</div>}
    <section><div className="section-title"><h2>Index Model Decisions</h2><span>Separate model → option recommendation</span></div><div className="index-grid">{indices.map((d,i)=><IndexCard key={d?.name||i} data={d}/>)}</div></section>
  </main>
}

createRoot(document.getElementById('root')).render(<App/>);
