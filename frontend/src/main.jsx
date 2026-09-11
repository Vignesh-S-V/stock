import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const API = import.meta.env.VITE_API_URL || 'https://stock-algo-trading-api.onrender.com'
const STRATEGIES = ['Ensemble', 'EMA Crossover', 'RSI Reversion', 'MACD', 'Trend Momentum', 'Mean Reversion']
const DEFAULTS = { strategy: 'Ensemble', risk_pct: 1, brokerage_pct: 0.03, reward_r: 2, threshold: 90, strict: true, auto: true, capital: 100000 }

function fmt(x) {
  const n = Number(x)
  if (!Number.isFinite(n)) return '—'
  return n.toLocaleString('en-IN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  })
}

function fmt0(x) {
  const n = Number(x)
  return Number.isFinite(n) ? Math.round(n).toLocaleString('en-IN') : '—'
}

function cls(value) { return String(value || 'hold').toLowerCase().replace(/\s+/g, '-') }

function IndexCard({ data }) {
  const decision = data?.decision || 'HOLD'
  const option = data?.option
  return <article className="index-card">
    <div className="index-head"><div><small>{data?.name || 'INDEX'}</small><strong>₹{fmt(data?.price)}</strong></div><span className={'decision ' + cls(decision)}>{decision}</span></div>
    <div className="index-meta"><span>MODEL {data?.confidence ?? '—'}%</span><span>ENTRY ₹{fmt(data?.entry)}</span><span>TARGET ₹{fmt(data?.target)}</span></div>
    <div className="reasons">{(data?.reasons || []).slice(-3).map((r, i) => <span key={i}>• {r}</span>)}</div>
    {option?.available ? <div className="option-box">
      <div className="option-title"><b>{option.contract}</b><span>{option.expiry}</span></div>
      <div className="option-grid"><div><small>LIVE PREMIUM</small><b>₹{fmt(option.premium)}</b></div><div><small>BUY</small><b>₹{fmt(option.buy_price)}</b></div><div><small>SELL / TARGET</small><b>₹{fmt(option.target_price)}</b></div><div><small>STOP</small><b>₹{fmt(option.stop_price)}</b></div></div>
      <div className="option-foot"><span>IV {fmt(option.iv)}%</span><span>OI {fmt0(option.oi)}</span><span>VOL {fmt0(option.volume)}</span><span>{option.source}</span></div>
    </div> : <div className="option-unavailable">{decision === 'HOLD' ? 'No option trade — model is HOLD.' : (option?.reason || 'Live option premium unavailable.')}</div>}
  </article>
}

function App() {
  const [symbol, setSymbol] = useState('RELIANCE.NS')
  const [qty, setQty] = useState(1)
  const [config, setConfig] = useState(DEFAULTS)
  const [state, setState] = useState(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState('')
  const [lastTick, setLastTick] = useState(null)
  const wsRef = useRef(null)

  useEffect(() => {
    let ws
    let stopped = false
    let reconnect
    const connect = () => {
      const wsUrl = API.replace(/^http/, 'ws') + '/ws'
      ws = new WebSocket(wsUrl)
      wsRef.current = ws
      ws.onopen = () => {
        if (!stopped) {
          setConnected(true)
          ws.send(JSON.stringify({ symbol, ...config }))
        }
      }
      ws.onmessage = event => {
        try {
          const next = JSON.parse(event.data)
          setState(next)
          setLastTick(new Date())
          setError(next.type === 'error' ? next.message : '')
        } catch { setError('Invalid server response') }
      }
      ws.onerror = () => { if (!stopped) setError('WebSocket connection error') }
      ws.onclose = () => {
        if (!stopped) { setConnected(false); reconnect = setTimeout(connect, 3000) }
      }
    }
    connect()
    return () => { stopped = true; clearTimeout(reconnect); ws?.close(); wsRef.current = null }
  }, [symbol])

  useEffect(() => {
    if (!connected || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
    wsRef.current.send(JSON.stringify({ symbol, qty, ...config }))
  }, [config, qty, symbol, connected])

  const signal = state?.signal || {}
  const paper = state?.position || null
  const indices = useMemo(() => Object.values(state?.indices || {}), [state])
  const sendConfig = patch => setConfig(prev => ({ ...prev, ...patch }))

  return <main className="app">
    <header><div><h1>Algorithmic Trading Pro</h1><p>Live model signals · index options · paper execution</p></div><span className={'status ' + (connected ? 'ok' : 'bad')}>{connected ? 'LIVE' : 'OFFLINE'}</span></header>
    <section className="controls">
      <label>Symbol<input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} /></label>
      <label>Qty<input type="number" min="1" value={qty} onChange={e => setQty(Math.max(1, Number(e.target.value) || 1))} /></label>
      <label>Strategy<select value={config.strategy} onChange={e => sendConfig({ strategy: e.target.value })}>{STRATEGIES.map(s => <option key={s}>{s}</option>)}</select></label>
      <label>Risk %<input type="number" step="0.1" min="0.1" max="10" value={config.risk_pct} onChange={e => sendConfig({ risk_pct: Number(e.target.value) })} /></label>
      <label>Reward R<input type="number" step="0.5" min="1" max="10" value={config.reward_r} onChange={e => sendConfig({ reward_r: Number(e.target.value) })} /></label>
      <label>Threshold %<input type="number" min="50" max="100" value={config.threshold} onChange={e => sendConfig({ threshold: Number(e.target.value) })} /></label>
      <label>Auto Paper<select value={String(config.auto)} onChange={e => sendConfig({ auto: e.target.value === 'true' })}><option value="true">ON</option><option value="false">OFF</option></select></label>
    </section>
    <section className="hero">
      <div><small>SELECTED SYMBOL</small><h2>{symbol}</h2><strong>₹{fmt(state?.price)}</strong></div>
      <div><small>MODEL SIGNAL</small><b className={'signal ' + cls(signal.action)}>{signal.action || 'HOLD'}</b></div>
      <div><small>CONFIDENCE</small><b>{signal.confidence ?? '—'}%</b></div>
      <div><small>ENTRY</small><b>₹{fmt(signal.entry)}</b></div>
      <div><small>STOP</small><b>₹{fmt(signal.stop)}</b></div>
      <div><small>TARGET</small><b>₹{fmt(signal.target)}</b></div>
      <div><small>PAPER P/L</small><b>₹{fmt(state?.live_pnl)}</b></div>
    </section>
    {error && <div className="error">{error}</div>}
    <section><div className="section-title"><h2>Index Model Decisions</h2><span>Independent model → option recommendation</span></div><div className="index-grid">{indices.length ? indices.map((d, i) => <IndexCard key={d?.name || i} data={d} />) : <div className="empty">Waiting for index data…</div>}</div></section>
    <section className="lower-grid">
      <article className="panel"><h2>Paper Position</h2>{paper ? <div className="position-grid"><div><small>SIDE</small><b>{paper.side}</b></div><div><small>QTY</small><b>{fmt0(paper.qty)}</b></div><div><small>ENTRY</small><b>₹{fmt(paper.entry)}</b></div><div><small>STOP</small><b>₹{fmt(paper.stop)}</b></div><div><small>TARGET</small><b>₹{fmt(paper.target)}</b></div><div><small>CONFIDENCE</small><b>{paper.confidence}%</b></div></div> : <p className="muted">No open paper position.</p>}<p className="event">{state?.event || 'No recent execution event.'}</p></article>
      <article className="panel"><h2>Signal Reasons</h2><ul>{(signal.reasons || []).map((r, i) => <li key={i}>{r}</li>)}</ul></article>
      <article className="panel"><h2>Account</h2><div className="account-grid"><div><small>CASH</small><b>₹{fmt(state?.cash)}</b></div><div><small>REALIZED P/L</small><b>₹{fmt(state?.realized_pnl)}</b></div><div><small>LAST UPDATE</small><b>{lastTick ? lastTick.toLocaleTimeString('en-IN') : '—'}</b></div></div></article>
    </section>
  </main>
}

createRoot(document.getElementById('root')).render(<App />)
