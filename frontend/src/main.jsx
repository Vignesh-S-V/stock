import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const API = import.meta.env.VITE_API_URL || 'https://stock-algo-trading-api.onrender.com'
const INDEXES = ['NIFTY 50', 'BANK NIFTY', 'SENSEX']
const INDEX_SYMBOLS = ['^NSEI', '^NSEBANK', '^BSESN']
const SYMBOL_ALIASES = { 'NIFTY 50': '^NSEI', 'BANK NIFTY': '^NSEBANK', SENSEX: '^BSESN' }
const DEFAULTS = { strategy: 'Ensemble', risk_pct: 1, brokerage_pct: 0.03, reward_r: 2, threshold: 70, strict: true, auto: false, capital: 100000, paper_command: '' }

function fmt(x, digits = 2) { const n = Number(x); if (!Number.isFinite(n)) return '—'; return n.toLocaleString('en-IN', { minimumFractionDigits: digits, maximumFractionDigits: digits }) }
function fmt0(x) { const n = Number(x); return Number.isFinite(n) ? Math.round(n).toLocaleString('en-IN') : '—' }
function cls(value) { return String(value || 'hold').toLowerCase().replace(/\s+/g, '-') }
function resolveSymbol(value) { const raw = String(value || '').trim().toUpperCase(); return SYMBOL_ALIASES[raw] || raw }
function tradeLabels(action) { if (action === 'SELL') return { entry: 'SELL ENTRY', stop: 'BUY STOP', target: 'BUY-TO-COVER TARGET' }; if (action === 'BUY') return { entry: 'BUY ENTRY', stop: 'SELL STOP', target: 'SELL TARGET' }; return { entry: 'ENTRY', stop: 'STOP', target: 'TARGET' } }
function Pill({ children, tone = '' }) { return <span className={'pill ' + tone}>{children}</span> }

function App() {
  const [selected, setSelected] = useState('NIFTY 50')
  const [qty, setQty] = useState(1)
  const [config, setConfig] = useState(DEFAULTS)
  const [state, setState] = useState(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState('')
  const [lastTick, setLastTick] = useState(null)
  const [history, setHistory] = useState(() => { try { const raw = JSON.parse(localStorage.getItem('ml-index-paper-history') || '[]'); return Array.isArray(raw) ? raw.filter(x => INDEX_SYMBOLS.includes(x?.symbol)) : [] } catch { return [] } })
  const wsRef = useRef(null)
  const clientIdRef = useRef(null)
  const activeTradeRef = useRef(null)

  if (!clientIdRef.current) { try { clientIdRef.current = localStorage.getItem('algo-paper-client-id') || crypto.randomUUID(); localStorage.setItem('algo-paper-client-id', clientIdRef.current) } catch { clientIdRef.current = 'browser-paper-session' } }

  useEffect(() => {
    try { localStorage.setItem('ml-index-paper-history', JSON.stringify(history)) } catch {}
  }, [history])

  useEffect(() => {
    let ws, stopped = false, reconnect
    const connect = () => {
      const wsUrl = API.replace(/^http/, 'ws') + '/ws?client_id=' + encodeURIComponent(clientIdRef.current)
      ws = new WebSocket(wsUrl); wsRef.current = ws
      ws.onopen = () => { if (!stopped) { setConnected(true); setError(''); ws.send(JSON.stringify({ symbol: resolveSymbol(selected), qty, ...config })) } }
      ws.onmessage = event => { try { const next = JSON.parse(event.data); setState(prev => ({ ...next, indices: next.indices && Object.keys(next.indices).length ? next.indices : (prev?.indices || {}) })); setLastTick(new Date()); setError(next.type === 'error' ? next.message : '') } catch { setError('Invalid server response') } }
      ws.onerror = () => { if (!stopped) setError('WebSocket connection error') }
      ws.onclose = () => { if (!stopped) { setConnected(false); reconnect = setTimeout(connect, 3000) } }
    }
    connect(); return () => { stopped = true; clearTimeout(reconnect); ws?.close(); wsRef.current = null }
  }, [selected])

  useEffect(() => { if (!connected || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return; wsRef.current.send(JSON.stringify({ symbol: resolveSymbol(selected), qty, ...config })) }, [config, qty, selected, connected])

  const signal = state?.signal || {}
  const paper = state?.position || null
  const indices = useMemo(() => INDEXES.map(name => state?.indices?.[name] || { name }), [state])
  const action = signal.action || 'HOLD'
  const mlReady = state?.model?.trained === true
  const confidence = Number(signal.confidence)
  const labels = tradeLabels(action)
  const paperPnl = Number(state?.live_pnl)
  const signalIsQualified = mlReady && Number.isFinite(confidence) && confidence >= Number(config.threshold) && (action === 'BUY' || action === 'SELL')

  const sendConfig = patch => setConfig(prev => ({ ...prev, ...patch }))
  const selectIndex = name => { setSelected(name); setError('') }
  const paperOrder = command => {
    if (command !== 'CLOSE' && !signalIsQualified) { setError('Paper order blocked: ML must be READY with qualified BUY/SELL confidence.'); return }
    sendConfig({ paper_command: command })
  }

  const testIndex = d => {
    const name = d?.name || selected
    setSelected(name)
    setError('')
    const dReady = d?.model?.trained === true
    const dConfidence = Number(d?.confidence)
    const dAction = d?.decision || 'HOLD'
    if (!dReady || !Number.isFinite(dConfidence) || dConfidence < Number(config.threshold) || !['BUY', 'SELL'].includes(dAction)) {
      const readiness = dReady ? 'ML ready' : 'ML not ready'
      const confidenceText = Number.isFinite(dConfidence) ? `${fmt(dConfidence, 1)}%` : 'no confidence'
      setError(`${name}: PAPER TRADE NOT OPENED — ${readiness}; signal ${dAction}; confidence ${confidenceText}.`)
      return
    }
    sendConfig({ paper_command: dAction === 'BUY' ? 'BUY' : 'SELL' })
  }

  useEffect(() => {
    if (paper && !activeTradeRef.current) activeTradeRef.current = { symbol: paper.symbol, action: paper.action, entry: Number(paper.entry), qty: Number(paper.qty), confidence: Number(paper.confidence), ml: mlReady, openedAt: paper.opened_at || Date.now() / 1000 }
    if (!paper && activeTradeRef.current) {
      const t = activeTradeRef.current, exit = Number(state?.price)
      if (Number.isFinite(exit) && Number.isFinite(t.entry)) {
        const pnl = t.action === 'BUY' ? (exit - t.entry) * t.qty : (t.entry - exit) * t.qty
        const result = pnl > 0 ? 'CORRECT' : pnl < 0 ? 'WRONG' : 'FLAT'
        const row = { ...t, exit, pnl, result, closedAt: Date.now() }
        if (INDEX_SYMBOLS.includes(row.symbol)) setHistory(prev => [row, ...prev].slice(0, 100))
      }
      activeTradeRef.current = null
    }
  }, [paper, state?.price, mlReady])

  const mlTrades = history.filter(x => x.ml && (x.result === 'CORRECT' || x.result === 'WRONG'))
  const correct = mlTrades.filter(x => x.result === 'CORRECT').length
  const accuracy = mlTrades.length ? (correct / mlTrades.length) * 100 : null
  const mlPnl = mlTrades.reduce((sum, x) => sum + Number(x.pnl || 0), 0)

  return <main className="app">
    <header className="topbar"><div><div className="eyebrow">ML PAPER TRADING LAB</div><h1>Index Signal Tester</h1><p>Only NIFTY 50 · BANK NIFTY · SENSEX · ML signal verification</p></div><div className="top-right"><Pill tone={connected ? 'live' : 'bear'}>{connected ? '● MARKET LIVE' : '○ OFFLINE'}</Pill><span>{lastTick ? lastTick.toLocaleTimeString('en-IN') : '—'}</span></div></header>
    <section className="index-tabs">{INDEXES.map(name => <button key={name} className={selected === name ? 'active' : ''} onClick={() => selectIndex(name)}>{name}<small>{state?.indices?.[name]?.price != null ? `₹${fmt0(state.indices[name].price)}` : '—'}</small></button>)}</section>
    <section className="focus-grid">
      <article className={'signal-card ' + cls(action)}><div className="signal-head"><div><span className="eyebrow">ML RECOMMENDATION</span><h2>{state?.display_symbol || selected}</h2></div><span className="signal-state">{mlReady ? action : 'WAIT'}</span></div><div className="price">₹{fmt(state?.price)}</div><div className="status-row"><Pill tone={mlReady ? 'live' : 'neutral'}>{mlReady ? 'ML READY' : 'ML NOT READY'}</Pill><b>{mlReady && Number.isFinite(confidence) ? `${fmt(confidence, 1)}% confidence` : 'Confidence unavailable until ML is ready'}</b></div><div className="levels"><div><span>{labels.entry}</span><b>₹{fmt(signal.entry)}</b></div><div><span>{labels.stop}</span><b>₹{fmt(signal.stop)}</b></div><div><span>{labels.target}</span><b>₹{fmt(signal.target)}</b></div></div><div className="signal-note">{mlReady ? (action === 'BUY' ? 'ML says BUY: paper BUY is allowed when confidence reaches the threshold.' : action === 'SELL' ? 'ML says SELL: paper SHORT is allowed when confidence reaches the threshold.' : 'ML says HOLD: paper trade is blocked.') : 'ML is not ready: paper trade is blocked.'}</div></article>
      <article className="paper-card"><div className="card-head"><div><span className="eyebrow">PAPER EXECUTION</span><h2>{selected}</h2></div><Pill tone={paper ? 'bull' : 'muted'}>{paper ? `${paper.action} · OPEN` : 'FLAT'}</Pill></div><div className="order-row"><label>Units<input type="number" min="1" step="1" value={qty} onChange={e => setQty(Math.max(1, Number(e.target.value) || 1))} /></label><div className="rule">Every paper order shows OPEN / BLOCKED / CLOSED status here.</div></div><div className="buttons"><button className="buy" disabled={!signalIsQualified || action !== 'BUY'} onClick={() => paperOrder('BUY')}>PAPER BUY</button><button className="sell" disabled={!signalIsQualified || action !== 'SELL'} onClick={() => paperOrder('SELL')}>PAPER SHORT</button><button className="close" disabled={!paper} onClick={() => paperOrder('CLOSE')}>CLOSE</button></div>{paper ? <div className="position"><div><span>STATUS</span><b>OPEN · {paper.action === 'BUY' ? 'LONG' : 'SHORT'}</b></div><div><span>ENTRY</span><b>₹{fmt(paper.entry)}</b></div><div><span>LIVE P/L</span><b className={paperPnl >= 0 ? 'bull-text' : 'bear-text'}>₹{fmt(paperPnl)}</b></div><div><span>CONFIDENCE</span><b>{fmt(paper.confidence, 1)}%</b></div></div> : <div className="flat">FLAT · No open paper position. A BUY/SHORT button is enabled only when the selected index has a qualified ML signal.</div>}<div className="event">{state?.event || state?.execution_reason || 'Waiting for paper-trade action.'}</div></article>
    </section>
    <section className="verification"><div className="section-head"><div><span className="eyebrow">ML ACCURACY TRACKER</span><h2>Is the ML suggestion correct?</h2></div><Pill tone={mlTrades.length ? 'live' : 'neutral'}>{mlTrades.length ? `${fmt(accuracy, 1)}% ML accuracy` : 'NO ML TRADES YET'}</Pill></div><div className="stats"><div><span>ML trades tested</span><b>{mlTrades.length}</b></div><div><span>Correct</span><b className="bull-text">{correct}</b></div><div><span>Wrong</span><b className="bear-text">{mlTrades.length - correct}</b></div><div><span>Paper P/L</span><b className={mlPnl >= 0 ? 'bull-text' : 'bear-text'}>₹{fmt(mlPnl)}</b></div></div>{history.length ? <div className="history"><div className="history-title">Recent index ML signal results</div>{history.slice(0, 8).map((x, i) => <div className="history-row" key={`${x.closedAt}-${i}`}><b>{x.symbol}</b><span>{x.action}</span><span>{x.ml ? `ML ${fmt(x.confidence, 1)}%` : 'NOT ML'}</span><span className={x.result === 'CORRECT' ? 'bull-text' : x.result === 'WRONG' ? 'bear-text' : ''}>{x.result}</span><span>₹{fmt(x.pnl)}</span></div>)}</div> : <div className="empty">No completed index ML paper trades yet. Select NIFTY 50, BANK NIFTY, or SENSEX, wait for ML READY + BUY/SELL confidence above the threshold, then use TEST & PAPER and close the position to measure correctness.</div>}</section>
    <section className="mini-grid">{indices.map((d, i) => { const dAction = d?.decision || 'HOLD'; const dReady = d?.model?.trained === true; const dConfidence = Number(d?.confidence); const qualified = dReady && Number.isFinite(dConfidence) && dConfidence >= Number(config.threshold) && ['BUY', 'SELL'].includes(dAction); const status = !dReady ? 'ML NOT READY' : dAction === 'HOLD' ? 'HOLD — NO TRADE' : dConfidence < Number(config.threshold) ? `WAIT — ${fmt(dConfidence, 1)}%` : `${dAction} READY`; return <article key={d?.name || i} className="mini-card"><div className="mini-head"><div><span className="eyebrow">INDEX</span><h3>{d?.name || INDEXES[i]}</h3></div><Pill tone={d?.available ? 'live' : 'muted'}>{d?.available ? 'LIVE' : 'OFFLINE'}</Pill></div><strong>₹{fmt0(d?.price)}</strong><div className={'mini-decision ' + cls(dReady ? dAction : 'WAIT')}>{dReady ? dAction : 'WAIT'}</div><div className="mini-line"><span>ML status</span><b className={dReady ? 'bull-text' : 'bear-text'}>{status}</b></div><div className="mini-line"><span>ML confidence</span><b>{dReady && Number.isFinite(dConfidence) ? `${fmt(dConfidence, 1)}%` : '—'}</b></div><div className="mini-line"><span>Target</span><b>₹{fmt0(d?.target)}</b></div><button disabled={!qualified} onClick={() => testIndex(d)}>{qualified ? `TEST & PAPER ${dAction}` : 'TEST SIGNAL — NO TRADE'}</button></article> })}</section>
    {error && <div className="error-banner">{error}</div>}<footer>Paper simulation only · No real-money order is placed · ML accuracy is counted only for index trades opened while ML READY.</footer>
  </main>
}

createRoot(document.getElementById('root')).render(<App />)
