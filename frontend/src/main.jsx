import React, { useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const API = import.meta.env.VITE_API_URL || 'https://stock-algo-trading-api.onrender.com'
const STRATEGIES = ['Ensemble', 'EMA Crossover', 'RSI Reversion', 'MACD', 'Trend Momentum', 'Mean Reversion']
const DEFAULTS = { strategy: 'Ensemble', risk_pct: 1, brokerage_pct: 0.03, reward_r: 2, threshold: 95, strict: true, auto: true, capital: 100000 }

function fmt(x, digits = 2) {
  const n = Number(x)
  if (!Number.isFinite(n)) return '—'
  return n.toLocaleString('en-IN', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}
function fmt0(x) { const n = Number(x); return Number.isFinite(n) ? Math.round(n).toLocaleString('en-IN') : '—' }
function cls(value) { return String(value || 'hold').toLowerCase().replace(/\s+/g, '-') }
function timeAgo(value) { if (!value) return '—'; const d = new Date(value); if (Number.isNaN(d.getTime())) return value; const mins = Math.max(0, Math.round((Date.now() - d.getTime()) / 60000)); return mins < 1 ? 'now' : `${mins}m ago` }

function Pill({ children, tone = '' }) { return <span className={'pill ' + tone}>{children}</span> }

function IndexCard({ data }) {
  const decision = data?.decision || 'HOLD'
  const live = Boolean(data?.available)
  const option = data?.option
  const snapshot = option?.mode === 'snapshot'
  return <article className="index-card">
    <div className="index-card-top"><div><span className="eyebrow">INDEX</span><h3>{data?.name || 'INDEX'}</h3></div><Pill tone={live ? 'live' : 'muted'}>{live ? '● LIVE' : '○ OFFLINE'}</Pill></div>
    <div className="index-price-row"><strong>₹{fmt0(data?.price)}</strong><span className={'decision ' + cls(decision)}>{decision === 'BUY' ? 'BUY / UP' : decision === 'SELL' ? 'SELL / DOWN' : 'HOLD / SIDEWAYS'}</span></div>
    <div className="metric-row"><span>Model confidence</span><b>{data?.confidence ?? '—'}%</b></div>
    <div className="metric-row"><span>Target</span><b>{fmt0(data?.target)}</b></div>
    {snapshot ? <div className="option-box"><div className="option-title"><span>LIVE OPTION SNAPSHOT</span><b>{option.contract}</b></div><div className="option-grid"><div><small>CE LTP</small><b>₹{fmt(option.ce_premium)}</b></div><div><small>PE LTP</small><b>₹{fmt(option.pe_premium)}</b></div><div><small>CE IV</small><b>{fmt(option.ce_iv)}%</b></div><div><small>PE IV</small><b>{fmt(option.pe_iv)}%</b></div></div><div className="option-foot"><span>Expiry {option.expiry || '—'}</span><span>OI {fmt0(option.ce_oi)} / {fmt0(option.pe_oi)}</span></div></div> : option?.available ? <div className="option-box"><div className="option-title"><span>TRADE OPTION</span><b>{option.contract}</b></div><div className="option-grid"><div><small>PREMIUM</small><b>₹{fmt(option.premium)}</b></div><div><small>ENTRY</small><b>₹{fmt(option.buy_price)}</b></div><div><small>TARGET</small><b>₹{fmt(option.target_price)}</b></div><div><small>STOP</small><b>₹{fmt(option.stop_price)}</b></div></div><div className="option-foot"><span>IV {fmt(option.iv)}%</span><span>OI {fmt0(option.oi)}</span><span>VOL {fmt0(option.volume)}</span></div></div> : <div className="option-unavailable">{live ? (option?.reason || 'Live option-chain unavailable') : 'Live index data unavailable'}</div>}
  </article>
}

function NewsPanel({ news }) {
  const items = news?.items || []
  const score = Number(news?.score || 0)
  return <article className="panel news-panel">
    <div className="panel-head"><div><span className="eyebrow">MARKET INTELLIGENCE</span><h2>News & Sentiment</h2></div><Pill tone={score > 0.18 ? 'bull' : score < -0.18 ? 'bear' : 'neutral'}>{news?.bias || 'NEUTRAL'} {score >= 0 ? '+' : ''}{score.toFixed(2)}</Pill></div>
    <div className="news-summary"><div><span>Headline bias</span><b>{news?.bias || 'NEUTRAL'}</b></div><div><span>Positive</span><b>{news?.positive ?? 0}</b></div><div><span>Negative</span><b>{news?.negative ?? 0}</b></div><div><span>Feed</span><b>{news?.available ? 'LIVE' : 'OFFLINE'}</b></div></div>
    {items.length ? <div className="news-list">{items.slice(0, 6).map((item, i) => <a className="news-item" key={item.link || i} href={item.link || '#'} target="_blank" rel="noreferrer"><div className={'news-dot ' + (item.sentiment > 0.15 ? 'bull' : item.sentiment < -0.15 ? 'bear' : '')}></div><div><strong>{item.title}</strong><span>{item.source} · {item.published || 'recent'}</span></div></a>)}</div> : <p className="muted">{news?.reason || 'Waiting for headline feed…'}</p>}
    <p className="disclaimer">Headline sentiment is a confirmation/risk filter, not a standalone trading signal.</p>
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
    let ws, stopped = false, reconnect
    const connect = () => {
      const wsUrl = API.replace(/^http/, 'ws') + '/ws'
      ws = new WebSocket(wsUrl); wsRef.current = ws
      ws.onopen = () => { if (!stopped) { setConnected(true); setError(''); ws.send(JSON.stringify({ symbol, ...config })) } }
      ws.onmessage = event => { try { const next = JSON.parse(event.data); setState(next); setLastTick(new Date()); setError(next.type === 'error' ? next.message : '') } catch { setError('Invalid server response') } }
      ws.onerror = () => { if (!stopped) setError('WebSocket connection error') }
      ws.onclose = () => { if (!stopped) { setConnected(false); reconnect = setTimeout(connect, 3000) } }
    }
    connect()
    return () => { stopped = true; clearTimeout(reconnect); ws?.close(); wsRef.current = null }
  }, [symbol])

  useEffect(() => { if (!connected || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return; wsRef.current.send(JSON.stringify({ symbol, qty, ...config })) }, [config, qty, symbol, connected])

  const signal = state?.signal || {}
  const paper = state?.position || null
  const news = state?.news || {}
  const indices = useMemo(() => Object.values(state?.indices || {}), [state])
  const sendConfig = patch => setConfig(prev => ({ ...prev, ...patch }))
  const action = signal.action || 'HOLD'
  const newsScore = Number(news.score || 0)

  return <main className="app">
    <header className="topbar"><div className="brand"><div className="brand-mark">AP</div><div><h1>Algo Trading <span>Pro</span></h1><p>Indian market intelligence · ML · technicals · options</p></div></div><div className="top-actions"><Pill tone={connected ? 'live' : 'bear'}>{connected ? '● MARKET LIVE' : '○ OFFLINE'}</Pill><span className="update-time">Updated {lastTick ? lastTick.toLocaleTimeString('en-IN') : '—'}</span></div></header>

    <section className="control-bar"><div className="symbol-select"><span className="eyebrow">WATCH</span><input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} /></div><label>Strategy<select value={config.strategy} onChange={e => sendConfig({ strategy: e.target.value })}>{STRATEGIES.map(s => <option key={s}>{s}</option>)}</select></label><label>Risk<input type="number" step="0.1" min="0.1" max="10" value={config.risk_pct} onChange={e => sendConfig({ risk_pct: Number(e.target.value) })} /></label><label>R:R<input type="number" step="0.5" min="1" max="10" value={config.reward_r} onChange={e => sendConfig({ reward_r: Number(e.target.value) })} /></label><label>ML threshold<input type="number" min="50" max="100" value={config.threshold} onChange={e => sendConfig({ threshold: Number(e.target.value) })} /></label><label>Paper execution<select value={String(config.auto)} onChange={e => sendConfig({ auto: e.target.value === 'true' })}><option value="true">AUTO ON</option><option value="false">MANUAL</option></select></label></section>

    {error && <div className="error-banner">{error}</div>}

    <section className="hero-grid">
      <article className={'signal-card ' + cls(action)}><div className="signal-top"><div><span className="eyebrow">PRIMARY SIGNAL</span><h2>{symbol}</h2></div><span className="signal-state">{action}</span></div><div className="signal-price">₹{fmt(state?.price)}</div><div className="signal-sub">{action === 'BUY' ? 'Long bias' : action === 'SELL' ? 'Short bias' : 'No-trade / wait for confirmation'}</div><div className="signal-levels"><div><span>ENTRY</span><b>₹{fmt(signal.entry)}</b></div><div><span>STOP</span><b>₹{fmt(signal.stop)}</b></div><div><span>TARGET</span><b>₹{fmt(signal.target)}</b></div></div></article>
      <article className="score-card"><div className="panel-head"><div><span className="eyebrow">DECISION ENGINE</span><h2>Signal Stack</h2></div><Pill tone={signal.trained ? 'live' : 'neutral'}>{signal.trained ? 'ML READY' : 'TRAINING'}</Pill></div><div className="stack-row"><span>Technical structure</span><b>{action}</b></div><div className="stack-row"><span>ML probability</span><b>{signal.confidence ?? '—'}%</b></div><div className="stack-row"><span>News bias</span><b className={newsScore > .18 ? 'bull-text' : newsScore < -.18 ? 'bear-text' : ''}>{news.bias || 'NEUTRAL'} {news?.score != null ? `(${newsScore >= 0 ? '+' : ''}${newsScore.toFixed(2)})` : ''}</b></div><div className="stack-row"><span>Validation</span><b>{state?.model?.validation_accuracy != null ? `${(state.model.validation_accuracy * 100).toFixed(1)}%` : '—'}</b></div><div className="engine-note">{state?.model?.engine || 'Calibrated ensemble'} · {fmt0(state?.model?.samples)} training rows</div></article>
      <article className="risk-card"><span className="eyebrow">RISK CONTROL</span><h2>₹{fmt(config.capital, 0)}</h2><div className="risk-grid"><div><span>Risk / trade</span><b>{config.risk_pct}%</b></div><div><span>Reward</span><b>{config.reward_r}R</b></div><div><span>Paper P/L</span><b className={Number(state?.live_pnl) >= 0 ? 'bull-text' : 'bear-text'}>₹{fmt(state?.live_pnl)}</b></div><div><span>Realized</span><b>₹{fmt(state?.realized_pnl)}</b></div></div></article>
    </section>

    <section><div className="section-title"><div><span className="eyebrow">MARKET PULSE</span><h2>Indian Index Command Center</h2></div><span>NIFTY 50 · BANK NIFTY · SENSEX</span></div><div className="index-grid">{indices.length ? indices.map((d, i) => <IndexCard key={d?.name || i} data={d} />) : <div className="empty">Waiting for index data…</div>}</div></section>

    <section className="content-grid"><NewsPanel news={news} /><article className="panel"><div className="panel-head"><div><span className="eyebrow">EXECUTION</span><h2>Paper Position</h2></div><Pill tone={paper ? 'bull' : 'muted'}>{paper ? paper.side : 'FLAT'}</Pill></div>{paper ? <div className="position-grid"><div><small>QTY</small><b>{fmt0(paper.qty)}</b></div><div><small>ENTRY</small><b>₹{fmt(paper.entry)}</b></div><div><small>STOP</small><b>₹{fmt(paper.stop)}</b></div><div><small>TARGET</small><b>₹{fmt(paper.target)}</b></div><div><small>CONFIDENCE</small><b>{paper.confidence}%</b></div><div><small>LIVE P/L</small><b>₹{fmt(state?.live_pnl)}</b></div></div> : <div className="flat-state"><strong>No open paper position</strong><span>Strict mode waits for ML threshold + news confirmation.</span></div>}<div className="event-line">{state?.event || 'No recent execution event.'}</div></article></section>

    <section className="lower-grid"><article className="panel"><div className="panel-head"><div><span className="eyebrow">EXPLAINABILITY</span><h2>Why this signal?</h2></div></div><ul className="reason-list">{(signal.reasons || []).map((r, i) => <li key={i}>{r}</li>)}</ul></article><article className="panel"><div className="panel-head"><div><span className="eyebrow">ACCOUNT</span><h2>Paper Ledger</h2></div></div><div className="account-grid"><div><small>CASH</small><b>₹{fmt(state?.cash)}</b></div><div><small>REALIZED P/L</small><b>₹{fmt(state?.realized_pnl)}</b></div><div><small>QTY</small><b>{fmt0(paper?.qty)}</b></div><div><small>MODE</small><b>{config.auto ? 'AUTO PAPER' : 'MANUAL'}</b></div></div></article></section>

    <nav className="mobile-nav"><a href="#top">⌂<span>Overview</span></a><a href="#signals">◎<span>Signals</span></a><a href="#news">◈<span>News</span></a><a href="#risk">◌<span>Risk</span></a></nav>
    <footer>Research / paper-trading interface only. No real-money orders are placed by this dashboard.</footer>
  </main>
}

createRoot(document.getElementById('root')).render(<App />)
