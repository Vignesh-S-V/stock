# Stock Algo Trading Platform

A research and paper-trading dashboard built with Python, Streamlit and Yahoo Finance data.

## Included

- Market data download with caching
- EMA/SMA, RSI, MACD, ATR, Bollinger Bands, VWAP, ROC, volume spike and ADX
- Explainable BUY / SELL / HOLD signal engine
- Ensemble and single-strategy modes
- ATR-based stop loss and risk/reward target
- Risk-based paper position sizing
- Backtesting with P&L, return, drawdown, trades and win rate
- Intraday-history guardrails
- Automated pytest + GitHub Actions CI
- Render deployment configuration
- No real-money broker execution by default

## Correctness safeguards

- EMA_21 is explicitly calculated; a missing EMA_21 is never treated as zero.
- Signal confidence is **evidence strength**, not a claimed probability of profit.
- Backtests use the same indicator/signal engine as the live dashboard to reduce logic drift.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Test

```bash
pytest -q
```

## Render

The included `render.yaml` uses:

```text
Build: pip install -r requirements.txt
Start: streamlit run app/main.py --server.address=0.0.0.0 --server.port=$PORT
```

## Risk notice

This is a research/paper-trading system. Backtests can be affected by slippage, transaction costs, data quality, execution assumptions and regime changes. Signals are not guarantees of returns.
