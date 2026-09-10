# Stock Algo Trading Platform

A modular, explainable research and paper-trading dashboard built with Python, Streamlit and Yahoo Finance data.

## Included
- Market data with caching
- EMA/SMA, RSI, MACD, ATR, Bollinger Bands, VWAP, ROC, volume spike and ADX
- Ensemble and single-strategy BUY / SELL / HOLD engine
- Explainable signal reasoning
- ATR-based stop loss and risk/reward target
- Risk-based paper position sizing
- Backtesting with P&L, return, drawdown, trade count and win rate
- Intraday-history guardrails
- Automated tests
- Render deployment configuration
- No real-money broker execution by default

## Correctness fixes
The engine explicitly calculates EMA_21, preventing missing values from silently becoming zero. Signal confidence is shown as evidence strength, not a claimed probability of profit.

## Run
```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Test
```bash
pytest -q
```

## Render
Build: `pip install -r requirements.txt`
Start: `streamlit run app/main.py --server.address=0.0.0.0 --server.port=$PORT`

## Risk
This is research/paper-trading software. Backtests can be affected by slippage, fees, data quality and model assumptions. Signals are not guarantees of future returns.
