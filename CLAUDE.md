# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WEEX Multi-Agent Ensemble System - An AI-powered cryptocurrency trading system built for the WEEX AI Wars Hackathon. Uses three specialized trading agents combined via a gradient boosting meta-learner to generate trade signals.

## Common Commands

```bash
npm install           # Install dependencies
npm run build         # Compile TypeScript to dist/
npm run dev           # Run with ts-node (development)
npm start             # Run compiled version (production)
npm test              # Run Jest tests
npm test -- agents.test.ts  # Run specific test file
npm run lint          # ESLint checks
```

## Architecture

```
Market Data (WEEX API)
    ↓
Feature Engineering (FeatureStore)
    ↓
┌─── MOMENTUM AGENT (LSTM)
├─── MEAN REVERSION AGENT (XGBoost + Bayesian)
└─── VOLATILITY AGENT (GARCH + CNN + RL)
    ↓
ENSEMBLE META-LEARNER (LightGBM Gradient Boosting)
    ↓
RISK MANAGEMENT → EXECUTION (WEEX API) → AI LOGGING
```

### Key Directories

- `src/agents/` - Three trading agents, each extending `BaseAgent.ts`
- `src/meta-learner/` - Ensemble combiner and performance tracking
- `src/execution/` - WEEX API client, order management, risk management, paper trading engine
- `src/data/` - Market data service and feature engineering
- `src/logging/` - Competition compliance logging (AILogger)
- `src/database/` - SQLite database and repositories for trade persistence
- `src/control/` - TradingController for enable/disable trading
- `src/types/` - Shared TypeScript interfaces
- `data/` - SQLite database file (paper_trading.db)

### Entry Point

`src/index.ts` - TradingSystem class runs the main trading loop (5-second intervals by default). Initializes all components and handles graceful shutdown.

### Agent Models

- **MomentumAgent**: 2-layer LSTM (64 units each), 100 timesteps × 6 features
- **MeanReversionAgent**: Random Forest regime classifier + XGBoost parameter optimizer + Bayesian reversion probability
- **VolatilityAgent**: GARCH(1,1) forecast + 1D CNN breakout classifier + Q-Learning DQN for entry timing

### Key Configuration (src/config.ts)

- Trading mode: paper/live (paper is default)
- Max leverage: 20x (competition limit)
- Risk per trade: 2% default
- Approved pairs: BTC, ETH, BNB, XRP, ADA, SOL, LTC, DOGE
- Ensemble min confidence: 55%

## Environment Variables

Copy `.env.example` to `.env` and configure:
- `WEEX_API_KEY`, `WEEX_API_SECRET`, `WEEX_API_PASSPHRASE` - API credentials
- `TRADING_MODE` - 'paper' or 'live'
- `MAX_LEVERAGE` - Max 20x
- `RISK_PER_TRADE` - Default 0.02 (2%)
- `DB_PATH` - SQLite database path (default: ./data/paper_trading.db)
- `PAPER_INITIAL_BALANCE` - Starting balance for paper trading (default: 10000)

## Paper Trading & Trading Control

**Trading is DISABLED by default** (paper trading mode). This is a safety feature.

- `TradingController` manages enable/disable state (persisted in SQLite)
- When disabled → uses `PaperTradingEngine` for simulated trades
- When enabled → routes to live WEEX API
- P&L is tracked per-trade with entry/exit details

**Database Tables:**
- `trading_control` - Enable/disable state
- `accounts` - Simulated account balance
- `positions` - Open positions
- `trades` - Trade history with P&L
- `orders` - Order history
- `daily_stats` - Daily performance snapshots

## Important Constraints

- Strict TypeScript mode enabled - all code must be type-safe
- All trades must be logged via AILogger for competition compliance
- Risk limits enforced: 20x leverage max, 10% position size max, 5% daily drawdown halt
- Agents must return signals with confidence scores (0-1 normalized)

## Common Development Tasks

**Add a new agent**: Extend `BaseAgent` class in `src/agents/`, implement `generateSignal()` method

**Modify risk rules**: Edit `RiskManager` in `src/execution/RiskManager.ts`

**Add technical indicators**: Extend `MarketDataService` or `FeatureStore` in `src/data/`

**Adjust ensemble weights**: Modify `Ensemble.ts` or `PerformanceTracker.ts` in `src/meta-learner/`

---

## WEEX Contract API Reference

**Base URL**: `https://api-contract.weex.com`

**Documentation**:
- https://www.weex.com/api-doc/contract/introduction/APIBriefIntroduction
- https://www.weex.com/api-doc/ai/QuickStart/RequestInteraction

### Authentication

All private endpoints require these headers:
```
ACCESS-KEY: <api_key>
ACCESS-SIGN: <signature>
ACCESS-TIMESTAMP: <unix_timestamp_ms>
ACCESS-PASSPHRASE: <passphrase>
Content-Type: application/json
```

**Signature**: HMAC-SHA256 of `timestamp + method + path + queryString + body`, Base64 encoded.

### Symbol Format

Contract symbols use format: `cmt_<base>usdt` (lowercase)
- BTC → `cmt_btcusdt`
- ETH → `cmt_ethusdt`
- SOL → `cmt_solusdt`

### Public Endpoints (No Auth)

#### Get Ticker
```
GET /capi/v2/market/ticker?symbol=cmt_btcusdt
```
**Response** (direct, not wrapped):
```json
{
  "symbol": "cmt_btcusdt",
  "last": "90953.6",
  "best_ask": "90953.7",
  "best_bid": "90953.6",
  "high_24h": "93378.7",
  "low_24h": "90631.4",
  "base_volume": "49303.1325",
  "volume_24h": "4531013070.98703",
  "priceChangePercent": "-0.021420",
  "markPrice": "90940.6",
  "indexPrice": "90994.751",
  "timestamp": "1768915571152"
}
```

#### Get Candles
```
GET /capi/v2/market/candles?symbol=cmt_btcusdt&granularity=5m&limit=100
```
**Response**: Array of `[timestamp, open, high, low, close, volume, turnover]`

### Private Endpoints (Auth Required)

#### Get Account Balance
```
GET /capi/v2/account/assets
```
**Response** (array, direct):
```json
[
  {
    "coinName": "USDT",
    "available": "1000.00000000",
    "equity": "1000.00000000",
    "frozen": "0.00000000",
    "unrealizePnl": "0"
  }
]
```

#### Set Leverage
```
POST /capi/v2/account/leverage
```
**Body**:
```json
{
  "symbol": "cmt_btcusdt",
  "marginMode": "1",
  "longLeverage": "10",
  "shortLeverage": "10"
}
```
- `marginMode`: 1 = Cross, 3 = Isolated

#### Place Order
```
POST /capi/v2/order/placeOrder
```
**Body**:
```json
{
  "symbol": "cmt_btcusdt",
  "client_oid": "unique_id",
  "size": "0.001",
  "type": "1",
  "order_type": "0",
  "match_price": "1",
  "marginMode": "1",
  "price": "90000"
}
```

**Parameters**:
| Field | Values |
|-------|--------|
| `type` | 1=Open long, 2=Open short, 3=Close long, 4=Close short |
| `order_type` | 0=Normal, 1=Post-Only, 2=FOK, 3=IOC |
| `match_price` | 0=Limit, 1=Market |
| `marginMode` | 1=Cross, 3=Isolated |
| `price` | Required for limit orders only |

**Optional**: `presetStopLossPrice`, `presetTakeProfitPrice`

**Response** (direct):
```json
{
  "order_id": "708484453776753078",
  "client_oid": "test_1768915856587"
}
```

#### Cancel Order
```
POST /capi/v2/order/cancel_order
```
**Body**:
```json
{
  "symbol": "cmt_btcusdt",
  "orderId": "708484453776753078"
}
```

#### Get Current Orders
```
GET /capi/v2/order/current?symbol=cmt_btcusdt
```

#### Get Order History
```
GET /capi/v2/order/history?symbol=cmt_btcusdt&limit=100
```

#### Get Order Detail
```
GET /capi/v2/order/detail?symbol=cmt_btcusdt&orderId=<order_id>
```

#### Get Position
```
GET /capi/v2/account/position/singlePosition?symbol=cmt_btcusdt
```

#### Get Trade Fills
```
GET /capi/v2/trade/fills?symbol=cmt_btcusdt&limit=100
```

### Response Formats

**Important**: WEEX Contract API has inconsistent response formats:

1. **Public endpoints** (ticker, candles): Return data directly
2. **Most private endpoints** (assets, orders): Return data directly as array/object
3. **Some endpoints** (leverage): Return wrapped `{code, msg, requestTime}`

The `WeexClient` handles both formats automatically.

### Hackathon Requirements

- **Test Account Balance**: 1000 USDT
- **Minimum Trading**: 10 USDT to pass API test
- **Max Leverage**: 20x
- **Allowed Pairs**: BTC, ETH, BNB, XRP, ADA, SOL, LTC, DOGE

### Error Codes

| Code | Meaning |
|------|---------|
| `00000` | Success (wrapped responses) |
| `200` | Success (some endpoints) |
| `40001` | Invalid parameter |
| `40002` | Insufficient balance |
| `40003` | Order not found |
| `40004` | Position not found |
| `40007` | Leverage exceeds limit |
| `40014` | Order size too small |
| `43011` | Insufficient margin |
| `50001` | System error |

### Rate Limits

| Endpoint Type | IP Weight | UID Weight |
|---------------|-----------|------------|
| Market Data | 1-2 | - |
| Account | 5 | 10 |
| Place Order | 2 | 5 |
| Cancel Order | 2 | 3 |
| Order Query | 2 | 2 |

Recommend minimum 100ms between requests.

### Common Issues & Solutions

1. **404 on endpoints**: Ensure correct path (e.g., `/capi/v2/order/placeOrder` not `/capi/v2/order/place`)

2. **Empty response for positions**: Use `/capi/v2/account/position/singlePosition` with symbol parameter

3. **Order rejected**: Check:
   - Minimum order size (usually 0.001 for BTC)
   - Sufficient margin (position value / leverage)
   - Leverage is set for the symbol

4. **Signature errors**: Ensure timestamp is in milliseconds and string concatenation order is correct

### Minimum Order Sizes

| Symbol | Min Size |
|--------|----------|
| BTC | 0.001 |
| ETH | 0.01 |
| SOL | 0.1 |
| Others | Check API |

### Test Scripts

```bash
# Verify API connection and all endpoints
npx ts-node scripts/verify-api.ts

# Run full API test with order placement
npx ts-node scripts/test-api.ts

# Close any open positions
npx ts-node scripts/close-position.ts
```

---

## WebSocket API (Future Reference)

**URL**: `wss://ws-contract.weex.com`

Channels:
- `ticker.<symbol>` - Real-time price updates
- `candle.<interval>.<symbol>` - Candle updates
- `depth.<symbol>` - Order book
- `trade.<symbol>` - Recent trades

*Note: WebSocket not yet implemented in current codebase.*
