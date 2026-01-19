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
