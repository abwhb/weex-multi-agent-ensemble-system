# WEEX Multi-Agent Alpha

> Multi-Agent Ensemble AI Trading System for WEEX AI Wars Hackathon

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

A sophisticated AI-powered trading system that employs multiple specialized agents working in concert to make trading decisions. Each agent focuses on a specific market pattern (momentum, mean reversion, volatility breakouts), and a meta-learner optimally combines their signals.

## 🏗️ Architecture

```
Market Data → Feature Engineering → [Agents] → Meta-Learner → Risk Management → Execution
```

**Agents:**
- **Momentum Agent** - LSTM neural network for trend prediction
- **Mean Reversion Agent** - XGBoost + Bayesian probability estimation
- **Volatility Agent** - GARCH + CNN + Reinforcement Learning

**Meta-Learner:**
- Gradient-boosted signal combination
- Online learning for adaptive weights
- Kelly criterion position sizing

## 📁 Project Structure

```
weex-multi-agent-alpha/
├── src/
│   ├── agents/           # Trading agents (Momentum, MeanReversion, Volatility)
│   ├── meta-learner/     # Ensemble and performance tracking
│   ├── execution/        # WEEX API client, risk management, orders
│   ├── data/             # Market data and feature engineering
│   ├── logging/          # AI decision logging for competition
│   └── types/            # TypeScript type definitions
├── docs/
│   ├── STRATEGY.md       # Trading strategy documentation
│   └── AI_ARCHITECTURE.md # AI/ML architecture details
├── models/               # Trained model weights (gitignored)
├── logs/                 # AI decision logs
└── tests/                # Unit and integration tests
```

## 🚀 Quick Start

```bash
# Install dependencies
pnpm install

# Configure environment
cp .env.example .env
# Edit .env with your WEEX API credentials

# Build
pnpm build

# Run (paper trading mode)
pnpm start
```

## 📊 Supported Trading Pairs

Per competition rules, limited to:
- BTC, ETH, BNB, XRP, ADA, SOL, LTC, DOGE

## ⚠️ Risk Management

- Max leverage: 20x (competition limit)
- Position size: Max 10% per trade
- Daily drawdown: 5% halt threshold
- Minimum trades: 10 (competition requirement)

## 📝 Documentation

- [Trading Strategy](STRATEGY.md)
- [AI Architecture](AI_ARCHITECTURE.md)
- [WEEX API Docs](https://www.weex.com/api-doc/ai/intro)

## 🏆 Competition

Built for **AI Wars: WEEX Alpha Awakens** hackathon.

- Prize Pool: $880,000 USDT
- Duration: Nov 2025 - Mar 2026
- [Competition Page](https://dorahacks.io/hackathon/weex)

## 📄 License

MIT

