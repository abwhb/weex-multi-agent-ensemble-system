# WEEX Multi-Agent Alpha

> 🏆 Multi-Agent Ensemble AI Trading System for WEEX AI Wars Hackathon

[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue.svg)](https://www.typescriptlang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-20+-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

A sophisticated AI-powered trading system that employs multiple specialized agents working in concert to make trading decisions. Each agent focuses on a specific market pattern, and a meta-learner optimally combines their signals using machine learning.

### Key Features

- **Multi-Agent Architecture**: Three specialized agents (Momentum, Mean Reversion, Volatility)
- **Ensemble Meta-Learning**: Gradient boosted signal combination with online adaptation
- **Full AI Traceability**: Complete logging for competition compliance
- **Risk Management**: Built-in position limits, drawdown protection, correlation filters
- **TypeScript**: Fully typed with comprehensive documentation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MARKET DATA                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│ MOMENTUM  │  │   MEAN    │  │ VOLATILITY│
│   AGENT   │  │ REVERSION │  │   AGENT   │
│  (LSTM)   │  │  (XGBoost)│  │  (GARCH)  │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │              │              │
      └──────────────┼──────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    META-LEARNER                              │
│  • Signal combination  • Adaptive weighting  • Kelly sizing │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│           RISK MANAGEMENT → EXECUTION → AI LOGGING          │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
weex-multi-agent-alpha/
├── src/
│   ├── agents/           # Trading agents (Momentum, MeanReversion, Volatility)
│   ├── meta-learner/     # Ensemble and performance tracking
│   ├── execution/        # WEEX API client, risk management, orders
│   ├── data/             # Market data and feature engineering
│   ├── logging/          # AI decision logging for competition
│   ├── types/            # TypeScript type definitions
│   ├── config.ts         # Configuration management
│   └── index.ts          # Main entry point
├── docs/
│   ├── STRATEGY.md       # Trading strategy documentation
│   ├── AI_ARCHITECTURE.md # AI/ML architecture details
│   └── README.md         # Documentation index
├── models/               # Trained model weights
├── logs/                 # AI decision logs
└── tests/                # Unit and integration tests
```

## 🚀 Quick Start

### Prerequisites

- Node.js 20+
- pnpm (recommended) or npm

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/weex-multi-agent-alpha.git
cd weex-multi-agent-alpha

# Install dependencies
pnpm install

# Configure environment
cp .env.example .env
# Edit .env with your WEEX API credentials
```

### Running

```bash
# Build the project
pnpm build

# Run in paper trading mode (default)
pnpm start

# Run in development mode
pnpm dev

# Run tests
pnpm test

# Lint code
pnpm lint
```

## 🤖 AI/ML Components

### Momentum Agent (LSTM)
- 2-layer LSTM neural network for trend prediction
- Input: 6 features × 100 timesteps
- Confidence calibration via Platt scaling

### Mean Reversion Agent (XGBoost + Bayesian)
- Random Forest regime classifier
- XGBoost parameter optimizer
- Bayesian probability estimation

### Volatility Agent (GARCH + CNN + RL)
- GARCH(1,1) volatility forecasting
- 1D CNN breakout pattern recognition
- Q-learning entry timing optimization

### Meta-Learner (Gradient Boosting)
- LightGBM signal combination
- Online gradient descent weight adaptation
- Kelly criterion position sizing

## 📊 Supported Trading Pairs

Per competition rules:
- BTC, ETH, BNB, XRP, ADA, SOL, LTC, DOGE

## ⚠️ Risk Management

| Rule | Limit |
|------|-------|
| Max Leverage | 20x |
| Position Size | 10% of capital |
| Daily Drawdown | 5% halt |
| Correlation Limit | 60% to correlated assets |
| Minimum Trades | 10 |

## 📝 Configuration

Key environment variables:

```bash
# WEEX API Credentials
WEEX_API_KEY=your_api_key
WEEX_API_SECRET=your_api_secret
WEEX_API_PASSPHRASE=your_passphrase

# Trading Settings
TRADING_MODE=paper          # paper or live
RISK_PER_TRADE=0.02         # 2% risk per trade

# Logging
LOG_LEVEL=info              # debug, info, warn, error
```

## 📄 Documentation

- [Trading Strategy](docs/STRATEGY.md) - Complete strategy documentation
- [AI Architecture](docs/AI_ARCHITECTURE.md) - ML model details
- [WEEX API Docs](https://www.weex.com/api-doc/ai/intro) - Official API reference

## 🏆 Competition

Built for **AI Wars: WEEX Alpha Awakens** hackathon.

- **Prize Pool**: $880,000 USDT
- **Duration**: November 2025 - March 2026
- **Link**: [DoraHacks Competition Page](https://dorahacks.io/hackathon/weex)

### Competition Compliance

- ✅ All trades use WEEX OpenAPI
- ✅ Maximum 20x leverage
- ✅ Approved trading pairs only
- ✅ Complete AI decision logging
- ✅ Minimum 10 trades requirement

## 🧪 Testing

```bash
# Run all tests
pnpm test

# Run with coverage
pnpm test -- --coverage

# Run specific test file
pnpm test -- tests/agents.test.ts
```

## 📈 Performance Tracking

The system tracks:
- Agent-specific win rates and Sharpe ratios
- Regime-specific performance breakdown
- Real-time PnL attribution
- Adaptive weight changes

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and competition purposes only. Trading cryptocurrencies involves significant risk. The authors are not responsible for any financial losses incurred through the use of this software.

---

Built with 🤖 for the WEEX AI Wars Hackathon

