# Multi-Agent Ensemble Trading Strategy

## Overview

This system employs a multi-agent architecture where specialized AI agents collaborate to make trading decisions. Each agent focuses on a specific market pattern, and a meta-learner combines their signals for optimal execution.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MARKET DATA                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                        │
│  • Technical indicators  • Normalized features  • Regime    │
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
│  • Signal combination (Gradient Boosting)                   │
│  • Adaptive weighting (Online Learning)                     │
│  • Position sizing (Kelly Criterion)                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 RISK MANAGEMENT                              │
│  • Max 20x leverage  • Position limits  • Drawdown control  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   EXECUTION                                  │
│  • WEEX API  • Order management  • AI logging               │
└─────────────────────────────────────────────────────────────┘
```

## Agent Descriptions

### 1. Momentum Agent (LSTM Neural Network)

**Purpose:** Predict short-term price momentum and trend continuation.

**AI/ML Components:**
- 2-layer LSTM network (64 hidden units)
- Input features: price returns, RSI, volume momentum, volatility
- Output: Momentum score (-1 to 1) with calibrated confidence

**Trading Logic:**
- Long signal when momentum > 0.6 with high confidence
- Short signal when momentum < -0.6 with high confidence
- Neutral otherwise

### 2. Mean Reversion Agent (XGBoost + Bayesian)

**Purpose:** Identify overextended moves and predict mean reversion.

**AI/ML Components:**
- Random Forest for regime classification
- XGBoost for dynamic parameter optimization
- Bayesian probability estimation for reversion likelihood

**Trading Logic:**
- Detect when price exceeds dynamically-adjusted Bollinger Bands
- Calculate probability of reversion within N candles
- Trade when probability exceeds threshold and regime is favorable

### 3. Volatility Agent (GARCH + CNN + RL)

**Purpose:** Predict volatility expansions and trade breakouts.

**AI/ML Components:**
- GARCH(1,1) for volatility forecasting
- 1D CNN for breakout pattern classification
- Q-learning agent for entry timing optimization

**Trading Logic:**
- Forecast next-period volatility using GARCH
- Classify breakout probability using CNN on price/volume patterns
- Time entry using RL-optimized policy

## Meta-Learner

**Signal Combination:**
A gradient-boosted model takes all agent signals plus market features as input and outputs optimal signal weights. The model is trained on historical data to learn which agents perform best under different conditions.

**Adaptive Weighting:**
Online gradient descent continuously updates agent weights based on recent performance. Agents that are performing well get higher weights; underperforming agents are downweighted.

**Position Sizing:**
Kelly Criterion applied to ensemble confidence score:
```
size = kelly_fraction * confidence * max_position
```

## Risk Management

- **Leverage:** Capped at 20x per competition rules
- **Position Size:** Max 10% of capital per trade
- **Daily Drawdown:** Trading halted at 5% daily loss
- **Correlation:** Max 60% exposure to correlated assets

## Trading Pairs

Limited to competition-approved pairs:
ADA, SOL, LTC, DOGE, BTC, ETH, XRP, BNB

## Differentiation

1. **Multi-agent diversity** - Captures different market patterns
2. **Adaptive meta-learning** - Adjusts to changing conditions
3. **Comprehensive AI stack** - LSTM, XGBoost, GARCH, CNN, RL
4. **Robust risk management** - Built for consistency over home runs

