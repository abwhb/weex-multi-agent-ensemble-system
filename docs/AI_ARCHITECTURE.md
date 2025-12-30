# AI/ML Architecture Documentation

## Overview

This document details how Artificial Intelligence and Machine Learning drive all trading decisions in our multi-agent system. Every trade is the result of AI model inference, not manual intervention.

## AI Components by Module

### 1. Momentum Agent - LSTM Neural Network

**Model Architecture:**
```
Input Layer: 6 features × 100 timesteps
    ↓
LSTM Layer 1: 64 units, return_sequences=True
    ↓
Dropout: 0.2
    ↓
LSTM Layer 2: 64 units
    ↓
Dropout: 0.2
    ↓
Dense Layer: 32 units, ReLU
    ↓
Output Layer: 1 unit, tanh activation
```

**Input Features:**
| Feature | Description | Normalization |
|---------|-------------|---------------|
| price_returns_5m | 5-minute returns | Z-score |
| price_returns_15m | 15-minute returns | Z-score |
| price_returns_1h | 1-hour returns | Z-score |
| rsi_14 | 14-period RSI | Min-max to [0,1] |
| volume_ratio | Volume vs 20-period MA | Log transform |
| volatility_20 | 20-period realized vol | Z-score |

**Training:**
- Dataset: 2 years of minute-level data per pair
- Loss: MSE on next-period return prediction
- Optimizer: Adam, lr=0.001
- Early stopping on validation loss

**Confidence Calibration:**
Platt scaling applied to raw predictions using held-out calibration set.

---

### 2. Mean Reversion Agent - Ensemble ML

**Regime Classifier (Random Forest):**
```
Input: 15 market features
Trees: 100
Max Depth: 10
Output: [trending, ranging, volatile]
```

**Parameter Optimizer (XGBoost):**
```
Input: Regime + 20 market features
Output: Optimal (BB_period, BB_stddev)
Trained on: Historical performance by parameter set
```

**Reversion Probability (Bayesian):**
```
Prior: Beta(2, 2)
Likelihood: Historical reversion rate given deviation size
Posterior: Updated probability of reversion within N candles
```

---

### 3. Volatility Agent - GARCH + Deep Learning + RL

**Volatility Forecast (GARCH(1,1)):**
```
σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)

Parameters fitted via MLE on rolling 500-period window
Outputs: Point forecast + 95% confidence interval
```

**Breakout Classifier (1D CNN):**
```
Input: 50 candles × 5 features (OHLCV)
    ↓
Conv1D: 32 filters, kernel=3, ReLU
    ↓
MaxPool1D: pool_size=2
    ↓
Conv1D: 64 filters, kernel=3, ReLU
    ↓
GlobalMaxPool1D
    ↓
Dense: 32 units, ReLU
    ↓
Output: sigmoid (breakout probability)
```

**Entry Timing (Q-Learning):**
```
State: [volatility_forecast, breakout_prob, price_position, time_since_signal]
Actions: [enter_now, wait_1_candle, wait_3_candles, skip]
Reward: Trade PnL if entered, 0 if skipped
Learning: DQN with experience replay
```

---

### 4. Meta-Learner - Gradient Boosting

**Signal Combiner (LightGBM):**
```
Input Features:
- momentum_signal, momentum_confidence
- reversion_signal, reversion_confidence  
- volatility_signal, volatility_confidence
- market_regime
- cross_asset_correlation
- recent_agent_performance (3 features)

Output: Optimal position direction and size

Trained on: Historical agent signals → actual returns
```

**Online Weight Adaptation:**
```
For each completed trade:
  1. Calculate PnL attribution to each agent
  2. Update agent weight: w(t+1) = w(t) + η·gradient
  3. Normalize weights to sum to 1
```

---

## AI Decision Flow

```
1. Market data arrives
     ↓
2. Feature engineering extracts normalized inputs
     ↓
3. Each agent runs inference:
   - MomentumAgent.predict() → LSTM forward pass
   - MeanReversionAgent.predict() → RF + XGB + Bayesian
   - VolatilityAgent.predict() → GARCH + CNN + DQN
     ↓
4. Meta-learner combines signals:
   - LightGBM inference on agent outputs
   - Apply online-learned weights
     ↓
5. Position sizing:
   - Kelly criterion with ensemble confidence
     ↓
6. Risk check:
   - Validate against all risk rules
     ↓
7. Execute via WEEX API
     ↓
8. Log AI decision (model inputs, outputs, reasoning)
```

---

## Model Versioning

All models are versioned and tracked:
```
models/
├── momentum/
│   └── lstm_v1.0.0.pt
├── mean_reversion/
│   ├── regime_classifier_v1.0.0.pkl
│   └── param_optimizer_v1.0.0.pkl
├── volatility/
│   ├── breakout_cnn_v1.0.0.pt
│   └── entry_dqn_v1.0.0.pt
└── meta/
    └── signal_combiner_v1.0.0.pkl
```

---

## AI Logging Format

Every decision is logged with full AI traceability:

```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "model_versions": {
    "momentum_lstm": "1.0.0",
    "regime_classifier": "1.0.0",
    "breakout_cnn": "1.0.0",
    "meta_learner": "1.0.0"
  },
  "inputs": {
    "symbol": "BTC",
    "features": [0.12, -0.05, 0.23, ...],
    "regime": "trending"
  },
  "agent_outputs": {
    "momentum": {"signal": 0.72, "confidence": 0.85},
    "reversion": {"signal": -0.15, "confidence": 0.45},
    "volatility": {"signal": 0.60, "confidence": 0.70}
  },
  "meta_output": {
    "direction": "long",
    "size": 0.05,
    "confidence": 0.78
  },
  "reasoning": "Strong momentum signal with high confidence. Volatility agent confirms breakout. Mean reversion agent neutral. Meta-learner weighted toward momentum (0.45) and volatility (0.35) based on recent performance in trending regime.",
  "order_id": "ord_abc123"
}
```

---

## Why This Architecture?

1. **Genuine AI-driven decisions** - Every trade results from model inference
2. **Diverse signal sources** - Captures momentum, reversion, and volatility patterns
3. **Adaptive combination** - Meta-learner adjusts to market conditions
4. **Full traceability** - Complete audit log for competition compliance
5. **Production-ready design** - Clean separation of concerns, versioned models

