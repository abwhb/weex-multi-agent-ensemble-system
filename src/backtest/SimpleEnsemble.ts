/**
 * @fileoverview Simplified ensemble for backtesting
 * @module backtest/SimpleEnsemble
 * 
 * A working implementation of the ensemble that uses direct indicator
 * calculations rather than complex agent abstractions. This allows
 * backtesting without full system initialization.
 */

import { MarketData, AllowedPair, Direction, OHLCV } from '../types';
import { 
  RSI, 
  MACD, 
  BollingerBands, 
  ATR, 
  EMA, 
  SMA,
  VolumeRatio,
  BBSqueeze,
  normalize,
  ADX
} from '../indicators';
import { TradeDecision } from '../meta-learner/Ensemble';

/**
 * Agent signal structure for internal use.
 */
interface SimpleSignal {
  direction: Direction;
  strength: number;
  confidence: number;
}

/**
 * Simple working ensemble for backtesting.
 * 
 * Implements the three-agent strategy using direct indicator calculations:
 * 1. Momentum - RSI, MACD, EMA crossovers
 * 2. Mean Reversion - Bollinger Bands, RSI extremes
 * 3. Volatility - ATR expansion, BB squeeze breakouts
 */
export class SimpleEnsemble {
  private minConfidence = 0.30; // Lowered for more activity
  private maxPositionSize = 0.10;
  
  // Agent weights (adapted over time)
  private weights = {
    momentum: 0.40,
    meanReversion: 0.35,
    volatility: 0.25
  };
  
  // Performance tracking for weight adaptation
  private recentResults: { agent: string; profit: number }[] = [];

  /**
   * Generate a trading decision.
   */
  async decide(marketData: MarketData, symbol: AllowedPair): Promise<TradeDecision | null> {
    const { ohlcv } = marketData;
    
    if (ohlcv.length < 50) {
      return null; // Not enough data
    }
    
    // Get signals from each "agent"
    const momentumSignal = this.getMomentumSignal(ohlcv);
    const reversionSignal = this.getMeanReversionSignal(ohlcv);
    const volatilitySignal = this.getVolatilitySignal(ohlcv);
    
    // Combine signals
    const combined = this.combineSignals(momentumSignal, reversionSignal, volatilitySignal);
    
    // Check if we should trade
    if (combined.confidence < this.minConfidence) {
      return null;
    }
    
    if (combined.direction === 'neutral') {
      return null;
    }
    
    // Calculate position size
    const size = this.calculatePositionSize(combined);
    
    // Calculate stop loss and take profit
    const currentPrice = ohlcv[ohlcv.length - 1].close;
    const atr = ATR(ohlcv, 14);
    
    const direction = combined.direction;
    const stopMultiplier = direction === 'long' ? -2 : 2;
    const tpMultiplier = direction === 'long' ? 3 : -3;
    
    const stopLoss = currentPrice + stopMultiplier * atr;
    const takeProfit = currentPrice + tpMultiplier * atr;
    
    // Generate reasoning
    const reasoning = this.generateReasoning(
      momentumSignal, 
      reversionSignal, 
      volatilitySignal,
      combined,
      symbol
    );
    
    return {
      action: direction === 'long' ? 'buy' : 'sell',
      symbol,
      size,
      confidence: combined.confidence,
      stopLoss,
      takeProfit,
      agentContributions: {
        momentum: momentumSignal.strength * this.weights.momentum,
        meanReversion: reversionSignal.strength * this.weights.meanReversion,
        volatility: volatilitySignal.strength * this.weights.volatility
      },
      reasoning,
      timestamp: Date.now(),
      currentPrice,
      regime: this.detectRegime(ohlcv)
    };
  }

  /**
   * Get momentum signal using RSI, MACD, and EMA.
   */
  private getMomentumSignal(ohlcv: OHLCV[]): SimpleSignal {
    const closes = ohlcv.map(c => c.close);
    const volumes = ohlcv.map(c => c.volume);
    
    // Calculate indicators
    const rsi = RSI(closes, 14);
    const macd = MACD(closes);
    const ema12 = EMA(closes, 12);
    const ema26 = EMA(closes, 26);
    const currentPrice = closes[closes.length - 1];
    const volumeRatio = VolumeRatio(volumes, 20);
    
    // RSI signal (-1 to 1)
    const rsiSignal = normalize(rsi - 50, 0.04);
    
    // MACD signal (-1 to 1)
    const macdSignal = normalize(macd.histogram, 0.01);
    
    // EMA crossover signal (-1 to 1)
    const emaDiff = (ema12 - ema26) / currentPrice;
    const emaSignal = normalize(emaDiff, 100);
    
    // Price momentum (returns)
    const returns5 = (closes[closes.length - 1] - closes[closes.length - 6]) / closes[closes.length - 6];
    const returns20 = (closes[closes.length - 1] - closes[closes.length - 21]) / closes[closes.length - 21];
    const returnSignal = normalize(returns5, 20) * 0.6 + normalize(returns20, 10) * 0.4;
    
    // Volume confirmation
    const volumeConfirmation = volumeRatio > 1.2 ? 1.2 : volumeRatio < 0.8 ? 0.8 : 1.0;
    
    // Combine components
    const rawStrength = (
      0.25 * rsiSignal +
      0.30 * macdSignal +
      0.25 * emaSignal +
      0.20 * returnSignal
    ) * volumeConfirmation;
    
    const strength = Math.max(-1, Math.min(1, rawStrength));
    
    // Determine direction (lowered threshold for more signals)
    let direction: Direction = 'neutral';
    if (strength > 0.15) direction = 'long';
    else if (strength < -0.15) direction = 'short';
    
    // Confidence based on signal agreement
    const signalAgreement = Math.abs(rsiSignal) > 0.3 && 
                           Math.sign(rsiSignal) === Math.sign(macdSignal) &&
                           Math.sign(rsiSignal) === Math.sign(emaSignal);
    const confidence = Math.abs(strength) * (signalAgreement ? 1.2 : 0.9);
    
    return {
      direction,
      strength,
      confidence: Math.min(1, confidence)
    };
  }

  /**
   * Get mean reversion signal using Bollinger Bands and RSI.
   */
  private getMeanReversionSignal(ohlcv: OHLCV[]): SimpleSignal {
    const closes = ohlcv.map(c => c.close);
    
    // Calculate indicators
    const bb = BollingerBands(closes, 20, 2);
    const rsi = RSI(closes, 14);
    const currentPrice = closes[closes.length - 1];
    
    // Check for regime (only trade reversion in ranging markets)
    const adx = ADX(ohlcv, 14);
    const isTrending = adx > 25;
    
    if (isTrending) {
      // Don't do mean reversion in trending markets
      return { direction: 'neutral', strength: 0, confidence: 0.3 };
    }
    
    // BB position (0 = lower band, 1 = upper band)
    const bbPosition = bb.percentB;
    
    // Check for extreme readings
    let direction: Direction = 'neutral';
    let strength = 0;
    
    if (bbPosition < 0.1 && rsi < 35) {
      // Oversold - expect reversion up
      direction = 'long';
      strength = (0.1 - bbPosition) * 10 + (35 - rsi) / 35;
    } else if (bbPosition > 0.9 && rsi > 65) {
      // Overbought - expect reversion down
      direction = 'short';
      strength = (bbPosition - 0.9) * 10 + (rsi - 65) / 35;
    }
    
    strength = Math.max(-1, Math.min(1, strength));
    
    // Confidence based on how extreme the reading is
    const extremeRSI = Math.abs(rsi - 50) / 50;
    const extremeBB = Math.abs(bbPosition - 0.5) * 2;
    const confidence = (extremeRSI + extremeBB) / 2;
    
    return {
      direction,
      strength,
      confidence: Math.min(1, confidence)
    };
  }

  /**
   * Get volatility/breakout signal.
   */
  private getVolatilitySignal(ohlcv: OHLCV[]): SimpleSignal {
    const closes = ohlcv.map(c => c.close);
    const volumes = ohlcv.map(c => c.volume);
    
    // Calculate indicators
    const bb = BollingerBands(closes, 20, 2);
    const atr = ATR(ohlcv, 14);
    const atr50 = ATR(ohlcv.slice(-50), 14);
    const currentPrice = closes[closes.length - 1];
    const volumeRatio = VolumeRatio(volumes, 20);
    
    // Check for squeeze (low volatility consolidation)
    const isSqueeze = BBSqueeze(closes, 20, 20);
    
    // Check for ATR expansion (breakout)
    const atrExpansion = atr50 > 0 ? atr / atr50 : 1;
    const isExpanding = atrExpansion > 1.3;
    
    // Volume confirmation
    const hasVolumeSpike = volumeRatio > 1.5;
    
    let direction: Direction = 'neutral';
    let strength = 0;
    let confidence = 0;
    
    // Breakout conditions
    if (isExpanding && hasVolumeSpike) {
      // Determine direction based on price relative to bands
      if (currentPrice > bb.upper) {
        direction = 'long';
        strength = Math.min(1, (currentPrice - bb.upper) / (bb.upper - bb.middle) + 0.5);
      } else if (currentPrice < bb.lower) {
        direction = 'short';
        strength = Math.min(1, (bb.lower - currentPrice) / (bb.middle - bb.lower) + 0.5);
      }
      
      confidence = Math.min(1, (atrExpansion - 1) * 2 + volumeRatio * 0.2);
    } else if (isSqueeze) {
      // In squeeze - prepare for breakout but don't trade yet
      direction = 'neutral';
      strength = 0;
      confidence = 0.3;
    }
    
    return {
      direction,
      strength,
      confidence
    };
  }

  /**
   * Combine signals from all agents.
   */
  private combineSignals(
    momentum: SimpleSignal,
    reversion: SimpleSignal,
    volatility: SimpleSignal
  ): { direction: Direction; strength: number; confidence: number } {
    // Weight-adjusted signal combination
    const weightedStrength = 
      momentum.strength * this.weights.momentum +
      reversion.strength * this.weights.meanReversion +
      volatility.strength * this.weights.volatility;
    
    // Conflict detection - if agents disagree, reduce confidence
    const directions = [
      { dir: momentum.direction, conf: momentum.confidence * this.weights.momentum },
      { dir: reversion.direction, conf: reversion.confidence * this.weights.meanReversion },
      { dir: volatility.direction, conf: volatility.confidence * this.weights.volatility }
    ].filter(d => d.dir !== 'neutral');
    
    const longWeight = directions.filter(d => d.dir === 'long').reduce((s, d) => s + d.conf, 0);
    const shortWeight = directions.filter(d => d.dir === 'short').reduce((s, d) => s + d.conf, 0);
    
    let direction: Direction = 'neutral';
    let confidence = 0;
    
    if (longWeight > shortWeight && longWeight > 0.2) {
      direction = 'long';
      confidence = longWeight / (longWeight + shortWeight + 0.1);
    } else if (shortWeight > longWeight && shortWeight > 0.2) {
      direction = 'short';
      confidence = shortWeight / (longWeight + shortWeight + 0.1);
    }
    
    // Boost confidence if multiple agents agree
    const agreeingAgents = directions.filter(d => d.dir === direction).length;
    if (agreeingAgents >= 2) {
      confidence *= 1.15;
    }
    
    return {
      direction,
      strength: weightedStrength,
      confidence: Math.min(1, confidence)
    };
  }

  /**
   * Calculate position size using simplified Kelly criterion.
   */
  private calculatePositionSize(signal: { direction: Direction; strength: number; confidence: number }): number {
    // Simplified Kelly: f = edge * confidence
    // Where edge is estimated from signal strength
    const winProb = 0.5 + signal.confidence * 0.2; // 50-70% estimated win rate
    const avgWinLoss = 1.5; // Target 1.5:1 risk-reward
    
    // Kelly formula
    const kelly = (winProb * avgWinLoss - (1 - winProb)) / avgWinLoss;
    
    // Apply half-Kelly for safety and scale by confidence
    const size = Math.max(0, kelly * 0.5 * signal.confidence);
    
    // Cap at max position size
    return Math.min(size, this.maxPositionSize);
  }

  /**
   * Detect market regime.
   */
  private detectRegime(ohlcv: OHLCV[]): 'trending' | 'ranging' | 'volatile' | 'quiet' {
    const closes = ohlcv.map(c => c.close);
    const adx = ADX(ohlcv, 14);
    const bb = BollingerBands(closes, 20, 2);
    
    if (adx > 25) {
      return 'trending';
    } else if (bb.width > 0.05) {
      return 'volatile';
    } else if (bb.width < 0.02) {
      return 'quiet';
    } else {
      return 'ranging';
    }
  }

  /**
   * Generate reasoning for the decision.
   */
  private generateReasoning(
    momentum: SimpleSignal,
    reversion: SimpleSignal,
    volatility: SimpleSignal,
    combined: { direction: Direction; strength: number; confidence: number },
    symbol: string
  ): string {
    const parts: string[] = [];
    
    parts.push(`[${symbol}] Ensemble decision: ${combined.direction.toUpperCase()}`);
    parts.push(`Confidence: ${(combined.confidence * 100).toFixed(1)}%`);
    
    if (momentum.direction !== 'neutral') {
      parts.push(`Momentum: ${momentum.direction} (${(momentum.confidence * 100).toFixed(0)}%)`);
    }
    if (reversion.direction !== 'neutral') {
      parts.push(`Reversion: ${reversion.direction} (${(reversion.confidence * 100).toFixed(0)}%)`);
    }
    if (volatility.direction !== 'neutral') {
      parts.push(`Volatility: ${volatility.direction} (${(volatility.confidence * 100).toFixed(0)}%)`);
    }
    
    return parts.join(' | ');
  }

  /**
   * Update weights based on trade result (for online learning).
   */
  updateWeights(agentContributions: Record<string, number>, pnl: number): void {
    const learningRate = 0.05;
    
    // Simple gradient-based update
    for (const [agent, contribution] of Object.entries(agentContributions)) {
      if (agent in this.weights) {
        const key = agent as keyof typeof this.weights;
        // Increase weight if agent contributed to profitable trade
        const update = learningRate * contribution * Math.sign(pnl) * 0.1;
        this.weights[key] = Math.max(0.1, Math.min(0.6, this.weights[key] + update));
      }
    }
    
    // Normalize weights
    const totalWeight = Object.values(this.weights).reduce((a, b) => a + b, 0);
    for (const key of Object.keys(this.weights) as Array<keyof typeof this.weights>) {
      this.weights[key] /= totalWeight;
    }
  }
}

