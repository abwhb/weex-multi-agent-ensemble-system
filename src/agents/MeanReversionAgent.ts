/**
 * @fileoverview Statistical mean reversion agent with ML-tuned parameters
 * @module agents/MeanReversionAgent
 * 
 * ## Overview
 * 
 * The MeanReversionAgent identifies overextended price moves and predicts
 * mean reversion opportunities. It combines traditional statistical methods
 * with machine learning for parameter optimization and probability estimation.
 * 
 * ## AI/ML Architecture
 * 
 * ### 1. Regime Classifier (Random Forest)
 * 
 * Classifies current market conditions to determine if mean reversion is viable:
 * - **Trending**: Avoid mean reversion (trends can extend)
 * - **Ranging**: Ideal for mean reversion strategies
 * - **Volatile**: Possible but with wider bands
 * 
 * ```
 * Input: 15 market features (momentum, volatility, volume, orderbook)
 * Model: Random Forest, 100 trees, max_depth=10
 * Output: Probability distribution over regimes
 * ```
 * 
 * ### 2. Parameter Optimizer (XGBoost)
 * 
 * Dynamically adjusts Bollinger Band parameters based on regime:
 * 
 * ```
 * Input: Regime + 20 market features
 * Output: Optimal (BB_period, BB_stddev)
 * Trained on: Historical performance by parameter set
 * ```
 * 
 * ### 3. Reversion Probability Estimator (Bayesian)
 * 
 * Calculates probability of price reverting to mean within N candles:
 * 
 * ```
 * Prior: Beta(2, 2) - weakly informative
 * Likelihood: Historical reversion rate given deviation size
 * Posterior: Updated probability of reversion
 * ```
 * 
 * ## Trading Logic
 * 
 * 1. Check if regime is favorable for mean reversion
 * 2. Calculate deviation from dynamically-adjusted Bollinger Bands
 * 3. Estimate probability of reversion
 * 4. Trade when probability exceeds threshold and deviation is significant
 * 
 * ## Market Regime Suitability
 * 
 * Best performance in ranging/sideways markets where price oscillates
 * around a mean. Performance degrades in strong trending conditions.
 */

import { BaseAgent, AgentSignal, AgentConfig } from './BaseAgent';
import { MarketData, MarketRegime, OHLCV } from '../types';
import { RSI, BollingerBands, ATR, ADX, VolumeRatio, SMA, normalize } from '../indicators';

// ============================================================================
// Types
// ============================================================================

/**
 * Extended configuration for MeanReversionAgent.
 */
export interface MeanReversionAgentConfig extends AgentConfig {
  /**
   * Default Bollinger Band period (will be optimized by ML).
   * @default 20
   */
  defaultBBPeriod?: number;

  /**
   * Default Bollinger Band standard deviation multiplier.
   * @default 2.0
   */
  defaultBBStdDev?: number;

  /**
   * Minimum probability of reversion to generate signal.
   * @default 0.65
   */
  minReversionProbability?: number;

  /**
   * Maximum candles to wait for reversion.
   * @default 12
   */
  reversionHorizon?: number;
}

/**
 * Regime classification output from Random Forest.
 */
interface RegimeClassification {
  regime: MarketRegime;
  probabilities: Record<MarketRegime, number>;
  features: number[];
}

/**
 * Optimized Bollinger Band parameters from XGBoost.
 */
interface OptimizedBBParams {
  period: number;
  stdDevMultiplier: number;
  confidence: number;
}

/**
 * Bayesian reversion probability estimate.
 */
interface ReversionEstimate {
  probability: number;
  expectedCandles: number;
  priorStrength: number;
}

/**
 * Bollinger Band values at current price.
 */
interface BollingerBands {
  upper: number;
  middle: number;
  lower: number;
  bandwidth: number;
  percentB: number; // Position within bands (0 = lower, 1 = upper)
}

// ============================================================================
// MeanReversionAgent Implementation
// ============================================================================

/**
 * Statistical mean reversion agent with ML-tuned parameters.
 * 
 * This agent combines classical technical analysis (Bollinger Bands) with
 * machine learning to optimize parameters and estimate reversion probability.
 * 
 * ## Key Capabilities
 * 
 * 1. **Regime Awareness**: Random Forest classifier determines if current
 *    market conditions favor mean reversion strategies.
 * 
 * 2. **Dynamic Parameters**: XGBoost model optimizes BB parameters based
 *    on current market characteristics, avoiding fixed-parameter pitfalls.
 * 
 * 3. **Probabilistic Trading**: Bayesian estimation provides calibrated
 *    probability of reversion, enabling risk-adjusted position sizing.
 * 
 * 4. **Volume Confirmation**: Incorporates volume profile analysis to
 *    confirm mean reversion setups.
 * 
 * @extends BaseAgent
 * 
 * @example
 * ```typescript
 * const agent = new MeanReversionAgent({
 *   lookbackPeriod: 50,
 *   threshold: 0.6,
 *   modelPath: 'models/mean_reversion/',
 *   minReversionProbability: 0.65
 * });
 * 
 * await agent.initialize();
 * const signal = await agent.analyze(marketData);
 * ```
 */
export class MeanReversionAgent extends BaseAgent {
  /**
   * Extended configuration.
   */
  private mrConfig: MeanReversionAgentConfig;

  /**
   * Random Forest regime classifier placeholder.
   * @placeholder Actual implementation would use scikit-learn export or similar
   */
  private regimeClassifier: unknown = null;

  /**
   * XGBoost parameter optimizer placeholder.
   * @placeholder Actual implementation would use XGBoost JS bindings
   */
  private paramOptimizer: unknown = null;

  /**
   * Bayesian prior parameters for reversion estimation.
   * Beta(alpha, beta) prior on reversion probability.
   */
  private betaPrior: { alpha: number; beta: number };

  /**
   * Historical reversion statistics by deviation bucket.
   * Used to update Bayesian posterior.
   */
  private reversionHistory: Map<string, { successes: number; trials: number }>;

  /**
   * Creates a new MeanReversionAgent.
   * 
   * @param config - Agent configuration
   * @param initialWeight - Starting weight in ensemble
   */
  constructor(config: MeanReversionAgentConfig, initialWeight: number = 0.33) {
    super('MeanReversionAgent', config, initialWeight);
    
    this.mrConfig = {
      defaultBBPeriod: 20,
      defaultBBStdDev: 2.0,
      minReversionProbability: 0.65,
      reversionHorizon: 12,
      ...config
    };

    // Weakly informative prior: Beta(2, 2) centered at 0.5
    this.betaPrior = { alpha: 2, beta: 2 };
    
    this.reversionHistory = new Map();
  }

  // ==========================================================================
  // Core Analysis Method
  // ==========================================================================

  /**
   * Analyze market data for mean reversion opportunities.
   * 
   * ## Process Flow
   * 
   * 1. **Regime Detection**: Classify market as trending/ranging/volatile
   * 2. **Parameter Optimization**: Get optimal BB parameters for regime
   * 3. **Deviation Calculation**: Measure price distance from mean
   * 4. **Probability Estimation**: Bayesian estimate of reversion
   * 5. **Signal Generation**: Combine all factors into trading signal
   * 
   * @param marketData - Current market data
   * @returns Trading signal with confidence and reasoning
   */
  async analyze(marketData: MarketData): Promise<AgentSignal> {
    this.ensureInitialized();

    try {
      // Step 1: Detect market regime
      const regimeResult = await this.detectRegime(marketData);
      
      // Step 2: Check if regime favors mean reversion
      if (regimeResult.regime === 'trending') {
        return this.createNeutralSignal(
          `Trending market detected (${(regimeResult.probabilities.trending * 100).toFixed(1)}% confidence). Mean reversion not recommended.`
        );
      }

      // Step 3: Optimize Bollinger Band parameters
      const bbParams = await this.optimizeParameters(regimeResult, marketData);

      // Step 4: Calculate Bollinger Bands with optimized parameters
      const bands = this.calculateBollingerBands(
        marketData.ohlcv,
        bbParams.period,
        bbParams.stdDevMultiplier
      );

      // Step 5: Check if price is at an extreme
      const deviation = this.calculateDeviation(bands);
      
      if (Math.abs(deviation) < 0.8) {
        return this.createNeutralSignal(
          `Price within normal range (deviation: ${deviation.toFixed(2)}). No mean reversion opportunity.`
        );
      }

      // Step 6: Calculate reversion probability
      const reversionEstimate = this.calculateReversionProbability(
        deviation,
        regimeResult.regime,
        marketData
      );

      // Step 7: Determine direction (opposite to current deviation)
      const direction = deviation > 0 ? 'short' : 'long';

      // Step 8: Calculate confidence
      const confidence = this.calculateConfidence(
        reversionEstimate,
        regimeResult,
        bbParams
      );

      // Step 9: Check minimum probability threshold
      if (reversionEstimate.probability < this.mrConfig.minReversionProbability!) {
        return this.createNeutralSignal(
          `Reversion probability (${(reversionEstimate.probability * 100).toFixed(1)}%) below threshold.`
        );
      }

      // Step 10: Calculate suggested size and generate reasoning
      const suggestedSize = this.calculateSuggestedSize(confidence, reversionEstimate);
      const reasoning = this.generateReasoning(
        deviation,
        reversionEstimate,
        regimeResult,
        bands,
        marketData
      );

      return {
        direction,
        confidence,
        suggestedSize,
        reasoning,
        timestamp: Date.now(),
        rawScore: reversionEstimate.probability,
        regime: regimeResult.regime
      };
    } catch (error) {
      return this.createNeutralSignal(
        `MeanReversionAgent error: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  // ==========================================================================
  // Regime Classification (Random Forest)
  // ==========================================================================

  /**
   * Classify current market regime using Random Forest.
   * 
   * ## AI/ML Details
   * 
   * The Random Forest classifier uses 15 features:
   * - Trend indicators: SMA slopes, ADX, directional movement
   * - Volatility indicators: ATR, BB width, historical vol
   * - Volume indicators: Volume momentum, OBV trend
   * - Mean reversion indicators: RSI extremes, distance from MA
   * 
   * @param marketData - Market data for classification
   * @returns Regime classification with probabilities
   */
  private async detectRegime(marketData: MarketData): Promise<RegimeClassification> {
    const features = this.extractRegimeFeatures(marketData);
    
    // =========================================================================
    // PLACEHOLDER: Random Forest Inference
    // =========================================================================
    // 
    // In production:
    // const probabilities = this.regimeClassifier.predictProba(features);
    // const regime = this.regimeClassifier.predict(features);
    // =========================================================================

    // Simplified regime detection based on features
    const { ohlcv } = marketData;
    const closes = ohlcv.slice(-50).map(c => c.close);
    
    // Calculate trend strength
    const trendStrength = this.calculateTrendStrength(closes);
    
    // Calculate volatility
    const volatility = this.calculateVolatilityMetric(ohlcv.slice(-20));
    
    // Determine regime
    let regime: MarketRegime;
    const probabilities: Record<MarketRegime, number> = {
      trending: 0,
      ranging: 0,
      volatile: 0,
      quiet: 0
    };

    if (volatility > 0.03) {
      regime = 'volatile';
      probabilities.volatile = 0.7;
      probabilities.trending = 0.2;
      probabilities.ranging = 0.1;
    } else if (trendStrength > 0.6) {
      regime = 'trending';
      probabilities.trending = 0.75;
      probabilities.ranging = 0.15;
      probabilities.volatile = 0.1;
    } else if (volatility < 0.01) {
      regime = 'quiet';
      probabilities.quiet = 0.6;
      probabilities.ranging = 0.3;
      probabilities.trending = 0.1;
    } else {
      regime = 'ranging';
      probabilities.ranging = 0.65;
      probabilities.trending = 0.2;
      probabilities.volatile = 0.15;
    }

    return { regime, probabilities, features };
  }

  /**
   * Extract features for regime classification.
   * 
   * @param marketData - Market data
   * @returns Feature vector for Random Forest
   */
  private extractRegimeFeatures(marketData: MarketData): number[] {
    const { ohlcv } = marketData;
    const closes = ohlcv.slice(-50).map(c => c.close);
    const volumes = ohlcv.slice(-50).map(c => c.volume);
    
    // Feature extraction (15 features)
    const features: number[] = [];
    
    // Trend features (5)
    features.push(this.calculateTrendStrength(closes));
    features.push(this.calculateSMASlope(closes, 10));
    features.push(this.calculateSMASlope(closes, 20));
    features.push(this.calculateADX(ohlcv.slice(-20)));
    features.push(closes[closes.length - 1] / closes[0] - 1); // Period return
    
    // Volatility features (5)
    features.push(this.calculateVolatilityMetric(ohlcv.slice(-20)));
    features.push(this.calculateATR(ohlcv.slice(-14)));
    features.push(this.calculateBollingerWidth(closes, 20, 2));
    features.push(this.calculateVolatilityRatio(ohlcv.slice(-50))); // Short vs long vol
    features.push(this.calculateMaxDrawdown(closes));
    
    // Volume features (3)
    features.push(this.calculateVolumeMA(volumes, 10) / this.calculateVolumeMA(volumes, 30));
    features.push(this.calculateOBVTrend(ohlcv.slice(-20)));
    features.push(this.calculateVolumeVolatility(volumes));
    
    // Mean reversion features (2)
    features.push(this.calculateRSI(ohlcv.slice(-14)));
    features.push(this.calculateDistanceFromMA(closes, 20));

    return features;
  }

  // ==========================================================================
  // Parameter Optimization (XGBoost)
  // ==========================================================================

  /**
   * Optimize Bollinger Band parameters using XGBoost.
   * 
   * ## AI/ML Details
   * 
   * The XGBoost model predicts optimal (period, stdDev) based on:
   * - Current market regime
   * - Volatility characteristics
   * - Recent mean reversion success rates
   * 
   * Trained on historical data where each (regime, features) → (params)
   * mapping was evaluated by backtested performance.
   * 
   * @param regime - Current regime classification
   * @param marketData - Market data
   * @returns Optimized BB parameters
   */
  private async optimizeParameters(
    regime: RegimeClassification,
    marketData: MarketData
  ): Promise<OptimizedBBParams> {
    // =========================================================================
    // PLACEHOLDER: XGBoost Parameter Optimization
    // =========================================================================
    // 
    // In production:
    // const input = [...regime.features, ...additionalFeatures];
    // const [period, stdDev, confidence] = this.paramOptimizer.predict(input);
    // =========================================================================

    // Heuristic parameter selection based on regime
    let period: number;
    let stdDevMultiplier: number;
    let confidence: number;

    switch (regime.regime) {
      case 'ranging':
        // Tighter bands in ranging markets
        period = 15;
        stdDevMultiplier = 1.8;
        confidence = 0.8;
        break;
      case 'volatile':
        // Wider bands in volatile markets
        period = 25;
        stdDevMultiplier = 2.5;
        confidence = 0.6;
        break;
      case 'quiet':
        // Standard bands in quiet markets
        period = 20;
        stdDevMultiplier = 2.0;
        confidence = 0.7;
        break;
      default:
        // Default parameters
        period = this.mrConfig.defaultBBPeriod!;
        stdDevMultiplier = this.mrConfig.defaultBBStdDev!;
        confidence = 0.5;
    }

    return { period, stdDevMultiplier, confidence };
  }

  // ==========================================================================
  // Bollinger Band Calculations
  // ==========================================================================

  /**
   * Calculate Bollinger Bands with given parameters.
   * 
   * @param ohlcv - OHLCV data
   * @param period - Moving average period
   * @param stdDevMultiplier - Standard deviation multiplier
   * @returns Bollinger Band values
   */
  private calculateBollingerBands(
    ohlcv: OHLCV[],
    period: number,
    stdDevMultiplier: number
  ): BollingerBands {
    const closes = ohlcv.slice(-period).map(c => c.close);
    const currentClose = closes[closes.length - 1];
    
    // Calculate middle band (SMA)
    const middle = closes.reduce((a, b) => a + b, 0) / closes.length;
    
    // Calculate standard deviation
    const variance = closes.reduce((sum, c) => sum + Math.pow(c - middle, 2), 0) / closes.length;
    const stdDev = Math.sqrt(variance);
    
    // Calculate bands
    const upper = middle + stdDevMultiplier * stdDev;
    const lower = middle - stdDevMultiplier * stdDev;
    const bandwidth = (upper - lower) / middle;
    
    // Calculate %B (position within bands)
    const percentB = (currentClose - lower) / (upper - lower);

    return { upper, middle, lower, bandwidth, percentB };
  }

  /**
   * Calculate deviation from mean in standard deviation units.
   * 
   * @param bands - Bollinger Band values
   * @returns Deviation score (-2 to 2 typical range)
   */
  private calculateDeviation(bands: BollingerBands): number {
    // Convert percentB to z-score-like deviation
    // percentB of 0 = at lower band = -2 std dev (approximately)
    // percentB of 1 = at upper band = +2 std dev (approximately)
    // percentB of 0.5 = at middle = 0 std dev
    return (bands.percentB - 0.5) * 4;
  }

  // ==========================================================================
  // Bayesian Reversion Probability
  // ==========================================================================

  /**
   * Calculate probability of price reverting to mean using Bayesian estimation.
   * 
   * ## Bayesian Approach
   * 
   * We maintain a Beta prior on reversion probability and update it
   * with observed reversion outcomes:
   * 
   * ```
   * Prior: Beta(α, β)
   * After observing s successes in n trials:
   * Posterior: Beta(α + s, β + n - s)
   * Expected probability: (α + s) / (α + β + n)
   * ```
   * 
   * The prior is updated separately for different deviation buckets
   * to capture that larger deviations may have different reversion rates.
   * 
   * @param deviation - Current deviation from mean
   * @param regime - Market regime
   * @param marketData - Market data
   * @returns Reversion probability estimate
   */
  private calculateReversionProbability(
    deviation: number,
    regime: MarketRegime,
    marketData: MarketData
  ): ReversionEstimate {
    const deviationBucket = this.getDeviationBucket(deviation);
    const historyKey = `${regime}_${deviationBucket}`;
    
    // Get historical data for this bucket
    const history = this.reversionHistory.get(historyKey) || { successes: 0, trials: 0 };
    
    // Calculate posterior probability
    const posteriorAlpha = this.betaPrior.alpha + history.successes;
    const posteriorBeta = this.betaPrior.beta + (history.trials - history.successes);
    const probability = posteriorAlpha / (posteriorAlpha + posteriorBeta);
    
    // Adjust for regime
    let regimeMultiplier = 1.0;
    if (regime === 'ranging') {
      regimeMultiplier = 1.2; // Higher reversion rate in ranging markets
    } else if (regime === 'volatile') {
      regimeMultiplier = 0.9; // Slightly lower in volatile markets
    }
    
    // Adjust for deviation magnitude
    const absDeviation = Math.abs(deviation);
    let deviationMultiplier = 1.0;
    if (absDeviation > 2.5) {
      // Extreme deviations more likely to revert
      deviationMultiplier = 1.1;
    } else if (absDeviation < 1.0) {
      // Small deviations less reliable
      deviationMultiplier = 0.8;
    }
    
    const adjustedProbability = Math.min(0.95, probability * regimeMultiplier * deviationMultiplier);
    
    // Estimate expected candles to reversion
    const expectedCandles = Math.round(this.mrConfig.reversionHorizon! * (1 - adjustedProbability + 0.5));
    
    // Prior strength indicates how much data we have
    const priorStrength = Math.min(1, (history.trials + this.betaPrior.alpha + this.betaPrior.beta) / 50);

    return {
      probability: adjustedProbability,
      expectedCandles,
      priorStrength
    };
  }

  /**
   * Bucket deviations for Bayesian tracking.
   * 
   * @param deviation - Deviation value
   * @returns Bucket identifier
   */
  private getDeviationBucket(deviation: number): string {
    const absDeviation = Math.abs(deviation);
    if (absDeviation < 1.0) return 'small';
    if (absDeviation < 1.5) return 'medium';
    if (absDeviation < 2.0) return 'large';
    return 'extreme';
  }

  /**
   * Update reversion history with trade outcome.
   * Call this after a mean reversion trade closes.
   * 
   * @param deviation - Original deviation at entry
   * @param regime - Regime at entry
   * @param success - Whether reversion occurred
   */
  public updateReversionHistory(
    deviation: number,
    regime: MarketRegime,
    success: boolean
  ): void {
    const bucket = this.getDeviationBucket(deviation);
    const key = `${regime}_${bucket}`;
    
    const current = this.reversionHistory.get(key) || { successes: 0, trials: 0 };
    this.reversionHistory.set(key, {
      successes: current.successes + (success ? 1 : 0),
      trials: current.trials + 1
    });
  }

  // ==========================================================================
  // Confidence & Sizing
  // ==========================================================================

  /**
   * Calculate overall confidence from multiple factors.
   * 
   * @param reversion - Reversion probability estimate
   * @param regime - Regime classification
   * @param bbParams - Optimized BB parameters
   * @returns Combined confidence score
   */
  private calculateConfidence(
    reversion: ReversionEstimate,
    regime: RegimeClassification,
    bbParams: OptimizedBBParams
  ): number {
    // Weight contributions from different factors
    const reversionWeight = 0.5;
    const regimeWeight = 0.3;
    const paramWeight = 0.2;
    
    const reversionConfidence = reversion.probability * reversion.priorStrength;
    const regimeConfidence = 1 - regime.probabilities.trending; // Inverse of trending
    const paramConfidence = bbParams.confidence;
    
    return (
      reversionWeight * reversionConfidence +
      regimeWeight * regimeConfidence +
      paramWeight * paramConfidence
    );
  }

  /**
   * Calculate suggested position size.
   * 
   * @param confidence - Overall confidence
   * @param reversion - Reversion estimate
   * @returns Suggested position size
   */
  private calculateSuggestedSize(
    confidence: number,
    reversion: ReversionEstimate
  ): number {
    const baseSize = 0.02;
    const maxSize = 0.08; // Conservative for mean reversion
    
    // Scale by confidence and probability
    const scaleFactor = confidence * reversion.probability;
    
    // Reduce size if prior is weak (less data)
    const priorPenalty = 0.5 + 0.5 * reversion.priorStrength;
    
    return Math.min(maxSize, baseSize + (maxSize - baseSize) * scaleFactor * priorPenalty);
  }

  // ==========================================================================
  // Helper Methods
  // ==========================================================================

  private calculateTrendStrength(closes: number[]): number {
    const n = closes.length;
    const sumX = (n * (n - 1)) / 2;
    const sumY = closes.reduce((a, b) => a + b, 0);
    const sumXY = closes.reduce((sum, y, i) => sum + i * y, 0);
    const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6;
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const avgPrice = sumY / n;
    
    return Math.min(1, Math.abs(slope / avgPrice) * 100);
  }

  private calculateSMASlope(closes: number[], period: number): number {
    const sma1 = closes.slice(-period).reduce((a, b) => a + b, 0) / period;
    const sma2 = closes.slice(-period * 2, -period).reduce((a, b) => a + b, 0) / period;
    return (sma1 - sma2) / sma2;
  }

  private calculateADX(ohlcv: OHLCV[]): number {
    // Simplified ADX calculation
    let sumDM = 0;
    let sumTR = 0;
    for (let i = 1; i < ohlcv.length; i++) {
      const high = ohlcv[i].high;
      const low = ohlcv[i].low;
      const prevHigh = ohlcv[i - 1].high;
      const prevLow = ohlcv[i - 1].low;
      const prevClose = ohlcv[i - 1].close;
      
      sumDM += Math.abs((high - prevHigh) - (prevLow - low));
      sumTR += Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
    }
    return sumTR > 0 ? sumDM / sumTR : 0;
  }

  private calculateVolatilityMetric(ohlcv: OHLCV[]): number {
    const returns = ohlcv.slice(1).map((c, i) => 
      (c.close - ohlcv[i].close) / ohlcv[i].close
    );
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    return Math.sqrt(variance);
  }

  private calculateATR(ohlcv: OHLCV[]): number {
    let atr = 0;
    for (let i = 1; i < ohlcv.length; i++) {
      const tr = Math.max(
        ohlcv[i].high - ohlcv[i].low,
        Math.abs(ohlcv[i].high - ohlcv[i - 1].close),
        Math.abs(ohlcv[i].low - ohlcv[i - 1].close)
      );
      atr += tr;
    }
    return atr / (ohlcv.length - 1) / ohlcv[ohlcv.length - 1].close;
  }

  private calculateBollingerWidth(closes: number[], period: number, stdDev: number): number {
    const bands = this.calculateBollingerBands(
      closes.slice(-period).map((c, i) => ({ close: c, open: c, high: c, low: c, volume: 0, timestamp: i })),
      period,
      stdDev
    );
    return bands.bandwidth;
  }

  private calculateVolatilityRatio(ohlcv: OHLCV[]): number {
    const shortVol = this.calculateVolatilityMetric(ohlcv.slice(-10));
    const longVol = this.calculateVolatilityMetric(ohlcv);
    return shortVol / (longVol || 0.001);
  }

  private calculateMaxDrawdown(closes: number[]): number {
    let peak = closes[0];
    let maxDD = 0;
    for (const close of closes) {
      if (close > peak) peak = close;
      const dd = (peak - close) / peak;
      if (dd > maxDD) maxDD = dd;
    }
    return maxDD;
  }

  private calculateVolumeMA(volumes: number[], period: number): number {
    return volumes.slice(-period).reduce((a, b) => a + b, 0) / period;
  }

  private calculateOBVTrend(ohlcv: OHLCV[]): number {
    let obv = 0;
    for (let i = 1; i < ohlcv.length; i++) {
      if (ohlcv[i].close > ohlcv[i - 1].close) {
        obv += ohlcv[i].volume;
      } else if (ohlcv[i].close < ohlcv[i - 1].close) {
        obv -= ohlcv[i].volume;
      }
    }
    return obv / ohlcv.reduce((sum, c) => sum + c.volume, 0);
  }

  private calculateVolumeVolatility(volumes: number[]): number {
    const mean = volumes.reduce((a, b) => a + b, 0) / volumes.length;
    const variance = volumes.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / volumes.length;
    return Math.sqrt(variance) / mean;
  }

  private calculateRSI(ohlcv: OHLCV[]): number {
    let gains = 0;
    let losses = 0;
    for (let i = 1; i < ohlcv.length; i++) {
      const change = ohlcv[i].close - ohlcv[i - 1].close;
      if (change > 0) gains += change;
      else losses -= change;
    }
    const rs = gains / (losses || 0.001);
    return (100 - 100 / (1 + rs)) / 100;
  }

  private calculateDistanceFromMA(closes: number[], period: number): number {
    const ma = closes.slice(-period).reduce((a, b) => a + b, 0) / period;
    return (closes[closes.length - 1] - ma) / ma;
  }

  // ==========================================================================
  // Reasoning Generation
  // ==========================================================================

  /**
   * Generate human-readable reasoning for the signal.
   */
  private generateReasoning(
    deviation: number,
    reversion: ReversionEstimate,
    regime: RegimeClassification,
    bands: BollingerBands,
    marketData: MarketData
  ): string {
    const direction = deviation > 0 ? 'above' : 'below';
    const deviationStr = Math.abs(deviation).toFixed(2);
    const currentPrice = marketData.ohlcv[marketData.ohlcv.length - 1].close;
    
    return [
      `MeanReversionAgent detected price ${direction} mean by ${deviationStr} std devs.`,
      `Current price: ${currentPrice.toFixed(2)}, Mean: ${bands.middle.toFixed(2)}.`,
      `Regime: ${regime.regime} (${(regime.probabilities[regime.regime] * 100).toFixed(0)}% confidence).`,
      `Bayesian reversion probability: ${(reversion.probability * 100).toFixed(1)}%,`,
      `expected within ${reversion.expectedCandles} candles.`,
      `Prior strength: ${(reversion.priorStrength * 100).toFixed(0)}% (based on historical data).`,
      `Symbol: ${marketData.symbol}.`
    ].join(' ');
  }

  // ==========================================================================
  // Lifecycle Methods
  // ==========================================================================

  async initialize(): Promise<void> {
    console.log(`Initializing ${this.name}`);
    
    // Load regime classifier and parameter optimizer
    // PLACEHOLDER: Model loading code
    
    await new Promise(resolve => setTimeout(resolve, 100));
    
    this.isInitialized = true;
    console.log(`${this.name} initialized successfully`);
  }

  async shutdown(): Promise<void> {
    console.log(`Shutting down ${this.name}`);
    
    this.regimeClassifier = null;
    this.paramOptimizer = null;
    this.reversionHistory.clear();
    this.isInitialized = false;
    
    console.log(`${this.name} shutdown complete`);
  }
}

