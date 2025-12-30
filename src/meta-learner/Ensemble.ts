/**
 * @fileoverview Meta-learner that combines agent signals using gradient boosting
 * @module meta-learner/Ensemble
 * 
 * ## Overview
 * 
 * The Ensemble is the core meta-learner that aggregates signals from all
 * specialized trading agents. It uses a trained gradient boosted model to
 * optimally weight agent predictions based on current market features.
 * 
 * ## AI/ML Architecture
 * 
 * ### Signal Combiner (LightGBM)
 * 
 * ```
 * Input Features:
 * - momentum_signal, momentum_confidence
 * - reversion_signal, reversion_confidence
 * - volatility_signal, volatility_confidence
 * - market_regime (one-hot encoded)
 * - cross_asset_correlation
 * - recent_agent_performance (3 features)
 * 
 * Output: Optimal position direction and size
 * 
 * Training: Historical agent signals → actual returns
 * ```
 * 
 * ### Online Weight Adaptation
 * 
 * ```
 * For each completed trade:
 *   1. Calculate PnL attribution to each agent
 *   2. Update agent weight: w(t+1) = w(t) + η·gradient
 *   3. Normalize weights to sum to 1
 * ```
 * 
 * ### Position Sizing (Kelly Criterion)
 * 
 * ```
 * f* = (p * b - q) / b
 * 
 * Where:
 * - p = win probability (from ensemble confidence)
 * - q = 1 - p
 * - b = average win/loss ratio
 * 
 * Final size = f* * ensemble_confidence * max_position
 * ```
 */

import { BaseAgent, AgentSignal } from '../agents/BaseAgent';
import { MarketData, MarketRegime, Direction, AllowedPair } from '../types';

// ============================================================================
// Interfaces
// ============================================================================

/**
 * Configuration for the Ensemble meta-learner.
 */
export interface EnsembleConfig {
  /** Minimum confidence to consider trading. @default 0.55 */
  minConfidence: number;
  /** Learning rate for online weight adaptation. @default 0.05 */
  learningRate: number;
  /** Window size for performance-based weighting. @default 50 */
  performanceWindow: number;
  /** Maximum position size as fraction of capital. @default 0.10 */
  maxPositionSize: number;
  /** Correlation threshold for diversification filter. @default 0.7 */
  correlationThreshold: number;
  /** Path to trained meta-model weights. */
  modelPath?: string;
}

/**
 * Final trade decision output from the ensemble.
 * This is what gets passed to the execution layer.
 */
export interface TradeDecision {
  /** Final action to take. */
  action: 'buy' | 'sell' | 'hold';
  /** Trading symbol. */
  symbol: AllowedPair;
  /** Position size in base currency or as fraction. */
  size: number;
  /** Ensemble confidence (0-1). */
  confidence: number;
  /** Suggested stop loss price. */
  stopLoss: number;
  /** Suggested take profit price. */
  takeProfit: number;
  /** Each agent's contribution to the decision. */
  agentContributions: Record<string, number>;
  /** AI-generated explanation for audit log. */
  reasoning: string;
  /** Timestamp of decision. */
  timestamp: number;
  /** Current market price at decision time. */
  currentPrice: number;
  /** Detected market regime. */
  regime: MarketRegime;
}

/**
 * Internal signal aggregation structure.
 */
interface AggregatedSignals {
  weightedDirection: number;  // -1 (short) to 1 (long)
  combinedConfidence: number;
  agentContributions: Record<string, number>;
  consensus: number;  // How much agents agree (0-1)
}

// ============================================================================
// Ensemble Implementation
// ============================================================================

/**
 * Meta-learner that combines agent signals into optimal trading decisions.
 * 
 * ## Key Responsibilities
 * 
 * 1. **Signal Aggregation**: Combine signals from all registered agents
 * 2. **Adaptive Weighting**: Adjust agent weights based on performance
 * 3. **Position Sizing**: Calculate optimal position size using Kelly criterion
 * 4. **Risk Filtering**: Apply correlation and confidence filters
 * 5. **Decision Generation**: Produce final trade decisions with reasoning
 * 
 * ## AI/ML Integration
 * 
 * The ensemble uses a gradient boosted model (LightGBM) trained on historical
 * data to learn optimal signal combination weights. The model adapts online
 * as agent performance changes.
 * 
 * @example
 * ```typescript
 * const ensemble = new Ensemble({
 *   minConfidence: 0.6,
 *   learningRate: 0.05,
 *   performanceWindow: 50,
 *   maxPositionSize: 0.10
 * });
 * 
 * ensemble.registerAgent(momentumAgent);
 * ensemble.registerAgent(meanReversionAgent);
 * ensemble.registerAgent(volatilityAgent);
 * 
 * await ensemble.initialize();
 * const decision = await ensemble.decide(marketData, 'BTC');
 * ```
 */
export class Ensemble {
  /** Registered trading agents. */
  private agents: BaseAgent[] = [];
  
  /** Ensemble configuration. */
  private config: EnsembleConfig;
  
  /** LightGBM meta-model placeholder. */
  private metaModel: unknown = null;
  
  /** Recent trade results for online learning. */
  private recentResults: Array<{ agentWeights: Record<string, number>; pnl: number }> = [];
  
  /** Historical win rate for Kelly criterion. */
  private winRate: number = 0.5;
  
  /** Average win/loss ratio for Kelly criterion. */
  private avgWinLossRatio: number = 1.0;
  
  /** Correlation matrix between agent signals. */
  private signalCorrelation: Map<string, Map<string, number>> = new Map();
  
  /** Whether the ensemble is initialized. */
  private isInitialized: boolean = false;

  /**
   * Create a new Ensemble instance.
   * 
   * @param config - Ensemble configuration
   */
  constructor(config: Partial<EnsembleConfig> = {}) {
    this.config = {
      minConfidence: 0.55,
      learningRate: 0.05,
      performanceWindow: 50,
      maxPositionSize: 0.10,
      correlationThreshold: 0.7,
      ...config
    };
  }

  // ==========================================================================
  // Agent Management
  // ==========================================================================

  /**
   * Register a trading agent with the ensemble.
   * 
   * @param agent - Agent to register
   */
  registerAgent(agent: BaseAgent): void {
    this.agents.push(agent);
    console.log(`Registered agent: ${agent.name} (weight: ${agent.weight})`);
  }

  /**
   * Get all registered agents.
   */
  getAgents(): BaseAgent[] {
    return [...this.agents];
  }

  // ==========================================================================
  // Main Decision Method
  // ==========================================================================

  /**
   * Generate a trading decision by combining all agent signals.
   * 
   * ## Process Flow
   * 
   * 1. Collect signals from all agents
   * 2. Combine signals using meta-model
   * 3. Apply confidence and correlation filters
   * 4. Calculate position size (Kelly criterion)
   * 5. Generate stop loss and take profit levels
   * 6. Create reasoning summary
   * 
   * @param marketData - Current market data
   * @param symbol - Trading symbol
   * @returns Trade decision or null if no trade should be taken
   */
  async decide(marketData: MarketData, symbol: AllowedPair): Promise<TradeDecision | null> {
    this.ensureInitialized();

    // Step 1: Collect signals from all agents
    const signals = await this.collectSignals(marketData);
    
    // Step 2: Combine signals using meta-model
    const aggregated = await this.combineSignals(signals, marketData);
    
    // Step 3: Check if we should trade
    if (!this.shouldTrade(aggregated)) {
      return null;
    }

    // Step 4: Determine direction
    const direction = this.determineDirection(aggregated);
    if (direction === 'neutral') {
      return null;
    }

    // Step 5: Calculate position size
    const size = this.calculatePositionSize(aggregated);

    // Step 6: Calculate stop loss and take profit
    const currentPrice = marketData.ohlcv[marketData.ohlcv.length - 1].close;
    const { stopLoss, takeProfit } = this.calculateLevels(
      direction,
      currentPrice,
      marketData
    );

    // Step 7: Detect regime
    const regime = this.detectConsensusRegime(signals);

    // Step 8: Generate reasoning
    const reasoning = this.generateReasoning(
      signals,
      aggregated,
      direction,
      size,
      symbol
    );

    return {
      action: direction === 'long' ? 'buy' : 'sell',
      symbol,
      size,
      confidence: aggregated.combinedConfidence,
      stopLoss,
      takeProfit,
      agentContributions: aggregated.agentContributions,
      reasoning,
      timestamp: Date.now(),
      currentPrice,
      regime
    };
  }

  // ==========================================================================
  // Signal Collection & Combination
  // ==========================================================================

  /**
   * Collect signals from all registered agents.
   */
  private async collectSignals(marketData: MarketData): Promise<Map<string, AgentSignal>> {
    const signals = new Map<string, AgentSignal>();
    
    const signalPromises = this.agents.map(async (agent) => {
      try {
        const signal = await agent.analyze(marketData);
        return { name: agent.name, signal };
      } catch (error) {
        console.error(`Error collecting signal from ${agent.name}:`, error);
        return { name: agent.name, signal: null };
      }
    });

    const results = await Promise.all(signalPromises);
    
    for (const { name, signal } of results) {
      if (signal) {
        signals.set(name, signal);
      }
    }

    return signals;
  }

  /**
   * Combine agent signals using the meta-model.
   * 
   * ## AI/ML Details
   * 
   * The meta-model (LightGBM) takes as input:
   * - All agent signals and confidences
   * - Market features (regime, volatility, etc.)
   * - Recent agent performance metrics
   * 
   * It outputs optimal weights for combining the signals.
   */
  private async combineSignals(
    signals: Map<string, AgentSignal>,
    marketData: MarketData
  ): Promise<AggregatedSignals> {
    // Extract features for meta-model
    const features = this.extractMetaFeatures(signals, marketData);
    
    // Get agent weights (would use meta-model in production)
    const weights = await this.getOptimalWeights(features);
    
    // Calculate weighted direction
    let weightedDirection = 0;
    let totalWeight = 0;
    let totalConfidence = 0;
    const contributions: Record<string, number> = {};

    for (const [agentName, signal] of signals) {
      const agent = this.agents.find(a => a.name === agentName);
      const weight = weights.get(agentName) || (agent?.weight || 0);
      
      // Convert direction to numeric
      const directionValue = signal.direction === 'long' ? 1 :
                            signal.direction === 'short' ? -1 : 0;
      
      const contribution = directionValue * signal.confidence * weight;
      contributions[agentName] = contribution;
      
      weightedDirection += contribution;
      totalWeight += weight;
      totalConfidence += signal.confidence * weight;
    }

    // Normalize
    if (totalWeight > 0) {
      weightedDirection /= totalWeight;
      totalConfidence /= totalWeight;
    }

    // Calculate consensus (how much agents agree)
    const consensus = this.calculateConsensus(signals);

    return {
      weightedDirection,
      combinedConfidence: totalConfidence * consensus, // Reduce confidence if disagreement
      agentContributions: contributions,
      consensus
    };
  }

  /**
   * Extract features for the meta-model.
   */
  private extractMetaFeatures(
    signals: Map<string, AgentSignal>,
    marketData: MarketData
  ): number[] {
    const features: number[] = [];
    
    // Agent signals and confidences (6 features)
    for (const agent of this.agents) {
      const signal = signals.get(agent.name);
      if (signal) {
        const dirValue = signal.direction === 'long' ? 1 : 
                        signal.direction === 'short' ? -1 : 0;
        features.push(dirValue);
        features.push(signal.confidence);
      } else {
        features.push(0, 0);
      }
    }
    
    // Recent agent performance (3 features)
    for (const agent of this.agents) {
      features.push(agent.getConfidence());
    }
    
    // Market features (use first signal's regime or detect)
    const regimeValues = { trending: 1, ranging: 0, volatile: -1, quiet: 0.5 };
    const regime = Array.from(signals.values())[0]?.regime || 'ranging';
    features.push(regimeValues[regime]);
    
    // Volatility feature
    const returns = marketData.ohlcv.slice(-20).map((c, i, arr) => 
      i > 0 ? (c.close - arr[i-1].close) / arr[i-1].close : 0
    ).slice(1);
    const volatility = Math.sqrt(returns.reduce((sum, r) => sum + r * r, 0) / returns.length);
    features.push(volatility);

    return features;
  }

  /**
   * Get optimal weights from meta-model or use performance-based heuristic.
   */
  private async getOptimalWeights(features: number[]): Promise<Map<string, number>> {
    // PLACEHOLDER: Meta-model inference
    // In production: const weights = this.metaModel.predict(features);
    
    // Use performance-based weighting as heuristic
    const weights = new Map<string, number>();
    let totalWeight = 0;
    
    for (const agent of this.agents) {
      const performanceWeight = 0.5 + 0.5 * agent.getConfidence();
      weights.set(agent.name, agent.weight * performanceWeight);
      totalWeight += agent.weight * performanceWeight;
    }
    
    // Normalize weights
    for (const [name, weight] of weights) {
      weights.set(name, weight / totalWeight);
    }
    
    return weights;
  }

  /**
   * Calculate consensus among agents.
   */
  private calculateConsensus(signals: Map<string, AgentSignal>): number {
    const signalValues = Array.from(signals.values());
    if (signalValues.length === 0) return 0;
    
    // Count directions
    let longCount = 0;
    let shortCount = 0;
    let neutralCount = 0;
    
    for (const signal of signalValues) {
      if (signal.direction === 'long') longCount++;
      else if (signal.direction === 'short') shortCount++;
      else neutralCount++;
    }
    
    const total = signalValues.length;
    const maxCount = Math.max(longCount, shortCount, neutralCount);
    
    return maxCount / total;
  }

  // ==========================================================================
  // Trading Decision Logic
  // ==========================================================================

  /**
   * Determine if we should trade based on aggregated signals.
   */
  private shouldTrade(aggregated: AggregatedSignals): boolean {
    // Check minimum confidence
    if (aggregated.combinedConfidence < this.config.minConfidence) {
      return false;
    }
    
    // Check minimum consensus
    if (aggregated.consensus < 0.5) {
      return false; // Agents disagree too much
    }
    
    // Check direction strength
    if (Math.abs(aggregated.weightedDirection) < 0.3) {
      return false; // Direction not clear enough
    }
    
    return true;
  }

  /**
   * Determine trading direction from aggregated signals.
   */
  private determineDirection(aggregated: AggregatedSignals): Direction {
    if (aggregated.weightedDirection > 0.3) {
      return 'long';
    } else if (aggregated.weightedDirection < -0.3) {
      return 'short';
    }
    return 'neutral';
  }

  // ==========================================================================
  // Position Sizing (Kelly Criterion)
  // ==========================================================================

  /**
   * Calculate optimal position size using Kelly Criterion.
   * 
   * ## Kelly Criterion
   * 
   * f* = (p * b - q) / b
   * 
   * Where:
   * - p = probability of winning
   * - q = probability of losing (1 - p)
   * - b = win/loss ratio
   * 
   * We apply a fractional Kelly (typically 25-50%) for safety.
   */
  calculatePositionSize(aggregated: AggregatedSignals): number {
    const p = (aggregated.combinedConfidence + 1) / 2; // Convert to probability
    const q = 1 - p;
    const b = this.avgWinLossRatio;
    
    // Kelly fraction
    let kelly = (p * b - q) / b;
    kelly = Math.max(0, kelly); // Can't be negative
    
    // Apply fractional Kelly (25%)
    const fractionalKelly = kelly * 0.25;
    
    // Scale by confidence and cap at max position size
    const scaledSize = fractionalKelly * aggregated.combinedConfidence;
    
    return Math.min(this.config.maxPositionSize, Math.max(0.01, scaledSize));
  }

  // ==========================================================================
  // Stop Loss & Take Profit
  // ==========================================================================

  /**
   * Calculate stop loss and take profit levels.
   */
  private calculateLevels(
    direction: Direction,
    currentPrice: number,
    marketData: MarketData
  ): { stopLoss: number; takeProfit: number } {
    // Calculate ATR for volatility-based levels
    const atr = this.calculateATR(marketData.ohlcv.slice(-14));
    
    // Default to 2x ATR stop loss, 3x ATR take profit (1.5 R:R)
    const stopDistance = atr * 2;
    const profitDistance = atr * 3;
    
    if (direction === 'long') {
      return {
        stopLoss: currentPrice - stopDistance,
        takeProfit: currentPrice + profitDistance
      };
    } else {
      return {
        stopLoss: currentPrice + stopDistance,
        takeProfit: currentPrice - profitDistance
      };
    }
  }

  private calculateATR(ohlcv: import('../types').OHLCV[]): number {
    let atr = 0;
    for (let i = 1; i < ohlcv.length; i++) {
      const tr = Math.max(
        ohlcv[i].high - ohlcv[i].low,
        Math.abs(ohlcv[i].high - ohlcv[i - 1].close),
        Math.abs(ohlcv[i].low - ohlcv[i - 1].close)
      );
      atr += tr;
    }
    return atr / (ohlcv.length - 1);
  }

  // ==========================================================================
  // Online Weight Adaptation
  // ==========================================================================

  /**
   * Update agent weights based on trade result.
   * 
   * ## Online Learning
   * 
   * After each trade completes, we update agent weights using gradient descent:
   * ```
   * w(t+1) = w(t) + η * gradient
   * ```
   * 
   * Where the gradient is proportional to each agent's contribution to PnL.
   */
  updateWeights(tradeResult: { pnl: number; agentContributions: Record<string, number> }): void {
    const { pnl, agentContributions } = tradeResult;
    
    // Store for batch analysis
    this.recentResults.push({ agentWeights: agentContributions, pnl });
    if (this.recentResults.length > this.config.performanceWindow) {
      this.recentResults.shift();
    }
    
    // Update win rate and win/loss ratio
    this.updateWinStatistics(pnl);
    
    // Gradient update for each agent
    for (const agent of this.agents) {
      const contribution = agentContributions[agent.name] || 0;
      const gradient = contribution * Math.sign(pnl) * Math.abs(pnl);
      
      // Update weight
      agent.weight += this.config.learningRate * gradient;
      agent.weight = Math.max(0.1, Math.min(0.6, agent.weight)); // Clamp weights
      
      // Update agent's own performance tracking
      agent.updatePerformance(pnl * Math.abs(contribution));
    }
    
    // Normalize weights
    this.normalizeWeights();
  }

  private updateWinStatistics(pnl: number): void {
    const wins = this.recentResults.filter(r => r.pnl > 0);
    const losses = this.recentResults.filter(r => r.pnl < 0);
    
    this.winRate = wins.length / Math.max(1, this.recentResults.length);
    
    if (wins.length > 0 && losses.length > 0) {
      const avgWin = wins.reduce((sum, r) => sum + r.pnl, 0) / wins.length;
      const avgLoss = Math.abs(losses.reduce((sum, r) => sum + r.pnl, 0) / losses.length);
      this.avgWinLossRatio = avgWin / (avgLoss || 1);
    }
  }

  private normalizeWeights(): void {
    const totalWeight = this.agents.reduce((sum, a) => sum + a.weight, 0);
    for (const agent of this.agents) {
      agent.weight /= totalWeight;
    }
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  private detectConsensusRegime(signals: Map<string, AgentSignal>): MarketRegime {
    const regimeCounts: Record<MarketRegime, number> = {
      trending: 0,
      ranging: 0,
      volatile: 0,
      quiet: 0
    };
    
    for (const signal of signals.values()) {
      if (signal.regime) {
        regimeCounts[signal.regime]++;
      }
    }
    
    let maxRegime: MarketRegime = 'ranging';
    let maxCount = 0;
    
    for (const [regime, count] of Object.entries(regimeCounts)) {
      if (count > maxCount) {
        maxCount = count;
        maxRegime = regime as MarketRegime;
      }
    }
    
    return maxRegime;
  }

  /**
   * Generate reasoning for the trade decision.
   */
  private generateReasoning(
    signals: Map<string, AgentSignal>,
    aggregated: AggregatedSignals,
    direction: Direction,
    size: number,
    symbol: AllowedPair
  ): string {
    const parts: string[] = [
      `Ensemble decision for ${symbol}: ${direction.toUpperCase()}.`
    ];
    
    // Add agent contributions
    for (const [name, contribution] of Object.entries(aggregated.agentContributions)) {
      const signal = signals.get(name);
      if (signal) {
        parts.push(
          `${name}: ${signal.direction} (conf: ${(signal.confidence * 100).toFixed(0)}%, ` +
          `contrib: ${(contribution * 100).toFixed(1)}%).`
        );
      }
    }
    
    parts.push(
      `Consensus: ${(aggregated.consensus * 100).toFixed(0)}%.`,
      `Combined confidence: ${(aggregated.combinedConfidence * 100).toFixed(1)}%.`,
      `Position size: ${(size * 100).toFixed(2)}% of capital.`
    );
    
    return parts.join(' ');
  }

  // ==========================================================================
  // Lifecycle Methods
  // ==========================================================================

  private ensureInitialized(): void {
    if (!this.isInitialized) {
      throw new Error('Ensemble not initialized. Call initialize() first.');
    }
  }

  /**
   * Initialize the ensemble and all agents.
   */
  async initialize(): Promise<void> {
    console.log('Initializing Ensemble meta-learner...');
    
    // Initialize all agents
    await Promise.all(this.agents.map(agent => agent.initialize()));
    
    // Load meta-model
    // PLACEHOLDER: this.metaModel = await loadModel(this.config.modelPath);
    
    this.isInitialized = true;
    console.log(`Ensemble initialized with ${this.agents.length} agents`);
  }

  /**
   * Shutdown the ensemble and all agents.
   */
  async shutdown(): Promise<void> {
    console.log('Shutting down Ensemble...');
    
    await Promise.all(this.agents.map(agent => agent.shutdown()));
    
    this.metaModel = null;
    this.recentResults = [];
    this.isInitialized = false;
    
    console.log('Ensemble shutdown complete');
  }
}

