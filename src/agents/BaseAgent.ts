/**
 * @fileoverview Abstract base class for all trading agents in the multi-agent ensemble
 * @module agents/BaseAgent
 * 
 * This module defines the foundational interface and abstract implementation that all
 * specialized trading agents must extend. It establishes the contract for signal
 * generation, confidence scoring, and performance tracking that enables the meta-learner
 * to effectively combine diverse trading strategies.
 * 
 * ## AI/ML Architecture
 * 
 * Each agent encapsulates a specific ML model or strategy:
 * - **MomentumAgent**: LSTM neural network for trend prediction
 * - **MeanReversionAgent**: XGBoost + Bayesian probability estimation
 * - **VolatilityAgent**: GARCH + CNN + Reinforcement Learning
 * 
 * The base class provides:
 * 1. Standardized signal format for ensemble combination
 * 2. Confidence calibration framework
 * 3. Performance tracking for adaptive weighting
 * 4. Lifecycle management for model loading/unloading
 */

import { MarketData, Direction, MarketRegime } from '../types';

// ============================================================================
// Interfaces
// ============================================================================

/**
 * Trading signal produced by an agent after analyzing market data.
 * This is the standardized output format that the meta-learner consumes.
 * 
 * @interface AgentSignal
 * 
 * @example
 * ```typescript
 * const signal: AgentSignal = {
 *   direction: 'long',
 *   confidence: 0.85,
 *   suggestedSize: 0.05,
 *   reasoning: 'Strong momentum detected with RSI divergence confirmation',
 *   timestamp: Date.now()
 * };
 * ```
 */
export interface AgentSignal {
  /**
   * Predicted market direction.
   * - 'long': Expecting price to rise
   * - 'short': Expecting price to fall
   * - 'neutral': No clear directional edge
   */
  direction: Direction;

  /**
   * Confidence score from 0 to 1.
   * 
   * This should be a calibrated probability reflecting the agent's certainty.
   * Agents should use techniques like Platt scaling or isotonic regression
   * to ensure confidence scores are well-calibrated.
   * 
   * Interpretation:
   * - 0.9+: Very high confidence, strong signal
   * - 0.7-0.9: High confidence, tradeable signal
   * - 0.5-0.7: Moderate confidence, consider with other signals
   * - <0.5: Low confidence, likely not tradeable alone
   */
  confidence: number;

  /**
   * Suggested position size as a fraction of available capital.
   * 
   * This is a recommendation based on the agent's view of opportunity size.
   * The meta-learner and risk manager may adjust this based on:
   * - Ensemble confidence
   * - Current portfolio exposure
   * - Risk limits
   */
  suggestedSize: number;

  /**
   * Human-readable explanation of the trading rationale.
   * 
   * This is critical for:
   * 1. Competition AI log requirements
   * 2. Debugging and strategy refinement
   * 3. Building trust in AI decisions
   */
  reasoning: string;

  /**
   * Unix timestamp when this signal was generated.
   */
  timestamp: number;

  /**
   * Optional: Raw model output before thresholding.
   * Useful for the meta-learner to access continuous predictions.
   */
  rawScore?: number;

  /**
   * Optional: Detected market regime at time of signal.
   * Helps meta-learner weight agents appropriately.
   */
  regime?: MarketRegime;
}

/**
 * Configuration options for agent initialization.
 * Each agent type may extend this with strategy-specific parameters.
 * 
 * @interface AgentConfig
 */
export interface AgentConfig {
  /**
   * Number of historical candles to consider for analysis.
   * Longer lookbacks capture more context but may be slower.
   */
  lookbackPeriod: number;

  /**
   * Confidence threshold below which signals are considered neutral.
   * Helps filter out weak signals before they reach the ensemble.
   */
  threshold: number;

  /**
   * Path to pre-trained model weights.
   * Models are version-controlled and loaded at initialization.
   * 
   * @example 'models/momentum/lstm_v1.0.0.pt'
   */
  modelPath: string;

  /**
   * Optional: Enable debug logging for this agent.
   */
  debug?: boolean;

  /**
   * Optional: Override default feature set.
   */
  features?: string[];
}

// ============================================================================
// Abstract Base Class
// ============================================================================

/**
 * Abstract base class for all trading agents in the multi-agent ensemble.
 * 
 * ## Overview
 * 
 * BaseAgent defines the interface that all specialized trading agents must implement.
 * It provides common functionality for:
 * - Performance tracking and reporting
 * - Confidence calibration
 * - Model lifecycle management
 * - Signal generation contract
 * 
 * ## AI/ML Integration
 * 
 * Each concrete agent implementation encapsulates a specific ML approach:
 * 
 * | Agent | Primary Model | Backup/Ensemble |
 * |-------|---------------|-----------------|
 * | Momentum | LSTM | Gradient features |
 * | MeanReversion | XGBoost | Bayesian estimation |
 * | Volatility | GARCH + CNN | Q-learning |
 * 
 * ## Performance-Based Weighting
 * 
 * The meta-learner adjusts agent weights based on recent performance.
 * Each agent tracks its own performance history, which is used for:
 * 1. Online weight adaptation (better performers get higher weights)
 * 2. Regime-specific performance analysis
 * 3. Confidence calibration refinement
 * 
 * @abstract
 * 
 * @example
 * ```typescript
 * class MyCustomAgent extends BaseAgent {
 *   async analyze(marketData: MarketData): Promise<AgentSignal> {
 *     // Custom ML model inference
 *     const prediction = await this.model.predict(marketData.features);
 *     return this.formatSignal(prediction);
 *   }
 * }
 * ```
 */
export abstract class BaseAgent {
  /**
   * Unique identifier for this agent.
   * Used in logging, performance tracking, and ensemble attribution.
   */
  public readonly name: string;

  /**
   * Agent-specific configuration.
   */
  protected config: AgentConfig;

  /**
   * Current voting weight in the ensemble (0 to 1).
   * 
   * This weight is dynamically adjusted by the meta-learner based on:
   * - Recent trading performance
   * - Current market regime fit
   * - Signal correlation with other agents
   * 
   * Initial weight is typically 1/N where N is number of agents.
   */
  public weight: number;

  /**
   * Rolling window of recent PnL values for adaptive weighting.
   * 
   * The meta-learner uses this history to:
   * 1. Calculate performance metrics (Sharpe, win rate)
   * 2. Detect regime-specific performance patterns
   * 3. Adjust weights via online learning
   */
  protected recentPerformance: number[];

  /**
   * Maximum size of the performance history window.
   */
  private readonly performanceWindowSize: number = 100;

  /**
   * Flag indicating if the agent's model is loaded and ready.
   */
  protected isInitialized: boolean = false;

  /**
   * Creates a new agent instance.
   * 
   * @param name - Unique identifier for this agent
   * @param config - Agent-specific configuration
   * @param initialWeight - Starting weight in ensemble (default: 0.33 for 3 agents)
   */
  constructor(name: string, config: AgentConfig, initialWeight: number = 0.33) {
    this.name = name;
    this.config = config;
    this.weight = initialWeight;
    this.recentPerformance = [];
  }

  // ==========================================================================
  // Abstract Methods (Must be implemented by subclasses)
  // ==========================================================================

  /**
   * Analyze market data and produce a trading signal.
   * 
   * This is the core method that each agent must implement. It should:
   * 1. Extract relevant features from market data
   * 2. Run ML model inference
   * 3. Calibrate confidence scores
   * 4. Generate human-readable reasoning
   * 
   * ## AI/ML Implementation Notes
   * 
   * - Feature extraction should be deterministic for reproducibility
   * - Model inference should handle edge cases gracefully
   * - Confidence should be calibrated (use Platt scaling, isotonic regression)
   * - Reasoning should explain the key factors driving the signal
   * 
   * @abstract
   * @param marketData - Current market data including OHLCV and indicators
   * @returns Promise resolving to a trading signal
   * 
   * @example
   * ```typescript
   * async analyze(marketData: MarketData): Promise<AgentSignal> {
   *   const features = this.extractFeatures(marketData);
   *   const rawPrediction = await this.model.forward(features);
   *   const calibratedConfidence = this.calibrate(rawPrediction);
   *   
   *   return {
   *     direction: rawPrediction > 0.5 ? 'long' : 'short',
   *     confidence: calibratedConfidence,
   *     suggestedSize: this.calculateSize(calibratedConfidence),
   *     reasoning: this.generateReasoning(features, rawPrediction),
   *     timestamp: Date.now()
   *   };
   * }
   * ```
   */
  abstract analyze(marketData: MarketData): Promise<AgentSignal>;

  /**
   * Initialize the agent, loading any required models.
   * 
   * This method is called once during system startup. Implementations should:
   * 1. Load pre-trained model weights from disk
   * 2. Initialize any required preprocessing pipelines
   * 3. Warm up the model with a test inference
   * 4. Set isInitialized = true when ready
   * 
   * @abstract
   * @returns Promise that resolves when initialization is complete
   */
  abstract initialize(): Promise<void>;

  /**
   * Clean up resources when the agent is no longer needed.
   * 
   * Implementations should:
   * 1. Unload models from memory
   * 2. Close any open connections
   * 3. Flush any pending logs
   * 
   * @abstract
   * @returns Promise that resolves when cleanup is complete
   */
  abstract shutdown(): Promise<void>;

  // ==========================================================================
  // Performance Tracking Methods
  // ==========================================================================

  /**
   * Update the agent's performance history with a new trade result.
   * 
   * Called by the meta-learner after each trade to track how well
   * this agent's signals are performing. This data is used for:
   * 1. Adaptive weight adjustment
   * 2. Regime-specific performance analysis
   * 3. Overall strategy evaluation
   * 
   * @param pnl - Profit/loss from the trade (positive = profit)
   * 
   * @example
   * ```typescript
   * // After a winning trade
   * agent.updatePerformance(0.02); // 2% profit
   * 
   * // After a losing trade
   * agent.updatePerformance(-0.01); // 1% loss
   * ```
   */
  updatePerformance(pnl: number): void {
    this.recentPerformance.push(pnl);
    
    // Maintain rolling window
    if (this.recentPerformance.length > this.performanceWindowSize) {
      this.recentPerformance.shift();
    }
  }

  /**
   * Get the agent's current confidence level based on recent performance.
   * 
   * This is a meta-confidence that reflects how well the agent is performing
   * in current market conditions. It's used by the meta-learner to:
   * 1. Weight signals from agents performing well
   * 2. Reduce influence of agents in drawdown
   * 3. Detect regime changes (performance degradation)
   * 
   * @returns Confidence score from 0 to 1 based on recent performance
   */
  getConfidence(): number {
    if (this.recentPerformance.length === 0) {
      return 0.5; // Neutral confidence when no history
    }

    // Calculate rolling win rate
    const wins = this.recentPerformance.filter(pnl => pnl > 0).length;
    const winRate = wins / this.recentPerformance.length;

    // Calculate rolling Sharpe-like metric
    const mean = this.recentPerformance.reduce((a, b) => a + b, 0) / this.recentPerformance.length;
    const variance = this.recentPerformance.reduce((sum, pnl) => sum + Math.pow(pnl - mean, 2), 0) / this.recentPerformance.length;
    const std = Math.sqrt(variance) || 0.001; // Avoid division by zero
    const sharpeProxy = mean / std;

    // Combine win rate and Sharpe into confidence score
    // This is a simplified heuristic; production would use more sophisticated calibration
    const rawConfidence = 0.5 * winRate + 0.5 * Math.tanh(sharpeProxy);
    
    // Clamp to [0, 1]
    return Math.max(0, Math.min(1, rawConfidence));
  }

  /**
   * Get detailed performance metrics for this agent.
   * 
   * @returns Object containing various performance statistics
   */
  getPerformanceMetrics(): {
    winRate: number;
    avgPnl: number;
    sharpeRatio: number;
    maxDrawdown: number;
    tradeCount: number;
  } {
    const trades = this.recentPerformance;
    const tradeCount = trades.length;

    if (tradeCount === 0) {
      return {
        winRate: 0,
        avgPnl: 0,
        sharpeRatio: 0,
        maxDrawdown: 0,
        tradeCount: 0
      };
    }

    const wins = trades.filter(pnl => pnl > 0).length;
    const winRate = wins / tradeCount;
    const avgPnl = trades.reduce((a, b) => a + b, 0) / tradeCount;
    
    // Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    const mean = avgPnl;
    const variance = trades.reduce((sum, pnl) => sum + Math.pow(pnl - mean, 2), 0) / tradeCount;
    const std = Math.sqrt(variance) || 0.001;
    const sharpeRatio = (mean / std) * Math.sqrt(252); // Annualized

    // Calculate max drawdown
    let peak = 0;
    let maxDrawdown = 0;
    let cumulative = 0;
    for (const pnl of trades) {
      cumulative += pnl;
      if (cumulative > peak) {
        peak = cumulative;
      }
      const drawdown = (peak - cumulative) / (peak || 1);
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }

    return {
      winRate,
      avgPnl,
      sharpeRatio,
      maxDrawdown,
      tradeCount
    };
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  /**
   * Validate that the agent is ready to process signals.
   * 
   * @throws Error if the agent is not initialized
   */
  protected ensureInitialized(): void {
    if (!this.isInitialized) {
      throw new Error(`Agent '${this.name}' is not initialized. Call initialize() first.`);
    }
  }

  /**
   * Create a neutral signal (no trading recommendation).
   * 
   * Used when:
   * - Market conditions are unclear
   * - Model confidence is below threshold
   * - Data quality issues are detected
   * 
   * @param reason - Explanation for why signal is neutral
   * @returns Neutral AgentSignal
   */
  protected createNeutralSignal(reason: string): AgentSignal {
    return {
      direction: 'neutral',
      confidence: 0,
      suggestedSize: 0,
      reasoning: reason,
      timestamp: Date.now()
    };
  }

  /**
   * Get a string representation of the agent for logging.
   */
  toString(): string {
    return `${this.name}(weight=${this.weight.toFixed(3)}, confidence=${this.getConfidence().toFixed(3)})`;
  }
}

// ============================================================================
// Exports
// ============================================================================

export type { MarketData } from '../types';

