/**
 * @fileoverview Performance tracking for agents and ensemble
 * @module meta-learner/PerformanceTracker
 * 
 * ## Overview
 * 
 * The PerformanceTracker monitors real-time performance of each agent and the
 * overall ensemble. It calculates rolling metrics (Sharpe ratio, win rate,
 * drawdown) and tracks regime-specific performance for adaptive weighting.
 * 
 * ## Key Features
 * 
 * 1. **Trade Attribution**: Attributes PnL to individual agents based on contribution
 * 2. **Rolling Metrics**: Maintains sliding window statistics
 * 3. **Regime Analysis**: Tracks performance by market regime
 * 4. **Export for Training**: Outputs data for meta-model retraining
 */

import { BaseAgent } from '../agents/BaseAgent';
import { TradeDecision } from './Ensemble';
import { MarketRegime, PerformanceMetrics } from '../types';

// ============================================================================
// Interfaces
// ============================================================================

/**
 * Complete record of a trade for performance analysis.
 */
export interface TradeRecord {
  /** Unique trade identifier. */
  id: string;
  /** Trading symbol. */
  symbol: string;
  /** Entry timestamp. */
  entryTime: number;
  /** Exit timestamp. */
  exitTime?: number;
  /** Entry price. */
  entryPrice: number;
  /** Exit price. */
  exitPrice?: number;
  /** Position direction. */
  direction: 'long' | 'short';
  /** Position size. */
  size: number;
  /** Realized PnL (after exit). */
  pnl?: number;
  /** Ensemble confidence at entry. */
  confidence: number;
  /** Agent contributions to this trade. */
  agentContributions: Record<string, number>;
  /** Market regime at entry. */
  regime: MarketRegime;
  /** AI reasoning for audit. */
  reasoning: string;
  /** Status of the trade. */
  status: 'open' | 'closed' | 'cancelled';
}

/**
 * Performance summary for an agent.
 */
export interface AgentPerformanceSummary {
  name: string;
  metrics: PerformanceMetrics;
  regimePerformance: Record<MarketRegime, PerformanceMetrics>;
  recentTrend: 'improving' | 'stable' | 'declining';
  lastUpdated: number;
}

/**
 * Training data export format.
 */
export interface TrainingDataExport {
  version: string;
  exportTime: number;
  trades: TradeRecord[];
  agentMetrics: AgentPerformanceSummary[];
  ensembleMetrics: PerformanceMetrics;
}

// ============================================================================
// PerformanceTracker Implementation
// ============================================================================

/**
 * Tracks and analyzes performance of agents and the ensemble.
 * 
 * ## Responsibilities
 * 
 * 1. Record all trades with full attribution
 * 2. Calculate rolling performance metrics
 * 3. Track regime-specific performance
 * 4. Export data for meta-model training
 * 5. Detect performance trend changes
 * 
 * @example
 * ```typescript
 * const tracker = new PerformanceTracker();
 * 
 * // Record a trade
 * const tradeId = tracker.recordTradeEntry(decision, currentPrice);
 * 
 * // Close the trade
 * tracker.recordTradeExit(tradeId, exitPrice);
 * 
 * // Get agent metrics
 * const metrics = tracker.getAgentMetrics('MomentumAgent');
 * ```
 */
export class PerformanceTracker {
  /** All recorded trades. */
  private trades: Map<string, TradeRecord> = new Map();
  
  /** Agent performance cache. */
  private agentMetrics: Map<string, AgentPerformanceSummary> = new Map();
  
  /** Rolling window size for metrics. */
  private windowSize: number = 100;
  
  /** Trade ID counter. */
  private tradeCounter: number = 0;

  constructor(windowSize: number = 100) {
    this.windowSize = windowSize;
  }

  // ==========================================================================
  // Trade Recording
  // ==========================================================================

  /**
   * Record a new trade entry.
   * 
   * @param decision - Trade decision from ensemble
   * @param entryPrice - Actual entry price
   * @returns Trade ID for later reference
   */
  recordTradeEntry(decision: TradeDecision, entryPrice: number): string {
    const id = `trade_${Date.now()}_${++this.tradeCounter}`;
    
    const record: TradeRecord = {
      id,
      symbol: decision.symbol,
      entryTime: Date.now(),
      entryPrice,
      direction: decision.action === 'buy' ? 'long' : 'short',
      size: decision.size,
      confidence: decision.confidence,
      agentContributions: decision.agentContributions,
      regime: decision.regime,
      reasoning: decision.reasoning,
      status: 'open'
    };
    
    this.trades.set(id, record);
    console.log(`Trade recorded: ${id}`);
    
    return id;
  }

  /**
   * Record trade exit and calculate PnL.
   * 
   * @param tradeId - ID of the trade to close
   * @param exitPrice - Exit price
   * @returns Trade PnL or null if trade not found
   */
  recordTradeExit(tradeId: string, exitPrice: number): number | null {
    const trade = this.trades.get(tradeId);
    if (!trade || trade.status !== 'open') {
      console.warn(`Trade not found or already closed: ${tradeId}`);
      return null;
    }
    
    trade.exitTime = Date.now();
    trade.exitPrice = exitPrice;
    trade.status = 'closed';
    
    // Calculate PnL
    const priceChange = exitPrice - trade.entryPrice;
    const direction = trade.direction === 'long' ? 1 : -1;
    trade.pnl = priceChange * direction * trade.size / trade.entryPrice; // As percentage
    
    // Update agent metrics
    this.updateAgentMetrics(trade);
    
    console.log(`Trade closed: ${tradeId}, PnL: ${(trade.pnl * 100).toFixed(2)}%`);
    
    return trade.pnl;
  }

  /**
   * Cancel an open trade (e.g., stop loss hit).
   */
  cancelTrade(tradeId: string, reason: string): void {
    const trade = this.trades.get(tradeId);
    if (trade) {
      trade.status = 'cancelled';
      trade.reasoning += ` | Cancelled: ${reason}`;
    }
  }

  // ==========================================================================
  // Metrics Calculation
  // ==========================================================================

  /**
   * Update agent metrics after a trade closes.
   */
  private updateAgentMetrics(trade: TradeRecord): void {
    if (trade.pnl === undefined) return;
    
    for (const [agentName, contribution] of Object.entries(trade.agentContributions)) {
      // Attribute PnL based on contribution
      const attributedPnl = trade.pnl * Math.abs(contribution);
      
      let summary = this.agentMetrics.get(agentName);
      if (!summary) {
        summary = this.createEmptySummary(agentName);
        this.agentMetrics.set(agentName, summary);
      }
      
      // Update overall metrics
      this.updateMetrics(summary.metrics, attributedPnl);
      
      // Update regime-specific metrics
      if (!summary.regimePerformance[trade.regime]) {
        summary.regimePerformance[trade.regime] = this.createEmptyMetrics();
      }
      this.updateMetrics(summary.regimePerformance[trade.regime], attributedPnl);
      
      // Update trend
      summary.recentTrend = this.detectTrend(summary.metrics);
      summary.lastUpdated = Date.now();
    }
  }

  private updateMetrics(metrics: PerformanceMetrics, pnl: number): void {
    metrics.totalTrades++;
    if (pnl > 0) {
      metrics.winningTrades++;
    } else {
      metrics.losingTrades++;
    }
    metrics.winRate = metrics.winningTrades / metrics.totalTrades;
    metrics.totalPnl += pnl;
    
    // Update Sharpe ratio (simplified rolling calculation)
    // In production, maintain full return series
    const avgReturn = metrics.totalPnl / metrics.totalTrades;
    // Approximate std dev update would require more data
    metrics.sharpeRatio = avgReturn * Math.sqrt(252); // Annualized
    
    // Update max drawdown
    if (pnl < 0) {
      const drawdown = Math.abs(pnl);
      if (drawdown > metrics.maxDrawdown) {
        metrics.maxDrawdown = drawdown;
      }
    }
  }

  private detectTrend(metrics: PerformanceMetrics): 'improving' | 'stable' | 'declining' {
    // Would use rolling window comparison in production
    if (metrics.winRate > 0.55) return 'improving';
    if (metrics.winRate < 0.45) return 'declining';
    return 'stable';
  }

  private createEmptySummary(name: string): AgentPerformanceSummary {
    return {
      name,
      metrics: this.createEmptyMetrics(),
      regimePerformance: {} as Record<MarketRegime, PerformanceMetrics>,
      recentTrend: 'stable',
      lastUpdated: Date.now()
    };
  }

  private createEmptyMetrics(): PerformanceMetrics {
    return {
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      winRate: 0,
      totalPnl: 0,
      sharpeRatio: 0,
      maxDrawdown: 0,
      avgTradeDuration: 0
    };
  }

  // ==========================================================================
  // Metrics Retrieval
  // ==========================================================================

  /**
   * Get performance metrics for a specific agent.
   */
  getAgentMetrics(agentName: string): AgentPerformanceSummary | null {
    return this.agentMetrics.get(agentName) || null;
  }

  /**
   * Get performance breakdown by market regime.
   */
  getRegimePerformance(agentName?: string): Record<MarketRegime, PerformanceMetrics> {
    if (agentName) {
      return this.agentMetrics.get(agentName)?.regimePerformance || 
        {} as Record<MarketRegime, PerformanceMetrics>;
    }
    
    // Aggregate across all agents
    const aggregate: Record<string, PerformanceMetrics> = {};
    for (const summary of this.agentMetrics.values()) {
      for (const [regime, metrics] of Object.entries(summary.regimePerformance)) {
        if (!aggregate[regime]) {
          aggregate[regime] = this.createEmptyMetrics();
        }
        aggregate[regime].totalTrades += metrics.totalTrades;
        aggregate[regime].winningTrades += metrics.winningTrades;
        aggregate[regime].totalPnl += metrics.totalPnl;
      }
    }
    
    // Calculate rates
    for (const metrics of Object.values(aggregate)) {
      if (metrics.totalTrades > 0) {
        metrics.winRate = metrics.winningTrades / metrics.totalTrades;
      }
    }
    
    return aggregate as Record<MarketRegime, PerformanceMetrics>;
  }

  /**
   * Get ensemble-wide performance metrics.
   */
  getEnsembleMetrics(): PerformanceMetrics {
    const closedTrades = Array.from(this.trades.values())
      .filter(t => t.status === 'closed' && t.pnl !== undefined);
    
    if (closedTrades.length === 0) {
      return this.createEmptyMetrics();
    }
    
    const wins = closedTrades.filter(t => (t.pnl || 0) > 0);
    const totalPnl = closedTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
    const avgDuration = closedTrades.reduce((sum, t) => 
      sum + ((t.exitTime || 0) - t.entryTime), 0) / closedTrades.length;
    
    // Calculate Sharpe
    const returns = closedTrades.map(t => t.pnl || 0);
    const avgReturn = totalPnl / closedTrades.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    const std = Math.sqrt(variance) || 0.001;
    const sharpe = (avgReturn / std) * Math.sqrt(252);
    
    // Calculate max drawdown
    let peak = 0;
    let maxDD = 0;
    let cumulative = 0;
    for (const trade of closedTrades.sort((a, b) => a.entryTime - b.entryTime)) {
      cumulative += trade.pnl || 0;
      if (cumulative > peak) peak = cumulative;
      const dd = peak > 0 ? (peak - cumulative) / peak : 0;
      if (dd > maxDD) maxDD = dd;
    }
    
    return {
      totalTrades: closedTrades.length,
      winningTrades: wins.length,
      losingTrades: closedTrades.length - wins.length,
      winRate: wins.length / closedTrades.length,
      totalPnl,
      sharpeRatio: sharpe,
      maxDrawdown: maxDD,
      avgTradeDuration: avgDuration
    };
  }

  // ==========================================================================
  // Training Data Export
  // ==========================================================================

  /**
   * Export data in format suitable for meta-model retraining.
   * 
   * This output can be used to retrain the gradient boosted meta-learner
   * on recent performance data for continuous improvement.
   */
  exportForTraining(): TrainingDataExport {
    const closedTrades = Array.from(this.trades.values())
      .filter(t => t.status === 'closed');
    
    return {
      version: '1.0.0',
      exportTime: Date.now(),
      trades: closedTrades,
      agentMetrics: Array.from(this.agentMetrics.values()),
      ensembleMetrics: this.getEnsembleMetrics()
    };
  }

  /**
   * Get trade history for a symbol.
   */
  getTradeHistory(symbol?: string): TradeRecord[] {
    let trades = Array.from(this.trades.values());
    if (symbol) {
      trades = trades.filter(t => t.symbol === symbol);
    }
    return trades.sort((a, b) => b.entryTime - a.entryTime);
  }

  /**
   * Clear old trade records beyond retention window.
   */
  cleanup(retentionDays: number = 30): void {
    const cutoff = Date.now() - retentionDays * 24 * 60 * 60 * 1000;
    for (const [id, trade] of this.trades) {
      if (trade.entryTime < cutoff && trade.status !== 'open') {
        this.trades.delete(id);
      }
    }
  }
}

/**
 * Module exports
 */
export { PerformanceTracker as default };

