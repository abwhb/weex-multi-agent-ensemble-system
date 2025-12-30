/**
 * @fileoverview Risk management and position limits
 * @module execution/RiskManager
 * 
 * ## Overview
 * 
 * Enforces risk rules required by the competition. All orders pass through
 * risk checks before execution to ensure compliance and protect capital.
 * 
 * ## Competition Risk Rules
 * 
 * 1. **Max leverage**: 20x (competition rule)
 * 2. **Max position size**: 10% of capital per trade
 * 3. **Daily drawdown limit**: 5%
 * 4. **Correlation limit**: Max 60% exposure to correlated assets
 * 
 * ## Additional Risk Controls
 * 
 * - Per-trade stop loss requirements
 * - Maximum concurrent positions
 * - Volatility-adjusted sizing
 * - Exposure monitoring
 */

import { OrderParams } from './WeexClient';
import { Position, AllowedPair, ALLOWED_PAIRS } from '../types';

// ============================================================================
// Interfaces
// ============================================================================

/**
 * Risk manager configuration.
 */
export interface RiskConfig {
  /** Maximum leverage allowed (competition max: 20x). */
  maxLeverage: number;
  /** Maximum position size as fraction of capital. */
  maxPositionSize: number;
  /** Maximum daily drawdown before trading halt. */
  maxDailyDrawdown: number;
  /** Maximum exposure to correlated assets. */
  maxCorrelatedExposure: number;
  /** Maximum number of concurrent positions. */
  maxConcurrentPositions: number;
  /** Require stop loss on all trades. */
  requireStopLoss: boolean;
  /** Maximum stop loss distance as fraction of entry. */
  maxStopLossDistance: number;
}

/**
 * Result of order validation.
 */
export interface ValidationResult {
  /** Whether the order passes all risk checks. */
  valid: boolean;
  /** Reason for rejection if invalid. */
  reason?: string;
  /** Adjusted order parameters if modifications were needed. */
  adjustedParams?: OrderParams;
  /** Warnings that don't block the order. */
  warnings: string[];
}

/**
 * Current portfolio exposure.
 */
export interface PortfolioExposure {
  /** Total exposure as fraction of capital. */
  totalExposure: number;
  /** Long exposure as fraction of capital. */
  longExposure: number;
  /** Short exposure as fraction of capital. */
  shortExposure: number;
  /** Net exposure (long - short). */
  netExposure: number;
  /** Exposure by asset. */
  byAsset: Record<AllowedPair, number>;
  /** Correlated group exposures. */
  correlatedGroups: Record<string, number>;
}

/**
 * Daily PnL tracking.
 */
interface DailyStats {
  date: string;
  startingBalance: number;
  currentBalance: number;
  pnl: number;
  drawdown: number;
  peakBalance: number;
  tradeCount: number;
}

// ============================================================================
// RiskManager Implementation
// ============================================================================

/**
 * Risk management system for trade validation and portfolio protection.
 * 
 * ## Responsibilities
 * 
 * 1. Validate all orders against risk rules
 * 2. Calculate and monitor portfolio exposure
 * 3. Track daily drawdown and halt trading if exceeded
 * 4. Enforce position limits and correlation constraints
 * 
 * @example
 * ```typescript
 * const riskManager = new RiskManager({
 *   maxLeverage: 20,
 *   maxPositionSize: 0.10,
 *   maxDailyDrawdown: 0.05
 * }, 10000);
 * 
 * const validation = riskManager.validateOrder(orderParams, currentPositions);
 * if (!validation.valid) {
 *   console.log('Order rejected:', validation.reason);
 * }
 * ```
 */
export class RiskManager {
  /** Risk configuration. */
  private config: RiskConfig;
  
  /** Current account balance. */
  private balance: number;
  
  /** Daily statistics. */
  private dailyStats: DailyStats;
  
  /** Asset correlation groups. */
  private correlationGroups: Map<string, AllowedPair[]>;
  
  /** Trading halted flag. */
  private tradingHalted: boolean = false;
  
  /** Halt reason if halted. */
  private haltReason: string = '';

  /**
   * Create a new RiskManager.
   * 
   * @param config - Risk configuration
   * @param initialBalance - Starting account balance
   */
  constructor(config: Partial<RiskConfig> = {}, initialBalance: number = 10000) {
    this.config = {
      maxLeverage: 20,
      maxPositionSize: 0.10,
      maxDailyDrawdown: 0.05,
      maxCorrelatedExposure: 0.60,
      maxConcurrentPositions: 5,
      requireStopLoss: true,
      maxStopLossDistance: 0.05,
      ...config
    };
    
    this.balance = initialBalance;
    
    // Initialize daily stats
    this.dailyStats = {
      date: new Date().toISOString().split('T')[0],
      startingBalance: initialBalance,
      currentBalance: initialBalance,
      pnl: 0,
      drawdown: 0,
      peakBalance: initialBalance,
      tradeCount: 0
    };
    
    // Define correlation groups (assets that tend to move together)
    this.correlationGroups = new Map([
      ['major_crypto', ['BTC', 'ETH']],
      ['alt_layer1', ['SOL', 'ADA']],
      ['meme', ['DOGE']],
      ['defi', ['BNB']],
      ['payment', ['XRP', 'LTC']]
    ]);
  }

  // ==========================================================================
  // Order Validation
  // ==========================================================================

  /**
   * Validate an order against all risk rules.
   * 
   * @param params - Order parameters to validate
   * @param currentPositions - Current open positions
   * @param currentPrice - Current market price
   * @returns Validation result
   */
  validateOrder(
    params: OrderParams,
    currentPositions: Position[],
    currentPrice: number
  ): ValidationResult {
    const warnings: string[] = [];
    
    // Check if trading is halted
    if (this.tradingHalted) {
      return {
        valid: false,
        reason: `Trading halted: ${this.haltReason}`,
        warnings
      };
    }
    
    // Check daily drawdown
    if (this.dailyStats.drawdown >= this.config.maxDailyDrawdown) {
      this.haltTrading('Daily drawdown limit exceeded');
      return {
        valid: false,
        reason: `Daily drawdown limit (${this.config.maxDailyDrawdown * 100}%) exceeded`,
        warnings
      };
    }
    
    // Check leverage
    const leverage = params.leverage || 1;
    if (leverage > this.config.maxLeverage) {
      return {
        valid: false,
        reason: `Leverage ${leverage}x exceeds maximum ${this.config.maxLeverage}x`,
        warnings
      };
    }
    
    // Check position size
    const positionValue = params.size * currentPrice * leverage;
    const positionFraction = positionValue / this.balance;
    if (positionFraction > this.config.maxPositionSize) {
      return {
        valid: false,
        reason: `Position size (${(positionFraction * 100).toFixed(1)}%) exceeds maximum (${this.config.maxPositionSize * 100}%)`,
        warnings
      };
    }
    
    // Check stop loss requirement
    if (this.config.requireStopLoss && !params.stopLoss) {
      return {
        valid: false,
        reason: 'Stop loss is required for all orders',
        warnings
      };
    }
    
    // Validate stop loss distance
    if (params.stopLoss) {
      const stopDistance = Math.abs(params.stopLoss - currentPrice) / currentPrice;
      if (stopDistance > this.config.maxStopLossDistance) {
        warnings.push(
          `Stop loss distance (${(stopDistance * 100).toFixed(1)}%) exceeds recommended maximum (${this.config.maxStopLossDistance * 100}%)`
        );
      }
    }
    
    // Check concurrent positions
    if (currentPositions.length >= this.config.maxConcurrentPositions) {
      // Check if this is reducing an existing position
      const existingPosition = currentPositions.find(p => p.symbol === params.symbol);
      const isReducing = existingPosition && params.reduceOnly;
      
      if (!isReducing) {
        return {
          valid: false,
          reason: `Maximum concurrent positions (${this.config.maxConcurrentPositions}) reached`,
          warnings
        };
      }
    }
    
    // Check correlated exposure
    const exposure = this.getExposure(currentPositions);
    const correlatedExposure = this.getCorrelatedExposure(params.symbol, exposure);
    const newExposure = correlatedExposure + positionFraction;
    
    if (newExposure > this.config.maxCorrelatedExposure) {
      return {
        valid: false,
        reason: `Correlated exposure (${(newExposure * 100).toFixed(1)}%) would exceed maximum (${this.config.maxCorrelatedExposure * 100}%)`,
        warnings
      };
    }
    
    // All checks passed
    return {
      valid: true,
      warnings
    };
  }

  // ==========================================================================
  // Position Sizing
  // ==========================================================================

  /**
   * Calculate maximum allowed position size.
   * 
   * @param symbol - Trading symbol
   * @param currentPositions - Current open positions
   * @param currentPrice - Current market price
   * @param confidence - AI confidence score (0-1)
   * @returns Maximum allowed size in base currency
   */
  calculateMaxSize(
    symbol: AllowedPair,
    currentPositions: Position[],
    currentPrice: number,
    confidence: number = 0.5
  ): number {
    // Base max size from config
    const maxPositionValue = this.balance * this.config.maxPositionSize;
    
    // Adjust for confidence (lower confidence = smaller position)
    const confidenceMultiplier = 0.5 + 0.5 * confidence;
    
    // Check remaining correlated exposure room
    const exposure = this.getExposure(currentPositions);
    const correlatedExposure = this.getCorrelatedExposure(symbol, exposure);
    const remainingRoom = this.config.maxCorrelatedExposure - correlatedExposure;
    const maxCorrelatedValue = this.balance * Math.max(0, remainingRoom);
    
    // Take minimum of all constraints
    const maxValue = Math.min(
      maxPositionValue * confidenceMultiplier,
      maxCorrelatedValue
    );
    
    return maxValue / currentPrice;
  }

  // ==========================================================================
  // Drawdown Monitoring
  // ==========================================================================

  /**
   * Check current drawdown against limits.
   * 
   * @returns Object with drawdown info and status
   */
  checkDrawdown(): { 
    currentDrawdown: number; 
    maxDrawdown: number; 
    isExceeded: boolean;
    remainingRoom: number;
  } {
    return {
      currentDrawdown: this.dailyStats.drawdown,
      maxDrawdown: this.config.maxDailyDrawdown,
      isExceeded: this.dailyStats.drawdown >= this.config.maxDailyDrawdown,
      remainingRoom: this.config.maxDailyDrawdown - this.dailyStats.drawdown
    };
  }

  /**
   * Update balance and recalculate drawdown.
   * 
   * @param newBalance - Updated account balance
   */
  updateBalance(newBalance: number): void {
    const previousBalance = this.dailyStats.currentBalance;
    this.dailyStats.currentBalance = newBalance;
    this.balance = newBalance;
    
    // Update peak
    if (newBalance > this.dailyStats.peakBalance) {
      this.dailyStats.peakBalance = newBalance;
    }
    
    // Calculate drawdown from starting balance
    this.dailyStats.pnl = newBalance - this.dailyStats.startingBalance;
    this.dailyStats.drawdown = this.dailyStats.pnl < 0 
      ? Math.abs(this.dailyStats.pnl) / this.dailyStats.startingBalance
      : 0;
    
    // Check if we've hit the limit
    if (this.dailyStats.drawdown >= this.config.maxDailyDrawdown) {
      this.haltTrading('Daily drawdown limit exceeded');
    }
  }

  /**
   * Record a completed trade.
   */
  recordTrade(pnl: number): void {
    this.dailyStats.tradeCount++;
    this.updateBalance(this.balance + pnl);
  }

  // ==========================================================================
  // Exposure Calculation
  // ==========================================================================

  /**
   * Calculate current portfolio exposure.
   * 
   * @param positions - Current open positions
   * @returns Portfolio exposure breakdown
   */
  getExposure(positions: Position[]): PortfolioExposure {
    let longExposure = 0;
    let shortExposure = 0;
    const byAsset: Record<string, number> = {};
    
    // Initialize all assets to 0
    for (const pair of ALLOWED_PAIRS) {
      byAsset[pair] = 0;
    }
    
    // Calculate exposures
    for (const position of positions) {
      const exposure = (position.size * position.entryPrice * position.leverage) / this.balance;
      
      if (position.side === 'long') {
        longExposure += exposure;
      } else {
        shortExposure += exposure;
      }
      
      byAsset[position.symbol] = (byAsset[position.symbol] || 0) + exposure;
    }
    
    // Calculate correlated group exposures
    const correlatedGroups: Record<string, number> = {};
    for (const [group, assets] of this.correlationGroups) {
      correlatedGroups[group] = assets.reduce(
        (sum, asset) => sum + (byAsset[asset] || 0), 
        0
      );
    }
    
    return {
      totalExposure: longExposure + shortExposure,
      longExposure,
      shortExposure,
      netExposure: longExposure - shortExposure,
      byAsset: byAsset as Record<AllowedPair, number>,
      correlatedGroups
    };
  }

  /**
   * Get current exposure to correlated assets.
   */
  private getCorrelatedExposure(
    symbol: AllowedPair,
    exposure: PortfolioExposure
  ): number {
    // Find which correlation group this symbol belongs to
    for (const [group, assets] of this.correlationGroups) {
      if (assets.includes(symbol)) {
        return exposure.correlatedGroups[group] || 0;
      }
    }
    return exposure.byAsset[symbol] || 0;
  }

  // ==========================================================================
  // Trading Halt Management
  // ==========================================================================

  /**
   * Halt trading with a reason.
   */
  haltTrading(reason: string): void {
    this.tradingHalted = true;
    this.haltReason = reason;
    console.warn(`TRADING HALTED: ${reason}`);
  }

  /**
   * Resume trading (e.g., next day).
   */
  resumeTrading(): void {
    this.tradingHalted = false;
    this.haltReason = '';
    console.log('Trading resumed');
  }

  /**
   * Check if trading is halted.
   */
  isTradingHalted(): boolean {
    return this.tradingHalted;
  }

  /**
   * Reset daily statistics (call at start of new trading day).
   */
  resetDaily(currentBalance: number): void {
    this.dailyStats = {
      date: new Date().toISOString().split('T')[0],
      startingBalance: currentBalance,
      currentBalance: currentBalance,
      pnl: 0,
      drawdown: 0,
      peakBalance: currentBalance,
      tradeCount: 0
    };
    
    // Resume trading if it was halted due to drawdown
    if (this.haltReason.includes('drawdown')) {
      this.resumeTrading();
    }
  }

  // ==========================================================================
  // Getters
  // ==========================================================================

  /**
   * Get current daily statistics.
   */
  getDailyStats(): DailyStats {
    return { ...this.dailyStats };
  }

  /**
   * Get risk configuration.
   */
  getConfig(): RiskConfig {
    return { ...this.config };
  }

  /**
   * Get current balance.
   */
  getBalance(): number {
    return this.balance;
  }
}

