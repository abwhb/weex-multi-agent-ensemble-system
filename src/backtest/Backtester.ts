/**
 * @fileoverview Backtesting engine for strategy validation
 * @module backtest/Backtester
 * 
 * Simulates trading on historical data to validate strategy performance.
 * Accounts for slippage, fees, and realistic execution.
 */

import { OHLCV, MarketData, AllowedPair, Direction } from '../types';
import { BaseAgent, AgentSignal } from '../agents/BaseAgent';
import { Ensemble, TradeDecision } from '../meta-learner/Ensemble';

// ============================================================================
// Types
// ============================================================================

/**
 * Configuration for backtesting.
 */
export interface BacktestConfig {
  /** Starting capital in USD. */
  initialCapital: number;
  /** Slippage as fraction (0.001 = 0.1%). */
  slippage: number;
  /** Trading fee as fraction (0.001 = 0.1%). */
  fee: number;
  /** Maximum leverage. */
  maxLeverage: number;
  /** Maximum position size as fraction of capital. */
  maxPositionSize: number;
}

/**
 * A single trade in the backtest.
 */
export interface BacktestTrade {
  id: number;
  symbol: string;
  direction: 'long' | 'short';
  entryTime: number;
  entryPrice: number;
  exitTime?: number;
  exitPrice?: number;
  size: number;
  pnl?: number;
  pnlPercent?: number;
  confidence: number;
  reasoning: string;
  status: 'open' | 'closed';
}

/**
 * Backtest performance metrics.
 */
export interface BacktestMetrics {
  /** Total return as percentage. */
  totalReturn: number;
  /** Annualized return percentage. */
  annualizedReturn: number;
  /** Sharpe ratio (annualized). */
  sharpeRatio: number;
  /** Sortino ratio (annualized). */
  sortinoRatio: number;
  /** Maximum drawdown as percentage. */
  maxDrawdown: number;
  /** Win rate as percentage. */
  winRate: number;
  /** Profit factor (gross profit / gross loss). */
  profitFactor: number;
  /** Total number of trades. */
  totalTrades: number;
  /** Number of winning trades. */
  winningTrades: number;
  /** Number of losing trades. */
  losingTrades: number;
  /** Average trade duration in hours. */
  avgTradeDuration: number;
  /** Average winning trade percentage. */
  avgWin: number;
  /** Average losing trade percentage. */
  avgLoss: number;
  /** Final portfolio value. */
  finalValue: number;
}

/**
 * Equity curve data point.
 */
export interface EquityPoint {
  timestamp: number;
  equity: number;
  drawdown: number;
}

/**
 * Complete backtest result.
 */
export interface BacktestResult {
  config: BacktestConfig;
  metrics: BacktestMetrics;
  trades: BacktestTrade[];
  equityCurve: EquityPoint[];
  startDate: Date;
  endDate: Date;
  symbol: string;
}

// ============================================================================
// Backtester Implementation
// ============================================================================

/**
 * Backtesting engine for validating trading strategies.
 * 
 * @example
 * ```typescript
 * const backtester = new Backtester(ensemble, {
 *   initialCapital: 10000,
 *   slippage: 0.001,
 *   fee: 0.001,
 *   maxLeverage: 5,
 *   maxPositionSize: 0.1
 * });
 * 
 * const result = await backtester.run(historicalData, 'BTC');
 * console.log('Sharpe Ratio:', result.metrics.sharpeRatio);
 * ```
 */
export class Backtester {
  private ensemble: Ensemble;
  private config: BacktestConfig;
  
  // State during backtest
  private capital: number = 0;
  private equity: number = 0;
  private peakEquity: number = 0;
  private position: BacktestTrade | null = null;
  private trades: BacktestTrade[] = [];
  private equityCurve: EquityPoint[] = [];
  private tradeCounter: number = 0;

  constructor(ensemble: Ensemble, config: Partial<BacktestConfig> = {}) {
    this.ensemble = ensemble;
    this.config = {
      initialCapital: 10000,
      slippage: 0.001,
      fee: 0.001,
      maxLeverage: 5,
      maxPositionSize: 0.1,
      ...config
    };
  }

  /**
   * Run backtest on historical data.
   * 
   * @param ohlcv - Historical OHLCV data
   * @param symbol - Trading symbol
   * @param warmupPeriod - Number of candles to skip for indicator warmup
   */
  async run(
    ohlcv: OHLCV[],
    symbol: AllowedPair,
    warmupPeriod: number = 100
  ): Promise<BacktestResult> {
    console.log(`Running backtest for ${symbol} on ${ohlcv.length} candles...`);
    
    // Reset state
    this.capital = this.config.initialCapital;
    this.equity = this.config.initialCapital;
    this.peakEquity = this.config.initialCapital;
    this.position = null;
    this.trades = [];
    this.equityCurve = [];
    this.tradeCounter = 0;

    // Main backtest loop
    for (let i = warmupPeriod; i < ohlcv.length; i++) {
      const currentCandle = ohlcv[i];
      const historicalData = ohlcv.slice(0, i + 1);
      
      // Update position P&L if we have one
      if (this.position) {
        this.updatePositionPnL(currentCandle);
      }
      
      // Record equity
      this.recordEquity(currentCandle.timestamp);
      
      // Create market data for agents
      const marketData: MarketData = {
        symbol,
        ohlcv: historicalData.slice(-warmupPeriod),
        indicators: {},
        features: []
      };
      
      // Get trading decision
      try {
        const decision = await this.ensemble.decide(marketData, symbol);
        
        if (decision) {
          await this.processDecision(decision, currentCandle);
        }
      } catch (error) {
        // Skip this candle if there's an error
        continue;
      }
    }
    
    // Close any remaining position at end
    if (this.position) {
      this.closePosition(ohlcv[ohlcv.length - 1]);
    }
    
    // Calculate metrics
    const metrics = this.calculateMetrics(ohlcv);
    
    const result: BacktestResult = {
      config: this.config,
      metrics,
      trades: this.trades,
      equityCurve: this.equityCurve,
      startDate: new Date(ohlcv[warmupPeriod].timestamp),
      endDate: new Date(ohlcv[ohlcv.length - 1].timestamp),
      symbol
    };
    
    console.log(`Backtest complete: ${this.trades.length} trades, ${metrics.totalReturn.toFixed(2)}% return`);
    
    return result;
  }

  /**
   * Process a trading decision.
   */
  private async processDecision(decision: TradeDecision, candle: OHLCV): Promise<void> {
    const direction = decision.action === 'buy' ? 'long' : 'short';
    
    // If we have a position in opposite direction, close it first
    if (this.position && this.position.direction !== direction) {
      this.closePosition(candle);
    }
    
    // If no position, open one
    if (!this.position && decision.action !== 'hold') {
      this.openPosition(decision, candle);
    }
  }

  /**
   * Open a new position.
   */
  private openPosition(decision: TradeDecision, candle: OHLCV): void {
    const direction = decision.action === 'buy' ? 'long' : 'short';
    
    // Calculate position size
    const maxSize = this.capital * this.config.maxPositionSize;
    const size = Math.min(decision.size * this.capital, maxSize);
    
    // Apply slippage to entry price
    const slippageMultiplier = direction === 'long' 
      ? (1 + this.config.slippage) 
      : (1 - this.config.slippage);
    const entryPrice = candle.close * slippageMultiplier;
    
    // Deduct fees
    const fee = size * this.config.fee;
    this.capital -= fee;
    
    this.position = {
      id: ++this.tradeCounter,
      symbol: decision.symbol,
      direction,
      entryTime: candle.timestamp,
      entryPrice,
      size,
      confidence: decision.confidence,
      reasoning: decision.reasoning,
      status: 'open'
    };
  }

  /**
   * Close the current position.
   */
  private closePosition(candle: OHLCV): void {
    if (!this.position) return;
    
    // Apply slippage to exit price
    const slippageMultiplier = this.position.direction === 'long' 
      ? (1 - this.config.slippage) 
      : (1 + this.config.slippage);
    const exitPrice = candle.close * slippageMultiplier;
    
    // Calculate P&L
    const priceChange = exitPrice - this.position.entryPrice;
    const direction = this.position.direction === 'long' ? 1 : -1;
    const pnl = (priceChange / this.position.entryPrice) * this.position.size * direction;
    const pnlPercent = (pnl / this.position.size) * 100;
    
    // Deduct exit fees
    const fee = this.position.size * this.config.fee;
    this.capital += pnl - fee;
    
    // Update position
    this.position.exitTime = candle.timestamp;
    this.position.exitPrice = exitPrice;
    this.position.pnl = pnl;
    this.position.pnlPercent = pnlPercent;
    this.position.status = 'closed';
    
    // Add to trades list
    this.trades.push({ ...this.position });
    
    // Clear position
    this.position = null;
  }

  /**
   * Update unrealized P&L for open position.
   */
  private updatePositionPnL(candle: OHLCV): void {
    if (!this.position) return;
    
    const priceChange = candle.close - this.position.entryPrice;
    const direction = this.position.direction === 'long' ? 1 : -1;
    const unrealizedPnl = (priceChange / this.position.entryPrice) * this.position.size * direction;
    
    this.equity = this.capital + unrealizedPnl;
    
    if (this.equity > this.peakEquity) {
      this.peakEquity = this.equity;
    }
  }

  /**
   * Record equity point.
   */
  private recordEquity(timestamp: number): void {
    const equity = this.position ? this.equity : this.capital;
    const drawdown = this.peakEquity > 0 
      ? (this.peakEquity - equity) / this.peakEquity 
      : 0;
    
    this.equityCurve.push({
      timestamp,
      equity,
      drawdown
    });
  }

  /**
   * Calculate performance metrics.
   */
  private calculateMetrics(ohlcv: OHLCV[]): BacktestMetrics {
    const closedTrades = this.trades.filter(t => t.status === 'closed');
    const winningTrades = closedTrades.filter(t => (t.pnl || 0) > 0);
    const losingTrades = closedTrades.filter(t => (t.pnl || 0) <= 0);
    
    // Calculate returns
    const finalValue = this.capital;
    const totalReturn = ((finalValue - this.config.initialCapital) / this.config.initialCapital) * 100;
    
    // Calculate duration in years
    const durationMs = ohlcv[ohlcv.length - 1].timestamp - ohlcv[0].timestamp;
    const durationYears = durationMs / (365.25 * 24 * 60 * 60 * 1000);
    const annualizedReturn = durationYears > 0 
      ? (Math.pow(finalValue / this.config.initialCapital, 1 / durationYears) - 1) * 100 
      : totalReturn;
    
    // Calculate Sharpe ratio
    const returns = this.calculateDailyReturns();
    const avgReturn = returns.length > 0 ? returns.reduce((a, b) => a + b, 0) / returns.length : 0;
    const stdReturn = returns.length > 1 
      ? Math.sqrt(returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / (returns.length - 1))
      : 0.01;
    const sharpeRatio = stdReturn > 0 ? (avgReturn / stdReturn) * Math.sqrt(252) : 0;
    
    // Calculate Sortino ratio (only downside deviation)
    const negativeReturns = returns.filter(r => r < 0);
    const downsideStd = negativeReturns.length > 1
      ? Math.sqrt(negativeReturns.reduce((sum, r) => sum + r * r, 0) / negativeReturns.length)
      : 0.01;
    const sortinoRatio = downsideStd > 0 ? (avgReturn / downsideStd) * Math.sqrt(252) : 0;
    
    // Max drawdown
    const maxDrawdown = Math.max(...this.equityCurve.map(e => e.drawdown)) * 100;
    
    // Win rate
    const winRate = closedTrades.length > 0 
      ? (winningTrades.length / closedTrades.length) * 100 
      : 0;
    
    // Profit factor
    const grossProfit = winningTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
    const grossLoss = Math.abs(losingTrades.reduce((sum, t) => sum + (t.pnl || 0), 0));
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? 999 : 0;
    
    // Average trade stats
    const avgWin = winningTrades.length > 0
      ? winningTrades.reduce((sum, t) => sum + (t.pnlPercent || 0), 0) / winningTrades.length
      : 0;
    const avgLoss = losingTrades.length > 0
      ? losingTrades.reduce((sum, t) => sum + (t.pnlPercent || 0), 0) / losingTrades.length
      : 0;
    
    // Average trade duration
    const avgTradeDuration = closedTrades.length > 0
      ? closedTrades.reduce((sum, t) => sum + ((t.exitTime || 0) - t.entryTime), 0) / closedTrades.length / (1000 * 60 * 60)
      : 0;
    
    return {
      totalReturn,
      annualizedReturn,
      sharpeRatio,
      sortinoRatio,
      maxDrawdown,
      winRate,
      profitFactor,
      totalTrades: closedTrades.length,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length,
      avgTradeDuration,
      avgWin,
      avgLoss,
      finalValue
    };
  }

  /**
   * Calculate daily returns from equity curve.
   */
  private calculateDailyReturns(): number[] {
    if (this.equityCurve.length < 2) return [];
    
    const dailyReturns: number[] = [];
    let lastDayEquity = this.equityCurve[0].equity;
    let lastDay = new Date(this.equityCurve[0].timestamp).toDateString();
    
    for (const point of this.equityCurve) {
      const currentDay = new Date(point.timestamp).toDateString();
      if (currentDay !== lastDay) {
        const dailyReturn = (point.equity - lastDayEquity) / lastDayEquity;
        dailyReturns.push(dailyReturn);
        lastDayEquity = point.equity;
        lastDay = currentDay;
      }
    }
    
    return dailyReturns;
  }
}

/**
 * Format metrics for console output.
 */
export function formatMetrics(metrics: BacktestMetrics): string {
  const lines = [
    '┌─────────────────────────────────────────────┐',
    '│         BACKTEST RESULTS                    │',
    '├─────────────────────────────────────────────┤',
    `│ Total Return:      ${metrics.totalReturn >= 0 ? '+' : ''}${metrics.totalReturn.toFixed(2)}%`.padEnd(46) + '│',
    `│ Annualized Return: ${metrics.annualizedReturn >= 0 ? '+' : ''}${metrics.annualizedReturn.toFixed(2)}%`.padEnd(46) + '│',
    `│ Sharpe Ratio:      ${metrics.sharpeRatio.toFixed(2)}`.padEnd(46) + '│',
    `│ Sortino Ratio:     ${metrics.sortinoRatio.toFixed(2)}`.padEnd(46) + '│',
    `│ Max Drawdown:      -${metrics.maxDrawdown.toFixed(2)}%`.padEnd(46) + '│',
    '├─────────────────────────────────────────────┤',
    `│ Win Rate:          ${metrics.winRate.toFixed(1)}%`.padEnd(46) + '│',
    `│ Profit Factor:     ${metrics.profitFactor.toFixed(2)}`.padEnd(46) + '│',
    `│ Total Trades:      ${metrics.totalTrades}`.padEnd(46) + '│',
    `│ Winning Trades:    ${metrics.winningTrades}`.padEnd(46) + '│',
    `│ Losing Trades:     ${metrics.losingTrades}`.padEnd(46) + '│',
    '├─────────────────────────────────────────────┤',
    `│ Avg Win:           +${metrics.avgWin.toFixed(2)}%`.padEnd(46) + '│',
    `│ Avg Loss:          ${metrics.avgLoss.toFixed(2)}%`.padEnd(46) + '│',
    `│ Avg Duration:      ${metrics.avgTradeDuration.toFixed(1)}h`.padEnd(46) + '│',
    '├─────────────────────────────────────────────┤',
    `│ Final Value:       $${metrics.finalValue.toFixed(2)}`.padEnd(46) + '│',
    '└─────────────────────────────────────────────┘'
  ];
  
  return lines.join('\n');
}



