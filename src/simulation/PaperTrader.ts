/**
 * @fileoverview Paper trading simulation with live market data
 * @module simulation/PaperTrader
 * 
 * Runs the full trading system with real-time prices from Binance
 * but simulates order execution without real money.
 */

import { BinanceClient } from '../data/BinanceClient';
import { SimpleEnsemble } from '../backtest/SimpleEnsemble';
import { AllowedPair, OHLCV, MarketData, Timeframe } from '../types';
import { TradeDecision } from '../meta-learner/Ensemble';
import { AILogger } from '../logging/AILogger';

// ============================================================================
// Types
// ============================================================================

export interface PaperPosition {
  id: number;
  symbol: AllowedPair;
  direction: 'long' | 'short';
  entryTime: number;
  entryPrice: number;
  size: number;
  confidence: number;
  unrealizedPnl: number;
  stopLoss: number;
  takeProfit: number;
}

export interface PaperTrade {
  id: number;
  symbol: AllowedPair;
  direction: 'long' | 'short';
  entryTime: number;
  entryPrice: number;
  exitTime: number;
  exitPrice: number;
  size: number;
  pnl: number;
  pnlPercent: number;
  confidence: number;
  reasoning: string;
}

export interface PaperTradingConfig {
  symbols: AllowedPair[];
  interval: Timeframe;
  initialCapital: number;
  maxPositionSize: number;
  updateIntervalMs: number;
}

export interface PortfolioState {
  capital: number;
  equity: number;
  positions: Map<string, PaperPosition>;
  trades: PaperTrade[];
  startTime: number;
  peakEquity: number;
}

// ============================================================================
// Paper Trader Implementation
// ============================================================================

/**
 * Paper trading simulator that uses live market data.
 * 
 * @example
 * ```typescript
 * const trader = new PaperTrader({
 *   symbols: ['BTC', 'ETH'],
 *   interval: '5m',
 *   initialCapital: 10000
 * });
 * 
 * await trader.start();
 * // ... runs until stopped
 * await trader.stop();
 * ```
 */
export class PaperTrader {
  private config: PaperTradingConfig;
  private client: BinanceClient;
  private ensemble: SimpleEnsemble;
  private portfolio: PortfolioState;
  private isRunning = false;
  private updateInterval: NodeJS.Timeout | null = null;
  private tradeCounter = 0;
  private candles: Map<string, OHLCV[]> = new Map();
  private aiLogger: AILogger;

  // Callbacks for UI updates
  private onUpdate: ((state: PortfolioState) => void) | null = null;
  private onSignal: ((symbol: string, decision: TradeDecision | null) => void) | null = null;
  private onTrade: ((trade: PaperTrade) => void) | null = null;

  constructor(config: Partial<PaperTradingConfig> = {}) {
    this.config = {
      symbols: ['BTC', 'ETH', 'SOL'],
      interval: '5m',
      initialCapital: 10000,
      maxPositionSize: 0.10,
      updateIntervalMs: 60000, // 1 minute default
      ...config
    };

    this.client = new BinanceClient();
    this.ensemble = new SimpleEnsemble();

    this.portfolio = {
      capital: this.config.initialCapital,
      equity: this.config.initialCapital,
      positions: new Map(),
      trades: [],
      startTime: Date.now(),
      peakEquity: this.config.initialCapital
    };

    // Initialize AI Logger for competition compliance
    this.aiLogger = new AILogger('./logs/paper_trading_ai_decisions.json');
    this.aiLogger.registerModelVersions({
      'MomentumAgent': '1.0.0',
      'MeanReversionAgent': '1.0.0',
      'VolatilityAgent': '1.0.0',
      'SimpleEnsemble': '1.0.0'
    });
  }

  /**
   * Start paper trading.
   */
  async start(): Promise<void> {
    console.log('\n╔═══════════════════════════════════════════════════════════╗');
    console.log('║   WEEX Multi-Agent Ensemble - Paper Trading Simulation    ║');
    console.log('╚═══════════════════════════════════════════════════════════╝\n');

    console.log('Configuration:');
    console.log(`  Symbols:     ${this.config.symbols.join(', ')}`);
    console.log(`  Interval:    ${this.config.interval}`);
    console.log(`  Capital:     $${this.config.initialCapital.toLocaleString()}`);
    console.log(`  Update:      Every ${this.config.updateIntervalMs / 1000}s`);
    console.log('');

    // Connect to Binance
    console.log('📡 Connecting to Binance...');
    await this.client.connect();

    // Fetch initial candle data for warmup
    console.log('📊 Fetching historical data for warmup...');
    for (const symbol of this.config.symbols) {
      const candles = await this.client.getHistoricalData(symbol, this.config.interval, 200);
      this.candles.set(symbol, candles);
      console.log(`  ${symbol}: ${candles.length} candles loaded`);
    }

    console.log('');
    console.log('🚀 Starting paper trading...');
    console.log('   Press Ctrl+C to stop\n');

    this.isRunning = true;
    this.portfolio.startTime = Date.now();

    // Start update loop
    this.updateInterval = setInterval(() => this.update(), this.config.updateIntervalMs);

    // Run first update immediately
    await this.update();
  }

  /**
   * Stop paper trading.
   */
  async stop(): Promise<void> {
    console.log('\n🛑 Stopping paper trading...');

    this.isRunning = false;

    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }

    // Close all positions at current prices
    for (const [symbol, position] of this.portfolio.positions) {
      const price = await this.client.getCurrentPrice(symbol);
      this.closePosition(symbol as AllowedPair, price, 'Session ended');
    }

    await this.client.disconnect();

    // Export AI logs for competition
    const logPath = await this.aiLogger.exportLog();
    console.log(`📋 AI decision logs exported to: ${logPath}`);

    this.printSummary();
  }

  /**
   * Main update loop - called every updateIntervalMs.
   */
  private async update(): Promise<void> {
    if (!this.isRunning) return;

    try {
      // Update candle data for each symbol
      for (const symbol of this.config.symbols) {
        await this.updateSymbol(symbol);
      }

      // Update equity
      this.updateEquity();

      // Display status
      this.displayStatus();

      // Trigger callback
      if (this.onUpdate) {
        this.onUpdate(this.portfolio);
      }
    } catch (error) {
      console.error('Update error:', error);
    }
  }

  /**
   * Update a single symbol.
   */
  private async updateSymbol(symbol: AllowedPair): Promise<void> {
    // Get latest candles
    const newCandles = await this.client.getHistoricalData(symbol, this.config.interval, 10);
    const existingCandles = this.candles.get(symbol) || [];

    // Merge new candles (avoid duplicates by timestamp)
    const existingTimestamps = new Set(existingCandles.map(c => c.timestamp));
    for (const candle of newCandles) {
      if (!existingTimestamps.has(candle.timestamp)) {
        existingCandles.push(candle);
      } else {
        // Update existing candle (might be incomplete)
        const idx = existingCandles.findIndex(c => c.timestamp === candle.timestamp);
        if (idx >= 0) {
          existingCandles[idx] = candle;
        }
      }
    }

    // Keep only last 200 candles
    if (existingCandles.length > 200) {
      existingCandles.splice(0, existingCandles.length - 200);
    }

    // Sort by timestamp
    existingCandles.sort((a, b) => a.timestamp - b.timestamp);
    this.candles.set(symbol, existingCandles);

    // Get current price
    const currentPrice = await this.client.getCurrentPrice(symbol);

    // Update position P&L if exists
    const position = this.portfolio.positions.get(symbol);
    if (position) {
      this.updatePositionPnl(position, currentPrice);

      // Check stop loss / take profit
      if (this.shouldClosePosition(position, currentPrice)) {
        this.closePosition(symbol, currentPrice, 'SL/TP triggered');
        return;
      }
    }

    // Create market data for ensemble
    const marketData: MarketData = {
      symbol,
      ohlcv: existingCandles.slice(-100),
      indicators: {},
      features: []
    };

    // Get trading decision
    const decision = await this.ensemble.decide(marketData, symbol);

    // Trigger signal callback
    if (this.onSignal) {
      this.onSignal(symbol, decision);
    }

    // Process decision
    if (decision) {
      // Log the AI decision for competition compliance
      this.aiLogger.logDecision(decision);
      this.processDecision(symbol, decision, currentPrice);
    }
  }

  /**
   * Process a trading decision.
   */
  private processDecision(
    symbol: AllowedPair,
    decision: TradeDecision,
    currentPrice: number
  ): void {
    const existingPosition = this.portfolio.positions.get(symbol);

    if (decision.action === 'hold') {
      return;
    }

    const newDirection = decision.action === 'buy' ? 'long' : 'short';

    // Close opposite position if exists
    if (existingPosition && existingPosition.direction !== newDirection) {
      this.closePosition(symbol, currentPrice, 'Direction reversal');
    }

    // Open new position if none exists
    if (!this.portfolio.positions.has(symbol)) {
      this.openPosition(symbol, decision, currentPrice);
    }
  }

  /**
   * Open a new position.
   */
  private openPosition(
    symbol: AllowedPair,
    decision: TradeDecision,
    currentPrice: number
  ): void {
    const direction = decision.action === 'buy' ? 'long' : 'short';
    const size = Math.min(
      decision.size * this.portfolio.capital,
      this.config.maxPositionSize * this.portfolio.capital
    );

    // Apply simulated slippage
    const slippage = 1 + (direction === 'long' ? 0.001 : -0.001);
    const entryPrice = currentPrice * slippage;

    const position: PaperPosition = {
      id: ++this.tradeCounter,
      symbol,
      direction,
      entryTime: Date.now(),
      entryPrice,
      size,
      confidence: decision.confidence,
      unrealizedPnl: 0,
      stopLoss: decision.stopLoss,
      takeProfit: decision.takeProfit
    };

    this.portfolio.positions.set(symbol, position);

    console.log(`\n  📈 OPENED ${direction.toUpperCase()} ${symbol} @ $${entryPrice.toFixed(2)} (size: $${size.toFixed(2)})`);
    console.log(`     ${decision.reasoning}`);
  }

  /**
   * Close a position.
   */
  private closePosition(
    symbol: AllowedPair,
    currentPrice: number,
    reason: string
  ): void {
    const position = this.portfolio.positions.get(symbol);
    if (!position) return;

    // Apply simulated slippage
    const slippage = 1 + (position.direction === 'long' ? -0.001 : 0.001);
    const exitPrice = currentPrice * slippage;

    // Calculate P&L
    const priceChange = exitPrice - position.entryPrice;
    const direction = position.direction === 'long' ? 1 : -1;
    const pnl = (priceChange / position.entryPrice) * position.size * direction;
    const pnlPercent = (pnl / position.size) * 100;

    // Apply fees
    const fees = position.size * 0.002; // 0.1% entry + 0.1% exit
    const netPnl = pnl - fees;

    // Update capital
    this.portfolio.capital += netPnl;

    // Create trade record
    const trade: PaperTrade = {
      id: position.id,
      symbol,
      direction: position.direction,
      entryTime: position.entryTime,
      entryPrice: position.entryPrice,
      exitTime: Date.now(),
      exitPrice,
      size: position.size,
      pnl: netPnl,
      pnlPercent,
      confidence: position.confidence,
      reasoning: reason
    };

    this.portfolio.trades.push(trade);
    this.portfolio.positions.delete(symbol);

    const pnlStr = netPnl >= 0 ? `+$${netPnl.toFixed(2)}` : `-$${Math.abs(netPnl).toFixed(2)}`;
    const icon = netPnl >= 0 ? '✅' : '❌';
    console.log(`\n  ${icon} CLOSED ${position.direction.toUpperCase()} ${symbol} @ $${exitPrice.toFixed(2)} (${pnlStr})`);
    console.log(`     Reason: ${reason}`);

    // Trigger callback
    if (this.onTrade) {
      this.onTrade(trade);
    }

    // Update ensemble weights based on trade result
    this.ensemble.updateWeights(
      {
        momentum: position.direction === 'long' ? 0.5 : -0.5,
        meanReversion: 0,
        volatility: 0.3
      },
      netPnl
    );
  }

  /**
   * Update position P&L.
   */
  private updatePositionPnl(position: PaperPosition, currentPrice: number): void {
    const priceChange = currentPrice - position.entryPrice;
    const direction = position.direction === 'long' ? 1 : -1;
    position.unrealizedPnl = (priceChange / position.entryPrice) * position.size * direction;
  }

  /**
   * Check if position should be closed (SL/TP).
   */
  private shouldClosePosition(position: PaperPosition, currentPrice: number): boolean {
    if (position.direction === 'long') {
      return currentPrice <= position.stopLoss || currentPrice >= position.takeProfit;
    } else {
      return currentPrice >= position.stopLoss || currentPrice <= position.takeProfit;
    }
  }

  /**
   * Update portfolio equity.
   */
  private updateEquity(): void {
    let unrealized = 0;
    for (const position of this.portfolio.positions.values()) {
      unrealized += position.unrealizedPnl;
    }

    this.portfolio.equity = this.portfolio.capital + unrealized;

    if (this.portfolio.equity > this.portfolio.peakEquity) {
      this.portfolio.peakEquity = this.portfolio.equity;
    }
  }

  /**
   * Display current status.
   */
  private displayStatus(): void {
    const equity = this.portfolio.equity;
    const startCapital = this.config.initialCapital;
    const totalReturn = ((equity - startCapital) / startCapital) * 100;
    const drawdown = ((this.portfolio.peakEquity - equity) / this.portfolio.peakEquity) * 100;
    const runtime = (Date.now() - this.portfolio.startTime) / 1000 / 60; // minutes

    console.clear();
    console.log('╔═══════════════════════════════════════════════════════════════╗');
    console.log('║       PAPER TRADING SIMULATION - LIVE                         ║');
    console.log('╠═══════════════════════════════════════════════════════════════╣');
    console.log(`║ Portfolio Value: $${equity.toFixed(2).padStart(12)} (${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%)`.padEnd(66) + '║');
    console.log(`║ Drawdown:        ${drawdown.toFixed(2).padStart(12)}%`.padEnd(66) + '║');
    console.log(`║ Runtime:         ${runtime.toFixed(1).padStart(12)} min`.padEnd(66) + '║');
    console.log('╠═══════════════════════════════════════════════════════════════╣');
    console.log('║ Open Positions:                                               ║');

    if (this.portfolio.positions.size === 0) {
      console.log('║   (none)                                                       ║');
    } else {
      for (const [symbol, pos] of this.portfolio.positions) {
        const pnlStr = pos.unrealizedPnl >= 0
          ? `+$${pos.unrealizedPnl.toFixed(2)}`
          : `-$${Math.abs(pos.unrealizedPnl).toFixed(2)}`;
        console.log(`║   ${symbol}: ${pos.direction.toUpperCase().padEnd(5)} @ $${pos.entryPrice.toFixed(2).padStart(10)} (${pnlStr})`.padEnd(66) + '║');
      }
    }

    console.log('╠═══════════════════════════════════════════════════════════════╣');
    console.log(`║ Completed Trades: ${this.portfolio.trades.length}`.padEnd(66) + '║');

    const winningTrades = this.portfolio.trades.filter(t => t.pnl > 0).length;
    const winRate = this.portfolio.trades.length > 0
      ? (winningTrades / this.portfolio.trades.length) * 100
      : 0;
    console.log(`║ Win Rate:         ${winRate.toFixed(1)}%`.padEnd(66) + '║');

    console.log('╠═══════════════════════════════════════════════════════════════╣');
    console.log('║ Recent Trades:                                                ║');

    const recentTrades = this.portfolio.trades.slice(-3);
    if (recentTrades.length === 0) {
      console.log('║   (none yet)                                                   ║');
    } else {
      for (const trade of recentTrades) {
        const time = new Date(trade.exitTime).toLocaleTimeString();
        const pnlStr = trade.pnl >= 0 ? `+$${trade.pnl.toFixed(2)}` : `-$${Math.abs(trade.pnl).toFixed(2)}`;
        console.log(`║   [${time}] ${trade.direction.toUpperCase().padEnd(5)} ${trade.symbol.padEnd(4)} ${pnlStr.padStart(10)}`.padEnd(66) + '║');
      }
    }

    console.log('╚═══════════════════════════════════════════════════════════════╝');
    console.log('\nPress Ctrl+C to stop\n');
  }

  /**
   * Print final summary.
   */
  private printSummary(): void {
    const trades = this.portfolio.trades;
    const equity = this.portfolio.equity;
    const startCapital = this.config.initialCapital;

    console.log('\n');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('                    SESSION SUMMARY                             ');
    console.log('═══════════════════════════════════════════════════════════════');

    console.log(`\nDuration: ${((Date.now() - this.portfolio.startTime) / 1000 / 60).toFixed(1)} minutes`);
    console.log(`Starting Capital: $${startCapital.toLocaleString()}`);
    console.log(`Final Equity: $${equity.toFixed(2)}`);
    console.log(`Total Return: ${((equity - startCapital) / startCapital * 100).toFixed(2)}%`);
    console.log(`Peak Equity: $${this.portfolio.peakEquity.toFixed(2)}`);
    console.log(`Max Drawdown: ${((this.portfolio.peakEquity - equity) / this.portfolio.peakEquity * 100).toFixed(2)}%`);

    console.log(`\nTotal Trades: ${trades.length}`);
    if (trades.length > 0) {
      const winningTrades = trades.filter(t => t.pnl > 0);
      console.log(`Winning Trades: ${winningTrades.length}`);
      console.log(`Losing Trades: ${trades.length - winningTrades.length}`);
      console.log(`Win Rate: ${(winningTrades.length / trades.length * 100).toFixed(1)}%`);

      const totalProfit = winningTrades.reduce((s, t) => s + t.pnl, 0);
      const totalLoss = Math.abs(trades.filter(t => t.pnl < 0).reduce((s, t) => s + t.pnl, 0));
      console.log(`Profit Factor: ${totalLoss > 0 ? (totalProfit / totalLoss).toFixed(2) : 'N/A'}`);
    }

    console.log('\n═══════════════════════════════════════════════════════════════\n');
  }

  /**
   * Set callback for portfolio updates.
   */
  onPortfolioUpdate(callback: (state: PortfolioState) => void): void {
    this.onUpdate = callback;
  }

  /**
   * Set callback for trading signals.
   */
  onTradingSignal(callback: (symbol: string, decision: TradeDecision | null) => void): void {
    this.onSignal = callback;
  }

  /**
   * Set callback for completed trades.
   */
  onTradeComplete(callback: (trade: PaperTrade) => void): void {
    this.onTrade = callback;
  }

  /**
   * Get current portfolio state.
   */
  getPortfolio(): PortfolioState {
    return { ...this.portfolio };
  }
}

