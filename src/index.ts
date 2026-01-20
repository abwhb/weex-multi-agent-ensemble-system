/**
 * @fileoverview Main entry point for the WEEX Multi-Agent Trading System
 * @module index
 * 
 * ## Overview
 * 
 * This is the application entry point that initializes all agents, the meta-learner,
 * and execution components. It runs the main trading loop that:
 * 
 * 1. Fetches market data
 * 2. Runs all agents to generate signals
 * 3. Combines signals via the ensemble meta-learner
 * 4. Executes trades via WEEX API
 * 5. Logs all AI decisions for competition compliance
 * 
 * ## Architecture
 * 
 * ```
 * ┌─────────────────────────────────────────────────────────────┐
 * │                      MARKET DATA                            │
 * └─────────────────────┬───────────────────────────────────────┘
 *                       │
 *                       ▼
 * ┌─────────────────────────────────────────────────────────────┐
 * │                   FEATURE ENGINEERING                        │
 * │  • Technical indicators  • Normalized features  • Regime    │
 * └─────────────────────┬───────────────────────────────────────┘
 *                       │
 *         ┌─────────────┼─────────────┐
 *         ▼             ▼             ▼
 * ┌───────────┐  ┌───────────┐  ┌───────────┐
 * │ MOMENTUM  │  │   MEAN    │  │ VOLATILITY│
 * │   AGENT   │  │ REVERSION │  │   AGENT   │
 * │  (LSTM)   │  │  (XGBoost)│  │  (GARCH)  │
 * └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
 *       │              │              │
 *       └──────────────┼──────────────┘
 *                      ▼
 * ┌─────────────────────────────────────────────────────────────┐
 * │                    META-LEARNER                              │
 * │  • Signal combination (Gradient Boosting)                   │
 * │  • Adaptive weighting (Online Learning)                     │
 * │  • Position sizing (Kelly Criterion)                        │
 * └─────────────────────┬───────────────────────────────────────┘
 *                       │
 *                       ▼
 * ┌─────────────────────────────────────────────────────────────┐
 * │                 RISK MANAGEMENT                              │
 * │  • Max 20x leverage  • Position limits  • Drawdown control  │
 * └─────────────────────┬───────────────────────────────────────┘
 *                       │
 *                       ▼
 * ┌─────────────────────────────────────────────────────────────┐
 * │                   EXECUTION                                  │
 * │  • WEEX API  • Order management  • AI logging               │
 * └─────────────────────────────────────────────────────────────┘
 * ```
 */

import { config, validateConfig, getConfigSummary } from './config';

// Agents
import { MomentumAgent } from './agents/MomentumAgent';
import { MeanReversionAgent } from './agents/MeanReversionAgent';
import { VolatilityAgent } from './agents/VolatilityAgent';

// Meta-learner
import { Ensemble } from './meta-learner/Ensemble';
import { PerformanceTracker } from './meta-learner/PerformanceTracker';

// Execution
import { WeexClient } from './execution/WeexClient';
import { RiskManager } from './execution/RiskManager';
import { OrderManager } from './execution/OrderManager';
import { PaperTradingEngine } from './execution/PaperTradingEngine';

// Data
import { MarketDataService } from './data/MarketDataService';
import { FeatureStore } from './data/FeatureStore';

// Logging
import { AILogger } from './logging/AILogger';
import { Logger, systemLog, tradeLog } from './logging/Logger';

// Database
import { AppDatabase } from './database';

// Control
import { TradingController } from './control';

// Types
import { AllowedPair, MarketData } from './types';

// ============================================================================
// Trading System Class
// ============================================================================

/**
 * Main trading system orchestrator.
 * 
 * Coordinates all components and runs the main trading loop.
 */
class TradingSystem {
  // Database
  private db: AppDatabase;

  // Control
  private tradingController: TradingController;

  // Paper Trading
  private paperEngine: PaperTradingEngine;

  // Components
  private weexClient: WeexClient;
  private riskManager: RiskManager;
  private aiLogger: AILogger;
  private orderManager: OrderManager;
  private dataService: MarketDataService;
  private featureStore: FeatureStore;
  private ensemble: Ensemble;
  private performanceTracker: PerformanceTracker;

  // Agents
  private momentumAgent: MomentumAgent;
  private meanReversionAgent: MeanReversionAgent;
  private volatilityAgent: VolatilityAgent;

  // State
  private isRunning: boolean = false;
  private loopInterval: NodeJS.Timeout | null = null;

  constructor() {
    systemLog.info('Initializing WEEX Multi-Agent Trading System...');
    systemLog.debug('Configuration:', getConfigSummary());

    // Initialize database (async initialization happens in initialize())
    this.db = new AppDatabase({
      path: config.database.path,
      runMigrations: config.database.runMigrations
    });

    // Placeholder initializations - real initialization happens in initialize()
    this.tradingController = null as any;
    this.paperEngine = null as any;
    this.weexClient = null as any;
    this.riskManager = null as any;
    this.aiLogger = null as any;
    this.orderManager = null as any;
    this.dataService = null as any;
    this.featureStore = null as any;
    this.momentumAgent = null as any;
    this.meanReversionAgent = null as any;
    this.volatilityAgent = null as any;
    this.ensemble = null as any;
    this.performanceTracker = null as any;
  }

  private async initializeComponents(): Promise<void> {
    // Initialize database asynchronously
    await this.db.initializeAsync();
    systemLog.info(`Database initialized at ${config.database.path}`);

    // Initialize trading controller (trading disabled by default)
    this.tradingController = new TradingController(this.db);
    const tradingStatus = this.tradingController.isEnabled() ? 'ENABLED' : 'DISABLED';
    systemLog.info(`Trading mode: ${this.tradingController.getMode()} (trading ${tradingStatus})`);

    // Initialize paper trading engine
    this.paperEngine = new PaperTradingEngine(this.db, config.paperTrading.initialBalance);
    systemLog.info(`Paper trading engine initialized with ${config.paperTrading.initialBalance} USDT`);

    // Initialize execution components with trading controller and paper engine
    this.weexClient = new WeexClient(this.tradingController, this.paperEngine);
    this.riskManager = new RiskManager({
      maxLeverage: config.trading.maxLeverage,
      maxPositionSize: config.trading.maxPositionSize,
      maxDailyDrawdown: config.trading.maxDailyDrawdown
    });
    this.aiLogger = new AILogger(config.logging.aiLogPath);
    this.orderManager = new OrderManager(
      this.weexClient,
      this.riskManager,
      this.aiLogger
    );

    // Initialize data components
    this.dataService = new MarketDataService(this.weexClient);
    this.featureStore = new FeatureStore();

    // Initialize agents
    this.momentumAgent = new MomentumAgent({
      lookbackPeriod: config.agents.momentum.lookback,
      threshold: config.agents.momentum.threshold,
      modelPath: config.agents.momentum.modelPath,
      momentumThreshold: config.agents.momentum.momentumThreshold
    });

    this.meanReversionAgent = new MeanReversionAgent({
      lookbackPeriod: config.agents.meanReversion.lookback,
      threshold: config.agents.meanReversion.threshold,
      modelPath: config.agents.meanReversion.modelPath,
      defaultBBStdDev: config.agents.meanReversion.stdDev,
      minReversionProbability: config.agents.meanReversion.minReversionProb
    });

    this.volatilityAgent = new VolatilityAgent({
      lookbackPeriod: config.agents.volatility.lookback,
      threshold: config.agents.volatility.threshold,
      modelPath: config.agents.volatility.modelPath,
      breakoutThreshold: config.agents.volatility.breakoutThreshold,
      volExpansionThreshold: config.agents.volatility.volExpansionThreshold
    });

    // Initialize ensemble
    this.ensemble = new Ensemble({
      minConfidence: config.ensemble.minConfidence,
      learningRate: config.ensemble.learningRate,
      performanceWindow: config.ensemble.performanceWindow,
      maxPositionSize: config.ensemble.maxPositionSize,
      correlationThreshold: config.ensemble.correlationThreshold
    });

    // Register agents with ensemble
    this.ensemble.registerAgent(this.momentumAgent);
    this.ensemble.registerAgent(this.meanReversionAgent);
    this.ensemble.registerAgent(this.volatilityAgent);

    // Initialize performance tracker
    this.performanceTracker = new PerformanceTracker();

    systemLog.info('Trading system components initialized');
  }

  /**
   * Initialize all components and connect to WEEX.
   */
  async initialize(): Promise<void> {
    try {
      systemLog.banner('WEEX Multi-Agent Trading System');

      // Show paper mode warning
      if (config.trading.mode === 'paper') {
        tradeLog.paperMode();
      }

      // Validate configuration
      validateConfig();

      // Initialize all components (including async database)
      await this.initializeComponents();

      systemLog.info('Connecting to WEEX API...');

      // Connect to exchange
      await this.weexClient.connect();

      // Initialize ensemble (which initializes all agents)
      await this.ensemble.initialize();

      systemLog.separator('Initialization Complete');
      systemLog.info(`Trading mode: ${config.trading.mode.toUpperCase()}`);
      systemLog.info(`Agents: ${this.ensemble.getAgents().map(a => a.name).join(', ')}`);
      systemLog.info(`Symbols: ${config.trading.symbols.join(', ')}`);
      systemLog.info(`Max leverage: ${config.trading.maxLeverage}x`);
      systemLog.info(`Risk per trade: ${(config.trading.riskPerTrade * 100).toFixed(1)}%`);
      systemLog.separator();
    } catch (error) {
      systemLog.error('Failed to initialize trading system', error);
      throw error;
    }
  }

  /**
   * Start the main trading loop.
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      systemLog.warn('Trading system is already running');
      return;
    }

    systemLog.separator('Starting Trading Loop');
    this.isRunning = true;

    // Run immediately, then on interval
    await this.runTradingCycle();

    this.loopInterval = setInterval(async () => {
      if (this.isRunning) {
        await this.runTradingCycle();
      }
    }, config.system.loopInterval);

    systemLog.info(`Trading loop started (interval: ${config.system.loopInterval / 1000}s)`);
  }

  /**
   * Execute one trading cycle.
   */
  private async runTradingCycle(): Promise<void> {
    const cycleStart = Date.now();

    try {
      // Check if trading is halted
      if (this.riskManager.isTradingHalted()) {
        systemLog.warn('Trading halted by risk manager');
        return;
      }

      // Process each symbol
      for (const symbol of config.trading.symbols) {
        await this.processSymbol(symbol as AllowedPair);
      }

      const cycleDuration = Date.now() - cycleStart;
      if (config.system.debug) {
        systemLog.debug(`Trading cycle completed in ${cycleDuration}ms`);
      }
    } catch (error) {
      systemLog.error('Error in trading cycle', error);
    }
  }

  /**
   * Process a single symbol through the trading pipeline.
   */
  private async processSymbol(symbol: AllowedPair): Promise<void> {
    try {
      // Step 1: Fetch market data
      const marketData = await this.dataService.getHistorical(
        symbol,
        config.trading.defaultTimeframe,
        config.agents.momentum.lookback
      );

      // Step 2: Get decision from ensemble
      const decision = await this.ensemble.decide(marketData, symbol);

      if (!decision) {
        // No trade signal
        if (config.system.debug) {
          systemLog.debug(`${symbol}: No trade signal`);
        }
        return;
      }

      // Log the trade signal
      const action = decision.action === 'buy' ? 'BUY' : decision.action === 'sell' ? 'SELL' : 'HOLD';
      tradeLog.trade(action, symbol, {
        price: decision.currentPrice,
        size: decision.size,
        confidence: decision.confidence,
        reason: decision.reasoning
      });

      // Step 3: Execute decision
      const result = await this.orderManager.executeDecision(decision);

      if (result.success && result.order) {
        tradeLog.info(`Order placed`, { orderId: result.order.id, symbol });

        // Record trade for performance tracking
        this.performanceTracker.recordTradeEntry(decision, result.fillPrice || decision.currentPrice);
      } else {
        tradeLog.warn(`Order failed: ${result.error}`, { symbol });
      }
    } catch (error) {
      systemLog.error(`Error processing ${symbol}`, error);
    }
  }

  /**
   * Stop the trading loop.
   */
  async stop(): Promise<void> {
    systemLog.info('Stopping trading system...');

    this.isRunning = false;

    if (this.loopInterval) {
      clearInterval(this.loopInterval);
      this.loopInterval = null;
    }

    // Flush logs
    this.aiLogger.flush();

    // Export final logs
    await this.aiLogger.exportLog();

    systemLog.info('Trading system stopped');
  }

  /**
   * Gracefully shutdown all components.
   */
  async shutdown(): Promise<void> {
    systemLog.separator('Shutting Down');
    systemLog.info('Shutting down trading system...');

    await this.stop();

    // Shutdown components
    await this.ensemble.shutdown();
    this.dataService.shutdown();
    await this.weexClient.disconnect();

    // Close database
    this.db.close();
    systemLog.info('Database connection closed');

    systemLog.info('Trading system shutdown complete');
  }

  /**
   * Get current system status.
   */
  getStatus(): Record<string, unknown> {
    return {
      isRunning: this.isRunning,
      tradingEnabled: this.tradingController.isEnabled(),
      tradingMode: this.tradingController.getMode(),
      tradingHalted: this.riskManager.isTradingHalted(),
      dailyStats: this.riskManager.getDailyStats(),
      ensembleMetrics: this.performanceTracker.getEnsembleMetrics(),
      agentWeights: Object.fromEntries(
        this.ensemble.getAgents().map(a => [a.name, a.weight])
      ),
      paperTradingPnl: this.paperEngine.getPnlSummary()
    };
  }

  /**
   * Enable live trading (switch from paper trading).
   * WARNING: This will send real orders to the exchange.
   */
  enableTrading(): void {
    this.tradingController.enableTrading();
  }

  /**
   * Disable live trading (switch to paper trading).
   * This is the default safe mode.
   */
  disableTrading(): void {
    this.tradingController.disableTrading();
  }

  /**
   * Get paper trading P&L summary.
   */
  getPnlSummary() {
    return this.paperEngine.getPnlSummary();
  }

  /**
   * Get paper trading trade history.
   */
  getTradeHistory(options?: { symbol?: string; limit?: number }) {
    return this.paperEngine.getTradeHistory(options);
  }
}

// ============================================================================
// Main Entry Point
// ============================================================================

/**
 * Main function - entry point for the application.
 */
async function main(): Promise<void> {
  systemLog.banner('WEEX Multi-Agent Alpha Trading System');
  systemLog.info('AI Wars Hackathon Submission');
  systemLog.separator();

  const system = new TradingSystem();

  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    systemLog.warn('\nReceived SIGINT, shutting down...');
    await system.shutdown();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    systemLog.warn('\nReceived SIGTERM, shutting down...');
    await system.shutdown();
    process.exit(0);
  });

  try {
    // Initialize and start
    await system.initialize();
    await system.start();

    systemLog.info('Trading system is running. Press Ctrl+C to stop.');

    // Keep the process running
    process.stdin.resume();
  } catch (error) {
    systemLog.error('Fatal error', error);
    await system.shutdown();
    process.exit(1);
  }
}

// Run main function
main().catch(console.error);

// Export for testing
export { TradingSystem };

