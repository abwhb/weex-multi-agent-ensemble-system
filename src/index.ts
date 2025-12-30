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

// Data
import { MarketDataService } from './data/MarketDataService';
import { FeatureStore } from './data/FeatureStore';

// Logging
import { AILogger } from './logging/AILogger';

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
    console.log('Initializing WEEX Multi-Agent Trading System...');
    console.log('Configuration:', getConfigSummary());
    
    // Initialize execution components
    this.weexClient = new WeexClient();
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
    
    console.log('Trading system components initialized');
  }

  /**
   * Initialize all components and connect to WEEX.
   */
  async initialize(): Promise<void> {
    console.log('Connecting to WEEX API...');
    
    try {
      // Validate configuration
      validateConfig();
      
      // Connect to exchange
      await this.weexClient.connect();
      
      // Initialize ensemble (which initializes all agents)
      await this.ensemble.initialize();
      
      console.log('Trading system initialized successfully');
      console.log(`Agents registered: ${this.ensemble.getAgents().map(a => a.name).join(', ')}`);
      console.log(`Trading symbols: ${config.trading.symbols.join(', ')}`);
    } catch (error) {
      console.error('Failed to initialize trading system:', error);
      throw error;
    }
  }

  /**
   * Start the main trading loop.
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      console.warn('Trading system is already running');
      return;
    }
    
    console.log('Starting trading loop...');
    this.isRunning = true;
    
    // Run immediately, then on interval
    await this.runTradingCycle();
    
    this.loopInterval = setInterval(async () => {
      if (this.isRunning) {
        await this.runTradingCycle();
      }
    }, config.system.loopInterval);
    
    console.log(`Trading loop started (interval: ${config.system.loopInterval}ms)`);
  }

  /**
   * Execute one trading cycle.
   */
  private async runTradingCycle(): Promise<void> {
    const cycleStart = Date.now();
    
    try {
      // Check if trading is halted
      if (this.riskManager.isTradingHalted()) {
        console.log('Trading halted by risk manager');
        return;
      }
      
      // Process each symbol
      for (const symbol of config.trading.symbols) {
        await this.processSymbol(symbol as AllowedPair);
      }
      
      const cycleDuration = Date.now() - cycleStart;
      if (config.system.debug) {
        console.log(`Trading cycle completed in ${cycleDuration}ms`);
      }
    } catch (error) {
      console.error('Error in trading cycle:', error);
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
          console.log(`${symbol}: No trade signal`);
        }
        return;
      }
      
      console.log(`${symbol}: ${decision.action.toUpperCase()} signal (confidence: ${(decision.confidence * 100).toFixed(1)}%)`);
      
      // Step 3: Execute decision
      const result = await this.orderManager.executeDecision(decision);
      
      if (result.success && result.order) {
        console.log(`${symbol}: Order placed - ${result.order.id}`);
        
        // Record trade for performance tracking
        this.performanceTracker.recordTradeEntry(decision, result.fillPrice || decision.currentPrice);
      } else {
        console.log(`${symbol}: Order failed - ${result.error}`);
      }
    } catch (error) {
      console.error(`Error processing ${symbol}:`, error);
    }
  }

  /**
   * Stop the trading loop.
   */
  async stop(): Promise<void> {
    console.log('Stopping trading system...');
    
    this.isRunning = false;
    
    if (this.loopInterval) {
      clearInterval(this.loopInterval);
      this.loopInterval = null;
    }
    
    // Flush logs
    this.aiLogger.flush();
    
    // Export final logs
    await this.aiLogger.exportLog();
    
    console.log('Trading system stopped');
  }

  /**
   * Gracefully shutdown all components.
   */
  async shutdown(): Promise<void> {
    console.log('Shutting down trading system...');
    
    await this.stop();
    
    // Shutdown components
    await this.ensemble.shutdown();
    this.dataService.shutdown();
    await this.weexClient.disconnect();
    
    console.log('Trading system shutdown complete');
  }

  /**
   * Get current system status.
   */
  getStatus(): Record<string, unknown> {
    return {
      isRunning: this.isRunning,
      tradingHalted: this.riskManager.isTradingHalted(),
      dailyStats: this.riskManager.getDailyStats(),
      ensembleMetrics: this.performanceTracker.getEnsembleMetrics(),
      agentWeights: Object.fromEntries(
        this.ensemble.getAgents().map(a => [a.name, a.weight])
      )
    };
  }
}

// ============================================================================
// Main Entry Point
// ============================================================================

/**
 * Main function - entry point for the application.
 */
async function main(): Promise<void> {
  console.log('='.repeat(60));
  console.log('WEEX Multi-Agent Alpha Trading System');
  console.log('AI Wars Hackathon Submission');
  console.log('='.repeat(60));
  
  const system = new TradingSystem();
  
  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    console.log('\nReceived SIGINT, shutting down...');
    await system.shutdown();
    process.exit(0);
  });
  
  process.on('SIGTERM', async () => {
    console.log('\nReceived SIGTERM, shutting down...');
    await system.shutdown();
    process.exit(0);
  });
  
  try {
    // Initialize and start
    await system.initialize();
    await system.start();
    
    console.log('\nTrading system is running. Press Ctrl+C to stop.');
    
    // Keep the process running
    process.stdin.resume();
  } catch (error) {
    console.error('Fatal error:', error);
    await system.shutdown();
    process.exit(1);
  }
}

// Run main function
main().catch(console.error);

// Export for testing
export { TradingSystem };

