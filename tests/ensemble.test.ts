/**
 * @fileoverview Meta-learner and ensemble unit tests
 * @module tests/ensemble
 * 
 * Unit tests for the Ensemble meta-learner and related components.
 */

import { Ensemble, EnsembleConfig, TradeDecision } from '../src/meta-learner/Ensemble';
import { PerformanceTracker, TradeRecord } from '../src/meta-learner/PerformanceTracker';
import { MomentumAgent } from '../src/agents/MomentumAgent';
import { MeanReversionAgent } from '../src/agents/MeanReversionAgent';
import { VolatilityAgent } from '../src/agents/VolatilityAgent';
import { MarketData, OHLCV, AllowedPair } from '../src/types';

// ============================================================================
// Test Utilities
// ============================================================================

function generateMockOHLCV(length: number): OHLCV[] {
  const ohlcv: OHLCV[] = [];
  let price = 50000;
  
  for (let i = 0; i < length; i++) {
    price += (Math.random() - 0.5) * 100;
    ohlcv.push({
      timestamp: Date.now() - (length - i) * 60000,
      open: price - 20,
      high: price + 50,
      low: price - 50,
      close: price,
      volume: Math.random() * 1000000
    });
  }
  
  return ohlcv;
}

function generateMockMarketData(symbol: AllowedPair = 'BTC'): MarketData {
  return {
    symbol,
    ohlcv: generateMockOHLCV(100),
    indicators: {},
    features: Array(10).fill(0).map(() => Math.random() - 0.5)
  };
}

function createMockDecision(symbol: AllowedPair = 'BTC'): TradeDecision {
  return {
    action: 'buy',
    symbol,
    size: 0.05,
    confidence: 0.75,
    stopLoss: 49000,
    takeProfit: 52000,
    agentContributions: {
      MomentumAgent: 0.4,
      MeanReversionAgent: 0.2,
      VolatilityAgent: 0.4
    },
    reasoning: 'Test decision',
    timestamp: Date.now(),
    currentPrice: 50000,
    regime: 'trending'
  };
}

// ============================================================================
// Ensemble Tests
// ============================================================================

describe('Ensemble', () => {
  let ensemble: Ensemble;
  
  beforeEach(async () => {
    const config: Partial<EnsembleConfig> = {
      minConfidence: 0.55,
      learningRate: 0.05,
      performanceWindow: 50,
      maxPositionSize: 0.10
    };
    
    ensemble = new Ensemble(config);
    
    // Register mock agents
    ensemble.registerAgent(new MomentumAgent({
      lookbackPeriod: 100,
      threshold: 0.6,
      modelPath: ''
    }));
    
    ensemble.registerAgent(new MeanReversionAgent({
      lookbackPeriod: 50,
      threshold: 0.6,
      modelPath: ''
    }));
    
    ensemble.registerAgent(new VolatilityAgent({
      lookbackPeriod: 30,
      threshold: 0.6,
      modelPath: ''
    }));
    
    await ensemble.initialize();
  });
  
  afterEach(async () => {
    await ensemble.shutdown();
  });
  
  it('should combine agent signals', async () => {
    const marketData = generateMockMarketData();
    const decision = await ensemble.decide(marketData, 'BTC');
    
    // Decision might be null if confidence is too low
    if (decision) {
      expect(decision.action).toBeDefined();
      expect(['buy', 'sell', 'hold']).toContain(decision.action);
      expect(decision.symbol).toBe('BTC');
      expect(decision.confidence).toBeGreaterThanOrEqual(0);
      expect(decision.confidence).toBeLessThanOrEqual(1);
    }
  });

  it('should respect risk limits', async () => {
    const marketData = generateMockMarketData();
    const decision = await ensemble.decide(marketData, 'BTC');
    
    if (decision) {
      // Size should be within max position size
      expect(decision.size).toBeLessThanOrEqual(0.10);
      expect(decision.size).toBeGreaterThanOrEqual(0);
    }
  });
  
  it('should include agent contributions', async () => {
    const marketData = generateMockMarketData();
    const decision = await ensemble.decide(marketData, 'BTC');
    
    if (decision) {
      expect(decision.agentContributions).toBeDefined();
      expect(Object.keys(decision.agentContributions).length).toBeGreaterThan(0);
    }
  });
  
  it('should update weights after trade result', () => {
    const tradeResult = {
      pnl: 0.02,
      agentContributions: {
        MomentumAgent: 0.5,
        MeanReversionAgent: 0.2,
        VolatilityAgent: 0.3
      }
    };
    
    // Should not throw
    expect(() => ensemble.updateWeights(tradeResult)).not.toThrow();
  });
  
  it('should calculate position size using Kelly criterion', () => {
    const aggregated = {
      weightedDirection: 0.6,
      combinedConfidence: 0.75,
      agentContributions: {},
      consensus: 0.8
    };
    
    const size = ensemble.calculatePositionSize(aggregated);
    
    expect(size).toBeGreaterThanOrEqual(0);
    expect(size).toBeLessThanOrEqual(0.10);
  });
  
  it('should return null for low confidence signals', async () => {
    // Create ensemble with high confidence threshold
    const strictEnsemble = new Ensemble({ minConfidence: 0.99 });
    
    strictEnsemble.registerAgent(new MomentumAgent({
      lookbackPeriod: 100,
      threshold: 0.99, // Very high threshold
      modelPath: ''
    }));
    
    await strictEnsemble.initialize();
    
    const marketData = generateMockMarketData();
    const decision = await strictEnsemble.decide(marketData, 'BTC');
    
    // Most likely null due to high threshold
    // (unless random data happens to produce very high confidence)
    expect(decision === null || decision.confidence >= 0.99).toBe(true);
    
    await strictEnsemble.shutdown();
  });
  
  it('should generate valid AI log entries', async () => {
    const marketData = generateMockMarketData();
    const decision = await ensemble.decide(marketData, 'BTC');
    
    if (decision) {
      expect(decision.reasoning).toBeDefined();
      expect(decision.reasoning.length).toBeGreaterThan(0);
      expect(decision.timestamp).toBeGreaterThan(0);
    }
  });
});

// ============================================================================
// PerformanceTracker Tests
// ============================================================================

describe('PerformanceTracker', () => {
  let tracker: PerformanceTracker;
  
  beforeEach(() => {
    tracker = new PerformanceTracker();
  });
  
  it('should record trade entries', () => {
    const decision = createMockDecision();
    const tradeId = tracker.recordTradeEntry(decision, 50000);
    
    expect(tradeId).toBeDefined();
    expect(tradeId.startsWith('trade_')).toBe(true);
  });
  
  it('should record trade exits with PnL', () => {
    const decision = createMockDecision();
    const tradeId = tracker.recordTradeEntry(decision, 50000);
    
    const pnl = tracker.recordTradeExit(tradeId, 51000);
    
    expect(pnl).toBeDefined();
    expect(pnl).toBeGreaterThan(0); // Profit since exit > entry for long
  });
  
  it('should calculate ensemble metrics', () => {
    // Record some trades
    for (let i = 0; i < 5; i++) {
      const decision = createMockDecision();
      const tradeId = tracker.recordTradeEntry(decision, 50000);
      
      // Alternate wins and losses
      const exitPrice = i % 2 === 0 ? 51000 : 49500;
      tracker.recordTradeExit(tradeId, exitPrice);
    }
    
    const metrics = tracker.getEnsembleMetrics();
    
    expect(metrics.totalTrades).toBe(5);
    expect(metrics.winningTrades + metrics.losingTrades).toBe(5);
    expect(metrics.winRate).toBeGreaterThanOrEqual(0);
    expect(metrics.winRate).toBeLessThanOrEqual(1);
  });
  
  it('should track regime-specific performance', () => {
    // Record trades in different regimes
    const regimes = ['trending', 'ranging', 'volatile'] as const;
    
    for (const regime of regimes) {
      const decision = { ...createMockDecision(), regime };
      const tradeId = tracker.recordTradeEntry(decision, 50000);
      tracker.recordTradeExit(tradeId, 51000);
    }
    
    const regimePerformance = tracker.getRegimePerformance();
    
    expect(Object.keys(regimePerformance).length).toBeGreaterThan(0);
  });
  
  it('should export training data', () => {
    const decision = createMockDecision();
    const tradeId = tracker.recordTradeEntry(decision, 50000);
    tracker.recordTradeExit(tradeId, 51000);
    
    const exportData = tracker.exportForTraining();
    
    expect(exportData.version).toBeDefined();
    expect(exportData.exportTime).toBeGreaterThan(0);
    expect(exportData.trades.length).toBeGreaterThan(0);
  });
  
  it('should get trade history', () => {
    // Record trades for different symbols
    const symbols: AllowedPair[] = ['BTC', 'ETH', 'SOL'];
    
    for (const symbol of symbols) {
      const decision = createMockDecision(symbol);
      const tradeId = tracker.recordTradeEntry(decision, 50000);
      tracker.recordTradeExit(tradeId, 51000);
    }
    
    // Get all history
    const allHistory = tracker.getTradeHistory();
    expect(allHistory.length).toBe(3);
    
    // Get BTC only
    const btcHistory = tracker.getTradeHistory('BTC');
    expect(btcHistory.length).toBe(1);
    expect(btcHistory[0].symbol).toBe('BTC');
  });
  
  it('should handle cancelled trades', () => {
    const decision = createMockDecision();
    const tradeId = tracker.recordTradeEntry(decision, 50000);
    
    tracker.cancelTrade(tradeId, 'Stop loss hit');
    
    const history = tracker.getTradeHistory();
    const trade = history.find(t => t.id === tradeId);
    
    expect(trade?.status).toBe('cancelled');
  });
});

// ============================================================================
// Integration Tests
// ============================================================================

describe('Ensemble Integration', () => {
  it('should handle complete trading cycle', async () => {
    const ensemble = new Ensemble({ minConfidence: 0.3 });
    const tracker = new PerformanceTracker();
    
    // Register agents
    ensemble.registerAgent(new MomentumAgent({
      lookbackPeriod: 100,
      threshold: 0.3,
      modelPath: ''
    }));
    
    await ensemble.initialize();
    
    // Get decision
    const marketData = generateMockMarketData();
    const decision = await ensemble.decide(marketData, 'BTC');
    
    if (decision) {
      // Record trade
      const tradeId = tracker.recordTradeEntry(decision, decision.currentPrice);
      
      // Simulate trade exit
      const exitPrice = decision.action === 'buy' 
        ? decision.currentPrice * 1.01 
        : decision.currentPrice * 0.99;
      
      const pnl = tracker.recordTradeExit(tradeId, exitPrice);
      
      // Update ensemble weights
      ensemble.updateWeights({
        pnl: pnl!,
        agentContributions: decision.agentContributions
      });
      
      // Verify metrics
      const metrics = tracker.getEnsembleMetrics();
      expect(metrics.totalTrades).toBe(1);
    }
    
    await ensemble.shutdown();
  });
});

