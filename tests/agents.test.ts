/**
 * @fileoverview Agent unit tests
 * @module tests/agents
 * 
 * Unit tests for all trading agents. Tests signal generation,
 * confidence scoring, and edge case handling.
 */

import { MomentumAgent, MomentumAgentConfig } from '../src/agents/MomentumAgent';
import { MeanReversionAgent, MeanReversionAgentConfig } from '../src/agents/MeanReversionAgent';
import { VolatilityAgent, VolatilityAgentConfig } from '../src/agents/VolatilityAgent';
import { MarketData, OHLCV } from '../src/types';

// ============================================================================
// Test Utilities
// ============================================================================

/**
 * Generate mock OHLCV data for testing.
 */
function generateMockOHLCV(length: number, basePrice: number = 50000): OHLCV[] {
  const ohlcv: OHLCV[] = [];
  let price = basePrice;
  
  for (let i = 0; i < length; i++) {
    const change = (Math.random() - 0.5) * 100;
    price += change;
    
    ohlcv.push({
      timestamp: Date.now() - (length - i) * 60000,
      open: price - Math.random() * 20,
      high: price + Math.random() * 50,
      low: price - Math.random() * 50,
      close: price,
      volume: Math.random() * 1000000
    });
  }
  
  return ohlcv;
}

/**
 * Generate mock MarketData for testing.
 */
function generateMockMarketData(symbol: string = 'BTC'): MarketData {
  const ohlcv = generateMockOHLCV(100);
  
  return {
    symbol,
    ohlcv,
    indicators: {
      rsi_14: [50],
      sma_20: [ohlcv[ohlcv.length - 1].close]
    },
    features: Array(10).fill(0).map(() => Math.random() - 0.5)
  };
}

// ============================================================================
// MomentumAgent Tests
// ============================================================================

describe('MomentumAgent', () => {
  let agent: MomentumAgent;
  
  beforeEach(async () => {
    const config: MomentumAgentConfig = {
      lookbackPeriod: 100,
      threshold: 0.6,
      modelPath: 'models/momentum/lstm_v1.0.0.pt',
      momentumThreshold: 0.6
    };
    
    agent = new MomentumAgent(config);
    await agent.initialize();
  });
  
  afterEach(async () => {
    await agent.shutdown();
  });
  
  it('should generate signal from market data', async () => {
    const marketData = generateMockMarketData();
    const signal = await agent.analyze(marketData);
    
    expect(signal).toBeDefined();
    expect(signal.timestamp).toBeGreaterThan(0);
    expect(['long', 'short', 'neutral']).toContain(signal.direction);
  });

  it('should return confidence between 0 and 1', async () => {
    const marketData = generateMockMarketData();
    const signal = await agent.analyze(marketData);
    
    expect(signal.confidence).toBeGreaterThanOrEqual(0);
    expect(signal.confidence).toBeLessThanOrEqual(1);
  });
  
  it('should include reasoning in signal', async () => {
    const marketData = generateMockMarketData();
    const signal = await agent.analyze(marketData);
    
    expect(signal.reasoning).toBeDefined();
    expect(signal.reasoning.length).toBeGreaterThan(0);
  });
  
  it('should track performance updates', () => {
    agent.updatePerformance(0.02);
    agent.updatePerformance(-0.01);
    agent.updatePerformance(0.03);
    
    const metrics = agent.getPerformanceMetrics();
    
    expect(metrics.tradeCount).toBe(3);
    expect(metrics.winRate).toBeCloseTo(0.67, 1);
  });
  
  it('should handle insufficient data gracefully', async () => {
    const marketData: MarketData = {
      symbol: 'BTC',
      ohlcv: generateMockOHLCV(10), // Not enough data
      indicators: {},
      features: []
    };
    
    const signal = await agent.analyze(marketData);
    
    expect(signal.direction).toBe('neutral');
  });
});

// ============================================================================
// MeanReversionAgent Tests
// ============================================================================

describe('MeanReversionAgent', () => {
  let agent: MeanReversionAgent;
  
  beforeEach(async () => {
    const config: MeanReversionAgentConfig = {
      lookbackPeriod: 50,
      threshold: 0.6,
      modelPath: 'models/mean_reversion/',
      defaultBBPeriod: 20,
      defaultBBStdDev: 2.0,
      minReversionProbability: 0.65
    };
    
    agent = new MeanReversionAgent(config);
    await agent.initialize();
  });
  
  afterEach(async () => {
    await agent.shutdown();
  });
  
  it('should detect regime correctly', async () => {
    const marketData = generateMockMarketData();
    const signal = await agent.analyze(marketData);
    
    expect(signal).toBeDefined();
    // Regime should be detected
    expect(['trending', 'ranging', 'volatile', 'quiet']).toContain(signal.regime);
  });
  
  it('should return neutral in trending markets', async () => {
    // Generate trending data
    const ohlcv: OHLCV[] = [];
    let price = 50000;
    
    for (let i = 0; i < 100; i++) {
      price += 50; // Consistent uptrend
      ohlcv.push({
        timestamp: Date.now() - (100 - i) * 60000,
        open: price - 10,
        high: price + 20,
        low: price - 20,
        close: price,
        volume: 1000000
      });
    }
    
    const marketData: MarketData = {
      symbol: 'BTC',
      ohlcv,
      indicators: {},
      features: []
    };
    
    const signal = await agent.analyze(marketData);
    
    // In a strong trend, mean reversion should be neutral
    expect(signal.direction).toBe('neutral');
  });
  
  it('should update reversion history', () => {
    // Test Bayesian update mechanism
    agent.updateReversionHistory(1.5, 'ranging', true);
    agent.updateReversionHistory(1.5, 'ranging', true);
    agent.updateReversionHistory(1.5, 'ranging', false);
    
    // History should be updated (internal state)
    // This tests the public interface
    expect(() => agent.updateReversionHistory(2.0, 'volatile', true)).not.toThrow();
  });
});

// ============================================================================
// VolatilityAgent Tests
// ============================================================================

describe('VolatilityAgent', () => {
  let agent: VolatilityAgent;
  
  beforeEach(async () => {
    const config = {
      lookbackPeriod: 30,
      threshold: 0.6,
      modelPath: 'models/volatility/',
      garchOmega: 0.00001,
      garchAlpha: 0.1,
      garchBeta: 0.85,
      breakoutThreshold: 0.6,
      volExpansionThreshold: 1.5
    };
    
    agent = new VolatilityAgent(config);
    await agent.initialize();
  });
  
  afterEach(async () => {
    await agent.shutdown();
  });
  
  it('should forecast volatility', async () => {
    const marketData = generateMockMarketData();
    const signal = await agent.analyze(marketData);
    
    expect(signal).toBeDefined();
    expect(signal.timestamp).toBeGreaterThan(0);
  });
  
  it('should detect volatility regime', async () => {
    // Generate volatile data
    const ohlcv: OHLCV[] = [];
    let price = 50000;
    
    for (let i = 0; i < 100; i++) {
      const bigMove = (Math.random() - 0.5) * 500; // Large moves
      price += bigMove;
      price = Math.max(price, 1000); // Prevent negative
      
      ohlcv.push({
        timestamp: Date.now() - (100 - i) * 60000,
        open: price - Math.abs(bigMove) * 0.3,
        high: price + Math.abs(bigMove) * 0.5,
        low: price - Math.abs(bigMove) * 0.5,
        close: price,
        volume: Math.random() * 2000000
      });
    }
    
    const marketData: MarketData = {
      symbol: 'BTC',
      ohlcv,
      indicators: {},
      features: []
    };
    
    const signal = await agent.analyze(marketData);
    
    // Should detect volatile or generate signal
    expect(signal).toBeDefined();
  });
  
  it('should update Q-values after trade', () => {
    const state = [2, 3, 1, 2];
    const action = 0;
    const reward = 0.05;
    
    // Should not throw
    expect(() => agent.updateQValues(state, action, reward)).not.toThrow();
  });
  
  it('should return confidence within valid range', async () => {
    const marketData = generateMockMarketData();
    const signal = await agent.analyze(marketData);
    
    expect(signal.confidence).toBeGreaterThanOrEqual(0);
    expect(signal.confidence).toBeLessThanOrEqual(1);
  });
});

// ============================================================================
// Cross-Agent Tests
// ============================================================================

describe('Agent Consistency', () => {
  it('all agents should produce valid signals', async () => {
    const agents = [
      new MomentumAgent({ lookbackPeriod: 100, threshold: 0.6, modelPath: '' }),
      new MeanReversionAgent({ lookbackPeriod: 50, threshold: 0.6, modelPath: '' }),
      new VolatilityAgent({ lookbackPeriod: 30, threshold: 0.6, modelPath: '' })
    ];
    
    // Initialize all
    await Promise.all(agents.map(a => a.initialize()));
    
    const marketData = generateMockMarketData();
    
    // All agents should produce valid signals
    for (const agent of agents) {
      const signal = await agent.analyze(marketData);
      
      expect(signal).toBeDefined();
      expect(signal.timestamp).toBeGreaterThan(0);
      expect(['long', 'short', 'neutral']).toContain(signal.direction);
      expect(signal.confidence).toBeGreaterThanOrEqual(0);
      expect(signal.confidence).toBeLessThanOrEqual(1);
      expect(signal.reasoning).toBeDefined();
    }
    
    // Shutdown all
    await Promise.all(agents.map(a => a.shutdown()));
  });
});

