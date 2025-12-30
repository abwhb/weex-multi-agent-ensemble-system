/**
 * @fileoverview Market data aggregation and feature engineering
 * @module data/MarketDataService
 * 
 * ## Overview
 * 
 * Aggregates real-time and historical market data, performs feature engineering
 * for agent consumption, and maintains a normalized data store for ML models.
 * 
 * ## Key Features
 * 
 * 1. **Multi-Timeframe Data**: Aggregates data across multiple timeframes
 * 2. **Technical Indicators**: Pre-computes common indicators (RSI, MA, BB, etc.)
 * 3. **Feature Normalization**: Applies rolling Z-score normalization
 * 4. **Real-Time Updates**: Supports WebSocket subscriptions for live data
 */

import { WeexClient } from '../execution/WeexClient';
import { OHLCV, MarketData, AllowedPair, ALLOWED_PAIRS, Timeframe } from '../types';

// ============================================================================
// Interfaces
// ============================================================================

/**
 * Technical indicator values.
 */
export interface Indicators {
  /** Simple Moving Averages. */
  sma: { [period: number]: number };
  /** Exponential Moving Averages. */
  ema: { [period: number]: number };
  /** Relative Strength Index. */
  rsi: { [period: number]: number };
  /** Bollinger Bands. */
  bollingerBands: {
    upper: number;
    middle: number;
    lower: number;
    width: number;
    percentB: number;
  };
  /** Average True Range. */
  atr: { [period: number]: number };
  /** MACD. */
  macd: {
    macd: number;
    signal: number;
    histogram: number;
  };
  /** Volume indicators. */
  volume: {
    sma: number;
    ratio: number;
    obv: number;
  };
}

/**
 * Subscription callback type.
 */
type DataCallback = (data: MarketData) => void;

/**
 * Subscription handle.
 */
interface Subscription {
  id: string;
  symbol: AllowedPair;
  timeframe: Timeframe;
  callback: DataCallback;
}

// ============================================================================
// MarketDataService Implementation
// ============================================================================

/**
 * Market data aggregation and feature engineering service.
 * 
 * ## Responsibilities
 * 
 * 1. Fetch and cache historical market data
 * 2. Subscribe to real-time data updates
 * 3. Calculate technical indicators
 * 4. Normalize features for ML models
 * 5. Aggregate multi-timeframe data
 * 
 * @example
 * ```typescript
 * const dataService = new MarketDataService(weexClient);
 * 
 * // Get historical data with indicators
 * const data = await dataService.getHistorical('BTC', '5m', 100);
 * 
 * // Subscribe to real-time updates
 * const subId = dataService.subscribe('BTC', '1m', (data) => {
 *   console.log('New data:', data);
 * });
 * ```
 */
export class MarketDataService {
  /** WEEX API client for data fetching. */
  private client: WeexClient;
  
  /** Cached OHLCV data by symbol and timeframe. */
  private cache: Map<string, OHLCV[]> = new Map();
  
  /** Active subscriptions. */
  private subscriptions: Map<string, Subscription> = new Map();
  
  /** Normalization statistics. */
  private normStats: Map<string, { mean: number; std: number }> = new Map();
  
  /** Cache expiry time in ms. */
  private cacheExpiry: number = 60000; // 1 minute
  
  /** Last cache update times. */
  private lastUpdate: Map<string, number> = new Map();
  
  /** Subscription counter. */
  private subCounter: number = 0;
  
  /** Update interval for subscriptions. */
  private updateInterval: NodeJS.Timeout | null = null;

  constructor(client: WeexClient) {
    this.client = client;
  }

  // ==========================================================================
  // Historical Data
  // ==========================================================================

  /**
   * Fetch historical OHLCV data with pre-computed indicators.
   * 
   * @param symbol - Trading symbol
   * @param timeframe - Candle timeframe
   * @param limit - Number of candles
   * @returns MarketData with OHLCV, indicators, and features
   */
  async getHistorical(
    symbol: AllowedPair,
    timeframe: Timeframe = '5m',
    limit: number = 100
  ): Promise<MarketData> {
    const cacheKey = `${symbol}_${timeframe}`;
    const now = Date.now();
    
    // Check cache
    let ohlcv = this.cache.get(cacheKey);
    const lastUpdate = this.lastUpdate.get(cacheKey) || 0;
    
    if (!ohlcv || now - lastUpdate > this.cacheExpiry || ohlcv.length < limit) {
      // Fetch from API
      ohlcv = await this.client.getMarketData(symbol, timeframe, limit);
      this.cache.set(cacheKey, ohlcv);
      this.lastUpdate.set(cacheKey, now);
    }
    
    // Calculate indicators
    const indicators = this.calculateIndicators(ohlcv);
    
    // Normalize features
    const features = this.normalizeFeatures(ohlcv, indicators, symbol);
    
    return {
      symbol,
      ohlcv,
      indicators: this.indicatorsToRecord(indicators),
      features
    };
  }

  /**
   * Get multi-timeframe data for a symbol.
   * 
   * @param symbol - Trading symbol
   * @param timeframes - Array of timeframes to fetch
   * @returns Map of timeframe to MarketData
   */
  async getMultiTimeframe(
    symbol: AllowedPair,
    timeframes: Timeframe[] = ['1m', '5m', '15m', '1h']
  ): Promise<Map<Timeframe, MarketData>> {
    const result = new Map<Timeframe, MarketData>();
    
    await Promise.all(timeframes.map(async (tf) => {
      const data = await this.getHistorical(symbol, tf, 100);
      result.set(tf, data);
    }));
    
    return result;
  }

  // ==========================================================================
  // Real-Time Subscriptions
  // ==========================================================================

  /**
   * Subscribe to real-time data updates for a symbol.
   * 
   * @param symbol - Trading symbol
   * @param timeframe - Candle timeframe
   * @param callback - Callback function for updates
   * @returns Subscription ID for unsubscribing
   */
  subscribe(
    symbol: AllowedPair,
    timeframe: Timeframe,
    callback: DataCallback
  ): string {
    const id = `sub_${++this.subCounter}`;
    
    this.subscriptions.set(id, {
      id,
      symbol,
      timeframe,
      callback
    });
    
    // Start polling if not already running
    if (!this.updateInterval) {
      this.startPolling();
    }
    
    console.log(`Subscribed to ${symbol} ${timeframe}: ${id}`);
    return id;
  }

  /**
   * Unsubscribe from data updates.
   * 
   * @param subscriptionId - ID returned from subscribe()
   */
  unsubscribe(subscriptionId: string): void {
    this.subscriptions.delete(subscriptionId);
    
    // Stop polling if no more subscriptions
    if (this.subscriptions.size === 0 && this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
    
    console.log(`Unsubscribed: ${subscriptionId}`);
  }

  /**
   * Start polling for data updates.
   */
  private startPolling(): void {
    this.updateInterval = setInterval(async () => {
      const uniqueKeys = new Set<string>();
      
      // Collect unique symbol/timeframe combinations
      for (const sub of this.subscriptions.values()) {
        uniqueKeys.add(`${sub.symbol}_${sub.timeframe}`);
      }
      
      // Fetch and distribute updates
      for (const key of uniqueKeys) {
        const [symbol, timeframe] = key.split('_') as [AllowedPair, Timeframe];
        
        try {
          const data = await this.getHistorical(symbol, timeframe, 100);
          
          // Notify all subscribers for this symbol/timeframe
          for (const sub of this.subscriptions.values()) {
            if (sub.symbol === symbol && sub.timeframe === timeframe) {
              sub.callback(data);
            }
          }
        } catch (error) {
          console.error(`Error fetching data for ${key}:`, error);
        }
      }
    }, 5000); // Poll every 5 seconds
  }

  // ==========================================================================
  // Technical Indicators
  // ==========================================================================

  /**
   * Calculate technical indicators from OHLCV data.
   * 
   * @param ohlcv - Candlestick data
   * @returns Calculated indicators
   */
  calculateIndicators(ohlcv: OHLCV[]): Indicators {
    const closes = ohlcv.map(c => c.close);
    const highs = ohlcv.map(c => c.high);
    const lows = ohlcv.map(c => c.low);
    const volumes = ohlcv.map(c => c.volume);
    
    return {
      sma: {
        10: this.sma(closes, 10),
        20: this.sma(closes, 20),
        50: this.sma(closes, 50)
      },
      ema: {
        12: this.ema(closes, 12),
        26: this.ema(closes, 26)
      },
      rsi: {
        14: this.rsi(closes, 14)
      },
      bollingerBands: this.bollingerBands(closes, 20, 2),
      atr: {
        14: this.atr(highs, lows, closes, 14)
      },
      macd: this.macd(closes),
      volume: {
        sma: this.sma(volumes, 20),
        ratio: volumes[volumes.length - 1] / this.sma(volumes, 20),
        obv: this.obv(closes, volumes)
      }
    };
  }

  /**
   * Simple Moving Average.
   */
  private sma(data: number[], period: number): number {
    if (data.length < period) return data[data.length - 1];
    const slice = data.slice(-period);
    return slice.reduce((a, b) => a + b, 0) / period;
  }

  /**
   * Exponential Moving Average.
   */
  private ema(data: number[], period: number): number {
    const multiplier = 2 / (period + 1);
    let ema = data[0];
    
    for (let i = 1; i < data.length; i++) {
      ema = (data[i] - ema) * multiplier + ema;
    }
    
    return ema;
  }

  /**
   * Relative Strength Index.
   */
  private rsi(data: number[], period: number): number {
    if (data.length < period + 1) return 50;
    
    let gains = 0;
    let losses = 0;
    
    for (let i = data.length - period; i < data.length; i++) {
      const change = data[i] - data[i - 1];
      if (change > 0) gains += change;
      else losses -= change;
    }
    
    const avgGain = gains / period;
    const avgLoss = losses / period;
    
    if (avgLoss === 0) return 100;
    const rs = avgGain / avgLoss;
    
    return 100 - (100 / (1 + rs));
  }

  /**
   * Bollinger Bands.
   */
  private bollingerBands(
    data: number[],
    period: number,
    stdDev: number
  ): Indicators['bollingerBands'] {
    const slice = data.slice(-period);
    const middle = slice.reduce((a, b) => a + b, 0) / period;
    
    const variance = slice.reduce((sum, v) => sum + Math.pow(v - middle, 2), 0) / period;
    const std = Math.sqrt(variance);
    
    const upper = middle + stdDev * std;
    const lower = middle - stdDev * std;
    const current = data[data.length - 1];
    
    return {
      upper,
      middle,
      lower,
      width: (upper - lower) / middle,
      percentB: (current - lower) / (upper - lower)
    };
  }

  /**
   * Average True Range.
   */
  private atr(highs: number[], lows: number[], closes: number[], period: number): number {
    const trueRanges: number[] = [];
    
    for (let i = 1; i < highs.length; i++) {
      const tr = Math.max(
        highs[i] - lows[i],
        Math.abs(highs[i] - closes[i - 1]),
        Math.abs(lows[i] - closes[i - 1])
      );
      trueRanges.push(tr);
    }
    
    return this.sma(trueRanges.slice(-period), period);
  }

  /**
   * MACD.
   */
  private macd(data: number[]): Indicators['macd'] {
    const ema12 = this.ema(data, 12);
    const ema26 = this.ema(data, 26);
    const macdLine = ema12 - ema26;
    
    // Calculate signal line (9-period EMA of MACD)
    const macdHistory: number[] = [];
    for (let i = 26; i < data.length; i++) {
      const e12 = this.ema(data.slice(0, i + 1), 12);
      const e26 = this.ema(data.slice(0, i + 1), 26);
      macdHistory.push(e12 - e26);
    }
    
    const signal = macdHistory.length >= 9 ? this.ema(macdHistory, 9) : macdLine;
    
    return {
      macd: macdLine,
      signal,
      histogram: macdLine - signal
    };
  }

  /**
   * On-Balance Volume.
   */
  private obv(closes: number[], volumes: number[]): number {
    let obv = 0;
    
    for (let i = 1; i < closes.length; i++) {
      if (closes[i] > closes[i - 1]) {
        obv += volumes[i];
      } else if (closes[i] < closes[i - 1]) {
        obv -= volumes[i];
      }
    }
    
    return obv;
  }

  /**
   * Convert Indicators object to Record for storage.
   */
  private indicatorsToRecord(indicators: Indicators): Record<string, number[]> {
    return {
      sma_10: [indicators.sma[10]],
      sma_20: [indicators.sma[20]],
      sma_50: [indicators.sma[50]],
      ema_12: [indicators.ema[12]],
      ema_26: [indicators.ema[26]],
      rsi_14: [indicators.rsi[14]],
      bb_upper: [indicators.bollingerBands.upper],
      bb_middle: [indicators.bollingerBands.middle],
      bb_lower: [indicators.bollingerBands.lower],
      bb_width: [indicators.bollingerBands.width],
      bb_percentB: [indicators.bollingerBands.percentB],
      atr_14: [indicators.atr[14]],
      macd: [indicators.macd.macd],
      macd_signal: [indicators.macd.signal],
      macd_histogram: [indicators.macd.histogram],
      volume_sma: [indicators.volume.sma],
      volume_ratio: [indicators.volume.ratio]
    };
  }

  // ==========================================================================
  // Feature Normalization
  // ==========================================================================

  /**
   * Normalize features for ML model input.
   * 
   * Applies rolling Z-score normalization to ensure features
   * are on a consistent scale across different assets and time periods.
   * 
   * @param ohlcv - Candlestick data
   * @param indicators - Calculated indicators
   * @param symbol - Symbol for caching stats
   * @returns Normalized feature vector
   */
  normalizeFeatures(
    ohlcv: OHLCV[],
    indicators: Indicators,
    symbol: AllowedPair
  ): number[] {
    const features: number[] = [];
    const closes = ohlcv.map(c => c.close);
    const lastClose = closes[closes.length - 1];
    
    // Price returns at different horizons (normalized by recent volatility)
    const returns = [
      this.calculateReturn(closes, 1),
      this.calculateReturn(closes, 5),
      this.calculateReturn(closes, 10),
      this.calculateReturn(closes, 20)
    ];
    
    const volatility = this.calculateVolatility(closes, 20);
    features.push(...returns.map(r => this.normalize(r / (volatility || 0.01), `${symbol}_return`)));
    
    // RSI (already in 0-100 range, normalize to -1 to 1)
    features.push((indicators.rsi[14] - 50) / 50);
    
    // Bollinger Band position
    features.push((indicators.bollingerBands.percentB - 0.5) * 2);
    
    // MACD histogram (normalize by price)
    features.push(this.normalize(indicators.macd.histogram / lastClose, `${symbol}_macd`));
    
    // Volume ratio (log scale)
    features.push(Math.log(indicators.volume.ratio + 0.1));
    
    // Trend strength (price vs SMAs)
    const sma20 = indicators.sma[20];
    const sma50 = indicators.sma[50];
    features.push((lastClose - sma20) / sma20);
    features.push((lastClose - sma50) / sma50);
    
    // Volatility percentile (relative to recent history)
    features.push(this.normalize(volatility, `${symbol}_vol`));
    
    return features;
  }

  /**
   * Calculate return over period.
   */
  private calculateReturn(closes: number[], period: number): number {
    if (closes.length <= period) return 0;
    const current = closes[closes.length - 1];
    const past = closes[closes.length - 1 - period];
    return (current - past) / past;
  }

  /**
   * Calculate volatility.
   */
  private calculateVolatility(closes: number[], period: number): number {
    const returns = closes.slice(-period).map((c, i, arr) =>
      i > 0 ? (c - arr[i - 1]) / arr[i - 1] : 0
    ).slice(1);
    
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance);
  }

  /**
   * Apply rolling normalization.
   */
  private normalize(value: number, key: string): number {
    let stats = this.normStats.get(key);
    
    if (!stats) {
      stats = { mean: 0, std: 1 };
    }
    
    // Exponential moving stats update
    const alpha = 0.01;
    const newMean = stats.mean * (1 - alpha) + value * alpha;
    const newStd = Math.sqrt(stats.std * stats.std * (1 - alpha) + Math.pow(value - stats.mean, 2) * alpha);
    
    this.normStats.set(key, { mean: newMean, std: newStd || 1 });
    
    return (value - newMean) / (newStd || 1);
  }

  // ==========================================================================
  // Cleanup
  // ==========================================================================

  /**
   * Stop all subscriptions and clear cache.
   */
  shutdown(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
    
    this.subscriptions.clear();
    this.cache.clear();
    this.normStats.clear();
    
    console.log('MarketDataService shutdown complete');
  }
}

