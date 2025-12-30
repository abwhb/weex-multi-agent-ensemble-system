/**
 * @fileoverview Feature storage and retrieval for ML models
 * @module data/FeatureStore
 * 
 * ## Overview
 * 
 * Manages computed features for all agents. Handles feature versioning,
 * caching, and retrieval. Supports both real-time and batch feature computation.
 * 
 * ## Key Features
 * 
 * 1. **Feature Versioning**: Track feature schema versions
 * 2. **Caching**: In-memory cache with TTL
 * 3. **Batch Computation**: Efficient bulk feature calculation
 * 4. **Feature Registry**: Centralized feature definitions
 */

import { MarketData, AllowedPair, OHLCV } from '../types';

// ============================================================================
// Interfaces
// ============================================================================

/**
 * Feature definition for the registry.
 */
export interface FeatureDefinition {
  /** Unique feature name. */
  name: string;
  /** Feature description. */
  description: string;
  /** Data type. */
  type: 'continuous' | 'categorical' | 'binary';
  /** Normalization method. */
  normalization: 'z-score' | 'min-max' | 'none';
  /** Feature version. */
  version: string;
  /** Dependencies on other features. */
  dependencies: string[];
  /** Computation function. */
  compute: (data: MarketData) => number | number[];
}

/**
 * Cached feature entry.
 */
interface CacheEntry {
  features: number[];
  timestamp: number;
  version: string;
}

/**
 * Feature set for a symbol.
 */
export interface FeatureSet {
  symbol: AllowedPair;
  timestamp: number;
  features: Record<string, number[]>;
  version: string;
}

// ============================================================================
// FeatureStore Implementation
// ============================================================================

/**
 * Centralized feature storage and computation for ML models.
 * 
 * ## Responsibilities
 * 
 * 1. Register and manage feature definitions
 * 2. Compute features on demand with caching
 * 3. Ensure feature version consistency
 * 4. Support batch feature computation
 * 
 * @example
 * ```typescript
 * const store = new FeatureStore();
 * 
 * // Register a custom feature
 * store.registerFeature({
 *   name: 'price_momentum_5',
 *   description: '5-period price momentum',
 *   type: 'continuous',
 *   normalization: 'z-score',
 *   version: '1.0.0',
 *   dependencies: [],
 *   compute: (data) => {
 *     const closes = data.ohlcv.map(c => c.close);
 *     return (closes[closes.length - 1] - closes[closes.length - 6]) / closes[closes.length - 6];
 *   }
 * });
 * 
 * // Get features for a symbol
 * const features = store.getFeatures('BTC', marketData);
 * ```
 */
export class FeatureStore {
  /** Feature definitions registry. */
  private registry: Map<string, FeatureDefinition> = new Map();
  
  /** Feature cache by symbol. */
  private cache: Map<string, CacheEntry> = new Map();
  
  /** Cache TTL in milliseconds. */
  private cacheTTL: number = 5000; // 5 seconds
  
  /** Current schema version. */
  private schemaVersion: string = '1.0.0';

  constructor() {
    this.registerDefaultFeatures();
  }

  // ==========================================================================
  // Feature Registration
  // ==========================================================================

  /**
   * Register a new feature definition.
   * 
   * @param definition - Feature definition
   */
  registerFeature(definition: FeatureDefinition): void {
    if (this.registry.has(definition.name)) {
      console.warn(`Feature '${definition.name}' already registered, overwriting.`);
    }
    
    this.registry.set(definition.name, definition);
    console.log(`Registered feature: ${definition.name} v${definition.version}`);
  }

  /**
   * Register default features used by all agents.
   */
  private registerDefaultFeatures(): void {
    // Price returns at various horizons
    const returnPeriods = [1, 5, 10, 20];
    for (const period of returnPeriods) {
      this.registerFeature({
        name: `return_${period}`,
        description: `${period}-period price return`,
        type: 'continuous',
        normalization: 'z-score',
        version: '1.0.0',
        dependencies: [],
        compute: (data) => {
          const closes = data.ohlcv.map(c => c.close);
          if (closes.length <= period) return 0;
          const current = closes[closes.length - 1];
          const past = closes[closes.length - 1 - period];
          return (current - past) / past;
        }
      });
    }
    
    // RSI
    this.registerFeature({
      name: 'rsi_14',
      description: '14-period Relative Strength Index',
      type: 'continuous',
      normalization: 'min-max',
      version: '1.0.0',
      dependencies: [],
      compute: (data) => {
        const closes = data.ohlcv.map(c => c.close);
        return this.calculateRSI(closes, 14);
      }
    });
    
    // Bollinger Band position
    this.registerFeature({
      name: 'bb_position',
      description: 'Position within Bollinger Bands (-1 to 1)',
      type: 'continuous',
      normalization: 'none',
      version: '1.0.0',
      dependencies: [],
      compute: (data) => {
        const closes = data.ohlcv.map(c => c.close);
        const { percentB } = this.calculateBollingerBands(closes, 20, 2);
        return (percentB - 0.5) * 2;
      }
    });
    
    // Volume ratio
    this.registerFeature({
      name: 'volume_ratio',
      description: 'Current volume vs 20-period average',
      type: 'continuous',
      normalization: 'z-score',
      version: '1.0.0',
      dependencies: [],
      compute: (data) => {
        const volumes = data.ohlcv.map(c => c.volume);
        const avgVolume = volumes.slice(-20).reduce((a, b) => a + b, 0) / 20;
        return Math.log((volumes[volumes.length - 1] / avgVolume) + 0.1);
      }
    });
    
    // Volatility
    this.registerFeature({
      name: 'volatility_20',
      description: '20-period realized volatility',
      type: 'continuous',
      normalization: 'z-score',
      version: '1.0.0',
      dependencies: [],
      compute: (data) => {
        const closes = data.ohlcv.map(c => c.close);
        return this.calculateVolatility(closes, 20);
      }
    });
    
    // Trend strength
    this.registerFeature({
      name: 'trend_strength',
      description: 'Price position relative to 50-period SMA',
      type: 'continuous',
      normalization: 'z-score',
      version: '1.0.0',
      dependencies: [],
      compute: (data) => {
        const closes = data.ohlcv.map(c => c.close);
        const sma50 = closes.slice(-50).reduce((a, b) => a + b, 0) / 50;
        return (closes[closes.length - 1] - sma50) / sma50;
      }
    });
    
    console.log(`Registered ${this.registry.size} default features`);
  }

  // ==========================================================================
  // Feature Computation
  // ==========================================================================

  /**
   * Get all features for a symbol.
   * 
   * @param symbol - Trading symbol
   * @param data - Market data
   * @param features - Optional specific features to compute
   * @returns Feature set
   */
  getFeatures(
    symbol: AllowedPair,
    data: MarketData,
    features?: string[]
  ): FeatureSet {
    const now = Date.now();
    const cacheKey = symbol;
    
    // Check cache
    const cached = this.cache.get(cacheKey);
    if (cached && now - cached.timestamp < this.cacheTTL) {
      return {
        symbol,
        timestamp: cached.timestamp,
        features: this.arrayToRecord(cached.features),
        version: cached.version
      };
    }
    
    // Compute features
    const featureNames = features || Array.from(this.registry.keys());
    const computedFeatures: Record<string, number[]> = {};
    const featureVector: number[] = [];
    
    for (const name of featureNames) {
      const definition = this.registry.get(name);
      if (!definition) {
        console.warn(`Feature '${name}' not found in registry`);
        continue;
      }
      
      try {
        const value = definition.compute(data);
        const normalizedValue = this.normalizeValue(
          Array.isArray(value) ? value[0] : value,
          definition.normalization,
          name
        );
        
        computedFeatures[name] = [normalizedValue];
        featureVector.push(normalizedValue);
      } catch (error) {
        console.error(`Error computing feature '${name}':`, error);
        computedFeatures[name] = [0];
        featureVector.push(0);
      }
    }
    
    // Update cache
    this.cache.set(cacheKey, {
      features: featureVector,
      timestamp: now,
      version: this.schemaVersion
    });
    
    return {
      symbol,
      timestamp: now,
      features: computedFeatures,
      version: this.schemaVersion
    };
  }

  /**
   * Get feature vector as array (for ML model input).
   */
  getFeatureVector(symbol: AllowedPair, data: MarketData): number[] {
    const featureSet = this.getFeatures(symbol, data);
    return Object.values(featureSet.features).flat();
  }

  /**
   * Batch compute features for multiple symbols.
   */
  async batchCompute(
    dataMap: Map<AllowedPair, MarketData>
  ): Promise<Map<AllowedPair, FeatureSet>> {
    const results = new Map<AllowedPair, FeatureSet>();
    
    for (const [symbol, data] of dataMap) {
      results.set(symbol, this.getFeatures(symbol, data));
    }
    
    return results;
  }

  // ==========================================================================
  // Normalization
  // ==========================================================================

  /** Rolling stats for z-score normalization. */
  private normStats: Map<string, { mean: number; std: number; count: number }> = new Map();

  /**
   * Normalize a value based on the specified method.
   */
  private normalizeValue(
    value: number,
    method: 'z-score' | 'min-max' | 'none',
    featureName: string
  ): number {
    if (method === 'none') return value;
    
    if (method === 'min-max') {
      // Assume known ranges for specific features
      if (featureName === 'rsi_14') {
        return (value - 50) / 50; // RSI 0-100 -> -1 to 1
      }
      return value;
    }
    
    // Z-score normalization with rolling statistics
    let stats = this.normStats.get(featureName);
    
    if (!stats) {
      stats = { mean: 0, std: 1, count: 0 };
    }
    
    // Update rolling statistics
    const alpha = 0.01;
    stats.count++;
    
    if (stats.count === 1) {
      stats.mean = value;
      stats.std = 1;
    } else {
      const newMean = stats.mean * (1 - alpha) + value * alpha;
      const newStd = Math.sqrt(
        stats.std * stats.std * (1 - alpha) + 
        Math.pow(value - stats.mean, 2) * alpha
      );
      stats.mean = newMean;
      stats.std = newStd || 1;
    }
    
    this.normStats.set(featureName, stats);
    
    return (value - stats.mean) / stats.std;
  }

  // ==========================================================================
  // Helper Calculations
  // ==========================================================================

  private calculateRSI(closes: number[], period: number): number {
    if (closes.length < period + 1) return 50;
    
    let gains = 0;
    let losses = 0;
    
    for (let i = closes.length - period; i < closes.length; i++) {
      const change = closes[i] - closes[i - 1];
      if (change > 0) gains += change;
      else losses -= change;
    }
    
    const avgGain = gains / period;
    const avgLoss = losses / period;
    
    if (avgLoss === 0) return 100;
    const rs = avgGain / avgLoss;
    
    return 100 - (100 / (1 + rs));
  }

  private calculateBollingerBands(closes: number[], period: number, stdDev: number) {
    const slice = closes.slice(-period);
    const middle = slice.reduce((a, b) => a + b, 0) / period;
    
    const variance = slice.reduce((sum, v) => sum + Math.pow(v - middle, 2), 0) / period;
    const std = Math.sqrt(variance);
    
    const upper = middle + stdDev * std;
    const lower = middle - stdDev * std;
    const current = closes[closes.length - 1];
    
    return {
      upper,
      middle,
      lower,
      percentB: (current - lower) / (upper - lower)
    };
  }

  private calculateVolatility(closes: number[], period: number): number {
    const returns = closes.slice(-period).map((c, i, arr) =>
      i > 0 ? (c - arr[i - 1]) / arr[i - 1] : 0
    ).slice(1);
    
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance);
  }

  private arrayToRecord(arr: number[]): Record<string, number[]> {
    const featureNames = Array.from(this.registry.keys());
    const record: Record<string, number[]> = {};
    
    for (let i = 0; i < featureNames.length && i < arr.length; i++) {
      record[featureNames[i]] = [arr[i]];
    }
    
    return record;
  }

  // ==========================================================================
  // Utilities
  // ==========================================================================

  /**
   * Get all registered feature names.
   */
  getFeatureNames(): string[] {
    return Array.from(this.registry.keys());
  }

  /**
   * Get feature definition.
   */
  getFeatureDefinition(name: string): FeatureDefinition | undefined {
    return this.registry.get(name);
  }

  /**
   * Clear feature cache.
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Get schema version.
   */
  getSchemaVersion(): string {
    return this.schemaVersion;
  }
}

