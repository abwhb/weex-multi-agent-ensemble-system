/**
 * @fileoverview Configuration management for the trading system
 * @module config
 * 
 * Centralizes all configuration from environment variables with
 * sensible defaults. Validates required settings on load.
 */

import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();

/**
 * Main configuration object for the trading system.
 * 
 * ## Environment Variables
 * 
 * - `WEEX_API_KEY`: WEEX API key (required for live trading)
 * - `WEEX_API_SECRET`: WEEX API secret (required for live trading)
 * - `WEEX_API_PASSPHRASE`: WEEX API passphrase
 * - `TRADING_MODE`: 'paper' or 'live' (default: 'paper')
 * - `LOG_LEVEL`: 'debug', 'info', 'warn', 'error' (default: 'info')
 * - `RISK_PER_TRADE`: Risk per trade as decimal (default: 0.02)
 */
export const config = {
  /**
   * WEEX API configuration.
   */
  weex: {
    /** API key for authentication. */
    apiKey: process.env.WEEX_API_KEY || '',
    /** API secret for signing requests. */
    apiSecret: process.env.WEEX_API_SECRET || '',
    /** API passphrase. */
    passphrase: process.env.WEEX_API_PASSPHRASE || '',
    /** Base URL for API requests. */
    baseUrl: process.env.WEEX_BASE_URL || 'https://api.weex.com',
    /** WebSocket URL for real-time data. */
    wsUrl: process.env.WEEX_WS_URL || 'wss://ws.weex.com'
  },

  /**
   * Trading configuration.
   */
  trading: {
    /** Trading mode: 'paper' for simulation, 'live' for real trading. */
    mode: (process.env.TRADING_MODE || 'paper') as 'paper' | 'live',
    /** Maximum leverage (competition max: 20x). */
    maxLeverage: 20,
    /** Risk per trade as fraction of capital (default: 2%). */
    riskPerTrade: parseFloat(process.env.RISK_PER_TRADE || '0.02'),
    /** Minimum required trades per competition rules. */
    minTrades: 10,
    /** Maximum position size as fraction of capital. */
    maxPositionSize: 0.10,
    /** Daily drawdown limit before trading halt. */
    maxDailyDrawdown: 0.05,
    /** Default timeframe for analysis. */
    defaultTimeframe: '5m' as const,
    /** Symbols to trade (subset of allowed pairs). */
    symbols: ['BTC', 'ETH', 'SOL'] as const
  },

  /**
   * Agent-specific configuration.
   */
  agents: {
    /** MomentumAgent settings. */
    momentum: {
      /** Lookback period for feature calculation. */
      lookback: 100,
      /** Confidence threshold for signal generation. */
      threshold: 0.6,
      /** Path to trained LSTM model. */
      modelPath: 'models/momentum/lstm_v1.0.0.pt',
      /** Momentum threshold for direction. */
      momentumThreshold: 0.6
    },
    /** MeanReversionAgent settings. */
    meanReversion: {
      /** Lookback period for analysis. */
      lookback: 50,
      /** Bollinger Band standard deviation multiplier. */
      stdDev: 2,
      /** Confidence threshold. */
      threshold: 0.6,
      /** Path to regime classifier. */
      modelPath: 'models/mean_reversion/',
      /** Minimum reversion probability. */
      minReversionProb: 0.65
    },
    /** VolatilityAgent settings. */
    volatility: {
      /** Lookback period. */
      lookback: 30,
      /** Breakout probability threshold. */
      breakoutThreshold: 1.5,
      /** Confidence threshold. */
      threshold: 0.6,
      /** Path to models. */
      modelPath: 'models/volatility/',
      /** Volatility expansion threshold. */
      volExpansionThreshold: 1.5
    }
  },

  /**
   * Ensemble configuration.
   */
  ensemble: {
    /** Minimum confidence for trading. */
    minConfidence: 0.55,
    /** Learning rate for online weight adaptation. */
    learningRate: 0.05,
    /** Window size for performance-based weighting. */
    performanceWindow: 50,
    /** Maximum position size. */
    maxPositionSize: 0.10,
    /** Correlation threshold for diversification. */
    correlationThreshold: 0.7
  },

  /**
   * Logging configuration.
   */
  logging: {
    /** Log level: 'debug', 'info', 'warn', 'error'. */
    level: (process.env.LOG_LEVEL || 'info') as 'debug' | 'info' | 'warn' | 'error',
    /** Path to AI decision log file. */
    aiLogPath: process.env.AI_LOG_PATH || './logs/ai_decisions.json',
    /** Path to general application log. */
    appLogPath: process.env.APP_LOG_PATH || './logs/app.log',
    /** Whether to log to console. */
    console: process.env.LOG_CONSOLE !== 'false'
  },

  /**
   * System configuration.
   */
  system: {
    /** Main loop interval in milliseconds. */
    loopInterval: parseInt(process.env.LOOP_INTERVAL || '5000'),
    /** Maximum concurrent API requests. */
    maxConcurrentRequests: 5,
    /** API request timeout in milliseconds. */
    requestTimeout: 30000,
    /** Enable debug mode. */
    debug: process.env.DEBUG === 'true'
  }
};

/**
 * Validate configuration on startup.
 * 
 * @throws Error if required configuration is missing for live trading
 */
export function validateConfig(): void {
  const errors: string[] = [];
  
  // Check required credentials for live trading
  if (config.trading.mode === 'live') {
    if (!config.weex.apiKey) {
      errors.push('WEEX_API_KEY is required for live trading');
    }
    if (!config.weex.apiSecret) {
      errors.push('WEEX_API_SECRET is required for live trading');
    }
  }
  
  // Validate numeric ranges
  if (config.trading.maxLeverage > 20) {
    errors.push('maxLeverage cannot exceed 20 (competition rule)');
  }
  
  if (config.trading.riskPerTrade < 0 || config.trading.riskPerTrade > 0.1) {
    errors.push('riskPerTrade must be between 0 and 0.1');
  }
  
  if (errors.length > 0) {
    throw new Error(`Configuration errors:\n${errors.join('\n')}`);
  }
  
  console.log('Configuration validated successfully');
  console.log(`Trading mode: ${config.trading.mode}`);
  console.log(`Log level: ${config.logging.level}`);
}

/**
 * Get configuration summary for logging.
 */
export function getConfigSummary(): Record<string, unknown> {
  return {
    mode: config.trading.mode,
    symbols: config.trading.symbols,
    maxLeverage: config.trading.maxLeverage,
    riskPerTrade: config.trading.riskPerTrade,
    minConfidence: config.ensemble.minConfidence,
    logLevel: config.logging.level
  };
}

export default config;

