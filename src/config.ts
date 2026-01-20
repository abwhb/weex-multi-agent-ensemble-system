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
   * Using Contract/Futures API for AI Wars Hackathon.
   */
  weex: {
    /** API key for authentication. */
    apiKey: process.env.WEEX_API_KEY || '',
    /** API secret for signing requests. */
    apiSecret: process.env.WEEX_API_SECRET || '',
    /** API passphrase. */
    passphrase: process.env.WEEX_API_PASSPHRASE || '',
    /** Base URL for API requests (Contract/Futures API for hackathon). */
    baseUrl: process.env.WEEX_BASE_URL || 'https://api-contract.weex.com',
    /** WebSocket URL for real-time data. */
    wsUrl: process.env.WEEX_WS_URL || 'wss://ws-contract.weex.com'
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
    /** Symbols to trade (BTC only for focused trading). */
    symbols: ['BTC'] as const
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
    },
    /** GeminiAgent settings. */
    gemini: {
      /** Gemini API key (from environment). */
      apiKey: process.env.GEMINI_API_KEY || '',
      /** Gemini model to use (gemini-2.5-flash stable, or gemini-3-flash-preview for latest). */
      model: process.env.GEMINI_MODEL || 'gemini-2.5-flash',
      /** Temperature for AI responses (0-2, lower = more deterministic). */
      temperature: parseFloat(process.env.GEMINI_TEMPERATURE || '0.3'),
      /** Maximum tokens in response. */
      maxTokens: parseInt(process.env.GEMINI_MAX_TOKENS || '1024'),
      /** Number of candles to analyze. */
      candleWindow: 50,
      /** Confidence threshold. */
      threshold: 0.6,
      /** Lookback period. */
      lookback: 100,
      /** Model path (not used for Gemini but required by AgentConfig). */
      modelPath: ''
    },
    /** EMAStrategyAgent settings (User's custom strategy). */
    emaStrategy: {
      /** Gemini API key for AI enhancement. */
      apiKey: process.env.GEMINI_API_KEY || '',
      /** Gemini model to use (gemini-2.5-flash stable, or gemini-3-flash-preview for latest). */
      model: process.env.GEMINI_MODEL || 'gemini-2.5-flash',
      /** Temperature for AI (lower = more deterministic). */
      temperature: 0.2,
      /** Short EMA period. */
      shortEMA: 20,
      /** Long EMA period. */
      longEMA: 50,
      /** RSI period. */
      rsiPeriod: 14,
      /** RSI overbought threshold. */
      rsiOverbought: 70,
      /** RSI oversold threshold. */
      rsiOversold: 30,
      /** Volume ratio threshold for confirmation (1.5x = 150% of average). */
      volumeThreshold: 1.5,
      /** Confidence threshold. */
      threshold: 0.55,
      /** Lookback period for analysis. */
      lookback: 100,
      /** Model path (not used but required). */
      modelPath: ''
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
  },

  /**
   * Database configuration.
   */
  database: {
    /** Path to SQLite database file. */
    path: process.env.DB_PATH || './data/paper_trading.db',
    /** Whether to run migrations on startup. */
    runMigrations: process.env.DB_MIGRATE !== 'false'
  },

  /**
   * Paper trading configuration.
   */
  paperTrading: {
    /** Initial account balance for paper trading (USDT). */
    initialBalance: parseFloat(process.env.PAPER_INITIAL_BALANCE || '10000'),
    /** Fee rate for paper trading (0.001 = 0.1%). */
    feeRate: parseFloat(process.env.PAPER_FEE_RATE || '0.001'),
    /** Slippage rate for paper trading (0.0005 = 0.05%). */
    slippageRate: parseFloat(process.env.PAPER_SLIPPAGE_RATE || '0.0005')
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
    logLevel: config.logging.level,
    databasePath: config.database.path,
    paperInitialBalance: config.paperTrading.initialBalance
  };
}

export default config;

