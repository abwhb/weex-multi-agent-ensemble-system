/**
 * @fileoverview Shared TypeScript types for the WEEX Multi-Agent Trading System
 * @module types
 * 
 * This module defines all core data structures used throughout the trading system.
 * These types ensure type safety and consistency across all agents, the meta-learner,
 * and execution components.
 */

// ============================================================================
// Market Data Types
// ============================================================================

/**
 * OHLCV (Open, High, Low, Close, Volume) candlestick data structure.
 * This is the fundamental unit of market data used by all trading agents.
 * 
 * @interface OHLCV
 * @property {number} timestamp - Unix timestamp in milliseconds
 * @property {number} open - Opening price of the candle
 * @property {number} high - Highest price during the candle period
 * @property {number} low - Lowest price during the candle period
 * @property {number} close - Closing price of the candle
 * @property {number} volume - Trading volume during the candle period
 */
export interface OHLCV {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Comprehensive market data package passed to trading agents.
 * Contains raw OHLCV data, pre-computed technical indicators, and
 * normalized features ready for ML model consumption.
 * 
 * @interface MarketData
 * @property {string} symbol - Trading pair symbol (e.g., 'BTC', 'ETH')
 * @property {OHLCV[]} ohlcv - Array of historical candlestick data
 * @property {Record<string, number[]>} indicators - Pre-computed technical indicators
 * @property {number[]} features - Normalized feature vector for ML models
 */
export interface MarketData {
  symbol: string;
  ohlcv: OHLCV[];
  indicators: Record<string, number[]>;
  features: number[];
}

// ============================================================================
// Trading Types
// ============================================================================

/**
 * Trading direction type.
 * - 'long': Bullish position (buy low, sell high)
 * - 'short': Bearish position (sell high, buy low)
 * - 'neutral': No directional bias
 */
export type Direction = 'long' | 'short' | 'neutral';

/**
 * Order side for exchange API.
 * - 'buy': Purchase order
 * - 'sell': Sell order
 */
export type OrderSide = 'buy' | 'sell';

/**
 * Order execution type.
 * - 'market': Execute immediately at current market price
 * - 'limit': Execute only at specified price or better
 */
export type OrderType = 'market' | 'limit';

/**
 * Represents an order to be placed on the exchange.
 * 
 * @interface Order
 * @property {string} id - Unique order identifier
 * @property {string} symbol - Trading pair symbol
 * @property {OrderSide} side - Buy or sell
 * @property {OrderType} type - Market or limit order
 * @property {number} size - Position size in base currency
 * @property {number} [price] - Limit price (required for limit orders)
 * @property {number} [stopLoss] - Stop loss price level
 * @property {number} [takeProfit] - Take profit price level
 * @property {number} timestamp - Order creation timestamp
 */
export interface Order {
  id: string;
  symbol: string;
  side: OrderSide;
  type: OrderType;
  size: number;
  price?: number;
  stopLoss?: number;
  takeProfit?: number;
  timestamp: number;
}

/**
 * Represents an open position on the exchange.
 * 
 * @interface Position
 * @property {string} symbol - Trading pair symbol
 * @property {'long' | 'short'} side - Position direction
 * @property {number} size - Position size in base currency
 * @property {number} entryPrice - Average entry price
 * @property {number} unrealizedPnl - Current unrealized profit/loss
 * @property {number} leverage - Applied leverage (max 20x per competition rules)
 */
export interface Position {
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  unrealizedPnl: number;
  leverage: number;
}

// ============================================================================
// Competition Constraints
// ============================================================================

/**
 * Competition-approved trading pairs.
 * Trading is restricted to these pairs per WEEX AI Wars rules.
 * 
 * @constant
 */
export const ALLOWED_PAIRS = ['ADA', 'SOL', 'LTC', 'DOGE', 'BTC', 'ETH', 'XRP', 'BNB'] as const;

/**
 * Type representing any allowed trading pair.
 */
export type AllowedPair = typeof ALLOWED_PAIRS[number];

// ============================================================================
// Market Regime Types
// ============================================================================

/**
 * Market regime classification used by agents to adapt strategies.
 * 
 * - 'trending': Strong directional movement, favors momentum strategies
 * - 'ranging': Sideways price action, favors mean reversion
 * - 'volatile': High volatility with unclear direction, favors breakout strategies
 * - 'quiet': Low volatility, reduced trading opportunities
 */
export type MarketRegime = 'trending' | 'ranging' | 'volatile' | 'quiet';

/**
 * Timeframe for candle data.
 */
export type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h' | '1d';

// ============================================================================
// AI Logging Types (Competition Requirement)
// ============================================================================

/**
 * Structure for AI decision logging required by competition.
 * Every trading decision must be logged with full AI traceability.
 * 
 * @interface AILogEntry
 */
export interface AILogEntry {
  /** ISO timestamp of the decision */
  timestamp: string;
  /** Versions of all models involved in the decision */
  modelVersions: Record<string, string>;
  /** Input data and features used */
  inputs: {
    symbol: string;
    features: number[];
    regime: MarketRegime;
  };
  /** Individual agent outputs */
  agentOutputs: Record<string, {
    signal: number;
    confidence: number;
  }>;
  /** Final meta-learner output */
  metaOutput: {
    direction: Direction;
    size: number;
    confidence: number;
  };
  /** Natural language explanation of the decision */
  reasoning: string;
  /** Order ID if trade was executed */
  orderId?: string;
}

// ============================================================================
// Performance Tracking Types
// ============================================================================

/**
 * Performance metrics for an agent or the ensemble.
 * 
 * @interface PerformanceMetrics
 */
export interface PerformanceMetrics {
  /** Total number of trades */
  totalTrades: number;
  /** Number of winning trades */
  winningTrades: number;
  /** Number of losing trades */
  losingTrades: number;
  /** Win rate as a decimal (0-1) */
  winRate: number;
  /** Total profit/loss */
  totalPnl: number;
  /** Sharpe ratio (risk-adjusted return) */
  sharpeRatio: number;
  /** Maximum drawdown experienced */
  maxDrawdown: number;
  /** Average trade duration in milliseconds */
  avgTradeDuration: number;
}

// ============================================================================
// Configuration Types
// ============================================================================

/**
 * WEEX API configuration.
 */
export interface WeexConfig {
  apiKey: string;
  apiSecret: string;
  passphrase: string;
  baseUrl: string;
}

/**
 * Trading configuration parameters.
 */
export interface TradingConfig {
  /** 'paper' for simulation, 'live' for real trading */
  mode: 'paper' | 'live';
  /** Maximum leverage (competition max: 20x) */
  maxLeverage: number;
  /** Risk per trade as fraction of capital */
  riskPerTrade: number;
  /** Minimum required trades per competition rules */
  minTrades: number;
}

/**
 * Logging configuration.
 */
export interface LoggingConfig {
  level: 'debug' | 'info' | 'warn' | 'error';
  aiLogPath: string;
}

