/**
 * @fileoverview Abstract interface for market data providers
 * @module data/IMarketDataProvider
 * 
 * This interface allows swapping between different data sources:
 * - Binance (for testing/backtesting - no API keys needed)
 * - WEEX (for live trading in competition)
 * 
 * Both implement the same interface, so all trading logic works with either.
 */

import { OHLCV, AllowedPair, Timeframe } from '../types';

/**
 * Callback for real-time price updates.
 */
export type PriceCallback = (symbol: string, price: number, timestamp: number) => void;

/**
 * Callback for real-time candle updates.
 */
export type CandleCallback = (symbol: string, candle: OHLCV) => void;

/**
 * Subscription handle for cleanup.
 */
export interface Subscription {
  id: string;
  symbol: string;
  unsubscribe: () => void;
}

/**
 * Current ticker information.
 */
export interface TickerInfo {
  symbol: string;
  price: number;
  bidPrice: number;
  askPrice: number;
  volume24h: number;
  change24h: number;
  timestamp: number;
}

/**
 * Abstract interface for market data providers.
 * 
 * Implement this interface to add new data sources.
 * The trading system uses this interface, not concrete implementations,
 * so data sources can be swapped without changing trading logic.
 */
export interface IMarketDataProvider {
  /**
   * Provider name for logging.
   */
  readonly name: string;

  /**
   * Connect to the data source.
   */
  connect(): Promise<void>;

  /**
   * Disconnect from the data source.
   */
  disconnect(): Promise<void>;

  /**
   * Check if connected.
   */
  isConnected(): boolean;

  /**
   * Get historical OHLCV data.
   * 
   * @param symbol - Trading pair symbol (e.g., 'BTC', 'ETH')
   * @param interval - Candle interval ('1m', '5m', '15m', '1h', '4h', '1d')
   * @param limit - Number of candles to fetch (max varies by provider)
   * @returns Array of OHLCV candles, oldest first
   */
  getHistoricalData(
    symbol: string,
    interval: Timeframe,
    limit: number
  ): Promise<OHLCV[]>;

  /**
   * Get current price for a symbol.
   * 
   * @param symbol - Trading pair symbol
   * @returns Current price
   */
  getCurrentPrice(symbol: string): Promise<number>;

  /**
   * Get full ticker info for a symbol.
   * 
   * @param symbol - Trading pair symbol
   * @returns Ticker information
   */
  getTicker(symbol: string): Promise<TickerInfo>;

  /**
   * Subscribe to real-time price updates.
   * 
   * @param symbol - Trading pair symbol
   * @param callback - Function called on each price update
   * @returns Subscription handle for cleanup
   */
  subscribeToPrices(symbol: string, callback: PriceCallback): Subscription;

  /**
   * Subscribe to real-time candle updates.
   * 
   * @param symbol - Trading pair symbol
   * @param interval - Candle interval
   * @param callback - Function called on each candle update
   * @returns Subscription handle for cleanup
   */
  subscribeToCandles(
    symbol: string,
    interval: Timeframe,
    callback: CandleCallback
  ): Subscription;

  /**
   * Get list of supported symbols.
   */
  getSupportedSymbols(): string[];
}

/**
 * Map our standard symbol names to provider-specific format.
 * Override in concrete implementations if needed.
 */
export function formatSymbol(symbol: string, provider: 'binance' | 'weex'): string {
  const baseSymbol = symbol.toUpperCase();
  
  switch (provider) {
    case 'binance':
      return `${baseSymbol}USDT`;
    case 'weex':
      return `${baseSymbol}USDT`;
    default:
      return `${baseSymbol}USDT`;
  }
}

/**
 * Convert interval to provider-specific format.
 */
export function formatInterval(interval: Timeframe, provider: 'binance' | 'weex'): string {
  // Both Binance and WEEX use same format
  return interval;
}



