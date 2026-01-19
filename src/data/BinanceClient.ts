/**
 * @fileoverview Binance public API client for market data
 * @module data/BinanceClient
 * 
 * Binance public API requires NO API KEYS for market data endpoints:
 * - GET /api/v3/klines - Historical OHLCV
 * - GET /api/v3/ticker/price - Current prices
 * - GET /api/v3/ticker/24hr - 24h ticker stats
 * - wss://stream.binance.com - Real-time WebSocket
 * 
 * This client is used for backtesting and paper trading before
 * WEEX API access is granted. Same trading pairs, same price data.
 */

import axios, { AxiosInstance } from 'axios';
import WebSocket from 'ws';
import { OHLCV, Timeframe } from '../types';
import {
  IMarketDataProvider,
  PriceCallback,
  CandleCallback,
  Subscription,
  TickerInfo,
  formatSymbol,
  formatInterval
} from './IMarketDataProvider';

/**
 * Binance API response for klines (candlestick data).
 */
type BinanceKline = [
  number,   // 0: Open time
  string,   // 1: Open
  string,   // 2: High
  string,   // 3: Low
  string,   // 4: Close
  string,   // 5: Volume
  number,   // 6: Close time
  string,   // 7: Quote asset volume
  number,   // 8: Number of trades
  string,   // 9: Taker buy base asset volume
  string,   // 10: Taker buy quote asset volume
  string    // 11: Ignore
];

/**
 * Binance public API client.
 * 
 * No API keys required for market data.
 * Rate limits: 1200 requests per minute (very generous).
 * 
 * @implements IMarketDataProvider
 * 
 * @example
 * ```typescript
 * const client = new BinanceClient();
 * await client.connect();
 * 
 * // Get historical data
 * const candles = await client.getHistoricalData('BTC', '5m', 100);
 * 
 * // Subscribe to real-time prices
 * const sub = client.subscribeToPrices('BTC', (symbol, price) => {
 *   console.log(`${symbol}: $${price}`);
 * });
 * ```
 */
export class BinanceClient implements IMarketDataProvider {
  readonly name = 'Binance';
  
  private client: AxiosInstance;
  private wsConnections: Map<string, WebSocket> = new Map();
  private subscriptions: Map<string, Subscription> = new Map();
  private subCounter = 0;
  private connected = false;

  // Binance API endpoints
  private readonly REST_URL = 'https://api.binance.com';
  private readonly WS_URL = 'wss://stream.binance.com:9443/ws';

  // Supported symbols (matching competition pairs)
  private readonly SUPPORTED_SYMBOLS = [
    'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'LTC', 'DOGE'
  ];

  constructor() {
    this.client = axios.create({
      baseURL: this.REST_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  // ==========================================================================
  // Connection Management
  // ==========================================================================

  async connect(): Promise<void> {
    console.log('Connecting to Binance public API...');
    
    try {
      // Test connection with a simple ping
      await this.client.get('/api/v3/ping');
      this.connected = true;
      console.log('Connected to Binance successfully');
    } catch (error) {
      console.error('Failed to connect to Binance:', error);
      throw new Error('Binance connection failed');
    }
  }

  async disconnect(): Promise<void> {
    console.log('Disconnecting from Binance...');
    
    // Close all WebSocket connections
    for (const [id, ws] of this.wsConnections) {
      ws.close();
    }
    this.wsConnections.clear();
    this.subscriptions.clear();
    this.connected = false;
    
    console.log('Disconnected from Binance');
  }

  isConnected(): boolean {
    return this.connected;
  }

  // ==========================================================================
  // Historical Data
  // ==========================================================================

  /**
   * Get historical OHLCV data from Binance.
   * 
   * @param symbol - Base symbol (e.g., 'BTC')
   * @param interval - Candle interval
   * @param limit - Number of candles (max 1000)
   */
  async getHistoricalData(
    symbol: string,
    interval: Timeframe,
    limit: number = 100
  ): Promise<OHLCV[]> {
    const binanceSymbol = formatSymbol(symbol, 'binance');
    const binanceInterval = formatInterval(interval, 'binance');
    
    try {
      const response = await this.client.get<BinanceKline[]>('/api/v3/klines', {
        params: {
          symbol: binanceSymbol,
          interval: binanceInterval,
          limit: Math.min(limit, 1000)
        }
      });

      return response.data.map(this.parseKline);
    } catch (error) {
      console.error(`Error fetching historical data for ${symbol}:`, error);
      throw error;
    }
  }

  /**
   * Parse Binance kline to OHLCV format.
   */
  private parseKline(kline: BinanceKline): OHLCV {
    return {
      timestamp: kline[0],
      open: parseFloat(kline[1]),
      high: parseFloat(kline[2]),
      low: parseFloat(kline[3]),
      close: parseFloat(kline[4]),
      volume: parseFloat(kline[5])
    };
  }

  // ==========================================================================
  // Current Prices
  // ==========================================================================

  /**
   * Get current price for a symbol.
   */
  async getCurrentPrice(symbol: string): Promise<number> {
    const binanceSymbol = formatSymbol(symbol, 'binance');
    
    try {
      const response = await this.client.get<{ symbol: string; price: string }>(
        '/api/v3/ticker/price',
        { params: { symbol: binanceSymbol } }
      );
      
      return parseFloat(response.data.price);
    } catch (error) {
      console.error(`Error fetching price for ${symbol}:`, error);
      throw error;
    }
  }

  /**
   * Get full ticker info for a symbol.
   */
  async getTicker(symbol: string): Promise<TickerInfo> {
    const binanceSymbol = formatSymbol(symbol, 'binance');
    
    try {
      const response = await this.client.get<{
        symbol: string;
        lastPrice: string;
        bidPrice: string;
        askPrice: string;
        volume: string;
        priceChangePercent: string;
      }>('/api/v3/ticker/24hr', {
        params: { symbol: binanceSymbol }
      });

      return {
        symbol,
        price: parseFloat(response.data.lastPrice),
        bidPrice: parseFloat(response.data.bidPrice),
        askPrice: parseFloat(response.data.askPrice),
        volume24h: parseFloat(response.data.volume),
        change24h: parseFloat(response.data.priceChangePercent),
        timestamp: Date.now()
      };
    } catch (error) {
      console.error(`Error fetching ticker for ${symbol}:`, error);
      throw error;
    }
  }

  // ==========================================================================
  // Real-time Subscriptions
  // ==========================================================================

  /**
   * Subscribe to real-time price updates via WebSocket.
   */
  subscribeToPrices(symbol: string, callback: PriceCallback): Subscription {
    const binanceSymbol = formatSymbol(symbol, 'binance').toLowerCase();
    const streamName = `${binanceSymbol}@trade`;
    const wsUrl = `${this.WS_URL}/${streamName}`;
    
    const ws = new WebSocket(wsUrl);
    const subId = `price_${++this.subCounter}`;
    
    ws.on('open', () => {
      console.log(`WebSocket connected for ${symbol} prices`);
    });
    
    ws.on('message', (data: WebSocket.Data) => {
      try {
        const parsed = JSON.parse(data.toString());
        const price = parseFloat(parsed.p);
        const timestamp = parsed.T;
        callback(symbol, price, timestamp);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    });
    
    ws.on('error', (error) => {
      console.error(`WebSocket error for ${symbol}:`, error);
    });
    
    ws.on('close', () => {
      console.log(`WebSocket closed for ${symbol}`);
      this.wsConnections.delete(subId);
    });
    
    this.wsConnections.set(subId, ws);
    
    const subscription: Subscription = {
      id: subId,
      symbol,
      unsubscribe: () => {
        ws.close();
        this.wsConnections.delete(subId);
        this.subscriptions.delete(subId);
      }
    };
    
    this.subscriptions.set(subId, subscription);
    return subscription;
  }

  /**
   * Subscribe to real-time candle updates via WebSocket.
   */
  subscribeToCandles(
    symbol: string,
    interval: Timeframe,
    callback: CandleCallback
  ): Subscription {
    const binanceSymbol = formatSymbol(symbol, 'binance').toLowerCase();
    const streamName = `${binanceSymbol}@kline_${interval}`;
    const wsUrl = `${this.WS_URL}/${streamName}`;
    
    const ws = new WebSocket(wsUrl);
    const subId = `candle_${++this.subCounter}`;
    
    ws.on('open', () => {
      console.log(`WebSocket connected for ${symbol} ${interval} candles`);
    });
    
    ws.on('message', (data: WebSocket.Data) => {
      try {
        const parsed = JSON.parse(data.toString());
        const k = parsed.k;
        
        const candle: OHLCV = {
          timestamp: k.t,
          open: parseFloat(k.o),
          high: parseFloat(k.h),
          low: parseFloat(k.l),
          close: parseFloat(k.c),
          volume: parseFloat(k.v)
        };
        
        callback(symbol, candle);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    });
    
    ws.on('error', (error) => {
      console.error(`WebSocket error for ${symbol} candles:`, error);
    });
    
    ws.on('close', () => {
      console.log(`WebSocket closed for ${symbol} candles`);
      this.wsConnections.delete(subId);
    });
    
    this.wsConnections.set(subId, ws);
    
    const subscription: Subscription = {
      id: subId,
      symbol,
      unsubscribe: () => {
        ws.close();
        this.wsConnections.delete(subId);
        this.subscriptions.delete(subId);
      }
    };
    
    this.subscriptions.set(subId, subscription);
    return subscription;
  }

  /**
   * Get list of supported symbols.
   */
  getSupportedSymbols(): string[] {
    return [...this.SUPPORTED_SYMBOLS];
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  /**
   * Get multiple symbols' prices at once.
   */
  async getAllPrices(): Promise<Map<string, number>> {
    try {
      const response = await this.client.get<Array<{ symbol: string; price: string }>>(
        '/api/v3/ticker/price'
      );
      
      const prices = new Map<string, number>();
      
      for (const ticker of response.data) {
        // Only include our supported symbols
        for (const symbol of this.SUPPORTED_SYMBOLS) {
          if (ticker.symbol === `${symbol}USDT`) {
            prices.set(symbol, parseFloat(ticker.price));
          }
        }
      }
      
      return prices;
    } catch (error) {
      console.error('Error fetching all prices:', error);
      throw error;
    }
  }

  /**
   * Get server time (useful for syncing).
   */
  async getServerTime(): Promise<number> {
    const response = await this.client.get<{ serverTime: number }>('/api/v3/time');
    return response.data.serverTime;
  }
}



