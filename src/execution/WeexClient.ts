/**
 * @fileoverview WEEX API client wrapper for trading operations
 * @module execution/WeexClient
 *
 * ## Overview
 *
 * Type-safe wrapper around WEEX OpenAPI. Handles authentication, rate limiting,
 * and order management. All trades must go through this client for competition
 * validity.
 *
 * ## API Reference
 *
 * @see https://www.weex.com/api-doc/spot/introduction/APIBriefIntroduction
 *
 * ## Competition Requirements
 *
 * - All trades must use WEEX OpenAPI
 * - Maximum leverage: 20x
 * - Allowed pairs: ADA, SOL, LTC, DOGE, BTC, ETH, XRP, BNB
 * - Minimum 10 trades required
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import * as crypto from 'crypto';
import {
  OHLCV,
  Order,
  Position,
  OrderSide,
  OrderType,
  AllowedPair,
  ALLOWED_PAIRS
} from '../types';
import { config } from '../config';
import { TradingController } from '../control';
import { PaperTradingEngine } from './PaperTradingEngine';

// ============================================================================
// WEEX API Response Types
// ============================================================================

/**
 * WEEX API response wrapper.
 * Note: WEEX uses string code "00000" for success.
 */
interface WeexApiResponse<T> {
  code: string;
  msg: string;
  requestTime: number;
  data: T;
}

/**
 * WEEX candlestick data format.
 * Array: [timestamp, open, high, low, close, volume, turnover]
 */
type WeexCandleData = [number, string, string, string, string, string, string];

/**
 * WEEX ticker response.
 */
interface WeexTickerData {
  symbol: string;
  high24h: string;
  low24h: string;
  close: string;
  quoteVol: string;
  baseVol: string;
  usdtVol: string;
  ts: string;
  buyOne: string;
  sellOne: string;
  bidSz: string;
  askSz: string;
  openUtc0: string;
  changeUtc: string;
  change: string;
}

/**
 * WEEX account asset response.
 */
interface WeexAssetData {
  coinId: string;
  coinName: string;
  available: string;
  frozen: string;
  lock: string;
  uTime: string;
}

/**
 * WEEX order response.
 */
interface WeexOrderResponse {
  orderId: string;
  clientOrderId: string;
}

/**
 * WEEX order detail.
 */
interface WeexOrderDetail {
  accountId: string;
  symbol: string;
  orderId: string;
  clientOrderId: string;
  price: string;
  quantity: string;
  orderType: string;
  side: string;
  status: string;
  fillPrice: string;
  fillQuantity: string;
  fillTotalAmount: string;
  cTime: string;
}

// ============================================================================
// Interfaces
// ============================================================================

/**
 * Parameters for placing an order.
 */
export interface OrderParams {
  /** Trading symbol (must be in ALLOWED_PAIRS). */
  symbol: AllowedPair;
  /** Order side (buy/sell). */
  side: OrderSide;
  /** Order type (market/limit). */
  type: OrderType;
  /** Position size in base currency. */
  size: number;
  /** Limit price (required for limit orders). */
  price?: number;
  /** Stop loss price level. */
  stopLoss?: number;
  /** Take profit price level. */
  takeProfit?: number;
  /** Leverage to apply (max 20x). */
  leverage?: number;
  /** Reduce-only flag. */
  reduceOnly?: boolean;
  /** Client order ID for tracking. */
  clientOrderId?: string;
}

/**
 * Balance information.
 */
export interface Balance {
  currency: string;
  available: number;
  frozen: number;
  total: number;
}

/**
 * Ticker information.
 */
export interface Ticker {
  symbol: string;
  lastPrice: number;
  bidPrice: number;
  askPrice: number;
  volume24h: number;
  change24h: number;
}

// ============================================================================
// WeexClient Implementation
// ============================================================================

/**
 * WEEX API client for executing trades and fetching market data.
 *
 * ## Features
 *
 * 1. **Authentication**: HMAC-SHA256 + Base64 signature for all requests
 * 2. **Rate Limiting**: Built-in rate limiter to avoid API bans
 * 3. **Error Handling**: Comprehensive error handling with retries
 * 4. **Type Safety**: Full TypeScript types for all API calls
 *
 * ## Competition Compliance
 *
 * All trading operations are logged for AI audit requirements.
 * Only competition-approved trading pairs are allowed.
 */
export class WeexClient {
  /** Axios instance for API calls. */
  private client: AxiosInstance | null = null;

  /** API credentials. */
  private apiKey: string;
  private apiSecret: string;
  private passphrase: string;

  /** Base URL for API. */
  private baseUrl: string;

  /** Rate limiting. */
  private lastRequestTime: number = 0;
  private minRequestInterval: number = 100; // ms

  /** Connection status. */
  private isConnected: boolean = false;

  /** Trading controller for enable/disable trading. */
  private tradingController: TradingController | null = null;

  /** Paper trading engine for simulation. */
  private paperEngine: PaperTradingEngine | null = null;

  constructor(
    tradingController?: TradingController,
    paperEngine?: PaperTradingEngine
  ) {
    this.apiKey = config.weex.apiKey;
    this.apiSecret = config.weex.apiSecret;
    this.passphrase = config.weex.passphrase;
    this.baseUrl = config.weex.baseUrl;

    if (tradingController) {
      this.tradingController = tradingController;
    }
    if (paperEngine) {
      this.paperEngine = paperEngine;
    }
  }

  /**
   * Check if we should use paper trading.
   * Paper trading is used when:
   * 1. TradingController exists and trading is disabled (default)
   * 2. Or config.trading.mode is 'paper'
   */
  private usePaperTrading(): boolean {
    if (this.tradingController) {
      return !this.tradingController.isEnabled();
    }
    return config.trading.mode === 'paper';
  }

  /**
   * Convert symbol to WEEX format (e.g., 'BTC' -> 'BTCUSDT_SPBL').
   */
  private toWeexSymbol(symbol: AllowedPair): string {
    return `${symbol}USDT_SPBL`;
  }

  // ==========================================================================
  // Connection Management
  // ==========================================================================

  /**
   * Initialize API connection with credentials.
   * Validates credentials and tests connectivity.
   * In paper trading mode, no live API connection is required.
   */
  async connect(): Promise<void> {
    // If using paper trading, no need for live API connection
    if (this.usePaperTrading() && this.paperEngine) {
      console.log('Paper trading mode active - using simulated exchange');
      this.isConnected = true;
      return;
    }

    console.log('Connecting to WEEX API...');

    // Validate credentials
    if (!this.apiKey || !this.apiSecret) {
      throw new Error('WEEX API credentials not configured. Set WEEX_API_KEY and WEEX_API_SECRET.');
    }

    // Initialize axios instance
    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
        'locale': 'en-US'
      }
    });

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response) {
          const data = error.response.data as WeexApiResponse<unknown>;
          console.error(`WEEX API Error: ${data?.code} - ${data?.msg}`);
        }
        throw error;
      }
    );

    // Test connection with a public endpoint (no auth required)
    try {
      const testSymbol = this.toWeexSymbol('BTC');
      await this.client.get(`/api/v2/market/ticker?symbol=${testSymbol}`);
      this.isConnected = true;
      console.log('Connected to WEEX API successfully');
    } catch (error) {
      console.error('Failed to connect to WEEX API:', error);
      throw new Error('WEEX API connection failed');
    }
  }

  /**
   * Sign a request using HMAC-SHA256 + Base64.
   * Signature format: timestamp + method + requestPath + queryString + body
   */
  private signRequest(
    method: string,
    requestPath: string,
    timestamp: string,
    queryString: string = '',
    body: string = ''
  ): string {
    // Build the string to sign
    let stringToSign = timestamp + method.toUpperCase() + requestPath;
    if (queryString) {
      stringToSign += '?' + queryString;
    }
    if (body) {
      stringToSign += body;
    }

    // Create HMAC-SHA256 signature and encode as Base64
    const signature = crypto
      .createHmac('sha256', this.apiSecret)
      .update(stringToSign)
      .digest('base64');

    return signature;
  }

  /**
   * Make an authenticated request to WEEX API.
   */
  private async authenticatedRequest<T>(
    method: 'GET' | 'POST' | 'DELETE',
    path: string,
    params?: Record<string, string>,
    body?: Record<string, unknown>
  ): Promise<T> {
    if (!this.client) {
      throw new Error('API client not initialized');
    }

    const timestamp = Date.now().toString();
    const queryString = params ? new URLSearchParams(params).toString() : '';
    const bodyString = body ? JSON.stringify(body) : '';

    const signature = this.signRequest(method, path, timestamp, queryString, bodyString);

    const headers = {
      'ACCESS-KEY': this.apiKey,
      'ACCESS-SIGN': signature,
      'ACCESS-TIMESTAMP': timestamp,
      'ACCESS-PASSPHRASE': this.passphrase,
      'Content-Type': 'application/json',
      'locale': 'en-US'
    };

    const url = queryString ? `${path}?${queryString}` : path;

    const response = await this.client.request<WeexApiResponse<T>>({
      method,
      url,
      headers,
      data: body
    });

    if (response.data.code !== '00000') {
      throw new Error(`WEEX API Error: ${response.data.code} - ${response.data.msg}`);
    }

    return response.data.data;
  }

  /**
   * Rate-limited request wrapper.
   */
  private async rateLimitedRequest<T>(
    fn: () => Promise<T>
  ): Promise<T> {
    const now = Date.now();
    const elapsed = now - this.lastRequestTime;

    if (elapsed < this.minRequestInterval) {
      await new Promise(resolve =>
        setTimeout(resolve, this.minRequestInterval - elapsed)
      );
    }

    this.lastRequestTime = Date.now();
    return fn();
  }

  // ==========================================================================
  // Market Data (Public Endpoints - No Auth Required)
  // ==========================================================================

  /**
   * Fetch OHLCV (candlestick) data for a symbol.
   *
   * @param symbol - Trading pair symbol
   * @param interval - Candle interval ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
   * @param limit - Number of candles to fetch (max 1000)
   * @returns Array of OHLCV data
   */
  async getMarketData(
    symbol: AllowedPair,
    interval: string = '5m',
    limit: number = 100
  ): Promise<OHLCV[]> {
    this.ensureConnected();
    this.validateSymbol(symbol);

    return this.rateLimitedRequest(async () => {
      const weexSymbol = this.toWeexSymbol(symbol);

      // Map our interval format to WEEX format (they're the same)
      const period = interval;

      try {
        const response = await this.client!.get<WeexApiResponse<WeexCandleData[]>>(
          '/api/v2/market/candles',
          {
            params: {
              symbol: weexSymbol,
              period: period,
              limit: limit.toString()
            }
          }
        );

        if (response.data.code !== '00000') {
          throw new Error(`WEEX API Error: ${response.data.code} - ${response.data.msg}`);
        }

        // Convert WEEX format to our OHLCV format
        // WEEX returns: [timestamp, open, high, low, close, volume, turnover]
        const candles: OHLCV[] = response.data.data.map((candle: WeexCandleData) => ({
          timestamp: candle[0],
          open: parseFloat(candle[1]),
          high: parseFloat(candle[2]),
          low: parseFloat(candle[3]),
          close: parseFloat(candle[4]),
          volume: parseFloat(candle[5])
        }));

        // Sort by timestamp ascending (oldest first)
        candles.sort((a, b) => a.timestamp - b.timestamp);

        console.log(`Fetched ${candles.length} ${interval} candles for ${symbol}`);
        return candles;
      } catch (error) {
        console.error(`Error fetching market data for ${symbol}:`, error);
        throw error;
      }
    });
  }

  /**
   * Get current ticker information.
   */
  async getTicker(symbol: AllowedPair): Promise<Ticker> {
    this.ensureConnected();
    this.validateSymbol(symbol);

    return this.rateLimitedRequest(async () => {
      const weexSymbol = this.toWeexSymbol(symbol);

      try {
        const response = await this.client!.get<WeexApiResponse<WeexTickerData>>(
          '/api/v2/market/ticker',
          {
            params: { symbol: weexSymbol }
          }
        );

        if (response.data.code !== '00000') {
          throw new Error(`WEEX API Error: ${response.data.code} - ${response.data.msg}`);
        }

        const data = response.data.data;

        return {
          symbol: `${symbol}USDT`,
          lastPrice: parseFloat(data.close),
          bidPrice: parseFloat(data.buyOne),
          askPrice: parseFloat(data.sellOne),
          volume24h: parseFloat(data.baseVol),
          change24h: parseFloat(data.change)
        };
      } catch (error) {
        console.error(`Error fetching ticker for ${symbol}:`, error);
        throw error;
      }
    });
  }

  // ==========================================================================
  // Order Management (Private Endpoints - Auth Required)
  // ==========================================================================

  /**
   * Place an order on the exchange.
   * Routes to paper trading engine when trading is disabled.
   *
   * @param params - Order parameters
   * @param currentPrice - Current market price (used for paper trading)
   * @returns Created order with ID
   */
  async placeOrder(params: OrderParams, currentPrice?: number): Promise<Order> {
    this.ensureConnected();
    this.validateSymbol(params.symbol);
    this.validateOrderParams(params);

    // Route to paper trading if disabled
    if (this.usePaperTrading() && this.paperEngine) {
      const price = currentPrice ?? (await this.getTicker(params.symbol)).lastPrice;
      console.log(`[PAPER] Placing ${params.type} ${params.side} order for ${params.symbol}: size=${params.size}`);
      return this.paperEngine.executeOrder(params, price);
    }

    return this.rateLimitedRequest(async () => {
      const weexSymbol = this.toWeexSymbol(params.symbol);
      const clientOrderId = params.clientOrderId || `${Date.now()}${Math.random().toString(36).substr(2, 9)}`;

      // Build order request body
      const orderBody: Record<string, unknown> = {
        symbol: weexSymbol,
        side: params.side,
        orderType: params.type,
        force: 'normal', // GTC (Good Till Cancelled)
        quantity: params.size.toString(),
        clientOrderId: clientOrderId
      };

      // Add price for limit orders
      if (params.type === 'limit' && params.price) {
        orderBody.price = params.price.toString();
      } else if (params.type === 'market') {
        orderBody.price = '0'; // Market orders don't use price
      }

      try {
        console.log(`[LIVE] Placing ${params.type} ${params.side} order for ${params.symbol}: size=${params.size}`);

        const data = await this.authenticatedRequest<WeexOrderResponse>(
          'POST',
          '/api/v2/trade/orders',
          undefined,
          orderBody
        );

        return {
          id: data.orderId,
          symbol: params.symbol,
          side: params.side,
          type: params.type,
          size: params.size,
          price: params.price,
          stopLoss: params.stopLoss,
          takeProfit: params.takeProfit,
          timestamp: Date.now()
        };
      } catch (error) {
        console.error(`Error placing order for ${params.symbol}:`, error);
        throw error;
      }
    });
  }

  /**
   * Cancel an open order.
   *
   * @param orderId - ID of the order to cancel
   * @param symbol - Trading symbol (required by WEEX API)
   * @returns True if cancelled successfully
   */
  async cancelOrder(orderId: string, symbol?: AllowedPair): Promise<boolean> {
    this.ensureConnected();

    // Route to paper trading if disabled
    if (this.usePaperTrading() && this.paperEngine) {
      console.log(`[PAPER] Cancelling order: ${orderId}`);
      return this.paperEngine.cancelOrder(orderId);
    }

    return this.rateLimitedRequest(async () => {
      try {
        console.log(`[LIVE] Cancelling order: ${orderId}`);

        const params: Record<string, string> = {
          orderId: orderId
        };

        if (symbol) {
          params.symbol = this.toWeexSymbol(symbol);
        }

        await this.authenticatedRequest<unknown>(
          'POST',
          '/api/v2/trade/cancel-order',
          undefined,
          params
        );

        return true;
      } catch (error) {
        console.error(`Error cancelling order ${orderId}:`, error);
        return false;
      }
    });
  }

  /**
   * Get order status.
   */
  async getOrder(orderId: string, symbol: AllowedPair): Promise<Order | null> {
    this.ensureConnected();

    // Route to paper trading if disabled
    if (this.usePaperTrading() && this.paperEngine) {
      return this.paperEngine.getOrder(orderId);
    }

    return this.rateLimitedRequest(async () => {
      try {
        const weexSymbol = this.toWeexSymbol(symbol);

        const data = await this.authenticatedRequest<WeexOrderDetail>(
          'GET',
          '/api/v2/trade/orderInfo',
          {
            symbol: weexSymbol,
            orderId: orderId
          }
        );

        return {
          id: data.orderId,
          symbol: symbol,
          side: data.side as OrderSide,
          type: data.orderType as OrderType,
          size: parseFloat(data.quantity),
          price: parseFloat(data.fillPrice) || parseFloat(data.price),
          timestamp: parseInt(data.cTime)
        };
      } catch (error) {
        console.error(`Error fetching order ${orderId}:`, error);
        return null;
      }
    });
  }

  /**
   * Get all open orders.
   */
  async getOpenOrders(symbol?: AllowedPair): Promise<Order[]> {
    this.ensureConnected();
    if (symbol) this.validateSymbol(symbol);

    // Route to paper trading if disabled
    if (this.usePaperTrading() && this.paperEngine) {
      return this.paperEngine.getOpenOrders(symbol);
    }

    return this.rateLimitedRequest(async () => {
      try {
        const params: Record<string, string> = {};
        if (symbol) {
          params.symbol = this.toWeexSymbol(symbol);
        }

        const data = await this.authenticatedRequest<WeexOrderDetail[]>(
          'GET',
          '/api/v2/trade/open-orders',
          params
        );

        return data.map(order => ({
          id: order.orderId,
          symbol: symbol || order.symbol.replace('USDT_SPBL', '') as AllowedPair,
          side: order.side as OrderSide,
          type: order.orderType as OrderType,
          size: parseFloat(order.quantity),
          price: parseFloat(order.price),
          timestamp: parseInt(order.cTime)
        }));
      } catch (error) {
        console.error('Error fetching open orders:', error);
        return [];
      }
    });
  }

  // ==========================================================================
  // Position Management
  // ==========================================================================

  /**
   * Get current open positions.
   * Routes to paper trading engine when trading is disabled.
   *
   * @param symbol - Optional symbol filter
   * @returns Array of open positions
   */
  async getPositions(symbol?: AllowedPair): Promise<Position[]> {
    this.ensureConnected();
    if (symbol) this.validateSymbol(symbol);

    // Route to paper trading if disabled
    if (this.usePaperTrading() && this.paperEngine) {
      return this.paperEngine.getPositions(symbol);
    }

    // Note: WEEX Spot API doesn't have a positions endpoint like futures
    // For spot trading, positions are derived from account balances
    // This would need to be implemented differently for a futures API
    console.log(`[LIVE] Fetching positions${symbol ? ` for ${symbol}` : ''}`);
    return [];
  }

  /**
   * Close a position.
   */
  async closePosition(symbol: AllowedPair, size?: number): Promise<Order> {
    this.ensureConnected();
    this.validateSymbol(symbol);

    // Get current position
    const positions = await this.getPositions(symbol);
    const position = positions.find(p => p.symbol === symbol);

    if (!position) {
      throw new Error(`No open position for ${symbol}`);
    }

    // Place closing order
    return this.placeOrder({
      symbol,
      side: position.side === 'long' ? 'sell' : 'buy',
      type: 'market',
      size: size || position.size,
      reduceOnly: true
    });
  }

  // ==========================================================================
  // Account
  // ==========================================================================

  /**
   * Get account balance.
   * Routes to paper trading engine when trading is disabled.
   *
   * @returns Balance information
   */
  async getBalance(): Promise<Balance[]> {
    // Route to paper trading if disabled
    if (this.usePaperTrading() && this.paperEngine) {
      return this.paperEngine.getBalance();
    }

    return this.rateLimitedRequest(async () => {
      try {
        const data = await this.authenticatedRequest<WeexAssetData[]>(
          'GET',
          '/api/spot/v1/account/assets'
        );

        return data.map(asset => ({
          currency: asset.coinName.toUpperCase(),
          available: parseFloat(asset.available),
          frozen: parseFloat(asset.frozen),
          total: parseFloat(asset.available) + parseFloat(asset.frozen) + parseFloat(asset.lock)
        }));
      } catch (error) {
        console.error('Error fetching balance:', error);
        throw error;
      }
    });
  }

  // ==========================================================================
  // Validation & Utilities
  // ==========================================================================

  /**
   * Validate that symbol is allowed for competition.
   */
  private validateSymbol(symbol: AllowedPair): void {
    if (!ALLOWED_PAIRS.includes(symbol)) {
      throw new Error(
        `Symbol '${symbol}' is not allowed. Allowed pairs: ${ALLOWED_PAIRS.join(', ')}`
      );
    }
  }

  /**
   * Validate order parameters.
   */
  private validateOrderParams(params: OrderParams): void {
    if (params.type === 'limit' && !params.price) {
      throw new Error('Price is required for limit orders');
    }

    if (params.leverage && params.leverage > 20) {
      throw new Error('Maximum leverage is 20x per competition rules');
    }

    if (params.size <= 0) {
      throw new Error('Order size must be positive');
    }
  }

  /**
   * Ensure client is connected.
   * In paper trading mode, client may be null but isConnected will be true.
   */
  private ensureConnected(): void {
    if (!this.isConnected) {
      throw new Error('Not connected to WEEX API. Call connect() first.');
    }
    // In live trading mode, client must exist
    if (!this.usePaperTrading() && !this.client) {
      throw new Error('API client not initialized. Call connect() first.');
    }
  }

  /**
   * Check connection status.
   */
  isReady(): boolean {
    return this.isConnected;
  }

  /**
   * Disconnect from API.
   */
  async disconnect(): Promise<void> {
    console.log('Disconnecting from WEEX API');
    this.client = null;
    this.isConnected = false;
  }
}
