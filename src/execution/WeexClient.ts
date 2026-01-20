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
 * @see https://www.weex.com/api-doc/contract/introduction/APIBriefIntroduction
 * @see https://www.weex.com/api-doc/ai/QuickStart/RequestInteraction
 *
 * ## Competition Requirements (AI Wars Hackathon)
 *
 * - All trades must use WEEX Contract/Futures API
 * - Maximum leverage: 20x
 * - Allowed pairs: ADA, SOL, LTC, DOGE, BTC, ETH, XRP, BNB
 * - Minimum 10 USDT trading required for API test
 * - Test account balance: 1000 USDT
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
 * WEEX Contract ticker response.
 * Format from: /capi/v2/market/ticker
 */
interface WeexTickerData {
  symbol: string;
  last: string;
  best_ask: string;
  best_bid: string;
  high_24h: string;
  low_24h: string;
  base_volume: string;
  volume_24h: string;
  timestamp: string;
  priceChangePercent: string;
  indexPrice: string;
  markPrice: string;
}

/**
 * WEEX Contract account balance response.
 * Format from: /capi/v2/account/assets (returns array directly)
 */
interface WeexAccountData {
  coinName: string;
  available: string;
  equity: string;
  frozen: string;
  unrealizePnl: string;
}

/**
 * WEEX Contract order response.
 * Format from: /capi/v2/order/placeOrder (returns directly, not wrapped)
 */
interface WeexOrderResponse {
  order_id: string;
  client_oid: string;
}

/**
 * WEEX Contract order detail.
 * Format from: /capi/v2/order/current and /capi/v2/order/history
 * Note: API uses snake_case field names
 */
interface WeexOrderDetail {
  symbol: string;
  order_id: string;
  client_oid: string;
  price: string;
  size: string;
  type: string;        // 1=open_long, 2=open_short, 3=close_long, 4=close_short
  order_type: string;  // 0=normal, 1=post_only, 2=fok, 3=ioc
  match_price: string; // 0=limit, 1=market
  status: string;
  filled_qty: string;
  filled_amount: string;
  price_avg: string;
  leverage: string;
  fee: string;
  createTime: string;
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
   * Convert symbol to WEEX Contract format (e.g., 'BTC' -> 'cmt_btcusdt').
   * Contract API uses lowercase symbol with 'cmt_' prefix.
   */
  private toWeexSymbol(symbol: AllowedPair): string {
    return `cmt_${symbol.toLowerCase()}usdt`;
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
      await this.client.get(`/capi/v2/market/ticker?symbol=${testSymbol}`);
      this.isConnected = true;
      console.log('Connected to WEEX Contract API successfully');
    } catch (error) {
      console.error('Failed to connect to WEEX Contract API:', error);
      throw new Error('WEEX Contract API connection failed');
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
   * Note: Contract API returns data directly for most endpoints (not wrapped).
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

    const response = await this.client.request<T | WeexApiResponse<T>>({
      method,
      url,
      headers,
      data: body
    });

    // Handle both response formats:
    // 1. Direct data (most contract endpoints)
    // 2. Wrapped format {code, msg, data} (some endpoints)
    const data = response.data as any;

    // Check if response has error code
    if (data.code && data.code !== '00000' && data.code !== '200') {
      throw new Error(`WEEX API Error: ${data.code} - ${data.msg}`);
    }

    // If wrapped format, return data property; otherwise return directly
    if (data.data !== undefined) {
      return data.data as T;
    }

    return response.data as T;
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
        // Candles endpoint may return data directly or wrapped
        const response = await this.client!.get<WeexCandleData[] | WeexApiResponse<WeexCandleData[]>>(
          '/capi/v2/market/candles',
          {
            params: {
              symbol: weexSymbol,
              granularity: period,
              limit: limit.toString()
            }
          }
        );

        // Handle both response formats
        let candleData: WeexCandleData[];
        const respData = response.data as any;
        if (respData.code !== undefined) {
          // Wrapped format
          if (respData.code !== '00000') {
            throw new Error(`WEEX API Error: ${respData.code} - ${respData.msg}`);
          }
          candleData = respData.data;
        } else {
          // Direct array format
          candleData = respData;
        }

        // Convert WEEX format to our OHLCV format
        // WEEX returns: [timestamp, open, high, low, close, volume, turnover]
        const candles: OHLCV[] = candleData.map((candle: WeexCandleData) => ({
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
   * Note: Ticker endpoint returns data directly (not wrapped).
   */
  async getTicker(symbol: AllowedPair): Promise<Ticker> {
    this.ensureConnected();
    this.validateSymbol(symbol);

    return this.rateLimitedRequest(async () => {
      const weexSymbol = this.toWeexSymbol(symbol);

      try {
        // Ticker returns data directly, not wrapped
        const response = await this.client!.get<WeexTickerData>(
          '/capi/v2/market/ticker',
          {
            params: { symbol: weexSymbol }
          }
        );

        const data = response.data;

        return {
          symbol: `${symbol}USDT`,
          lastPrice: parseFloat(data.last),
          bidPrice: parseFloat(data.best_bid),
          askPrice: parseFloat(data.best_ask),
          volume24h: parseFloat(data.base_volume),
          change24h: parseFloat(data.priceChangePercent)
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
      const clientOid = params.clientOrderId || `${Date.now()}${Math.random().toString(36).substr(2, 9)}`;

      // Determine order type code:
      // 1 = Open long, 2 = Open short, 3 = Close long, 4 = Close short
      let typeCode = '1'; // Default: open long
      if (params.reduceOnly) {
        // Closing position
        typeCode = params.side === 'buy' ? '4' : '3'; // buy closes short, sell closes long
      } else {
        // Opening position
        typeCode = params.side === 'buy' ? '1' : '2'; // buy opens long, sell opens short
      }

      // Build order request body for Contract API
      const orderBody: Record<string, unknown> = {
        symbol: weexSymbol,
        client_oid: clientOid,
        size: params.size.toString(),
        type: typeCode,
        order_type: '0',  // 0 = Normal
        match_price: params.type === 'market' ? '1' : '0', // 1 = Market, 0 = Limit
        marginMode: '1'   // 1 = Cross margin
      };

      // Add price for limit orders
      if (params.type === 'limit' && params.price) {
        orderBody.price = params.price.toString();
      }

      // Add stop loss and take profit if specified
      if (params.stopLoss) {
        orderBody.presetStopLossPrice = params.stopLoss.toString();
      }
      if (params.takeProfit) {
        orderBody.presetTakeProfitPrice = params.takeProfit.toString();
      }

      try {
        console.log(`[LIVE] Placing ${params.type} ${params.side} order for ${params.symbol}: size=${params.size}`);

        const data = await this.authenticatedRequest<WeexOrderResponse>(
          'POST',
          '/capi/v2/order/placeOrder',
          undefined,
          orderBody
        );

        return {
          id: data.order_id || data.client_oid,
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

        if (!symbol) {
          throw new Error('Symbol is required to cancel order');
        }

        const body: Record<string, string> = {
          symbol: this.toWeexSymbol(symbol),
          orderId: orderId
        };

        await this.authenticatedRequest<unknown>(
          'POST',
          '/capi/v2/order/cancel_order',
          undefined,
          body
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
          '/capi/v2/order/detail',
          {
            symbol: weexSymbol,
            orderId: orderId
          }
        );

        // Map type code to order side: 1,3=long related, 2,4=short related
        const typeCode = parseInt(data.type);
        const side = (typeCode === 1 || typeCode === 3) ? 'buy' : 'sell';
        const orderType = data.match_price === '1' ? 'market' : 'limit';

        return {
          id: data.order_id,
          symbol: symbol,
          side: side as OrderSide,
          type: orderType as OrderType,
          size: parseFloat(data.size),
          price: parseFloat(data.price_avg) || parseFloat(data.price),
          timestamp: parseInt(data.createTime)
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
          '/capi/v2/order/current',
          params
        );

        // Handle empty response
        if (!data || !Array.isArray(data)) {
          return [];
        }

        return data.map(order => {
          // Extract symbol from contract format (cmt_btcusdt -> BTC)
          const rawSymbol = order.symbol.replace('cmt_', '').replace('usdt', '').toUpperCase();
          // Map type code to order side: 1,3=long related, 2,4=short related
          const typeCode = parseInt(order.type);
          const side = (typeCode === 1 || typeCode === 3) ? 'buy' : 'sell';
          const orderType = order.match_price === '1' ? 'market' : 'limit';

          return {
            id: order.order_id,
            symbol: (symbol || rawSymbol) as AllowedPair,
            side: side as OrderSide,
            type: orderType as OrderType,
            size: parseFloat(order.size),
            price: parseFloat(order.price),
            timestamp: parseInt(order.createTime)
          };
        });
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

    return this.rateLimitedRequest(async () => {
      try {
        console.log(`[LIVE] Fetching positions${symbol ? ` for ${symbol}` : ''}`);

        interface WeexPositionData {
          symbol: string;
          marginCoin: string;
          holdSide: string;
          margin: string;
          available: string;
          locked: string;
          total: string;
          leverage: string;
          averageOpenPrice: string;
          marginMode: string;
          unrealizedPL: string;
          liquidationPrice: string;
        }

        // If symbol specified, use single position endpoint
        if (symbol) {
          const params: Record<string, string> = {
            symbol: this.toWeexSymbol(symbol)
          };

          const data = await this.authenticatedRequest<WeexPositionData>(
            'GET',
            '/capi/v2/account/position/singlePosition',
            params
          );

          if (!data || !data.total || parseFloat(data.total) === 0) {
            return [];
          }

          return [{
            symbol: symbol,
            side: data.holdSide === 'long' ? 'long' : 'short',
            size: parseFloat(data.total),
            entryPrice: parseFloat(data.averageOpenPrice),
            markPrice: 0,
            unrealizedPnl: parseFloat(data.unrealizedPL || '0'),
            leverage: parseInt(data.leverage || '1'),
            liquidationPrice: parseFloat(data.liquidationPrice || '0')
          } as Position];
        }

        // For all positions, we need to check each symbol we trade
        const positions: Position[] = [];
        const tradingSymbols = config.trading.symbols;

        for (const sym of tradingSymbols) {
          try {
            const params: Record<string, string> = {
              symbol: this.toWeexSymbol(sym as AllowedPair)
            };

            const data = await this.authenticatedRequest<WeexPositionData>(
              'GET',
              '/capi/v2/account/position/singlePosition',
              params
            );

            if (data && data.total && parseFloat(data.total) > 0) {
              positions.push({
                symbol: sym as AllowedPair,
                side: data.holdSide === 'long' ? 'long' : 'short',
                size: parseFloat(data.total),
                entryPrice: parseFloat(data.averageOpenPrice),
                markPrice: 0,
                unrealizedPnl: parseFloat(data.unrealizedPL || '0'),
                leverage: parseInt(data.leverage || '1'),
                liquidationPrice: parseFloat(data.liquidationPrice || '0')
              } as Position);
            }
          } catch {
            // No position for this symbol, continue
          }
        }

        return positions;
      } catch (error) {
        console.error('Error fetching positions:', error);
        return [];
      }
    });
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
        const data = await this.authenticatedRequest<WeexAccountData[]>(
          'GET',
          '/capi/v2/account/assets'
        );

        return data.map(account => ({
          currency: account.coinName.toUpperCase(),
          available: parseFloat(account.available),
          frozen: parseFloat(account.frozen),
          total: parseFloat(account.equity)
        }));
      } catch (error) {
        console.error('Error fetching balance:', error);
        throw error;
      }
    });
  }

  /**
   * Set leverage for a symbol.
   * Required by hackathon guide step 4.
   *
   * @param symbol - Trading pair symbol
   * @param leverage - Leverage multiplier (max 20x per competition rules)
   * @param holdSide - Position side ('long', 'short', or 'both')
   */
  async setLeverage(
    symbol: AllowedPair,
    leverage: number,
    holdSide: 'long' | 'short' | 'both' = 'both'
  ): Promise<boolean> {
    this.ensureConnected();
    this.validateSymbol(symbol);

    if (leverage > 20) {
      throw new Error('Maximum leverage is 20x per competition rules');
    }

    // Paper trading doesn't need to set leverage via API
    if (this.usePaperTrading()) {
      console.log(`[PAPER] Setting leverage for ${symbol} to ${leverage}x`);
      return true;
    }

    return this.rateLimitedRequest(async () => {
      try {
        const weexSymbol = this.toWeexSymbol(symbol);

        await this.authenticatedRequest<unknown>(
          'POST',
          '/capi/v2/account/leverage',
          undefined,
          {
            symbol: weexSymbol,
            marginMode: '1', // 1 = Cross mode
            longLeverage: leverage.toString(),
            shortLeverage: leverage.toString()
          }
        );

        console.log(`[LIVE] Set leverage for ${symbol} to ${leverage}x`);
        return true;
      } catch (error) {
        console.error(`Error setting leverage for ${symbol}:`, error);
        return false;
      }
    });
  }

  /**
   * Get order history.
   * Required by hackathon guide step 10.
   */
  async getOrderHistory(symbol?: AllowedPair, limit: number = 100): Promise<Order[]> {
    this.ensureConnected();
    if (symbol) this.validateSymbol(symbol);

    // Route to paper trading if disabled
    if (this.usePaperTrading() && this.paperEngine) {
      // Paper engine can return empty for now
      return [];
    }

    return this.rateLimitedRequest(async () => {
      try {
        const params: Record<string, string> = {
          limit: limit.toString()
        };
        if (symbol) {
          params.symbol = this.toWeexSymbol(symbol);
        }

        const data = await this.authenticatedRequest<WeexOrderDetail[]>(
          'GET',
          '/capi/v2/order/history',
          params
        );

        // Handle empty response
        if (!data || !Array.isArray(data)) {
          return [];
        }

        return data.map(order => {
          const rawSymbol = order.symbol.replace('cmt_', '').replace('usdt', '').toUpperCase();
          // Map type code to order side
          const typeCode = parseInt(order.type);
          const side = (typeCode === 1 || typeCode === 3) ? 'buy' : 'sell';
          const orderType = order.match_price === '1' ? 'market' : 'limit';

          return {
            id: order.order_id,
            symbol: (symbol || rawSymbol) as AllowedPair,
            side: side as OrderSide,
            type: orderType as OrderType,
            size: parseFloat(order.size),
            price: parseFloat(order.price_avg) || parseFloat(order.price),
            timestamp: parseInt(order.createTime)
          };
        });
      } catch (error) {
        console.error('Error fetching order history:', error);
        return [];
      }
    });
  }

  /**
   * Get trade (fill) details.
   * Required by hackathon guide step 11.
   */
  async getTradeDetails(symbol?: AllowedPair, limit: number = 100): Promise<Array<{
    tradeId: string;
    orderId: string;
    symbol: AllowedPair;
    side: OrderSide;
    price: number;
    size: number;
    fee: number;
    timestamp: number;
  }>> {
    this.ensureConnected();
    if (symbol) this.validateSymbol(symbol);

    // Route to paper trading if disabled
    if (this.usePaperTrading()) {
      return [];
    }

    return this.rateLimitedRequest(async () => {
      try {
        const params: Record<string, string> = {
          pageSize: limit.toString()
        };
        if (symbol) {
          params.symbol = this.toWeexSymbol(symbol);
        }

        interface WeexTradeDetail {
          tradeId: string;
          orderId: string;
          symbol: string;
          side: string;
          price: string;
          size: string;
          fee: string;
          feeAsset: string;
          cTime: string;
        }

        const data = await this.authenticatedRequest<WeexTradeDetail[]>(
          'GET',
          '/capi/v2/trade/fills',
          params
        );

        return data.map(trade => {
          const rawSymbol = trade.symbol.replace('cmt_', '').replace('usdt', '').toUpperCase();
          const side = trade.side.includes('long') ? 'buy' : 'sell';

          return {
            tradeId: trade.tradeId,
            orderId: trade.orderId,
            symbol: (symbol || rawSymbol) as AllowedPair,
            side: side as OrderSide,
            price: parseFloat(trade.price),
            size: parseFloat(trade.size),
            fee: parseFloat(trade.fee),
            timestamp: parseInt(trade.cTime)
          };
        });
      } catch (error) {
        console.error('Error fetching trade details:', error);
        return [];
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
