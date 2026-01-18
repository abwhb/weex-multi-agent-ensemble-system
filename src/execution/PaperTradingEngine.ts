import {
  AppDatabase,
  AccountRepository,
  PositionRepository,
  TradeRepository,
  OrderRepository,
  PnlSummary
} from '../database';
import { Position, Order, OrderSide, OrderType, AllowedPair, Direction, MarketRegime } from '../types';

// Generate a unique ID
function generateId(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

export interface OrderParams {
  symbol: AllowedPair;
  side: OrderSide;
  type: OrderType;
  size: number;
  price?: number;
  stopLoss?: number;
  takeProfit?: number;
  reduceOnly?: boolean;
}

export interface TradeDecisionInput {
  symbol: AllowedPair;
  direction: Direction;
  size: number;
  confidence: number;
  stopLoss?: number;
  takeProfit?: number;
  regime?: MarketRegime;
  agentContributions?: Record<string, number>;
  reasoning?: string;
}

export interface Balance {
  currency: string;
  available: number;
  frozen: number;
  total: number;
}

export interface AccountState {
  balance: number;
  equity: number;
  unrealizedPnl: number;
  realizedPnl: number;
  marginUsed: number;
  freeMargin: number;
}

export interface TradeResult {
  tradeId: string;
  symbol: string;
  direction: 'long' | 'short';
  entryPrice: number;
  exitPrice: number;
  size: number;
  pnl: number;
  pnlPercent: number;
  duration: number;
}

export class PaperTradingEngine {
  private db: AppDatabase;
  private accountRepo: AccountRepository;
  private positionRepo: PositionRepository;
  private tradeRepo: TradeRepository;
  private orderRepo: OrderRepository;
  private initialBalance: number;

  constructor(db: AppDatabase, initialBalance: number = 10000) {
    this.db = db;
    this.initialBalance = initialBalance;
    this.accountRepo = new AccountRepository(db);
    this.positionRepo = new PositionRepository(db);
    this.tradeRepo = new TradeRepository(db);
    this.orderRepo = new OrderRepository(db);

    // Initialize account if not exists
    this.accountRepo.initializeAccount(initialBalance);
  }

  // ============================================================================
  // Account Management
  // ============================================================================

  getBalance(): Balance[] {
    const account = this.accountRepo.getAccount();
    if (!account) {
      return [{
        currency: 'USDT',
        available: this.initialBalance,
        frozen: 0,
        total: this.initialBalance
      }];
    }

    const unrealizedPnl = this.positionRepo.getTotalUnrealizedPnl();
    const equity = account.current_balance + unrealizedPnl;

    return [{
      currency: 'USDT',
      available: account.current_balance,
      frozen: 0,
      total: equity
    }];
  }

  getEquity(): number {
    const account = this.accountRepo.getAccount();
    if (!account) return this.initialBalance;

    const unrealizedPnl = this.positionRepo.getTotalUnrealizedPnl();
    return account.current_balance + unrealizedPnl;
  }

  getAccountState(): AccountState {
    const account = this.accountRepo.getAccount();
    const unrealizedPnl = this.positionRepo.getTotalUnrealizedPnl();

    if (!account) {
      return {
        balance: this.initialBalance,
        equity: this.initialBalance,
        unrealizedPnl: 0,
        realizedPnl: 0,
        marginUsed: 0,
        freeMargin: this.initialBalance
      };
    }

    const equity = account.current_balance + unrealizedPnl;
    const positions = this.positionRepo.getAllPositions();
    const marginUsed = positions.reduce((sum, p) => sum + (p.size * p.entry_price / (p.leverage || 1)), 0);

    return {
      balance: account.current_balance,
      equity,
      unrealizedPnl,
      realizedPnl: account.realized_pnl,
      marginUsed,
      freeMargin: equity - marginUsed
    };
  }

  // ============================================================================
  // Position Management
  // ============================================================================

  getPositions(symbol?: AllowedPair): Position[] {
    const dbPositions = symbol
      ? [this.positionRepo.getPosition(symbol)].filter(Boolean)
      : this.positionRepo.getAllPositions();

    return dbPositions.map(p => ({
      symbol: p!.symbol,
      side: p!.side,
      size: p!.size,
      entryPrice: p!.entry_price,
      unrealizedPnl: p!.unrealized_pnl,
      leverage: p!.leverage
    }));
  }

  updatePositionPrices(prices: Map<string, number>): void {
    for (const [symbol, price] of prices) {
      this.positionRepo.updateCurrentPrice(symbol, price);
    }

    // Update account equity
    const unrealizedPnl = this.positionRepo.getTotalUnrealizedPnl();
    const account = this.accountRepo.getAccount();
    if (account) {
      this.accountRepo.updateEquity(account.current_balance + unrealizedPnl, unrealizedPnl);
    }
  }

  checkStopLossTakeProfit(symbol: string, currentPrice: number): Order | null {
    const position = this.positionRepo.getPosition(symbol);
    if (!position) return null;

    const isLong = position.side === 'long';
    const positionSize = position.size;

    // Check stop loss
    if (position.stop_loss) {
      const stopHit = isLong
        ? currentPrice <= position.stop_loss
        : currentPrice >= position.stop_loss;

      if (stopHit) {
        const order = this.createCloseOrder(symbol, positionSize, currentPrice, 'stop_loss');
        return order; // may be null if position was already closed
      }
    }

    // Check take profit
    if (position.take_profit) {
      const tpHit = isLong
        ? currentPrice >= position.take_profit
        : currentPrice <= position.take_profit;

      if (tpHit) {
        const order = this.createCloseOrder(symbol, positionSize, currentPrice, 'take_profit');
        return order; // may be null if position was already closed
      }
    }

    return null;
  }

  private createCloseOrder(symbol: string, size: number, price: number, reason: string): Order | null {
    const position = this.positionRepo.getPosition(symbol);
    if (!position) {
      console.warn(`[PaperTradingEngine] Cannot close order for ${symbol}: no position found`);
      return null;
    }

    const side: OrderSide = position.side === 'long' ? 'sell' : 'buy';

    const order = this.orderRepo.createOrder({
      order_id: generateId('ord'),
      symbol,
      side,
      type: 'market',
      size,
      reduce_only: true
    });

    // Execute immediately for paper trading
    this.orderRepo.fillOrder(order.order_id, price);

    // Close position and record P&L
    this.closePositionInternal(symbol, price);

    return {
      id: order.order_id,
      symbol,
      side,
      type: 'market',
      size,
      timestamp: Date.now()
    };
  }

  // ============================================================================
  // Order Execution
  // ============================================================================

  executeOrder(params: OrderParams, currentPrice: number): Order {
    const orderId = generateId('ord');
    const fillPrice = params.type === 'market' ? currentPrice : (params.price ?? currentPrice);

    // Validate margin for non-reduce-only orders
    if (!params.reduceOnly) {
      const requiredMargin = params.size * fillPrice;
      const accountState = this.getAccountState();

      if (requiredMargin > accountState.freeMargin) {
        throw new Error(
          `Insufficient margin: required ${requiredMargin.toFixed(2)} USDT, ` +
          `available ${accountState.freeMargin.toFixed(2)} USDT`
        );
      }
    }

    // Create order record
    const dbOrder = this.orderRepo.createOrder({
      order_id: orderId,
      symbol: params.symbol,
      side: params.side,
      type: params.type,
      size: params.size,
      price: params.price,
      reduce_only: params.reduceOnly
    });

    // For paper trading, execute immediately if market order
    if (params.type === 'market') {
      this.orderRepo.fillOrder(orderId, fillPrice);

      // Update position
      this.updatePositionFromOrder(params, fillPrice);
    }

    return {
      id: orderId,
      symbol: params.symbol,
      side: params.side,
      type: params.type,
      size: params.size,
      price: fillPrice,
      stopLoss: params.stopLoss,
      takeProfit: params.takeProfit,
      timestamp: Date.now()
    };
  }

  private updatePositionFromOrder(params: OrderParams, fillPrice: number): void {
    // Wrap in transaction to prevent race conditions
    this.db.transaction(() => {
      const existingPosition = this.positionRepo.getPosition(params.symbol);
      const isOpeningTrade = params.side === 'buy' ? 'long' : 'short';

      if (params.reduceOnly && existingPosition) {
        // Closing or reducing position
        this.closePositionInternal(params.symbol, fillPrice);
        return;
      }

      if (existingPosition) {
        // Check if same direction (adding to position) or opposite (closing)
        if (existingPosition.side === isOpeningTrade) {
          // Add to existing position
          const newSize = existingPosition.size + params.size;
          const newEntryPrice = (existingPosition.entry_price * existingPosition.size + fillPrice * params.size) / newSize;
          this.positionRepo.updatePosition(params.symbol, {
            size: newSize,
            entry_price: newEntryPrice,
            stop_loss: params.stopLoss,
            take_profit: params.takeProfit
          });
        } else {
          // Opposite direction - close position first
          const existingSize = existingPosition.size;
          this.closePositionInternal(params.symbol, fillPrice);

          // If new size is larger, open remainder in new direction
          if (params.size > existingSize) {
            const remainingSize = params.size - existingSize;
            this.positionRepo.createPosition({
              symbol: params.symbol,
              side: isOpeningTrade,
              size: remainingSize,
              entry_price: fillPrice,
              stop_loss: params.stopLoss,
              take_profit: params.takeProfit
            });
          }
        }
      } else {
        // Open new position
        this.positionRepo.createPosition({
          symbol: params.symbol,
          side: isOpeningTrade,
          size: params.size,
          entry_price: fillPrice,
          stop_loss: params.stopLoss,
          take_profit: params.takeProfit
        });
      }
    });
  }

  private closePositionInternal(symbol: string, exitPrice: number): void {
    const position = this.positionRepo.getPosition(symbol);
    if (!position) return;

    // Calculate P&L
    const direction = position.side === 'long' ? 1 : -1;
    const pnl = (exitPrice - position.entry_price) * direction * position.size;

    // Record realized P&L
    this.accountRepo.recordRealizedPnl(pnl);

    // Close any open trades for this symbol
    const openTrades = this.tradeRepo.getOpenTrades().filter(t => t.symbol === symbol);
    for (const trade of openTrades) {
      this.tradeRepo.closeTrade(trade.trade_id, exitPrice, new Date().toISOString());
    }

    // Remove position
    this.positionRepo.closePosition(symbol);
  }

  cancelOrder(orderId: string): boolean {
    const order = this.orderRepo.getOrderById(orderId);
    if (!order || order.status !== 'pending') return false;

    this.orderRepo.cancelOrder(orderId);
    return true;
  }

  getOrder(orderId: string): Order | null {
    const dbOrder = this.orderRepo.getOrderById(orderId);
    if (!dbOrder) return null;

    return {
      id: dbOrder.order_id,
      symbol: dbOrder.symbol as AllowedPair,
      side: dbOrder.side,
      type: dbOrder.type,
      size: dbOrder.size,
      price: dbOrder.fill_price ?? dbOrder.price ?? undefined,
      timestamp: new Date(dbOrder.created_at).getTime()
    };
  }

  getOpenOrders(symbol?: string): Order[] {
    const dbOrders = this.orderRepo.getOpenOrders(symbol);
    return dbOrders.map(o => ({
      id: o.order_id,
      symbol: o.symbol as AllowedPair,
      side: o.side,
      type: o.type,
      size: o.size,
      price: o.price ?? undefined,
      timestamp: new Date(o.created_at).getTime()
    }));
  }

  // ============================================================================
  // Trade Management
  // ============================================================================

  openTrade(decision: TradeDecisionInput, fillPrice: number, orderId: string): string {
    // Validate direction - neutral should not open a trade
    if (decision.direction === 'neutral') {
      throw new Error('Cannot open trade with neutral direction');
    }

    const tradeId = generateId('trd');

    this.tradeRepo.createTrade({
      trade_id: tradeId,
      symbol: decision.symbol,
      direction: decision.direction,
      size: decision.size,
      entry_price: fillPrice,
      entry_time: new Date().toISOString(),
      stop_loss: decision.stopLoss,
      take_profit: decision.takeProfit,
      confidence: decision.confidence,
      regime: decision.regime,
      agent_contributions: decision.agentContributions,
      reasoning: decision.reasoning,
      order_id: orderId
    });

    return tradeId;
  }

  closeTrade(tradeId: string, exitPrice: number): TradeResult | null {
    const trade = this.tradeRepo.getTradeById(tradeId);
    if (!trade || trade.status !== 'open') return null;

    const closedTrade = this.tradeRepo.closeTrade(tradeId, exitPrice, new Date().toISOString());
    if (!closedTrade) return null;

    const duration = new Date(closedTrade.exit_time!).getTime() - new Date(closedTrade.entry_time).getTime();

    return {
      tradeId: closedTrade.trade_id,
      symbol: closedTrade.symbol,
      direction: closedTrade.direction,
      entryPrice: closedTrade.entry_price,
      exitPrice: closedTrade.exit_price!,
      size: closedTrade.size,
      pnl: closedTrade.pnl!,
      pnlPercent: closedTrade.pnl_percent!,
      duration
    };
  }

  // ============================================================================
  // Query Methods
  // ============================================================================

  getTradeHistory(options?: {
    symbol?: string;
    status?: 'open' | 'closed' | 'cancelled';
    limit?: number;
    offset?: number;
  }): Array<{
    id: string;
    symbol: string;
    direction: 'long' | 'short';
    size: number;
    entryPrice: number;
    exitPrice: number | null;
    entryTime: string;
    exitTime: string | null;
    pnl: number | null;
    pnlPercent: number | null;
    status: 'open' | 'closed' | 'cancelled';
    confidence: number | null;
    regime: string | null;
  }> {
    const trades = this.tradeRepo.getTradeHistory(options);
    return trades.map(t => ({
      id: t.trade_id,
      symbol: t.symbol,
      direction: t.direction,
      size: t.size,
      entryPrice: t.entry_price,
      exitPrice: t.exit_price,
      entryTime: t.entry_time,
      exitTime: t.exit_time,
      pnl: t.pnl,
      pnlPercent: t.pnl_percent,
      status: t.status,
      confidence: t.confidence,
      regime: t.regime
    }));
  }

  getPnlSummary(): PnlSummary {
    return this.tradeRepo.getPnlSummary();
  }

  getOpenTrades(): Array<{
    id: string;
    symbol: string;
    direction: 'long' | 'short';
    size: number;
    entryPrice: number;
    entryTime: string;
    confidence: number | null;
  }> {
    const trades = this.tradeRepo.getOpenTrades();
    return trades.map(t => ({
      id: t.trade_id,
      symbol: t.symbol,
      direction: t.direction,
      size: t.size,
      entryPrice: t.entry_price,
      entryTime: t.entry_time,
      confidence: t.confidence
    }));
  }

  // ============================================================================
  // Reset / Admin
  // ============================================================================

  resetAccount(): void {
    // Close all positions
    this.positionRepo.closeAllPositions();

    // Reset account balance
    this.accountRepo.resetAccount(this.initialBalance);
  }
}
