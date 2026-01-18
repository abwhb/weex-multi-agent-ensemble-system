import { AppDatabase } from '../Database';

export interface DbOrder {
  id: number;
  order_id: string;
  trade_id: string | null;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  size: number;
  price: number | null;
  fill_price: number | null;
  status: 'pending' | 'filled' | 'cancelled' | 'failed';
  reduce_only: number;
  created_at: string;
  filled_at: string | null;
}

export interface CreateOrderInput {
  order_id: string;
  trade_id?: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  size: number;
  price?: number;
  reduce_only?: boolean;
}

export interface OrderQueryOptions {
  symbol?: string;
  status?: 'pending' | 'filled' | 'cancelled' | 'failed';
  limit?: number;
  offset?: number;
}

export class OrderRepository {
  private db: AppDatabase;

  constructor(db: AppDatabase) {
    this.db = db;
  }

  createOrder(input: CreateOrderInput): DbOrder {
    // Input validation
    if (!input.order_id || input.order_id.trim() === '') {
      throw new Error('Order ID is required');
    }
    if (!input.symbol || input.symbol.trim() === '') {
      throw new Error('Order symbol is required');
    }
    if (input.size <= 0) {
      throw new Error('Order size must be positive');
    }
    if (input.type === 'limit' && (input.price === undefined || input.price <= 0)) {
      throw new Error('Limit orders require a positive price');
    }

    this.db.run(
      `INSERT INTO orders (order_id, trade_id, symbol, side, type, size, price, reduce_only, status)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')`,
      [
        input.order_id,
        input.trade_id ?? null,
        input.symbol,
        input.side,
        input.type,
        input.size,
        input.price ?? null,
        input.reduce_only ? 1 : 0
      ]
    );

    const order = this.getOrderById(input.order_id);
    if (!order) {
      throw new Error(`Failed to create order ${input.order_id}`);
    }
    return order;
  }

  getOrderById(orderId: string): DbOrder | undefined {
    return this.db.get<DbOrder>('SELECT * FROM orders WHERE order_id = ?', [orderId]);
  }

  fillOrder(orderId: string, fillPrice: number): DbOrder | undefined {
    this.db.run(
      `UPDATE orders
       SET status = 'filled', fill_price = ?, filled_at = datetime('now')
       WHERE order_id = ?`,
      [fillPrice, orderId]
    );

    return this.getOrderById(orderId);
  }

  cancelOrder(orderId: string): void {
    this.db.run(
      `UPDATE orders SET status = 'cancelled' WHERE order_id = ?`,
      [orderId]
    );
  }

  failOrder(orderId: string): void {
    this.db.run(
      `UPDATE orders SET status = 'failed' WHERE order_id = ?`,
      [orderId]
    );
  }

  getOpenOrders(symbol?: string): DbOrder[] {
    if (symbol) {
      return this.db.query<DbOrder>(
        `SELECT * FROM orders WHERE status = 'pending' AND symbol = ? ORDER BY created_at DESC`,
        [symbol]
      );
    }
    return this.db.query<DbOrder>(
      `SELECT * FROM orders WHERE status = 'pending' ORDER BY created_at DESC`
    );
  }

  getOrderHistory(options: OrderQueryOptions = {}): DbOrder[] {
    let sql = 'SELECT * FROM orders WHERE 1=1';
    const params: unknown[] = [];

    if (options.symbol) {
      sql += ' AND symbol = ?';
      params.push(options.symbol);
    }

    if (options.status) {
      sql += ' AND status = ?';
      params.push(options.status);
    }

    sql += ' ORDER BY created_at DESC';

    if (options.limit) {
      sql += ' LIMIT ?';
      params.push(options.limit);

      if (options.offset) {
        sql += ' OFFSET ?';
        params.push(options.offset);
      }
    }

    return this.db.query<DbOrder>(sql, params);
  }

  getOrdersByTradeId(tradeId: string): DbOrder[] {
    return this.db.query<DbOrder>(
      `SELECT * FROM orders WHERE trade_id = ? ORDER BY created_at ASC`,
      [tradeId]
    );
  }

  getRecentOrders(limit: number = 10): DbOrder[] {
    return this.getOrderHistory({ limit });
  }

  getOrderCount(): number {
    const result = this.db.get<{ count: number }>('SELECT COUNT(*) as count FROM orders');
    return result?.count ?? 0;
  }

  getPendingOrderCount(): number {
    const result = this.db.get<{ count: number }>(
      `SELECT COUNT(*) as count FROM orders WHERE status = 'pending'`
    );
    return result?.count ?? 0;
  }
}
