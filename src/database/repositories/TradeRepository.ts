import { AppDatabase } from '../Database';

export interface DbTrade {
  id: number;
  trade_id: string;
  symbol: string;
  direction: 'long' | 'short';
  size: number;
  entry_price: number;
  exit_price: number | null;
  entry_time: string;
  exit_time: string | null;
  pnl: number | null;
  pnl_percent: number | null;
  fees: number;
  status: 'open' | 'closed' | 'cancelled';
  stop_loss: number | null;
  take_profit: number | null;
  confidence: number | null;
  regime: string | null;
  agent_contributions: string | null;
  reasoning: string | null;
  order_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface CreateTradeInput {
  trade_id: string;
  symbol: string;
  direction: 'long' | 'short';
  size: number;
  entry_price: number;
  entry_time: string;
  stop_loss?: number;
  take_profit?: number;
  confidence?: number;
  regime?: string;
  agent_contributions?: Record<string, number>;
  reasoning?: string;
  order_id?: string;
}

export interface TradeQueryOptions {
  symbol?: string;
  status?: 'open' | 'closed' | 'cancelled';
  startDate?: string;
  endDate?: string;
  limit?: number;
  offset?: number;
}

export interface PnlSummary {
  totalPnl: number;
  realizedPnl: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  averageWin: number;
  averageLoss: number;
  largestWin: number;
  largestLoss: number;
}

export class TradeRepository {
  private db: AppDatabase;

  constructor(db: AppDatabase) {
    this.db = db;
  }

  createTrade(input: CreateTradeInput): DbTrade {
    // Input validation
    if (!input.trade_id || input.trade_id.trim() === '') {
      throw new Error('Trade ID is required');
    }
    if (!input.symbol || input.symbol.trim() === '') {
      throw new Error('Trade symbol is required');
    }
    if (input.size <= 0) {
      throw new Error('Trade size must be positive');
    }
    if (input.entry_price <= 0) {
      throw new Error('Entry price must be positive');
    }

    this.db.run(
      `INSERT INTO trades (
        trade_id, symbol, direction, size, entry_price, entry_time,
        stop_loss, take_profit, confidence, regime, agent_contributions,
        reasoning, order_id, status
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')`,
      [
        input.trade_id,
        input.symbol,
        input.direction,
        input.size,
        input.entry_price,
        input.entry_time,
        input.stop_loss ?? null,
        input.take_profit ?? null,
        input.confidence ?? null,
        input.regime ?? null,
        input.agent_contributions ? JSON.stringify(input.agent_contributions) : null,
        input.reasoning ?? null,
        input.order_id ?? null
      ]
    );

    const trade = this.getTradeById(input.trade_id);
    if (!trade) {
      throw new Error(`Failed to create trade ${input.trade_id}`);
    }
    return trade;
  }

  getTradeById(tradeId: string): DbTrade | undefined {
    return this.db.get<DbTrade>('SELECT * FROM trades WHERE trade_id = ?', [tradeId]);
  }

  closeTrade(tradeId: string, exitPrice: number, exitTime: string): DbTrade | undefined {
    const trade = this.getTradeById(tradeId);
    if (!trade || trade.status !== 'open') return undefined;

    const direction = trade.direction === 'long' ? 1 : -1;
    const pnl = (exitPrice - trade.entry_price) * direction * trade.size;
    const pnlPercent = ((exitPrice - trade.entry_price) / trade.entry_price) * direction * 100;

    // Use runAndSave for critical financial operations
    this.db.runAndSave(
      `UPDATE trades
       SET exit_price = ?, exit_time = ?, pnl = ?, pnl_percent = ?,
           status = 'closed', updated_at = datetime('now')
       WHERE trade_id = ?`,
      [exitPrice, exitTime, pnl, pnlPercent, tradeId]
    );

    return this.getTradeById(tradeId);
  }

  cancelTrade(tradeId: string): void {
    this.db.run(
      `UPDATE trades SET status = 'cancelled', updated_at = datetime('now') WHERE trade_id = ?`,
      [tradeId]
    );
  }

  getOpenTrades(): DbTrade[] {
    return this.db.query<DbTrade>(
      `SELECT * FROM trades WHERE status = 'open' ORDER BY entry_time DESC`
    );
  }

  getTradeHistory(options: TradeQueryOptions = {}): DbTrade[] {
    let sql = 'SELECT * FROM trades WHERE 1=1';
    const params: unknown[] = [];

    if (options.symbol) {
      sql += ' AND symbol = ?';
      params.push(options.symbol);
    }

    if (options.status) {
      sql += ' AND status = ?';
      params.push(options.status);
    }

    if (options.startDate) {
      sql += ' AND entry_time >= ?';
      params.push(options.startDate);
    }

    if (options.endDate) {
      sql += ' AND entry_time <= ?';
      params.push(options.endDate);
    }

    sql += ' ORDER BY entry_time DESC';

    if (options.limit) {
      sql += ' LIMIT ?';
      params.push(options.limit);

      if (options.offset) {
        sql += ' OFFSET ?';
        params.push(options.offset);
      }
    }

    return this.db.query<DbTrade>(sql, params);
  }

  getPnlSummary(): PnlSummary {
    const closedTrades = this.db.query<DbTrade>(
      `SELECT * FROM trades WHERE status = 'closed'`
    );

    const winningTrades = closedTrades.filter(t => (t.pnl ?? 0) > 0);
    const losingTrades = closedTrades.filter(t => (t.pnl ?? 0) < 0);

    const totalPnl = closedTrades.reduce((sum, t) => sum + (t.pnl ?? 0), 0);
    const avgWin = winningTrades.length > 0
      ? winningTrades.reduce((sum, t) => sum + (t.pnl ?? 0), 0) / winningTrades.length
      : 0;
    const avgLoss = losingTrades.length > 0
      ? losingTrades.reduce((sum, t) => sum + (t.pnl ?? 0), 0) / losingTrades.length
      : 0;

    const pnls = closedTrades.map(t => t.pnl ?? 0);
    const largestWin = pnls.length > 0 ? Math.max(...pnls, 0) : 0;
    const largestLoss = pnls.length > 0 ? Math.min(...pnls, 0) : 0;

    return {
      totalPnl,
      realizedPnl: totalPnl,
      totalTrades: closedTrades.length,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length,
      winRate: closedTrades.length > 0 ? winningTrades.length / closedTrades.length : 0,
      averageWin: avgWin,
      averageLoss: avgLoss,
      largestWin,
      largestLoss
    };
  }

  getTradesBySymbol(symbol: string): DbTrade[] {
    return this.getTradeHistory({ symbol });
  }

  getRecentTrades(limit: number = 10): DbTrade[] {
    return this.getTradeHistory({ limit });
  }

  getTradeCount(): number {
    const result = this.db.get<{ count: number }>('SELECT COUNT(*) as count FROM trades');
    return result?.count ?? 0;
  }

  getOpenTradeCount(): number {
    const result = this.db.get<{ count: number }>(
      `SELECT COUNT(*) as count FROM trades WHERE status = 'open'`
    );
    return result?.count ?? 0;
  }
}
