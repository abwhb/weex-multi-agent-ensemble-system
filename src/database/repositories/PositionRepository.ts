import { AppDatabase } from '../Database';

export interface DbPosition {
  id: number;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entry_price: number;
  current_price: number | null;
  unrealized_pnl: number;
  leverage: number;
  stop_loss: number | null;
  take_profit: number | null;
  opened_at: string;
  updated_at: string;
}

export interface CreatePositionInput {
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entry_price: number;
  leverage?: number;
  stop_loss?: number;
  take_profit?: number;
}

export class PositionRepository {
  private db: AppDatabase;

  constructor(db: AppDatabase) {
    this.db = db;
  }

  createPosition(input: CreatePositionInput): DbPosition {
    this.db.run(
      `INSERT INTO positions (symbol, side, size, entry_price, leverage, stop_loss, take_profit)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [
        input.symbol,
        input.side,
        input.size,
        input.entry_price,
        input.leverage ?? 1,
        input.stop_loss ?? null,
        input.take_profit ?? null
      ]
    );

    return this.getPosition(input.symbol)!;
  }

  getPosition(symbol: string): DbPosition | undefined {
    return this.db.get<DbPosition>('SELECT * FROM positions WHERE symbol = ?', [symbol]);
  }

  getAllPositions(): DbPosition[] {
    return this.db.query<DbPosition>('SELECT * FROM positions ORDER BY opened_at DESC');
  }

  updatePosition(symbol: string, updates: Partial<CreatePositionInput>): void {
    const setClauses: string[] = [];
    const params: unknown[] = [];

    if (updates.size !== undefined) {
      setClauses.push('size = ?');
      params.push(updates.size);
    }
    if (updates.entry_price !== undefined) {
      setClauses.push('entry_price = ?');
      params.push(updates.entry_price);
    }
    if (updates.stop_loss !== undefined) {
      setClauses.push('stop_loss = ?');
      params.push(updates.stop_loss);
    }
    if (updates.take_profit !== undefined) {
      setClauses.push('take_profit = ?');
      params.push(updates.take_profit);
    }
    if (updates.leverage !== undefined) {
      setClauses.push('leverage = ?');
      params.push(updates.leverage);
    }

    setClauses.push("updated_at = datetime('now')");

    if (setClauses.length > 1) {
      params.push(symbol);
      this.db.run(
        `UPDATE positions SET ${setClauses.join(', ')} WHERE symbol = ?`,
        params
      );
    }
  }

  updateCurrentPrice(symbol: string, currentPrice: number): void {
    const position = this.getPosition(symbol);
    if (!position) return;

    const direction = position.side === 'long' ? 1 : -1;
    const unrealizedPnl = (currentPrice - position.entry_price) * direction * position.size;

    this.db.run(
      `UPDATE positions
       SET current_price = ?, unrealized_pnl = ?, updated_at = datetime('now')
       WHERE symbol = ?`,
      [currentPrice, unrealizedPnl, symbol]
    );
  }

  closePosition(symbol: string): DbPosition | undefined {
    const position = this.getPosition(symbol);
    if (!position) return undefined;

    this.db.run('DELETE FROM positions WHERE symbol = ?', [symbol]);
    return position;
  }

  closeAllPositions(): DbPosition[] {
    const positions = this.getAllPositions();
    this.db.run('DELETE FROM positions');
    return positions;
  }

  getTotalUnrealizedPnl(): number {
    const result = this.db.get<{ total: number }>(
      'SELECT COALESCE(SUM(unrealized_pnl), 0) as total FROM positions'
    );
    return result?.total ?? 0;
  }

  getPositionCount(): number {
    const result = this.db.get<{ count: number }>('SELECT COUNT(*) as count FROM positions');
    return result?.count ?? 0;
  }
}
