import { AppDatabase } from '../Database';

export interface Account {
  id: number;
  initial_balance: number;
  current_balance: number;
  equity: number;
  unrealized_pnl: number;
  realized_pnl: number;
  created_at: string;
  updated_at: string;
}

export class AccountRepository {
  private db: AppDatabase;

  constructor(db: AppDatabase) {
    this.db = db;
  }

  getAccount(): Account | undefined {
    return this.db.get<Account>('SELECT * FROM accounts WHERE id = 1');
  }

  initializeAccount(initialBalance: number): Account {
    const existing = this.getAccount();
    if (existing) {
      return existing;
    }

    this.db.run(
      `INSERT INTO accounts (id, initial_balance, current_balance, equity, unrealized_pnl, realized_pnl)
       VALUES (1, ?, ?, ?, 0, 0)`,
      [initialBalance, initialBalance, initialBalance]
    );

    return this.getAccount()!;
  }

  updateBalance(currentBalance: number): void {
    this.db.run(
      `UPDATE accounts SET current_balance = ?, updated_at = datetime('now') WHERE id = 1`,
      [currentBalance]
    );
  }

  updateEquity(equity: number, unrealizedPnl: number): void {
    this.db.run(
      `UPDATE accounts SET equity = ?, unrealized_pnl = ?, updated_at = datetime('now') WHERE id = 1`,
      [equity, unrealizedPnl]
    );
  }

  recordRealizedPnl(pnl: number): void {
    this.db.run(
      `UPDATE accounts
       SET realized_pnl = realized_pnl + ?,
           current_balance = current_balance + ?,
           updated_at = datetime('now')
       WHERE id = 1`,
      [pnl, pnl]
    );
  }

  resetAccount(initialBalance: number): void {
    this.db.run(
      `UPDATE accounts
       SET initial_balance = ?,
           current_balance = ?,
           equity = ?,
           unrealized_pnl = 0,
           realized_pnl = 0,
           updated_at = datetime('now')
       WHERE id = 1`,
      [initialBalance, initialBalance, initialBalance]
    );
  }

  getBalanceHistory(): Array<{ balance: number; timestamp: string }> {
    // For now, return current state. Could be enhanced with a balance_history table
    const account = this.getAccount();
    if (!account) return [];
    return [{ balance: account.current_balance, timestamp: account.updated_at }];
  }
}
