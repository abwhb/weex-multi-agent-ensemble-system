import initSqlJs, { Database as SqlJsDatabase } from 'sql.js';
import path from 'path';
import fs from 'fs';

export interface DatabaseConfig {
  path: string;
  runMigrations: boolean;
}

export class AppDatabase {
  private db: SqlJsDatabase | null = null;
  private config: DatabaseConfig;
  private initialized: boolean = false;

  constructor(config: DatabaseConfig) {
    this.config = config;
  }

  async initializeAsync(): Promise<void> {
    if (this.initialized) return;

    // Ensure directory exists
    const dir = path.dirname(this.config.path);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    // Initialize SQL.js
    const SQL = await initSqlJs();

    // Load existing database or create new one
    if (fs.existsSync(this.config.path)) {
      const buffer = fs.readFileSync(this.config.path);
      this.db = new SQL.Database(buffer);
    } else {
      this.db = new SQL.Database();
    }

    // Enable foreign keys
    this.db.run('PRAGMA foreign_keys = ON');

    if (this.config.runMigrations) {
      this.runMigrations();
    }

    this.initialized = true;
  }

  initialize(): void {
    // Synchronous wrapper - for constructor usage, call initializeAsync() separately
    if (this.db) return;

    // We'll do a synchronous initialization using a workaround
    // In practice, call initializeAsync() after construction
    console.log('[Database] Call initializeAsync() to complete initialization');
  }

  private runMigrations(): void {
    if (!this.db) throw new Error('Database not initialized');

    // Create migrations tracking table
    this.db.run(`
      CREATE TABLE IF NOT EXISTS migrations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        applied_at TEXT NOT NULL DEFAULT (datetime('now'))
      )
    `);

    const migrations = this.getMigrations();
    const appliedMigrations = this.getAppliedMigrations();

    for (const migration of migrations) {
      if (!appliedMigrations.includes(migration.name)) {
        this.applyMigration(migration);
      }
    }

    // Save to disk after migrations
    this.save();
  }

  private getMigrations(): Array<{ name: string; sql: string }> {
    return [
      {
        name: '001_initial_schema',
        sql: `
          -- Trading control state (trading DISABLED by default)
          CREATE TABLE IF NOT EXISTS trading_control (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            enabled INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
          );

          -- Insert default row with trading disabled
          INSERT OR IGNORE INTO trading_control (id, enabled) VALUES (1, 0);

          -- Account state
          CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            initial_balance REAL NOT NULL,
            current_balance REAL NOT NULL,
            equity REAL NOT NULL,
            unrealized_pnl REAL NOT NULL DEFAULT 0,
            realized_pnl REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
          );

          -- Open positions
          CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            side TEXT NOT NULL CHECK(side IN ('long', 'short')),
            size REAL NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL,
            unrealized_pnl REAL DEFAULT 0,
            leverage INTEGER DEFAULT 1,
            stop_loss REAL,
            take_profit REAL,
            opened_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
          );

          -- Trade history with P&L
          CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT UNIQUE NOT NULL,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL CHECK(direction IN ('long', 'short')),
            size REAL NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            pnl REAL,
            pnl_percent REAL,
            fees REAL DEFAULT 0,
            status TEXT NOT NULL CHECK(status IN ('open', 'closed', 'cancelled')),
            stop_loss REAL,
            take_profit REAL,
            confidence REAL,
            regime TEXT,
            agent_contributions TEXT,
            reasoning TEXT,
            order_id TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
          );

          -- Order history
          CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT UNIQUE NOT NULL,
            trade_id TEXT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
            type TEXT NOT NULL CHECK(type IN ('market', 'limit')),
            size REAL NOT NULL,
            price REAL,
            fill_price REAL,
            status TEXT NOT NULL CHECK(status IN ('pending', 'filled', 'cancelled', 'failed')),
            reduce_only INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            filled_at TEXT,
            FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
          );

          -- Daily snapshots
          CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            starting_balance REAL NOT NULL,
            ending_balance REAL,
            pnl REAL DEFAULT 0,
            trade_count INTEGER DEFAULT 0,
            win_count INTEGER DEFAULT 0,
            loss_count INTEGER DEFAULT 0,
            max_drawdown REAL DEFAULT 0,
            peak_balance REAL
          );

          -- Create indexes for common queries
          CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
          CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
          CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
          CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
          CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
          CREATE INDEX IF NOT EXISTS idx_daily_stats_date ON daily_stats(date);
        `
      }
    ];
  }

  private getAppliedMigrations(): string[] {
    if (!this.db) return [];
    const result = this.db.exec('SELECT name FROM migrations');
    if (result.length === 0) return [];
    return result[0].values.map(row => row[0] as string);
  }

  private applyMigration(migration: { name: string; sql: string }): void {
    if (!this.db) throw new Error('Database not initialized');

    // Split SQL statements and execute each
    const statements = migration.sql.split(';').filter(s => s.trim());
    for (const statement of statements) {
      if (statement.trim()) {
        this.db.run(statement);
      }
    }
    this.db.run('INSERT INTO migrations (name) VALUES (?)', [migration.name]);
    console.log(`Applied migration: ${migration.name}`);
  }

  query<T>(sql: string, params: unknown[] = []): T[] {
    if (!this.db) throw new Error('Database not initialized');
    const result = this.db.exec(sql, params as any[]);
    if (result.length === 0) return [];

    const columns = result[0].columns;
    return result[0].values.map(row => {
      const obj: Record<string, unknown> = {};
      columns.forEach((col, i) => {
        obj[col] = row[i];
      });
      return obj as T;
    });
  }

  get<T>(sql: string, params: unknown[] = []): T | undefined {
    const results = this.query<T>(sql, params);
    return results[0];
  }

  run(sql: string, params: unknown[] = []): { changes: number; lastInsertRowid: number } {
    if (!this.db) throw new Error('Database not initialized');
    this.db.run(sql, params as any[]);
    const changes = this.db.getRowsModified();
    const lastIdResult = this.db.exec('SELECT last_insert_rowid()');
    const lastInsertRowid = lastIdResult.length > 0 ? (lastIdResult[0].values[0][0] as number) : 0;
    // Note: save() is NOT called automatically to avoid performance issues
    // Call save() explicitly or use transaction() which auto-saves on commit
    return { changes, lastInsertRowid };
  }

  /**
   * Execute a write operation and immediately save to disk.
   * Use this for critical operations where data loss is unacceptable.
   */
  runAndSave(sql: string, params: unknown[] = []): { changes: number; lastInsertRowid: number } {
    const result = this.run(sql, params);
    this.save();
    return result;
  }

  transaction<T>(fn: () => T): T {
    if (!this.db) throw new Error('Database not initialized');
    this.db.run('BEGIN TRANSACTION');
    try {
      const result = fn();
      this.db.run('COMMIT');
      this.save();
      return result;
    } catch (error) {
      this.db.run('ROLLBACK');
      throw error;
    }
  }

  save(): void {
    if (!this.db) return;
    const data = this.db.export();
    const buffer = Buffer.from(data);
    fs.writeFileSync(this.config.path, buffer);
  }

  close(): void {
    if (this.db) {
      this.save();
      this.db.close();
      this.db = null;
    }
    this.initialized = false;
  }

  isInitialized(): boolean {
    return this.initialized;
  }
}
