import { AppDatabase } from '../database';

export type TradingMode = 'paper' | 'live';

interface TradingControlRow {
  id: number;
  enabled: number;
  updated_at: string;
}

/**
 * TradingController manages the enable/disable state of live trading.
 *
 * By default, trading is DISABLED (paper mode).
 * When disabled → paper trading mode (simulation)
 * When enabled → live trading mode (real orders)
 *
 * State is persisted to SQLite database so it survives restarts.
 */
export class TradingController {
  private db: AppDatabase;
  private cachedEnabled: boolean | null = null;

  constructor(db: AppDatabase) {
    this.db = db;
    this.ensureInitialized();
  }

  /**
   * Ensure the trading_control table has the default row
   */
  private ensureInitialized(): void {
    const row = this.db.get<TradingControlRow>(
      'SELECT * FROM trading_control WHERE id = 1'
    );

    if (!row) {
      // Insert default row with trading disabled
      this.db.run(
        'INSERT INTO trading_control (id, enabled) VALUES (1, 0)'
      );
      this.cachedEnabled = false;
    } else {
      this.cachedEnabled = row.enabled === 1;
    }
  }

  /**
   * Enable live trading mode.
   * WARNING: This will route orders to the actual exchange API.
   */
  enableTrading(): void {
    this.db.run(
      `UPDATE trading_control SET enabled = 1, updated_at = datetime('now') WHERE id = 1`
    );
    this.cachedEnabled = true;
    console.log('[TradingController] Live trading ENABLED - orders will be sent to exchange');
  }

  /**
   * Disable live trading mode (switch to paper trading).
   * This is the safe default state.
   */
  disableTrading(): void {
    this.db.run(
      `UPDATE trading_control SET enabled = 0, updated_at = datetime('now') WHERE id = 1`
    );
    this.cachedEnabled = false;
    console.log('[TradingController] Live trading DISABLED - paper trading mode active');
  }

  /**
   * Check if live trading is enabled.
   * @returns true if live trading is enabled, false for paper trading
   */
  isEnabled(): boolean {
    if (this.cachedEnabled !== null) {
      return this.cachedEnabled;
    }

    const row = this.db.get<TradingControlRow>(
      'SELECT enabled FROM trading_control WHERE id = 1'
    );

    this.cachedEnabled = row?.enabled === 1;
    return this.cachedEnabled;
  }

  /**
   * Get the current trading mode as a string.
   * @returns 'live' if enabled, 'paper' if disabled
   */
  getMode(): TradingMode {
    return this.isEnabled() ? 'live' : 'paper';
  }

  /**
   * Get the last time the trading state was updated.
   */
  getLastUpdated(): string | null {
    const row = this.db.get<TradingControlRow>(
      'SELECT updated_at FROM trading_control WHERE id = 1'
    );
    return row?.updated_at ?? null;
  }

  /**
   * Get full trading control status.
   */
  getStatus(): {
    enabled: boolean;
    mode: TradingMode;
    lastUpdated: string | null;
  } {
    const row = this.db.get<TradingControlRow>(
      'SELECT * FROM trading_control WHERE id = 1'
    );

    return {
      enabled: row?.enabled === 1,
      mode: row?.enabled === 1 ? 'live' : 'paper',
      lastUpdated: row?.updated_at ?? null
    };
  }

  /**
   * Toggle the trading state.
   * @returns the new state (true = enabled/live, false = disabled/paper)
   */
  toggle(): boolean {
    if (this.isEnabled()) {
      this.disableTrading();
      return false;
    } else {
      this.enableTrading();
      return true;
    }
  }
}
