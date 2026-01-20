/**
 * @fileoverview Console logger for trading system
 * @module logging/Logger
 *
 * Provides formatted console output for trading activity.
 */

import { config } from '../config';

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3
};

const COLORS = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m',
  gray: '\x1b[90m'
};

/**
 * Formatted console logger for trading system.
 */
export class Logger {
  private context: string;
  private minLevel: number;

  constructor(context: string) {
    this.context = context;
    this.minLevel = LOG_LEVELS[config.logging.level] || LOG_LEVELS.info;
  }

  private formatTime(): string {
    return new Date().toISOString().replace('T', ' ').substring(0, 19);
  }

  private shouldLog(level: LogLevel): boolean {
    return LOG_LEVELS[level] >= this.minLevel;
  }

  private format(level: LogLevel, message: string, data?: unknown): string {
    const time = this.formatTime();
    const levelColors: Record<LogLevel, string> = {
      debug: COLORS.gray,
      info: COLORS.blue,
      warn: COLORS.yellow,
      error: COLORS.red
    };

    const levelStr = level.toUpperCase().padEnd(5);
    const color = levelColors[level];

    let output = `${COLORS.gray}${time}${COLORS.reset} ${color}${levelStr}${COLORS.reset} ${COLORS.cyan}[${this.context}]${COLORS.reset} ${message}`;

    if (data !== undefined) {
      if (typeof data === 'object') {
        output += `\n${COLORS.dim}${JSON.stringify(data, null, 2)}${COLORS.reset}`;
      } else {
        output += ` ${COLORS.dim}${data}${COLORS.reset}`;
      }
    }

    return output;
  }

  debug(message: string, data?: unknown): void {
    if (this.shouldLog('debug')) {
      console.log(this.format('debug', message, data));
    }
  }

  info(message: string, data?: unknown): void {
    if (this.shouldLog('info')) {
      console.log(this.format('info', message, data));
    }
  }

  warn(message: string, data?: unknown): void {
    if (this.shouldLog('warn')) {
      console.warn(this.format('warn', message, data));
    }
  }

  error(message: string, data?: unknown): void {
    if (this.shouldLog('error')) {
      console.error(this.format('error', message, data));
    }
  }

  // Special formatted logs for trading

  trade(action: 'BUY' | 'SELL' | 'HOLD', symbol: string, details: {
    price?: number;
    size?: number;
    confidence?: number;
    reason?: string;
    pnl?: number;
    orderId?: string;
  }): void {
    if (!this.shouldLog('info')) return;

    const actionColors = {
      BUY: COLORS.green,
      SELL: COLORS.red,
      HOLD: COLORS.yellow
    };

    const color = actionColors[action];
    const time = this.formatTime();

    let line = `${COLORS.gray}${time}${COLORS.reset} ${color}${COLORS.bright}${action.padEnd(4)}${COLORS.reset}`;
    line += ` ${COLORS.white}${symbol}${COLORS.reset}`;

    if (details.price) {
      line += ` @ ${COLORS.cyan}$${details.price.toFixed(2)}${COLORS.reset}`;
    }
    if (details.size) {
      line += ` x ${COLORS.magenta}${details.size}${COLORS.reset}`;
    }
    if (details.confidence) {
      const confColor = details.confidence > 0.7 ? COLORS.green : details.confidence > 0.5 ? COLORS.yellow : COLORS.red;
      line += ` (${confColor}${(details.confidence * 100).toFixed(1)}%${COLORS.reset})`;
    }
    if (details.pnl !== undefined) {
      const pnlColor = details.pnl >= 0 ? COLORS.green : COLORS.red;
      const pnlSign = details.pnl >= 0 ? '+' : '';
      line += ` P&L: ${pnlColor}${pnlSign}$${details.pnl.toFixed(2)}${COLORS.reset}`;
    }

    console.log(line);

    if (details.reason) {
      console.log(`${COLORS.dim}  -> ${details.reason}${COLORS.reset}`);
    }
    if (details.orderId) {
      console.log(`${COLORS.dim}  -> Order: ${details.orderId}${COLORS.reset}`);
    }
  }

  signal(agent: string, symbol: string, direction: string, confidence: number, reasoning: string): void {
    if (!this.shouldLog('debug')) return;

    const dirColor = direction === 'long' ? COLORS.green : direction === 'short' ? COLORS.red : COLORS.yellow;
    const time = this.formatTime();

    console.log(
      `${COLORS.gray}${time}${COLORS.reset} ${COLORS.dim}SIGNAL${COLORS.reset} ` +
      `${COLORS.cyan}[${agent}]${COLORS.reset} ${symbol} ` +
      `${dirColor}${direction.toUpperCase()}${COLORS.reset} ` +
      `(${(confidence * 100).toFixed(1)}%) - ${COLORS.dim}${reasoning}${COLORS.reset}`
    );
  }

  balance(available: number, equity: number, unrealizedPnl: number): void {
    if (!this.shouldLog('info')) return;

    const pnlColor = unrealizedPnl >= 0 ? COLORS.green : COLORS.red;
    const pnlSign = unrealizedPnl >= 0 ? '+' : '';

    console.log(
      `${COLORS.bright}${COLORS.white}BALANCE${COLORS.reset} ` +
      `Available: ${COLORS.cyan}$${available.toFixed(2)}${COLORS.reset} | ` +
      `Equity: ${COLORS.cyan}$${equity.toFixed(2)}${COLORS.reset} | ` +
      `Unrealized P&L: ${pnlColor}${pnlSign}$${unrealizedPnl.toFixed(2)}${COLORS.reset}`
    );
  }

  position(symbol: string, side: string, size: number, entryPrice: number, currentPrice: number, pnl: number): void {
    if (!this.shouldLog('info')) return;

    const sideColor = side === 'long' ? COLORS.green : COLORS.red;
    const pnlColor = pnl >= 0 ? COLORS.green : COLORS.red;
    const pnlSign = pnl >= 0 ? '+' : '';

    console.log(
      `${COLORS.dim}POSITION${COLORS.reset} ${symbol} ` +
      `${sideColor}${side.toUpperCase()}${COLORS.reset} ` +
      `${size} @ $${entryPrice.toFixed(2)} ` +
      `(now $${currentPrice.toFixed(2)}) ` +
      `${pnlColor}${pnlSign}$${pnl.toFixed(2)}${COLORS.reset}`
    );
  }

  separator(title?: string): void {
    if (!this.shouldLog('info')) return;

    const line = '─'.repeat(60);
    if (title) {
      const padding = Math.floor((60 - title.length - 2) / 2);
      console.log(`${COLORS.dim}${'─'.repeat(padding)} ${title} ${'─'.repeat(60 - padding - title.length - 2)}${COLORS.reset}`);
    } else {
      console.log(`${COLORS.dim}${line}${COLORS.reset}`);
    }
  }

  banner(text: string): void {
    const border = '═'.repeat(text.length + 4);
    console.log(`\n${COLORS.cyan}╔${border}╗${COLORS.reset}`);
    console.log(`${COLORS.cyan}║${COLORS.reset}  ${COLORS.bright}${text}${COLORS.reset}  ${COLORS.cyan}║${COLORS.reset}`);
    console.log(`${COLORS.cyan}╚${border}╝${COLORS.reset}\n`);
  }

  paperMode(): void {
    console.log(`\n${COLORS.yellow}${COLORS.bright}*** PAPER TRADING MODE ***${COLORS.reset}`);
    console.log(`${COLORS.yellow}All trades are simulated - no real orders will be placed${COLORS.reset}\n`);
  }
}

// Create default loggers
export const systemLog = new Logger('System');
export const tradeLog = new Logger('Trade');
export const apiLog = new Logger('API');
export const agentLog = new Logger('Agent');
