#!/usr/bin/env npx ts-node
/**
 * @fileoverview Paper trading runner CLI
 * @module simulation/run-paper-trading
 * 
 * Run live paper trading simulation.
 * 
 * Usage:
 *   npx ts-node src/simulation/run-paper-trading.ts
 *   npx ts-node src/simulation/run-paper-trading.ts --symbols BTC,ETH,SOL
 */

import { PaperTrader, PaperTradingConfig } from './PaperTrader';
import { AllowedPair, ALLOWED_PAIRS, Timeframe } from '../types';

// ============================================================================
// CLI Argument Parsing
// ============================================================================

interface CLIArgs {
  symbols: AllowedPair[];
  interval: Timeframe;
  capital: number;
  updateInterval: number;
}

function parseArgs(): CLIArgs {
  const args: CLIArgs = {
    symbols: ['BTC', 'ETH', 'SOL'],
    interval: '5m',
    capital: 10000,
    updateInterval: 60
  };
  
  const argv = process.argv.slice(2);
  
  for (let i = 0; i < argv.length; i++) {
    switch (argv[i]) {
      case '--symbols':
      case '-s':
        const symbolsStr = argv[++i];
        args.symbols = symbolsStr.split(',').map(s => s.trim().toUpperCase() as AllowedPair);
        break;
      case '--interval':
      case '-i':
        args.interval = argv[++i] as Timeframe;
        break;
      case '--capital':
      case '-c':
        args.capital = parseFloat(argv[++i]);
        break;
      case '--update':
      case '-u':
        args.updateInterval = parseInt(argv[++i], 10);
        break;
      case '--help':
      case '-h':
        printHelp();
        process.exit(0);
    }
  }
  
  return args;
}

function printHelp(): void {
  console.log(`
WEEX Multi-Agent Trading System - Paper Trading

Usage:
  npx ts-node src/simulation/run-paper-trading.ts [options]

Options:
  -s, --symbols <LIST>   Comma-separated symbols (default: BTC,ETH,SOL)
                         Allowed: ${ALLOWED_PAIRS.join(', ')}
  -i, --interval <INT>   Candle interval (default: 5m)
  -c, --capital <USD>    Starting capital (default: 10000)
  -u, --update <SEC>     Update interval in seconds (default: 60)
  -h, --help             Show this help

Examples:
  npx ts-node src/simulation/run-paper-trading.ts
  npx ts-node src/simulation/run-paper-trading.ts --symbols BTC,ETH --update 30
  npx ts-node src/simulation/run-paper-trading.ts -s SOL,ADA,XRP -c 50000
`);
}

// ============================================================================
// Main
// ============================================================================

async function main(): Promise<void> {
  const args = parseArgs();
  
  const config: Partial<PaperTradingConfig> = {
    symbols: args.symbols,
    interval: args.interval,
    initialCapital: args.capital,
    updateIntervalMs: args.updateInterval * 1000
  };
  
  const trader = new PaperTrader(config);
  
  // Handle graceful shutdown
  let shuttingDown = false;
  
  const shutdown = async () => {
    if (shuttingDown) return;
    shuttingDown = true;
    await trader.stop();
    process.exit(0);
  };
  
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
  
  // Start trading
  await trader.start();
  
  // Keep running until interrupted
  await new Promise(() => {}); // Never resolves
}

main().catch(err => {
  console.error('Paper trading failed:', err);
  process.exit(1);
});



