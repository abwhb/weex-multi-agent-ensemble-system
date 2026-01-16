#!/usr/bin/env npx ts-node
/**
 * @fileoverview Backtest runner CLI
 * @module backtest/run-backtest
 * 
 * Run backtests on historical data to validate strategy performance.
 * 
 * Usage:
 *   npx ts-node src/backtest/run-backtest.ts --symbol BTC --days 30
 *   npx ts-node src/backtest/run-backtest.ts --symbol ETH --days 90 --output ./results
 */

import { BinanceClient } from '../data/BinanceClient';
import { SimpleEnsemble } from './SimpleEnsemble';
import { Backtester, formatMetrics, BacktestConfig, BacktestResult } from './Backtester';
import { generateSummary, analyzeByMonth } from './PerformanceAnalyzer';
import { generateBacktestChart } from '../visualization/ChartGenerator';
import { AllowedPair, ALLOWED_PAIRS, OHLCV, MarketData, Timeframe } from '../types';
import * as fs from 'fs';
import * as path from 'path';

// ============================================================================
// CLI Argument Parsing
// ============================================================================

interface CLIArgs {
  symbol: AllowedPair;
  days: number;
  interval: Timeframe;
  output: string;
  capital: number;
  verbose: boolean;
}

function parseArgs(): CLIArgs {
  const args: CLIArgs = {
    symbol: 'BTC',
    days: 30,
    interval: '5m',
    output: './output',
    capital: 10000,
    verbose: false
  };
  
  const argv = process.argv.slice(2);
  
  for (let i = 0; i < argv.length; i++) {
    switch (argv[i]) {
      case '--symbol':
      case '-s':
        const sym = argv[++i]?.toUpperCase() as AllowedPair;
        if (ALLOWED_PAIRS.includes(sym as any)) {
          args.symbol = sym;
        } else {
          console.error(`Invalid symbol: ${sym}. Allowed: ${ALLOWED_PAIRS.join(', ')}`);
          process.exit(1);
        }
        break;
      case '--days':
      case '-d':
        args.days = parseInt(argv[++i], 10);
        break;
      case '--interval':
      case '-i':
        args.interval = argv[++i] as Timeframe;
        break;
      case '--output':
      case '-o':
        args.output = argv[++i];
        break;
      case '--capital':
      case '-c':
        args.capital = parseFloat(argv[++i]);
        break;
      case '--verbose':
      case '-v':
        args.verbose = true;
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
WEEX Multi-Agent Trading System - Backtester

Usage:
  npx ts-node src/backtest/run-backtest.ts [options]

Options:
  -s, --symbol <SYM>     Trading symbol (default: BTC)
                         Allowed: ${ALLOWED_PAIRS.join(', ')}
  -d, --days <N>         Number of days to backtest (default: 30)
  -i, --interval <INT>   Candle interval (default: 5m)
                         Allowed: 1m, 5m, 15m, 1h, 4h, 1d
  -o, --output <DIR>     Output directory for results (default: ./output)
  -c, --capital <USD>    Starting capital (default: 10000)
  -v, --verbose          Show detailed output
  -h, --help             Show this help

Examples:
  npx ts-node src/backtest/run-backtest.ts --symbol BTC --days 30
  npx ts-node src/backtest/run-backtest.ts -s ETH -d 90 -v
  npx ts-node src/backtest/run-backtest.ts --symbol SOL --days 60 --capital 50000
`);
}

// ============================================================================
// Main Backtest Runner
// ============================================================================

async function main(): Promise<void> {
  const args = parseArgs();
  
  console.log('\n╔═══════════════════════════════════════════════════════════╗');
  console.log('║   WEEX Multi-Agent Ensemble - Backtesting System          ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');
  
  console.log(`Configuration:`);
  console.log(`  Symbol:      ${args.symbol}/USDT`);
  console.log(`  Period:      ${args.days} days`);
  console.log(`  Interval:    ${args.interval}`);
  console.log(`  Capital:     $${args.capital.toLocaleString()}`);
  console.log('');
  
  // Step 1: Fetch historical data from Binance
  console.log('📡 Fetching historical data from Binance...');
  const client = new BinanceClient();
  await client.connect();
  
  // Calculate number of candles needed
  const intervalMinutes: Record<Timeframe, number> = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '1h': 60,
    '4h': 240,
    '1d': 1440
  };
  
  const candlesNeeded = Math.ceil((args.days * 24 * 60) / intervalMinutes[args.interval]);
  
  // Binance limit is 1000 per request, so we may need multiple fetches
  let allCandles: OHLCV[] = [];
  const batchSize = 1000;
  let remaining = candlesNeeded;
  
  console.log(`  Requesting ${candlesNeeded} candles...`);
  
  while (remaining > 0) {
    const limit = Math.min(remaining, batchSize);
    const candles = await client.getHistoricalData(args.symbol, args.interval, limit);
    
    if (candles.length === 0) break;
    
    // Prepend older candles
    if (allCandles.length === 0) {
      allCandles = candles;
    } else {
      // We need to fetch older data
      // This simple version just uses what we get
      break;
    }
    
    remaining -= candles.length;
    
    if (candles.length < limit) break;
  }
  
  console.log(`  Received ${allCandles.length} candles`);
  
  if (allCandles.length < 100) {
    console.error('❌ Not enough historical data. Need at least 100 candles.');
    process.exit(1);
  }
  
  // Sort by timestamp
  allCandles.sort((a, b) => a.timestamp - b.timestamp);
  
  const startDate = new Date(allCandles[0].timestamp);
  const endDate = new Date(allCandles[allCandles.length - 1].timestamp);
  console.log(`  Period: ${startDate.toISOString()} to ${endDate.toISOString()}`);
  console.log('');
  
  // Step 2: Initialize ensemble and backtester
  console.log('🤖 Initializing trading system...');
  const ensemble = new SimpleEnsemble();
  
  const backtestConfig: Partial<BacktestConfig> = {
    initialCapital: args.capital,
    slippage: 0.001, // 0.1%
    fee: 0.001, // 0.1%
    maxLeverage: 5,
    maxPositionSize: 0.10
  };
  
  // Create a backtester-compatible ensemble wrapper
  const ensembleWrapper = {
    async decide(marketData: MarketData, symbol: AllowedPair) {
      return ensemble.decide(marketData, symbol);
    }
  };
  
  const backtester = new Backtester(ensembleWrapper as any, backtestConfig);
  
  console.log('  Slippage: 0.1%');
  console.log('  Fees: 0.1%');
  console.log('  Max Leverage: 5x');
  console.log('  Max Position: 10%');
  console.log('');
  
  // Step 3: Run backtest
  console.log('📊 Running backtest...');
  const warmupPeriod = 100;
  
  const result = await backtester.run(allCandles, args.symbol, warmupPeriod);
  
  // Step 4: Display results
  console.log('\n');
  console.log(formatMetrics(result.metrics));
  console.log('');
  
  // Show detailed summary if verbose
  if (args.verbose) {
    console.log(generateSummary(result));
  }
  
  // Step 5: Save results
  if (!fs.existsSync(args.output)) {
    fs.mkdirSync(args.output, { recursive: true });
  }
  
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const resultFile = path.join(args.output, `backtest_${args.symbol}_${args.days}d_${timestamp}.json`);
  
  // Save detailed results
  const resultData = {
    config: {
      symbol: args.symbol,
      days: args.days,
      interval: args.interval,
      capital: args.capital,
      timestamp: new Date().toISOString()
    },
    metrics: result.metrics,
    trades: result.trades,
    monthly: analyzeByMonth(result)
  };
  
  fs.writeFileSync(resultFile, JSON.stringify(resultData, null, 2));
  console.log(`\n📁 Results saved to: ${resultFile}`);
  
  // Generate HTML chart
  const chartFile = path.join(args.output, `chart_${args.symbol}_${args.days}d_${timestamp}.html`);
  generateBacktestChart(result, allCandles, chartFile);
  
  // Step 6: Generate assessment
  console.log('\n');
  console.log('═══════════════════════════════════════════════════════════');
  console.log('                      ASSESSMENT                           ');
  console.log('═══════════════════════════════════════════════════════════');
  
  const m = result.metrics;
  
  if (m.sharpeRatio >= 1.5 && m.totalReturn > 0 && m.maxDrawdown < 15) {
    console.log('✅ EXCELLENT - Strategy shows strong risk-adjusted returns');
  } else if (m.sharpeRatio >= 1.0 && m.totalReturn > 0 && m.maxDrawdown < 20) {
    console.log('✅ GOOD - Strategy is profitable with acceptable risk');
  } else if (m.sharpeRatio >= 0.5 && m.totalReturn > -5) {
    console.log('⚠️  MARGINAL - Strategy needs optimization');
  } else {
    console.log('❌ POOR - Strategy is not performing well');
  }
  
  console.log('');
  console.log('Competition Readiness:');
  
  const checks = [
    { name: 'Sharpe > 1.0', pass: m.sharpeRatio >= 1.0 },
    { name: 'Win Rate > 50%', pass: m.winRate >= 50 },
    { name: 'Max Drawdown < 10%', pass: m.maxDrawdown < 10 },
    { name: 'Min 10 Trades', pass: m.totalTrades >= 10 },
    { name: 'Positive Return', pass: m.totalReturn > 0 }
  ];
  
  for (const check of checks) {
    const icon = check.pass ? '✅' : '❌';
    console.log(`  ${icon} ${check.name}`);
  }
  
  const passCount = checks.filter(c => c.pass).length;
  console.log(`\nScore: ${passCount}/${checks.length} checks passed`);
  
  if (passCount >= 4) {
    console.log('\n🏆 Ready for paper trading!');
  } else if (passCount >= 2) {
    console.log('\n⚙️  Needs more tuning before paper trading.');
  } else {
    console.log('\n🔧 Significant improvements needed.');
  }
  
  console.log('');
  
  await client.disconnect();
}

// Run
main().catch(err => {
  console.error('Backtest failed:', err);
  process.exit(1);
});

