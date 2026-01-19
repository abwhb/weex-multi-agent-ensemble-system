/**
 * @fileoverview Performance analysis utilities
 * @module backtest/PerformanceAnalyzer
 * 
 * Additional analysis tools for backtest results.
 */

import { BacktestResult, BacktestTrade, BacktestMetrics } from './Backtester';

/**
 * Monthly performance breakdown.
 */
export interface MonthlyPerformance {
  month: string;
  return: number;
  trades: number;
  winRate: number;
}

/**
 * Analyze performance by month.
 */
export function analyzeByMonth(result: BacktestResult): MonthlyPerformance[] {
  const monthlyData = new Map<string, { pnl: number; trades: number; wins: number }>();
  
  for (const trade of result.trades) {
    const date = new Date(trade.entryTime);
    const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
    
    const current = monthlyData.get(monthKey) || { pnl: 0, trades: 0, wins: 0 };
    current.pnl += trade.pnl || 0;
    current.trades++;
    if ((trade.pnl || 0) > 0) current.wins++;
    monthlyData.set(monthKey, current);
  }
  
  return Array.from(monthlyData.entries())
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([month, data]) => ({
      month,
      return: (data.pnl / result.config.initialCapital) * 100,
      trades: data.trades,
      winRate: data.trades > 0 ? (data.wins / data.trades) * 100 : 0
    }));
}

/**
 * Calculate rolling Sharpe ratio.
 */
export function calculateRollingSharpe(
  result: BacktestResult,
  windowDays: number = 30
): Array<{ timestamp: number; sharpe: number }> {
  const points: Array<{ timestamp: number; sharpe: number }> = [];
  const windowMs = windowDays * 24 * 60 * 60 * 1000;
  
  for (let i = 0; i < result.equityCurve.length; i++) {
    const currentTime = result.equityCurve[i].timestamp;
    const windowStart = currentTime - windowMs;
    
    // Get equity points in window
    const windowPoints = result.equityCurve.filter(
      p => p.timestamp >= windowStart && p.timestamp <= currentTime
    );
    
    if (windowPoints.length < 2) continue;
    
    // Calculate returns in window
    const returns: number[] = [];
    for (let j = 1; j < windowPoints.length; j++) {
      const ret = (windowPoints[j].equity - windowPoints[j - 1].equity) / windowPoints[j - 1].equity;
      returns.push(ret);
    }
    
    if (returns.length === 0) continue;
    
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const stdReturn = Math.sqrt(
      returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
    ) || 0.01;
    
    const sharpe = (avgReturn / stdReturn) * Math.sqrt(252);
    
    points.push({ timestamp: currentTime, sharpe });
  }
  
  return points;
}

/**
 * Analyze trade duration distribution.
 */
export function analyzeTradeDurations(
  trades: BacktestTrade[]
): { bucket: string; count: number }[] {
  const buckets: { [key: string]: number } = {
    '< 1h': 0,
    '1-4h': 0,
    '4-12h': 0,
    '12-24h': 0,
    '1-3d': 0,
    '> 3d': 0
  };
  
  for (const trade of trades) {
    if (!trade.exitTime) continue;
    
    const durationHours = (trade.exitTime - trade.entryTime) / (1000 * 60 * 60);
    
    if (durationHours < 1) buckets['< 1h']++;
    else if (durationHours < 4) buckets['1-4h']++;
    else if (durationHours < 12) buckets['4-12h']++;
    else if (durationHours < 24) buckets['12-24h']++;
    else if (durationHours < 72) buckets['1-3d']++;
    else buckets['> 3d']++;
  }
  
  return Object.entries(buckets).map(([bucket, count]) => ({ bucket, count }));
}

/**
 * Find best and worst trades.
 */
export function findExtremeTrades(
  trades: BacktestTrade[],
  n: number = 5
): { best: BacktestTrade[]; worst: BacktestTrade[] } {
  const sorted = [...trades]
    .filter(t => t.pnlPercent !== undefined)
    .sort((a, b) => (b.pnlPercent || 0) - (a.pnlPercent || 0));
  
  return {
    best: sorted.slice(0, n),
    worst: sorted.slice(-n).reverse()
  };
}

/**
 * Calculate win/loss streaks.
 */
export function calculateStreaks(
  trades: BacktestTrade[]
): { maxWinStreak: number; maxLossStreak: number; currentStreak: number } {
  let maxWinStreak = 0;
  let maxLossStreak = 0;
  let currentStreak = 0;
  let lastWin = true;
  
  for (const trade of trades) {
    const isWin = (trade.pnl || 0) > 0;
    
    if (isWin === lastWin) {
      currentStreak++;
    } else {
      if (lastWin && currentStreak > maxWinStreak) {
        maxWinStreak = currentStreak;
      } else if (!lastWin && currentStreak > maxLossStreak) {
        maxLossStreak = currentStreak;
      }
      currentStreak = 1;
      lastWin = isWin;
    }
  }
  
  // Check final streak
  if (lastWin && currentStreak > maxWinStreak) {
    maxWinStreak = currentStreak;
  } else if (!lastWin && currentStreak > maxLossStreak) {
    maxLossStreak = currentStreak;
  }
  
  return { maxWinStreak, maxLossStreak, currentStreak };
}

/**
 * Generate performance summary text.
 */
export function generateSummary(result: BacktestResult): string {
  const m = result.metrics;
  const monthly = analyzeByMonth(result);
  const streaks = calculateStreaks(result.trades);
  const extreme = findExtremeTrades(result.trades, 3);
  
  const lines = [
    `\n=== BACKTEST SUMMARY: ${result.symbol} ===\n`,
    `Period: ${result.startDate.toLocaleDateString()} to ${result.endDate.toLocaleDateString()}`,
    `\n--- Performance ---`,
    `Total Return: ${m.totalReturn >= 0 ? '+' : ''}${m.totalReturn.toFixed(2)}%`,
    `Sharpe Ratio: ${m.sharpeRatio.toFixed(2)}`,
    `Max Drawdown: -${m.maxDrawdown.toFixed(2)}%`,
    `\n--- Trading ---`,
    `Total Trades: ${m.totalTrades}`,
    `Win Rate: ${m.winRate.toFixed(1)}%`,
    `Profit Factor: ${m.profitFactor.toFixed(2)}`,
    `\n--- Streaks ---`,
    `Max Win Streak: ${streaks.maxWinStreak}`,
    `Max Loss Streak: ${streaks.maxLossStreak}`,
    `\n--- Best Trades ---`
  ];
  
  extreme.best.forEach((t, i) => {
    lines.push(`  ${i + 1}. ${t.direction.toUpperCase()} +${t.pnlPercent?.toFixed(2)}%`);
  });
  
  lines.push(`\n--- Worst Trades ---`);
  extreme.worst.forEach((t, i) => {
    lines.push(`  ${i + 1}. ${t.direction.toUpperCase()} ${t.pnlPercent?.toFixed(2)}%`);
  });
  
  if (monthly.length > 0) {
    lines.push(`\n--- Monthly Performance ---`);
    monthly.forEach(m => {
      lines.push(`  ${m.month}: ${m.return >= 0 ? '+' : ''}${m.return.toFixed(2)}% (${m.trades} trades, ${m.winRate.toFixed(0)}% win)`);
    });
  }
  
  return lines.join('\n');
}



