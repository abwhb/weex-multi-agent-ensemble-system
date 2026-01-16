/**
 * @fileoverview Backtest module exports
 * @module backtest
 */

export { 
  Backtester, 
  BacktestConfig, 
  BacktestTrade, 
  BacktestMetrics, 
  BacktestResult,
  EquityPoint,
  formatMetrics 
} from './Backtester';

export {
  analyzeByMonth,
  calculateRollingSharpe,
  analyzeTradeDurations,
  findExtremeTrades,
  calculateStreaks,
  generateSummary,
  MonthlyPerformance
} from './PerformanceAnalyzer';

export { SimpleEnsemble } from './SimpleEnsemble';

