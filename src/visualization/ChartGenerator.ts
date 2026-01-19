/**
 * @fileoverview HTML chart generator for backtest visualization
 * @module visualization/ChartGenerator
 * 
 * Generates standalone HTML files with interactive charts showing:
 * - Price chart with trade markers
 * - Equity curve
 * - Drawdown chart
 * - Trade log table
 */

import { BacktestResult, BacktestTrade, EquityPoint } from '../backtest/Backtester';
import { OHLCV } from '../types';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Generate an HTML file with backtest visualization.
 */
export function generateBacktestChart(
  result: BacktestResult,
  ohlcv: OHLCV[],
  outputPath: string
): void {
  const html = buildHTML(result, ohlcv);
  
  const dir = path.dirname(outputPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  
  fs.writeFileSync(outputPath, html);
  console.log(`📊 Chart generated: ${outputPath}`);
}

/**
 * Build the complete HTML document.
 */
function buildHTML(result: BacktestResult, ohlcv: OHLCV[]): string {
  const { metrics, trades, equityCurve } = result;
  
  // Prepare data for charts
  const priceData = ohlcv.map(c => ({
    x: c.timestamp,
    o: c.open,
    h: c.high,
    l: c.low,
    c: c.close
  }));
  
  const equityData = equityCurve.map(e => ({
    x: e.timestamp,
    y: e.equity
  }));
  
  const drawdownData = equityCurve.map(e => ({
    x: e.timestamp,
    y: -e.drawdown * 100
  }));
  
  const buyTrades = trades.filter(t => t.direction === 'long');
  const sellTrades = trades.filter(t => t.direction === 'short');
  
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Backtest Results - ${result.symbol} | WEEX Multi-Agent System</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.2.1/dist/chartjs-chart-financial.min.js"></script>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    
    body {
      font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
      background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0a0a0f 100%);
      color: #e0e0e0;
      min-height: 100vh;
      padding: 20px;
    }
    
    .container {
      max-width: 1400px;
      margin: 0 auto;
    }
    
    header {
      text-align: center;
      padding: 30px;
      margin-bottom: 30px;
      background: rgba(255, 255, 255, 0.03);
      border-radius: 16px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    h1 {
      font-size: 2rem;
      background: linear-gradient(135deg, #00ff88, #00d4ff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 10px;
    }
    
    .subtitle {
      color: #888;
      font-size: 0.9rem;
    }
    
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 15px;
      margin-bottom: 30px;
    }
    
    .metric-card {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 20px;
      text-align: center;
    }
    
    .metric-value {
      font-size: 1.8rem;
      font-weight: bold;
      margin-bottom: 5px;
    }
    
    .metric-value.positive { color: #00ff88; }
    .metric-value.negative { color: #ff4757; }
    .metric-value.neutral { color: #ffd93d; }
    
    .metric-label {
      font-size: 0.75rem;
      color: #888;
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    
    .chart-section {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
    }
    
    .chart-title {
      font-size: 1rem;
      color: #888;
      margin-bottom: 15px;
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    
    .chart-container {
      height: 300px;
      position: relative;
    }
    
    .chart-container.tall {
      height: 400px;
    }
    
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }
    
    th, td {
      padding: 12px;
      text-align: left;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    th {
      color: #888;
      text-transform: uppercase;
      font-size: 0.7rem;
      letter-spacing: 1px;
    }
    
    tr:hover {
      background: rgba(255, 255, 255, 0.02);
    }
    
    .tag {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 0.7rem;
      font-weight: bold;
    }
    
    .tag.long { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
    .tag.short { background: rgba(255, 71, 87, 0.2); color: #ff4757; }
    
    .pnl.positive { color: #00ff88; }
    .pnl.negative { color: #ff4757; }
    
    .footer {
      text-align: center;
      padding: 30px;
      color: #555;
      font-size: 0.8rem;
    }
    
    .status-badge {
      display: inline-block;
      padding: 8px 16px;
      border-radius: 8px;
      font-weight: bold;
      margin-top: 15px;
    }
    
    .status-badge.excellent { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
    .status-badge.good { background: rgba(0, 212, 255, 0.2); color: #00d4ff; }
    .status-badge.marginal { background: rgba(255, 217, 61, 0.2); color: #ffd93d; }
    .status-badge.poor { background: rgba(255, 71, 87, 0.2); color: #ff4757; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>WEEX Multi-Agent Ensemble</h1>
      <p class="subtitle">
        Backtest Results: ${result.symbol}/USDT | 
        ${result.startDate.toLocaleDateString()} - ${result.endDate.toLocaleDateString()}
      </p>
      ${getStatusBadge(metrics)}
    </header>
    
    <div class="metrics-grid">
      ${metricCard('Total Return', formatPercent(metrics.totalReturn), metrics.totalReturn)}
      ${metricCard('Sharpe Ratio', metrics.sharpeRatio.toFixed(2), metrics.sharpeRatio - 1)}
      ${metricCard('Max Drawdown', formatPercent(-metrics.maxDrawdown), -1)}
      ${metricCard('Win Rate', formatPercent(metrics.winRate), metrics.winRate - 50)}
      ${metricCard('Profit Factor', metrics.profitFactor.toFixed(2), metrics.profitFactor - 1)}
      ${metricCard('Total Trades', metrics.totalTrades.toString(), 0)}
      ${metricCard('Avg Win', formatPercent(metrics.avgWin), 1)}
      ${metricCard('Avg Loss', formatPercent(metrics.avgLoss), -1)}
    </div>
    
    <div class="chart-section">
      <h3 class="chart-title">Price Chart with Trades</h3>
      <div class="chart-container tall">
        <canvas id="priceChart"></canvas>
      </div>
    </div>
    
    <div class="chart-section">
      <h3 class="chart-title">Equity Curve</h3>
      <div class="chart-container">
        <canvas id="equityChart"></canvas>
      </div>
    </div>
    
    <div class="chart-section">
      <h3 class="chart-title">Drawdown</h3>
      <div class="chart-container">
        <canvas id="drawdownChart"></canvas>
      </div>
    </div>
    
    <div class="chart-section">
      <h3 class="chart-title">Trade Log</h3>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Direction</th>
            <th>Entry Time</th>
            <th>Entry Price</th>
            <th>Exit Price</th>
            <th>P&L</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          ${trades.slice(-50).map(t => tradeRow(t)).join('')}
        </tbody>
      </table>
      ${trades.length > 50 ? '<p style="text-align:center;padding:15px;color:#666;">Showing last 50 trades</p>' : ''}
    </div>
    
    <div class="footer">
      <p>Generated by WEEX Multi-Agent Ensemble Trading System</p>
      <p>Report generated: ${new Date().toISOString()}</p>
    </div>
  </div>
  
  <script>
    // Price data
    const priceData = ${JSON.stringify(priceData.slice(-500))};
    const buyTrades = ${JSON.stringify(buyTrades.map(t => ({ x: t.entryTime, y: t.entryPrice })))};
    const sellTrades = ${JSON.stringify(sellTrades.map(t => ({ x: t.entryTime, y: t.entryPrice })))};
    
    // Equity data
    const equityData = ${JSON.stringify(equityData.filter((_, i) => i % 10 === 0))};
    const drawdownData = ${JSON.stringify(drawdownData.filter((_, i) => i % 10 === 0))};
    
    // Chart defaults
    Chart.defaults.color = '#888';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
    
    // Price Chart
    new Chart(document.getElementById('priceChart'), {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'Price',
            data: priceData.map(d => ({ x: d.x, y: d.c })),
            borderColor: 'rgba(0, 212, 255, 0.8)',
            backgroundColor: 'rgba(0, 212, 255, 0.1)',
            fill: true,
            tension: 0.1,
            pointRadius: 0
          },
          {
            label: 'Long Entry',
            data: buyTrades,
            borderColor: '#00ff88',
            backgroundColor: '#00ff88',
            pointStyle: 'triangle',
            pointRadius: 8,
            pointHoverRadius: 10,
            showLine: false
          },
          {
            label: 'Short Entry',
            data: sellTrades,
            borderColor: '#ff4757',
            backgroundColor: '#ff4757',
            pointStyle: 'triangle',
            pointRadius: 8,
            pointHoverRadius: 10,
            rotation: 180,
            showLine: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: 'time',
            time: { unit: 'day' }
          },
          y: {
            position: 'right'
          }
        },
        plugins: {
          legend: { position: 'top' }
        }
      }
    });
    
    // Equity Chart
    new Chart(document.getElementById('equityChart'), {
      type: 'line',
      data: {
        datasets: [{
          label: 'Portfolio Value ($)',
          data: equityData,
          borderColor: '#00ff88',
          backgroundColor: 'rgba(0, 255, 136, 0.1)',
          fill: true,
          tension: 0.2,
          pointRadius: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { type: 'time', time: { unit: 'day' } },
          y: { position: 'right' }
        }
      }
    });
    
    // Drawdown Chart
    new Chart(document.getElementById('drawdownChart'), {
      type: 'line',
      data: {
        datasets: [{
          label: 'Drawdown (%)',
          data: drawdownData,
          borderColor: '#ff4757',
          backgroundColor: 'rgba(255, 71, 87, 0.2)',
          fill: true,
          tension: 0.2,
          pointRadius: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { type: 'time', time: { unit: 'day' } },
          y: { position: 'right', max: 0 }
        }
      }
    });
  </script>
</body>
</html>`;
}

function metricCard(label: string, value: string, sentiment: number): string {
  const colorClass = sentiment > 0 ? 'positive' : sentiment < 0 ? 'negative' : 'neutral';
  return `
    <div class="metric-card">
      <div class="metric-value ${colorClass}">${value}</div>
      <div class="metric-label">${label}</div>
    </div>
  `;
}

function formatPercent(value: number): string {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}%`;
}

function getStatusBadge(metrics: any): string {
  let status = 'poor';
  let text = 'NEEDS IMPROVEMENT';
  
  if (metrics.sharpeRatio >= 1.5 && metrics.totalReturn > 0 && metrics.maxDrawdown < 15) {
    status = 'excellent';
    text = 'EXCELLENT';
  } else if (metrics.sharpeRatio >= 1.0 && metrics.totalReturn > 0 && metrics.maxDrawdown < 20) {
    status = 'good';
    text = 'GOOD';
  } else if (metrics.sharpeRatio >= 0.5 && metrics.totalReturn > -5) {
    status = 'marginal';
    text = 'MARGINAL';
  }
  
  return `<span class="status-badge ${status}">${text}</span>`;
}

function tradeRow(trade: BacktestTrade): string {
  const pnlClass = (trade.pnl || 0) >= 0 ? 'positive' : 'negative';
  const pnlStr = trade.pnl !== undefined 
    ? `${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)}`
    : '-';
  
  return `
    <tr>
      <td>${trade.id}</td>
      <td><span class="tag ${trade.direction}">${trade.direction.toUpperCase()}</span></td>
      <td>${new Date(trade.entryTime).toLocaleString()}</td>
      <td>$${trade.entryPrice.toFixed(2)}</td>
      <td>${trade.exitPrice ? `$${trade.exitPrice.toFixed(2)}` : '-'}</td>
      <td class="pnl ${pnlClass}">${pnlStr}</td>
      <td>${(trade.confidence * 100).toFixed(0)}%</td>
    </tr>
  `;
}



