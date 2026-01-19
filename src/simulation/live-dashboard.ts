#!/usr/bin/env npx ts-node
/**
 * @fileoverview Live trading dashboard with web UI
 * @module simulation/live-dashboard
 * 
 * Real-time web dashboard for paper trading visualization.
 */

import * as http from 'http';
import { BinanceClient } from '../data/BinanceClient';
import { SimpleEnsemble } from '../backtest/SimpleEnsemble';
import { AllowedPair, OHLCV, MarketData, Timeframe } from '../types';
import { TradeDecision } from '../meta-learner/Ensemble';
import { AILogger } from '../logging/AILogger';

// ============================================================================
// State
// ============================================================================

interface Position {
  symbol: string;
  direction: 'long' | 'short';
  entryPrice: number;
  entryTime: number;
  size: number;
  unrealizedPnl: number;
}

interface Trade {
  id: number;
  symbol: string;
  direction: 'long' | 'short';
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  time: number;
}

interface Signal {
  time: number;
  symbol: string;
  action: string;
  confidence: number;
  reasoning: string;
}

const state = {
  capital: 10000,
  equity: 10000,
  peakEquity: 10000,
  positions: new Map<string, Position>(),
  trades: [] as Trade[],
  signals: [] as Signal[],
  prices: new Map<string, number>(),
  priceHistory: new Map<string, { time: number; price: number }[]>(),
  startTime: Date.now(),
  tradeCounter: 0,
  lastUpdate: Date.now()
};

// ============================================================================
// Trading Logic
// ============================================================================

const client = new BinanceClient();
const ensemble = new SimpleEnsemble();
const aiLogger = new AILogger('./logs/live_dashboard_ai_decisions.json');
const candles = new Map<string, OHLCV[]>();
const SYMBOLS: AllowedPair[] = ['BTC', 'ETH', 'SOL'];

// Register model versions for competition
aiLogger.registerModelVersions({
  'MomentumAgent': '1.0.0',
  'MeanReversionAgent': '1.0.0',
  'VolatilityAgent': '1.0.0',
  'SimpleEnsemble': '1.0.0'
});

async function updateTrading() {
  for (const symbol of SYMBOLS) {
    try {
      // Fetch latest candles
      const newCandles = await client.getHistoricalData(symbol, '5m', 100);
      candles.set(symbol, newCandles);

      // Get current price
      const price = await client.getCurrentPrice(symbol);
      state.prices.set(symbol, price);

      // Update price history
      const history = state.priceHistory.get(symbol) || [];
      history.push({ time: Date.now(), price });
      if (history.length > 100) history.shift();
      state.priceHistory.set(symbol, history);

      // Update position P&L
      const position = state.positions.get(symbol);
      if (position) {
        const change = price - position.entryPrice;
        const dir = position.direction === 'long' ? 1 : -1;
        position.unrealizedPnl = (change / position.entryPrice) * position.size * dir;
      }

      // Get trading signal
      const marketData: MarketData = {
        symbol,
        ohlcv: newCandles,
        indicators: {},
        features: []
      };

      const decision = await ensemble.decide(marketData, symbol);

      // Always log what the system is seeing
      state.signals.unshift({
        time: Date.now(),
        symbol,
        action: decision ? decision.action : 'hold',
        confidence: decision ? decision.confidence : 0,
        reasoning: decision ? decision.reasoning : `${symbol}: No clear signal (market neutral/low confidence)`
      });
      if (state.signals.length > 50) state.signals.pop();

      // Process decision if we should trade
      if (decision) {
        // Log AI decision for competition
        aiLogger.logDecision(decision);
        processDecision(symbol, decision, price);
      }
    } catch (err) {
      console.error(`Error updating ${symbol}:`, err);
    }
  }

  // Update equity
  let unrealized = 0;
  for (const pos of state.positions.values()) {
    unrealized += pos.unrealizedPnl;
  }
  state.equity = state.capital + unrealized;
  if (state.equity > state.peakEquity) state.peakEquity = state.equity;

  state.lastUpdate = Date.now();
}

function processDecision(symbol: AllowedPair, decision: TradeDecision, price: number) {
  const existing = state.positions.get(symbol);
  const newDir = decision.action === 'buy' ? 'long' : 'short';

  // Close opposite position
  if (existing && existing.direction !== newDir) {
    closePosition(symbol, price);
  }

  // Open new position
  if (!state.positions.has(symbol) && decision.action !== 'hold') {
    const size = Math.min(decision.size * state.capital, 1000);
    state.positions.set(symbol, {
      symbol,
      direction: newDir,
      entryPrice: price,
      entryTime: Date.now(),
      size,
      unrealizedPnl: 0
    });
  }
}

function closePosition(symbol: string, price: number) {
  const pos = state.positions.get(symbol);
  if (!pos) return;

  const change = price - pos.entryPrice;
  const dir = pos.direction === 'long' ? 1 : -1;
  const pnl = (change / pos.entryPrice) * pos.size * dir - pos.size * 0.002;

  state.capital += pnl;
  state.trades.unshift({
    id: ++state.tradeCounter,
    symbol,
    direction: pos.direction,
    entryPrice: pos.entryPrice,
    exitPrice: price,
    pnl,
    time: Date.now()
  });
  if (state.trades.length > 50) state.trades.pop();

  state.positions.delete(symbol);
}

// ============================================================================
// Web Server
// ============================================================================

const HTML = `<!DOCTYPE html>
<html>
<head>
  <title>WEEX Trading - LIVE</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { 
      font-family: 'SF Mono', 'Fira Code', monospace;
      background: #0a0a0f;
      color: #e0e0e0;
      padding: 20px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    
    header {
      text-align: center;
      padding: 20px;
      margin-bottom: 20px;
      background: rgba(255,255,255,0.03);
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.05);
    }
    h1 {
      font-size: 1.5rem;
      background: linear-gradient(135deg, #00ff88, #00d4ff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .live-badge {
      display: inline-block;
      padding: 4px 12px;
      background: #ff4757;
      color: white;
      border-radius: 4px;
      font-size: 0.7rem;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 15px;
      margin-bottom: 20px;
    }
    .metric {
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px;
      padding: 20px;
      text-align: center;
    }
    .metric-value {
      font-size: 1.8rem;
      font-weight: bold;
    }
    .metric-value.positive { color: #00ff88; }
    .metric-value.negative { color: #ff4757; }
    .metric-label { color: #888; font-size: 0.75rem; text-transform: uppercase; }
    
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    .panel {
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px;
      padding: 20px;
    }
    .panel-title {
      font-size: 0.85rem;
      color: #888;
      margin-bottom: 15px;
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    
    .prices { display: flex; gap: 20px; flex-wrap: wrap; }
    .price-card {
      flex: 1;
      min-width: 150px;
      padding: 15px;
      background: rgba(0,0,0,0.3);
      border-radius: 8px;
    }
    .price-symbol { font-size: 1rem; color: #00d4ff; }
    .price-value { font-size: 1.5rem; font-weight: bold; }
    
    .position {
      padding: 10px;
      margin: 5px 0;
      background: rgba(0,0,0,0.3);
      border-radius: 6px;
      display: flex;
      justify-content: space-between;
    }
    .position.long { border-left: 3px solid #00ff88; }
    .position.short { border-left: 3px solid #ff4757; }
    
    .signal {
      padding: 8px;
      margin: 5px 0;
      background: rgba(0,0,0,0.2);
      border-radius: 4px;
      font-size: 0.8rem;
    }
    .signal-time { color: #666; }
    .signal-buy { color: #00ff88; }
    .signal-sell { color: #ff4757; }
    
    .trade {
      padding: 8px;
      margin: 5px 0;
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
    }
    .trade-pnl.positive { color: #00ff88; }
    .trade-pnl.negative { color: #ff4757; }
    
    .update-time {
      text-align: center;
      color: #555;
      margin-top: 20px;
      font-size: 0.8rem;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>WEEX Multi-Agent Ensemble</h1>
      <span class="live-badge">LIVE SIMULATION</span>
    </header>
    
    <div class="metrics">
      <div class="metric">
        <div class="metric-value" id="equity">$10,000.00</div>
        <div class="metric-label">Portfolio Value</div>
      </div>
      <div class="metric">
        <div class="metric-value" id="return">+0.00%</div>
        <div class="metric-label">Total Return</div>
      </div>
      <div class="metric">
        <div class="metric-value" id="drawdown">0.00%</div>
        <div class="metric-label">Drawdown</div>
      </div>
      <div class="metric">
        <div class="metric-value" id="trades">0</div>
        <div class="metric-label">Trades</div>
      </div>
    </div>
    
    <div class="panel" style="margin-bottom: 20px;">
      <div class="panel-title">Live Prices</div>
      <div class="prices" id="prices"></div>
    </div>
    
    <div class="grid">
      <div class="panel">
        <div class="panel-title">Open Positions</div>
        <div id="positions"><em style="color:#666">No open positions</em></div>
      </div>
      <div class="panel">
        <div class="panel-title">Recent Signals</div>
        <div id="signals"><em style="color:#666">Waiting for signals...</em></div>
      </div>
    </div>
    
    <div class="panel" style="margin-top: 20px;">
      <div class="panel-title">Trade History</div>
      <div id="trades-list"><em style="color:#666">No trades yet</em></div>
    </div>
    
    <div class="panel" style="margin-top: 20px; background: rgba(0,255,136,0.03);">
      <div class="panel-title" style="color: #00ff88;">🤖 AI Decision Logs (Competition)</div>
      <div id="ai-logs" style="max-height: 300px; overflow-y: auto; font-size: 0.75rem;"><em style="color:#666">AI logs will appear here as decisions are made...</em></div>
    </div>
    
    <div class="update-time">Last update: <span id="update-time">-</span> | AI Logs saved to: logs/live_dashboard_ai_decisions.json</div>
  </div>
  
  <script>
    async function refresh() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();
        
        // Update metrics
        document.getElementById('equity').textContent = '$' + data.equity.toFixed(2);
        document.getElementById('equity').className = 'metric-value ' + (data.return >= 0 ? 'positive' : 'negative');
        
        document.getElementById('return').textContent = (data.return >= 0 ? '+' : '') + data.return.toFixed(2) + '%';
        document.getElementById('return').className = 'metric-value ' + (data.return >= 0 ? 'positive' : 'negative');
        
        document.getElementById('drawdown').textContent = data.drawdown.toFixed(2) + '%';
        document.getElementById('drawdown').className = 'metric-value negative';
        
        document.getElementById('trades').textContent = data.tradeCount;
        
        // Prices
        let pricesHtml = '';
        for (const [sym, price] of Object.entries(data.prices)) {
          pricesHtml += '<div class="price-card"><div class="price-symbol">' + sym + '/USDT</div><div class="price-value">$' + Number(price).toLocaleString() + '</div></div>';
        }
        document.getElementById('prices').innerHTML = pricesHtml || '<em style="color:#666">Loading...</em>';
        
        // Positions
        if (data.positions.length > 0) {
          let posHtml = '';
          for (const pos of data.positions) {
            const pnlClass = pos.unrealizedPnl >= 0 ? 'positive' : 'negative';
            const pnlStr = (pos.unrealizedPnl >= 0 ? '+' : '') + '$' + pos.unrealizedPnl.toFixed(2);
            posHtml += '<div class="position ' + pos.direction + '"><span>' + pos.symbol + ' ' + pos.direction.toUpperCase() + ' @ $' + pos.entryPrice.toFixed(2) + '</span><span class="' + pnlClass + '">' + pnlStr + '</span></div>';
          }
          document.getElementById('positions').innerHTML = posHtml;
        } else {
          document.getElementById('positions').innerHTML = '<em style="color:#666">No open positions</em>';
        }
        
        // Signals
        if (data.signals.length > 0) {
          let sigHtml = '';
          for (const sig of data.signals.slice(0, 10)) {
            const time = new Date(sig.time).toLocaleTimeString();
            const actionClass = sig.action === 'buy' ? 'signal-buy' : sig.action === 'sell' ? 'signal-sell' : '';
            sigHtml += '<div class="signal"><span class="signal-time">[' + time + ']</span> <span class="' + actionClass + '">' + sig.symbol + ': ' + sig.action.toUpperCase() + '</span> (' + (sig.confidence * 100).toFixed(0) + '%)</div>';
          }
          document.getElementById('signals').innerHTML = sigHtml;
        }
        
        // Trades
        if (data.trades.length > 0) {
          let tradeHtml = '';
          for (const t of data.trades.slice(0, 10)) {
            const pnlClass = t.pnl >= 0 ? 'positive' : 'negative';
            const pnlStr = (t.pnl >= 0 ? '+' : '') + '$' + t.pnl.toFixed(2);
            tradeHtml += '<div class="trade"><span>' + t.symbol + ' ' + t.direction.toUpperCase() + '</span><span class="trade-pnl ' + pnlClass + '">' + pnlStr + '</span></div>';
          }
          document.getElementById('trades-list').innerHTML = tradeHtml;
        }
        
        // AI Decision Logs - show signals with full reasoning
        if (data.signals.length > 0) {
          let aiLogsHtml = '';
          for (const sig of data.signals.slice(0, 20)) {
            const time = new Date(sig.time).toLocaleTimeString();
            const actionColor = sig.action === 'buy' ? '#00ff88' : sig.action === 'sell' ? '#ff4757' : '#888';
            aiLogsHtml += '<div style="padding: 8px; margin: 4px 0; background: rgba(0,0,0,0.3); border-radius: 4px; border-left: 3px solid ' + actionColor + ';">';
            aiLogsHtml += '<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">';
            aiLogsHtml += '<span style="color: #00d4ff;">' + sig.symbol + '</span>';
            aiLogsHtml += '<span style="color: ' + actionColor + '; font-weight: bold;">' + sig.action.toUpperCase() + ' @ ' + (sig.confidence * 100).toFixed(0) + '%</span>';
            aiLogsHtml += '<span style="color: #666;">' + time + '</span></div>';
            aiLogsHtml += '<div style="color: #aaa; font-size: 0.7rem;">' + sig.reasoning + '</div></div>';
          }
          document.getElementById('ai-logs').innerHTML = aiLogsHtml;
        }
        
        document.getElementById('update-time').textContent = new Date().toLocaleTimeString();
      } catch (err) {
        console.error('Refresh error:', err);
      }
    }
    
    setInterval(refresh, 2000);
    refresh();
  </script>
</body>
</html>`;

function getState() {
  const totalReturn = ((state.equity - 10000) / 10000) * 100;
  const drawdown = state.peakEquity > 0 ? ((state.peakEquity - state.equity) / state.peakEquity) * 100 : 0;

  return {
    equity: state.equity,
    return: totalReturn,
    drawdown,
    tradeCount: state.trades.length,
    prices: Object.fromEntries(state.prices),
    positions: Array.from(state.positions.values()),
    signals: state.signals,
    trades: state.trades,
    runtime: (Date.now() - state.startTime) / 1000 / 60
  };
}

const server = http.createServer((req, res) => {
  if (req.url === '/api/state') {
    res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
    res.end(JSON.stringify(getState()));
  } else {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(HTML);
  }
});

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log('Starting live dashboard...');

  await client.connect();

  // Initial data fetch
  for (const symbol of SYMBOLS) {
    const data = await client.getHistoricalData(symbol, '5m', 100);
    candles.set(symbol, data);
    const price = await client.getCurrentPrice(symbol);
    state.prices.set(symbol, price);
  }

  // Start update loop
  setInterval(updateTrading, 10000); // Every 10 seconds
  updateTrading();

  // Start server
  const PORT = 3457;
  server.listen(PORT, () => {
    console.log('');
    console.log('╔═══════════════════════════════════════════════════════════╗');
    console.log('║   LIVE DASHBOARD RUNNING                                  ║');
    console.log('╠═══════════════════════════════════════════════════════════╣');
    console.log('║                                                           ║');
    console.log('║   Open in browser: http://localhost:' + PORT + '                ║');
    console.log('║                                                           ║');
    console.log('╚═══════════════════════════════════════════════════════════╝');
    console.log('');
  });
}

main().catch(console.error);

