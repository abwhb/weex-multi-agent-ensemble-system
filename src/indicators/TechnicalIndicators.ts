/**
 * @fileoverview Technical indicator calculations
 * @module indicators/TechnicalIndicators
 * 
 * Pure functions for calculating technical indicators.
 * These are the actual working implementations used by agents.
 */

import { OHLCV } from '../types';

/**
 * Calculate Simple Moving Average.
 */
export function SMA(data: number[], period: number): number {
  if (data.length < period) return data[data.length - 1] || 0;
  const slice = data.slice(-period);
  return slice.reduce((a, b) => a + b, 0) / period;
}

/**
 * Calculate Exponential Moving Average.
 */
export function EMA(data: number[], period: number): number {
  if (data.length === 0) return 0;
  if (data.length === 1) return data[0];
  
  const multiplier = 2 / (period + 1);
  let ema = data[0];
  
  for (let i = 1; i < data.length; i++) {
    ema = (data[i] - ema) * multiplier + ema;
  }
  
  return ema;
}

/**
 * Calculate array of EMA values (for MACD, etc.).
 */
export function EMAArray(data: number[], period: number): number[] {
  if (data.length === 0) return [];
  
  const result: number[] = [];
  const multiplier = 2 / (period + 1);
  let ema = data[0];
  result.push(ema);
  
  for (let i = 1; i < data.length; i++) {
    ema = (data[i] - ema) * multiplier + ema;
    result.push(ema);
  }
  
  return result;
}

/**
 * Calculate Relative Strength Index (RSI).
 * 
 * RSI = 100 - (100 / (1 + RS))
 * RS = Average Gain / Average Loss
 */
export function RSI(closes: number[], period: number = 14): number {
  if (closes.length < period + 1) return 50; // Neutral
  
  let gains = 0;
  let losses = 0;
  
  // Calculate initial average gain/loss
  for (let i = closes.length - period; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    if (change > 0) {
      gains += change;
    } else {
      losses -= change;
    }
  }
  
  const avgGain = gains / period;
  const avgLoss = losses / period;
  
  if (avgLoss === 0) return 100;
  
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

/**
 * Calculate MACD (Moving Average Convergence Divergence).
 * 
 * Returns { macd, signal, histogram }
 */
export function MACD(
  closes: number[],
  fastPeriod: number = 12,
  slowPeriod: number = 26,
  signalPeriod: number = 9
): { macd: number; signal: number; histogram: number } {
  if (closes.length < slowPeriod) {
    return { macd: 0, signal: 0, histogram: 0 };
  }
  
  const ema12 = EMA(closes, fastPeriod);
  const ema26 = EMA(closes, slowPeriod);
  const macdLine = ema12 - ema26;
  
  // Calculate MACD line history for signal line
  const macdHistory: number[] = [];
  const ema12Arr = EMAArray(closes, fastPeriod);
  const ema26Arr = EMAArray(closes, slowPeriod);
  
  for (let i = slowPeriod - 1; i < closes.length; i++) {
    macdHistory.push(ema12Arr[i] - ema26Arr[i]);
  }
  
  const signalLine = macdHistory.length >= signalPeriod 
    ? EMA(macdHistory, signalPeriod) 
    : macdLine;
  
  return {
    macd: macdLine,
    signal: signalLine,
    histogram: macdLine - signalLine
  };
}

/**
 * Calculate Bollinger Bands.
 */
export function BollingerBands(
  closes: number[],
  period: number = 20,
  stdDevMultiplier: number = 2
): {
  upper: number;
  middle: number;
  lower: number;
  width: number;
  percentB: number;
} {
  if (closes.length < period) {
    const lastPrice = closes[closes.length - 1] || 0;
    return {
      upper: lastPrice,
      middle: lastPrice,
      lower: lastPrice,
      width: 0,
      percentB: 0.5
    };
  }
  
  const slice = closes.slice(-period);
  const middle = slice.reduce((a, b) => a + b, 0) / period;
  
  const variance = slice.reduce((sum, val) => sum + Math.pow(val - middle, 2), 0) / period;
  const stdDev = Math.sqrt(variance);
  
  const upper = middle + stdDevMultiplier * stdDev;
  const lower = middle - stdDevMultiplier * stdDev;
  const currentPrice = closes[closes.length - 1];
  
  const width = (upper - lower) / middle;
  const percentB = (upper - lower) > 0 ? (currentPrice - lower) / (upper - lower) : 0.5;
  
  return { upper, middle, lower, width, percentB };
}

/**
 * Calculate Average True Range (ATR).
 */
export function ATR(ohlcv: OHLCV[], period: number = 14): number {
  if (ohlcv.length < 2) return 0;
  
  const trueRanges: number[] = [];
  
  for (let i = 1; i < ohlcv.length; i++) {
    const high = ohlcv[i].high;
    const low = ohlcv[i].low;
    const prevClose = ohlcv[i - 1].close;
    
    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );
    trueRanges.push(tr);
  }
  
  if (trueRanges.length < period) {
    return trueRanges.reduce((a, b) => a + b, 0) / trueRanges.length;
  }
  
  return SMA(trueRanges, period);
}

/**
 * Calculate Average Directional Index (ADX) - simplified.
 */
export function ADX(ohlcv: OHLCV[], period: number = 14): number {
  if (ohlcv.length < period + 1) return 25; // Neutral
  
  let plusDM = 0;
  let minusDM = 0;
  let tr = 0;
  
  for (let i = 1; i <= period; i++) {
    const idx = ohlcv.length - period - 1 + i;
    const highDiff = ohlcv[idx].high - ohlcv[idx - 1].high;
    const lowDiff = ohlcv[idx - 1].low - ohlcv[idx].low;
    
    if (highDiff > lowDiff && highDiff > 0) {
      plusDM += highDiff;
    }
    if (lowDiff > highDiff && lowDiff > 0) {
      minusDM += lowDiff;
    }
    
    const trueRange = Math.max(
      ohlcv[idx].high - ohlcv[idx].low,
      Math.abs(ohlcv[idx].high - ohlcv[idx - 1].close),
      Math.abs(ohlcv[idx].low - ohlcv[idx - 1].close)
    );
    tr += trueRange;
  }
  
  if (tr === 0) return 25;
  
  const plusDI = (plusDM / tr) * 100;
  const minusDI = (minusDM / tr) * 100;
  const diSum = plusDI + minusDI;
  
  if (diSum === 0) return 25;
  
  const dx = (Math.abs(plusDI - minusDI) / diSum) * 100;
  return dx;
}

/**
 * Calculate volume moving average ratio.
 */
export function VolumeRatio(volumes: number[], period: number = 20): number {
  if (volumes.length < period) return 1;
  
  const currentVolume = volumes[volumes.length - 1];
  const avgVolume = SMA(volumes.slice(0, -1), period);
  
  return avgVolume > 0 ? currentVolume / avgVolume : 1;
}

/**
 * Calculate price momentum (rate of change).
 */
export function ROC(closes: number[], period: number = 10): number {
  if (closes.length <= period) return 0;
  
  const current = closes[closes.length - 1];
  const past = closes[closes.length - 1 - period];
  
  return past !== 0 ? ((current - past) / past) * 100 : 0;
}

/**
 * Normalize a value to -1 to 1 range using tanh.
 */
export function normalize(value: number, scale: number = 1): number {
  return Math.tanh(value * scale);
}

/**
 * Calculate percentile of a value in a dataset.
 */
export function percentile(value: number, data: number[]): number {
  const sorted = [...data].sort((a, b) => a - b);
  const index = sorted.findIndex(v => v >= value);
  
  if (index === -1) return 100;
  return (index / sorted.length) * 100;
}

/**
 * Detect if Bollinger Bands are squeezing (low volatility).
 */
export function BBSqueeze(
  closes: number[],
  period: number = 20,
  percentileThreshold: number = 20
): boolean {
  if (closes.length < period * 2) return false;
  
  // Calculate historical BB widths
  const widths: number[] = [];
  for (let i = period; i < closes.length; i++) {
    const slice = closes.slice(i - period, i);
    const bb = BollingerBands(slice, period, 2);
    widths.push(bb.width);
  }
  
  const currentWidth = widths[widths.length - 1];
  const pct = percentile(currentWidth, widths);
  
  return pct <= percentileThreshold;
}

/**
 * Calculate On-Balance Volume trend.
 */
export function OBV(ohlcv: OHLCV[]): number[] {
  const obv: number[] = [0];
  
  for (let i = 1; i < ohlcv.length; i++) {
    let change = 0;
    if (ohlcv[i].close > ohlcv[i - 1].close) {
      change = ohlcv[i].volume;
    } else if (ohlcv[i].close < ohlcv[i - 1].close) {
      change = -ohlcv[i].volume;
    }
    obv.push(obv[obv.length - 1] + change);
  }
  
  return obv;
}

/**
 * Get trend direction from OBV.
 */
export function OBVTrend(ohlcv: OHLCV[], period: number = 10): number {
  const obv = OBV(ohlcv);
  if (obv.length < period) return 0;
  
  const recentOBV = obv.slice(-period);
  const firstOBV = recentOBV[0];
  const lastOBV = recentOBV[recentOBV.length - 1];
  
  // Normalize to -1 to 1
  const maxOBV = Math.max(...obv.map(Math.abs));
  if (maxOBV === 0) return 0;
  
  return (lastOBV - firstOBV) / maxOBV;
}



