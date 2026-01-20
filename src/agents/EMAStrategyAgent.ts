/**
 * @fileoverview EMA Crossover Strategy Agent with AI enhancement
 * @module agents/EMAStrategyAgent
 *
 * ## Overview
 *
 * Implements the user's trading strategy combining:
 * - 20/50 EMA crossover for trend direction
 * - Volume analysis for fakeout detection
 * - RSI for overbought/oversold conditions
 * - Support/Resistance levels (Smart Money Concepts)
 * - Gemini AI for enhanced decision making
 *
 * ## Strategy Rules
 *
 * **Long Signal:**
 * - 20 EMA crosses above 50 EMA (bullish crossover)
 * - Volume confirms the move (not a fakeout)
 * - RSI not overbought (< 70)
 * - Price has room to resistance
 *
 * **Short Signal:**
 * - 50 EMA goes above 20 EMA (bearish crossover)
 * - Volume confirms the move
 * - RSI not oversold (> 30)
 * - Price has room to support
 */

import { GoogleGenerativeAI, GenerativeModel } from '@google/generative-ai';
import { BaseAgent, AgentSignal, AgentConfig } from './BaseAgent';
import { MarketData, MarketRegime, OHLCV } from '../types';

// ============================================================================
// Types
// ============================================================================

/**
 * Extended configuration for EMAStrategyAgent.
 */
export interface EMAStrategyConfig extends AgentConfig {
  /**
   * Google Gemini API key for AI analysis.
   */
  apiKey: string;

  /**
   * Gemini model to use.
   * @default 'gemini-1.5-flash'
   */
  model?: string;

  /**
   * Temperature for response generation.
   * @default 0.2
   */
  temperature?: number;

  /**
   * Short EMA period.
   * @default 20
   */
  shortEMA?: number;

  /**
   * Long EMA period.
   * @default 50
   */
  longEMA?: number;

  /**
   * RSI period.
   * @default 14
   */
  rsiPeriod?: number;

  /**
   * RSI overbought threshold.
   * @default 70
   */
  rsiOverbought?: number;

  /**
   * RSI oversold threshold.
   * @default 30
   */
  rsiOversold?: number;

  /**
   * Volume ratio threshold (current vs average).
   * @default 1.5
   */
  volumeThreshold?: number;
}

/**
 * Technical analysis data structure.
 */
interface TechnicalData {
  ema20: number;
  ema50: number;
  emaCrossover: 'bullish' | 'bearish' | 'none';
  emaCrossoverStrength: number;
  rsi: number;
  rsiSignal: 'overbought' | 'oversold' | 'neutral';
  volumeRatio: number;
  volumeConfirms: boolean;
  support: number;
  resistance: number;
  currentPrice: number;
  distanceToSupport: number;
  distanceToResistance: number;
  trend: 'bullish' | 'bearish' | 'neutral';
  recentHigh: number;
  recentLow: number;
}

/**
 * AI analysis response structure.
 */
interface AIAnalysisResponse {
  direction: 'long' | 'short' | 'neutral';
  confidence: number;
  reasoning: string;
  keyFactors: string[];
  riskLevel: 'low' | 'medium' | 'high';
  isFakeout: boolean;
}

// ============================================================================
// EMAStrategyAgent Implementation
// ============================================================================

/**
 * EMA Crossover Strategy Agent with AI enhancement.
 *
 * This agent combines traditional technical analysis (EMA crossovers, RSI, volume)
 * with AI-powered decision making to generate high-quality trading signals.
 *
 * ## Key Features
 *
 * 1. **EMA Crossover Detection**: Identifies trend changes using 20/50 EMA
 * 2. **Volume Confirmation**: Filters out fakeouts using volume analysis
 * 3. **RSI Filtering**: Avoids trades in overbought/oversold conditions
 * 4. **Support/Resistance**: Identifies key price levels
 * 5. **AI Enhancement**: Uses Gemini to synthesize all signals
 *
 * @extends BaseAgent
 */
export class EMAStrategyAgent extends BaseAgent {
  private strategyConfig: EMAStrategyConfig;
  private genAI: GoogleGenerativeAI | null = null;
  private model: GenerativeModel | null = null;

  // EMA state for crossover detection
  private previousEMA20: number = 0;
  private previousEMA50: number = 0;

  constructor(config: EMAStrategyConfig, initialWeight: number = 0.4) {
    super('EMAStrategyAgent', config, initialWeight);

    this.strategyConfig = {
      model: 'gemini-2.5-flash',
      temperature: 0.2,
      shortEMA: 20,
      longEMA: 50,
      rsiPeriod: 14,
      rsiOverbought: 70,
      rsiOversold: 30,
      volumeThreshold: 1.5,
      ...config
    };
  }

  // ==========================================================================
  // Core Analysis
  // ==========================================================================

  async analyze(marketData: MarketData): Promise<AgentSignal> {
    this.ensureInitialized();

    try {
      // Step 1: Calculate all technical indicators
      const technicals = this.calculateTechnicals(marketData);

      // Step 2: Get rule-based signal
      const ruleBasedSignal = this.getRuleBasedSignal(technicals);

      // Step 3: Use AI to enhance decision (if API key available)
      let finalSignal: AgentSignal;

      if (this.model) {
        const aiAnalysis = await this.getAIAnalysis(technicals, marketData.symbol);
        finalSignal = this.combineSignals(ruleBasedSignal, aiAnalysis, technicals);
      } else {
        finalSignal = ruleBasedSignal;
      }

      return finalSignal;
    } catch (error) {
      console.error(`EMAStrategyAgent analysis error:`, error);
      return this.createNeutralSignal(
        `EMAStrategyAgent error: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  // ==========================================================================
  // Technical Analysis
  // ==========================================================================

  private calculateTechnicals(marketData: MarketData): TechnicalData {
    const candles = marketData.ohlcv;
    const closes = candles.map(c => c.close);
    const volumes = candles.map(c => c.volume);
    const highs = candles.map(c => c.high);
    const lows = candles.map(c => c.low);

    // Calculate EMAs
    const ema20 = this.calculateEMA(closes, this.strategyConfig.shortEMA!);
    const ema50 = this.calculateEMA(closes, this.strategyConfig.longEMA!);

    // Detect crossover
    const emaCrossover = this.detectCrossover(ema20, ema50);
    const emaCrossoverStrength = Math.abs(ema20 - ema50) / ema50 * 100;

    // Store for next iteration
    this.previousEMA20 = ema20;
    this.previousEMA50 = ema50;

    // Calculate RSI
    const rsi = this.calculateRSI(candles, this.strategyConfig.rsiPeriod!);
    const rsiSignal = rsi > this.strategyConfig.rsiOverbought! ? 'overbought' :
                      rsi < this.strategyConfig.rsiOversold! ? 'oversold' : 'neutral';

    // Volume analysis
    const recentVolume = volumes.slice(-5).reduce((a, b) => a + b, 0) / 5;
    const avgVolume = volumes.slice(-20).reduce((a, b) => a + b, 0) / 20;
    const volumeRatio = avgVolume > 0 ? recentVolume / avgVolume : 1;
    const volumeConfirms = volumeRatio >= this.strategyConfig.volumeThreshold!;

    // Support and Resistance (using recent swing highs/lows)
    const { support, resistance, recentHigh, recentLow } = this.calculateSupportResistance(candles);

    const currentPrice = closes[closes.length - 1];
    const distanceToSupport = ((currentPrice - support) / support) * 100;
    const distanceToResistance = ((resistance - currentPrice) / currentPrice) * 100;

    // Determine trend
    const trend = ema20 > ema50 ? 'bullish' : ema20 < ema50 ? 'bearish' : 'neutral';

    return {
      ema20,
      ema50,
      emaCrossover,
      emaCrossoverStrength,
      rsi,
      rsiSignal,
      volumeRatio,
      volumeConfirms,
      support,
      resistance,
      currentPrice,
      distanceToSupport,
      distanceToResistance,
      trend,
      recentHigh,
      recentLow
    };
  }

  private calculateEMA(prices: number[], period: number): number {
    if (prices.length < period) return prices[prices.length - 1] || 0;

    const multiplier = 2 / (period + 1);
    let ema = prices.slice(0, period).reduce((a, b) => a + b, 0) / period;

    for (let i = period; i < prices.length; i++) {
      ema = (prices[i] - ema) * multiplier + ema;
    }

    return ema;
  }

  private detectCrossover(ema20: number, ema50: number): 'bullish' | 'bearish' | 'none' {
    // Check if EMA20 just crossed above EMA50 (bullish)
    if (this.previousEMA20 > 0 && this.previousEMA50 > 0) {
      const wasBelowOrEqual = this.previousEMA20 <= this.previousEMA50;
      const isAboveNow = ema20 > ema50;

      if (wasBelowOrEqual && isAboveNow) {
        return 'bullish';
      }

      const wasAboveOrEqual = this.previousEMA20 >= this.previousEMA50;
      const isBelowNow = ema20 < ema50;

      if (wasAboveOrEqual && isBelowNow) {
        return 'bearish';
      }
    }

    return 'none';
  }

  private calculateRSI(candles: OHLCV[], period: number): number {
    if (candles.length < period + 1) return 50;

    let gains = 0;
    let losses = 0;

    for (let i = candles.length - period; i < candles.length; i++) {
      const change = candles[i].close - candles[i - 1].close;
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

  private calculateSupportResistance(candles: OHLCV[]): {
    support: number;
    resistance: number;
    recentHigh: number;
    recentLow: number;
  } {
    const recent = candles.slice(-20);
    const highs = recent.map(c => c.high);
    const lows = recent.map(c => c.low);

    // Find swing highs and lows (Smart Money Concepts style)
    const swingHighs: number[] = [];
    const swingLows: number[] = [];

    for (let i = 2; i < recent.length - 2; i++) {
      // Swing high: higher than 2 candles before and after
      if (recent[i].high > recent[i - 1].high &&
          recent[i].high > recent[i - 2].high &&
          recent[i].high > recent[i + 1].high &&
          recent[i].high > recent[i + 2].high) {
        swingHighs.push(recent[i].high);
      }

      // Swing low: lower than 2 candles before and after
      if (recent[i].low < recent[i - 1].low &&
          recent[i].low < recent[i - 2].low &&
          recent[i].low < recent[i + 1].low &&
          recent[i].low < recent[i + 2].low) {
        swingLows.push(recent[i].low);
      }
    }

    const recentHigh = Math.max(...highs);
    const recentLow = Math.min(...lows);

    // Use most recent swing levels or fallback to high/low
    const resistance = swingHighs.length > 0 ?
      swingHighs[swingHighs.length - 1] : recentHigh;
    const support = swingLows.length > 0 ?
      swingLows[swingLows.length - 1] : recentLow;

    return { support, resistance, recentHigh, recentLow };
  }

  // ==========================================================================
  // Rule-Based Signal Generation
  // ==========================================================================

  private getRuleBasedSignal(technicals: TechnicalData): AgentSignal {
    let direction: 'long' | 'short' | 'neutral' = 'neutral';
    let confidence = 0;
    const factors: string[] = [];

    // Check for bullish setup
    if (technicals.trend === 'bullish' || technicals.emaCrossover === 'bullish') {
      factors.push(`EMA20 (${technicals.ema20.toFixed(2)}) > EMA50 (${technicals.ema50.toFixed(2)})`);

      // Volume confirmation
      if (technicals.volumeConfirms) {
        factors.push(`Volume confirmed (${technicals.volumeRatio.toFixed(2)}x average)`);
        confidence += 0.3;
      } else {
        factors.push(`Low volume warning - possible fakeout`);
        confidence += 0.1;
      }

      // RSI check
      if (technicals.rsiSignal === 'overbought') {
        factors.push(`RSI overbought (${technicals.rsi.toFixed(1)}) - caution`);
        confidence -= 0.15;
      } else if (technicals.rsiSignal === 'oversold') {
        factors.push(`RSI oversold (${technicals.rsi.toFixed(1)}) - good entry`);
        confidence += 0.2;
      } else {
        factors.push(`RSI neutral (${technicals.rsi.toFixed(1)})`);
        confidence += 0.1;
      }

      // Room to resistance
      if (technicals.distanceToResistance > 2) {
        factors.push(`${technicals.distanceToResistance.toFixed(1)}% to resistance - room to run`);
        confidence += 0.15;
      } else {
        factors.push(`Only ${technicals.distanceToResistance.toFixed(1)}% to resistance - limited upside`);
        confidence -= 0.1;
      }

      // Fresh crossover bonus
      if (technicals.emaCrossover === 'bullish') {
        factors.push(`Fresh bullish EMA crossover detected`);
        confidence += 0.25;
      }

      if (confidence > 0.3) {
        direction = 'long';
        confidence = Math.min(0.95, 0.5 + confidence);
      }
    }
    // Check for bearish setup
    else if (technicals.trend === 'bearish' || technicals.emaCrossover === 'bearish') {
      factors.push(`EMA50 (${technicals.ema50.toFixed(2)}) > EMA20 (${technicals.ema20.toFixed(2)})`);

      // Volume confirmation
      if (technicals.volumeConfirms) {
        factors.push(`Volume confirmed (${technicals.volumeRatio.toFixed(2)}x average)`);
        confidence += 0.3;
      } else {
        factors.push(`Low volume warning - possible fakeout`);
        confidence += 0.1;
      }

      // RSI check
      if (technicals.rsiSignal === 'oversold') {
        factors.push(`RSI oversold (${technicals.rsi.toFixed(1)}) - caution`);
        confidence -= 0.15;
      } else if (technicals.rsiSignal === 'overbought') {
        factors.push(`RSI overbought (${technicals.rsi.toFixed(1)}) - good entry`);
        confidence += 0.2;
      } else {
        factors.push(`RSI neutral (${technicals.rsi.toFixed(1)})`);
        confidence += 0.1;
      }

      // Room to support
      if (technicals.distanceToSupport > 2) {
        factors.push(`${technicals.distanceToSupport.toFixed(1)}% to support - room to fall`);
        confidence += 0.15;
      } else {
        factors.push(`Only ${technicals.distanceToSupport.toFixed(1)}% to support - limited downside`);
        confidence -= 0.1;
      }

      // Fresh crossover bonus
      if (technicals.emaCrossover === 'bearish') {
        factors.push(`Fresh bearish EMA crossover detected`);
        confidence += 0.25;
      }

      if (confidence > 0.3) {
        direction = 'short';
        confidence = Math.min(0.95, 0.5 + confidence);
      }
    } else {
      factors.push(`No clear trend - EMAs converging`);
    }

    const reasoning = `EMA Strategy: ${factors.join('. ')}`;

    return {
      direction,
      confidence: Math.max(0, confidence),
      suggestedSize: this.calculatePositionSize(confidence, direction),
      reasoning,
      timestamp: Date.now(),
      rawScore: direction === 'long' ? confidence : direction === 'short' ? -confidence : 0,
      regime: this.detectRegime(technicals)
    };
  }

  // ==========================================================================
  // AI Enhancement
  // ==========================================================================

  private async getAIAnalysis(technicals: TechnicalData, symbol: string): Promise<AIAnalysisResponse> {
    if (!this.model) {
      return {
        direction: 'neutral',
        confidence: 0,
        reasoning: 'AI model not available',
        keyFactors: [],
        riskLevel: 'medium',
        isFakeout: false
      };
    }

    const prompt = this.constructPrompt(technicals, symbol);

    try {
      const result = await this.model.generateContent({
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: {
          temperature: this.strategyConfig.temperature,
          maxOutputTokens: 1024,
        }
      });

      const response = result.response.text();
      return this.parseAIResponse(response);
    } catch (error) {
      console.error('AI analysis failed:', error);
      return {
        direction: 'neutral',
        confidence: 0,
        reasoning: 'AI analysis failed',
        keyFactors: [],
        riskLevel: 'high',
        isFakeout: false
      };
    }
  }

  private constructPrompt(technicals: TechnicalData, symbol: string): string {
    return `You are an expert crypto trader using the EMA Crossover Strategy. Analyze the following data for ${symbol}:

## EMA Analysis
- EMA 20: $${technicals.ema20.toFixed(2)}
- EMA 50: $${technicals.ema50.toFixed(2)}
- Current Trend: ${technicals.trend.toUpperCase()}
- Crossover Event: ${technicals.emaCrossover === 'none' ? 'No recent crossover' : `${technicals.emaCrossover.toUpperCase()} crossover detected`}
- EMA Spread: ${technicals.emaCrossoverStrength.toFixed(2)}%

## Volume Analysis (Fakeout Detection)
- Volume Ratio (recent/average): ${technicals.volumeRatio.toFixed(2)}x
- Volume Confirms Move: ${technicals.volumeConfirms ? 'YES - High volume supports the move' : 'NO - Low volume, potential FAKEOUT'}

## RSI Analysis
- RSI (14): ${technicals.rsi.toFixed(1)}
- RSI Signal: ${technicals.rsiSignal.toUpperCase()}

## Support & Resistance (Smart Money Concepts)
- Current Price: $${technicals.currentPrice.toFixed(2)}
- Support Level: $${technicals.support.toFixed(2)} (${technicals.distanceToSupport.toFixed(1)}% away)
- Resistance Level: $${technicals.resistance.toFixed(2)} (${technicals.distanceToResistance.toFixed(1)}% away)
- Recent High: $${technicals.recentHigh.toFixed(2)}
- Recent Low: $${technicals.recentLow.toFixed(2)}

## Strategy Rules
1. LONG when EMA 20 crosses above EMA 50 (bullish) with volume confirmation
2. SHORT when EMA 50 goes above EMA 20 (bearish) with volume confirmation
3. If no volume, likely a FAKEOUT - avoid or reduce position
4. RSI > 70 is overbought (avoid longs), RSI < 30 is oversold (avoid shorts)

Based on this analysis, provide a trading recommendation in JSON format ONLY:

{
  "direction": "long" | "short" | "neutral",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<brief explanation>",
  "keyFactors": ["<factor1>", "<factor2>", "<factor3>"],
  "riskLevel": "low" | "medium" | "high",
  "isFakeout": true | false
}`;
  }

  private parseAIResponse(response: string): AIAnalysisResponse {
    try {
      let jsonStr = response.trim();
      if (jsonStr.startsWith('```')) {
        jsonStr = jsonStr.replace(/```json?\n?/g, '').replace(/```/g, '').trim();
      }

      const parsed = JSON.parse(jsonStr);

      return {
        direction: ['long', 'short', 'neutral'].includes(parsed.direction) ? parsed.direction : 'neutral',
        confidence: Math.max(0, Math.min(1, Number(parsed.confidence) || 0)),
        reasoning: parsed.reasoning || 'No reasoning provided',
        keyFactors: Array.isArray(parsed.keyFactors) ? parsed.keyFactors : [],
        riskLevel: ['low', 'medium', 'high'].includes(parsed.riskLevel) ? parsed.riskLevel : 'medium',
        isFakeout: Boolean(parsed.isFakeout)
      };
    } catch (error) {
      console.error('Failed to parse AI response:', response);
      return {
        direction: 'neutral',
        confidence: 0,
        reasoning: 'Failed to parse AI response',
        keyFactors: [],
        riskLevel: 'high',
        isFakeout: false
      };
    }
  }

  private combineSignals(
    ruleBasedSignal: AgentSignal,
    aiAnalysis: AIAnalysisResponse,
    technicals: TechnicalData
  ): AgentSignal {
    // If AI detects fakeout, reduce confidence significantly
    if (aiAnalysis.isFakeout && !technicals.volumeConfirms) {
      return {
        ...ruleBasedSignal,
        confidence: ruleBasedSignal.confidence * 0.3,
        reasoning: `${ruleBasedSignal.reasoning}. AI WARNING: Potential fakeout detected - ${aiAnalysis.reasoning}`,
        suggestedSize: ruleBasedSignal.suggestedSize * 0.3
      };
    }

    // If AI and rules agree, boost confidence
    if (ruleBasedSignal.direction === aiAnalysis.direction && aiAnalysis.direction !== 'neutral') {
      const combinedConfidence = Math.min(0.95, (ruleBasedSignal.confidence + aiAnalysis.confidence) / 2 + 0.1);
      return {
        ...ruleBasedSignal,
        confidence: combinedConfidence,
        reasoning: `${ruleBasedSignal.reasoning}. AI confirms: ${aiAnalysis.reasoning}. Key factors: ${aiAnalysis.keyFactors.join(', ')}.`,
        suggestedSize: this.calculatePositionSize(combinedConfidence, ruleBasedSignal.direction)
      };
    }

    // If AI disagrees, be cautious
    if (ruleBasedSignal.direction !== aiAnalysis.direction && aiAnalysis.confidence > 0.6) {
      return {
        direction: 'neutral',
        confidence: 0.3,
        suggestedSize: 0,
        reasoning: `Mixed signals: Rules suggest ${ruleBasedSignal.direction} but AI suggests ${aiAnalysis.direction}. Staying neutral.`,
        timestamp: Date.now(),
        rawScore: 0,
        regime: ruleBasedSignal.regime
      };
    }

    return ruleBasedSignal;
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  private calculatePositionSize(confidence: number, direction: 'long' | 'short' | 'neutral'): number {
    if (direction === 'neutral' || confidence < this.config.threshold) {
      return 0;
    }

    const baseSize = 0.02; // 2% base
    const maxSize = 0.08;  // 8% max
    const scale = (confidence - this.config.threshold) / (1 - this.config.threshold);

    return Math.min(maxSize, baseSize + (maxSize - baseSize) * scale);
  }

  private detectRegime(technicals: TechnicalData): MarketRegime {
    if (technicals.emaCrossoverStrength > 3) {
      return 'trending';
    } else if (technicals.rsiSignal !== 'neutral') {
      return 'volatile';
    } else if (technicals.emaCrossoverStrength < 0.5) {
      return 'quiet';
    }
    return 'ranging';
  }

  // ==========================================================================
  // Lifecycle Methods
  // ==========================================================================

  async initialize(): Promise<void> {
    console.log(`Initializing ${this.name}...`);

    // Initialize Gemini AI if API key provided
    if (this.strategyConfig.apiKey) {
      try {
        this.genAI = new GoogleGenerativeAI(this.strategyConfig.apiKey);
        this.model = this.genAI.getGenerativeModel({
          model: this.strategyConfig.model || 'gemini-1.5-flash'
        });

        // Test API connection
        const testResult = await this.model.generateContent({
          contents: [{ role: 'user', parts: [{ text: 'Reply OK' }] }],
          generationConfig: { maxOutputTokens: 10 }
        });
        console.log(`${this.name}: Gemini AI connected - ${testResult.response.text().substring(0, 20)}`);
      } catch (error) {
        console.warn(`${this.name}: Gemini AI not available, using rule-based only:`, error);
        this.genAI = null;
        this.model = null;
      }
    } else {
      console.log(`${this.name}: No API key, using rule-based strategy only`);
    }

    this.isInitialized = true;
    console.log(`${this.name} initialized successfully`);
  }

  async shutdown(): Promise<void> {
    console.log(`Shutting down ${this.name}`);
    this.genAI = null;
    this.model = null;
    this.isInitialized = false;
    console.log(`${this.name} shutdown complete`);
  }
}
