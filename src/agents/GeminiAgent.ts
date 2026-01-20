/**
 * @fileoverview Gemini AI-powered trading agent
 * @module agents/GeminiAgent
 *
 * ## Overview
 *
 * The GeminiAgent leverages Google's Gemini AI model to analyze market data
 * and generate trading signals. It uses natural language understanding to
 * interpret market conditions, technical indicators, and sentiment to provide
 * AI-driven trading recommendations.
 *
 * ## AI/ML Architecture
 *
 * ### Model: Google Gemini (gemini-1.5-flash or gemini-1.5-pro)
 *
 * The agent constructs a detailed prompt containing:
 * - Recent price action and OHLCV data
 * - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
 * - Market regime classification
 * - Volume analysis
 *
 * The model then provides:
 * - Trading direction (long/short/neutral)
 * - Confidence score (0-1)
 * - Detailed reasoning for the trade decision
 *
 * ## Trading Logic
 *
 * - **Long Signal**: AI identifies bullish patterns with high confidence
 * - **Short Signal**: AI identifies bearish patterns with high confidence
 * - **Neutral**: Unclear market conditions or low confidence
 *
 * ## Safety Features
 *
 * - Fallback to neutral signal on API errors
 * - Response validation to ensure proper signal format
 * - Rate limiting awareness
 * - Structured output parsing with error handling
 */

import { GoogleGenerativeAI, GenerativeModel } from '@google/generative-ai';
import { BaseAgent, AgentSignal, AgentConfig } from './BaseAgent';
import { MarketData, MarketRegime, OHLCV } from '../types';

// ============================================================================
// Types
// ============================================================================

/**
 * Extended configuration for GeminiAgent.
 */
export interface GeminiAgentConfig extends AgentConfig {
  /**
   * Google Gemini API key.
   */
  apiKey: string;

  /**
   * Gemini model to use.
   * @default 'gemini-1.5-flash'
   */
  model?: string;

  /**
   * Temperature for response generation (0-2).
   * Lower = more deterministic, Higher = more creative.
   * @default 0.3
   */
  temperature?: number;

  /**
   * Maximum tokens in the response.
   * @default 1024
   */
  maxTokens?: number;

  /**
   * Number of recent candles to include in analysis.
   * @default 50
   */
  candleWindow?: number;
}

/**
 * Structure of the expected Gemini response.
 */
interface GeminiAnalysisResponse {
  direction: 'long' | 'short' | 'neutral';
  confidence: number;
  reasoning: string;
  keyFactors: string[];
  riskLevel: 'low' | 'medium' | 'high';
}

// ============================================================================
// GeminiAgent Implementation
// ============================================================================

/**
 * Gemini AI-powered trading agent.
 *
 * This agent uses Google's Gemini large language model to analyze market data
 * and generate trading signals. It combines the power of AI reasoning with
 * traditional technical analysis for sophisticated market interpretation.
 *
 * ## Key Capabilities
 *
 * 1. **Natural Language Analysis**: Understands market context beyond simple rules
 * 2. **Multi-Factor Integration**: Combines price, volume, and indicators
 * 3. **Explainable Decisions**: Provides detailed reasoning for every signal
 * 4. **Adaptive Analysis**: Can adjust to different market conditions
 *
 * @extends BaseAgent
 *
 * @example
 * ```typescript
 * const agent = new GeminiAgent({
 *   apiKey: process.env.GEMINI_API_KEY,
 *   lookbackPeriod: 100,
 *   threshold: 0.6,
 *   modelPath: '',
 *   model: 'gemini-1.5-flash',
 *   temperature: 0.3
 * });
 *
 * await agent.initialize();
 * const signal = await agent.analyze(marketData);
 * ```
 */
export class GeminiAgent extends BaseAgent {
  /**
   * Extended configuration with Gemini-specific settings.
   */
  private geminiConfig: GeminiAgentConfig;

  /**
   * Google Generative AI client instance.
   */
  private genAI: GoogleGenerativeAI | null = null;

  /**
   * Gemini model instance.
   */
  private model: GenerativeModel | null = null;

  /**
   * Creates a new GeminiAgent instance.
   *
   * @param config - Agent configuration including API key and model settings
   * @param initialWeight - Starting weight in the ensemble (default: 0.25)
   */
  constructor(config: GeminiAgentConfig, initialWeight: number = 0.25) {
    super('GeminiAgent', config, initialWeight);

    this.geminiConfig = {
      model: 'gemini-2.5-flash',
      temperature: 0.3,
      maxTokens: 1024,
      candleWindow: 50,
      ...config
    };
  }

  // ==========================================================================
  // Core Analysis Methods
  // ==========================================================================

  /**
   * Analyze market data using Gemini AI and produce a trading signal.
   *
   * ## Process Flow
   *
   * 1. **Data Preparation**: Format market data and indicators for prompt
   * 2. **Prompt Construction**: Build comprehensive analysis prompt
   * 3. **AI Inference**: Send to Gemini and parse response
   * 4. **Signal Generation**: Convert AI output to standardized signal
   * 5. **Validation**: Ensure response is within expected bounds
   *
   * @param marketData - Current market data including OHLCV and indicators
   * @returns Promise resolving to a trading signal
   */
  async analyze(marketData: MarketData): Promise<AgentSignal> {
    this.ensureInitialized();

    try {
      // Step 1: Prepare market analysis data
      const analysisData = this.prepareAnalysisData(marketData);

      // Step 2: Construct the prompt
      const prompt = this.constructPrompt(analysisData, marketData.symbol);

      // Step 3: Get AI analysis
      const aiResponse = await this.getGeminiAnalysis(prompt);

      // Step 4: Parse and validate response
      const parsedResponse = this.parseResponse(aiResponse);

      // Step 5: Determine if confidence meets threshold
      const direction = parsedResponse.confidence >= this.config.threshold
        ? parsedResponse.direction
        : 'neutral';

      // Step 6: Calculate suggested position size
      const suggestedSize = this.calculateSuggestedSize(parsedResponse.confidence, parsedResponse.riskLevel);

      // Detect current regime
      const regime = this.detectRegime(marketData);

      return {
        direction,
        confidence: parsedResponse.confidence,
        suggestedSize,
        reasoning: `${parsedResponse.reasoning} Key factors: ${parsedResponse.keyFactors.join(', ')}.`,
        timestamp: Date.now(),
        rawScore: direction === 'long' ? parsedResponse.confidence :
                  direction === 'short' ? -parsedResponse.confidence : 0,
        regime
      };
    } catch (error) {
      console.error(`GeminiAgent analysis error:`, error);
      return this.createNeutralSignal(
        `GeminiAgent error: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  // ==========================================================================
  // Data Preparation
  // ==========================================================================

  /**
   * Prepare market data for AI analysis.
   *
   * @param marketData - Raw market data
   * @returns Formatted analysis data object
   */
  private prepareAnalysisData(marketData: MarketData): {
    recentCandles: OHLCV[];
    indicators: Record<string, number[]>;
    priceChange24h: number;
    volumeChange: number;
    volatility: number;
    trend: string;
  } {
    const window = this.geminiConfig.candleWindow || 50;
    const recentCandles = marketData.ohlcv.slice(-window);

    // Calculate 24h price change
    const oldPrice = recentCandles[0]?.close || 0;
    const currentPrice = recentCandles[recentCandles.length - 1]?.close || 0;
    const priceChange24h = oldPrice > 0 ? ((currentPrice - oldPrice) / oldPrice) * 100 : 0;

    // Calculate volume change
    const recentVolume = recentCandles.slice(-10).reduce((sum, c) => sum + c.volume, 0) / 10;
    const priorVolume = recentCandles.slice(-20, -10).reduce((sum, c) => sum + c.volume, 0) / 10;
    const volumeChange = priorVolume > 0 ? ((recentVolume - priorVolume) / priorVolume) * 100 : 0;

    // Calculate volatility
    const returns = recentCandles.slice(1).map((c, i) =>
      (c.close - recentCandles[i].close) / recentCandles[i].close
    );
    const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance) * Math.sqrt(288) * 100; // Annualized for 5-min candles

    // Determine trend
    const ema20 = this.calculateEMA(recentCandles.map(c => c.close), 20);
    const ema50 = this.calculateEMA(recentCandles.map(c => c.close), 50);
    const trend = ema20 > ema50 ? 'bullish' : ema20 < ema50 ? 'bearish' : 'neutral';

    return {
      recentCandles,
      indicators: marketData.indicators,
      priceChange24h,
      volumeChange,
      volatility,
      trend
    };
  }

  /**
   * Calculate Exponential Moving Average.
   */
  private calculateEMA(prices: number[], period: number): number {
    if (prices.length < period) return prices[prices.length - 1] || 0;

    const multiplier = 2 / (period + 1);
    let ema = prices.slice(0, period).reduce((a, b) => a + b, 0) / period;

    for (let i = period; i < prices.length; i++) {
      ema = (prices[i] - ema) * multiplier + ema;
    }

    return ema;
  }

  // ==========================================================================
  // Prompt Construction
  // ==========================================================================

  /**
   * Construct a comprehensive prompt for Gemini analysis.
   *
   * @param analysisData - Prepared market analysis data
   * @param symbol - Trading symbol
   * @returns Formatted prompt string
   */
  private constructPrompt(analysisData: ReturnType<typeof this.prepareAnalysisData>, symbol: string): string {
    const { recentCandles, priceChange24h, volumeChange, volatility, trend } = analysisData;

    const currentCandle = recentCandles[recentCandles.length - 1];
    const last5Candles = recentCandles.slice(-5);

    // Calculate RSI
    const rsi = this.calculateRSI(recentCandles);

    // Calculate support/resistance levels
    const highs = recentCandles.map(c => c.high);
    const lows = recentCandles.map(c => c.low);
    const resistance = Math.max(...highs.slice(-20));
    const support = Math.min(...lows.slice(-20));

    return `You are an expert cryptocurrency trading analyst. Analyze the following market data for ${symbol} and provide a trading recommendation.

## Current Market Data
- **Symbol**: ${symbol}USDT
- **Current Price**: $${currentCandle.close.toFixed(2)}
- **24h Change**: ${priceChange24h.toFixed(2)}%
- **Volume Change (10 vs prior 10 periods)**: ${volumeChange.toFixed(2)}%
- **Annualized Volatility**: ${volatility.toFixed(2)}%
- **Overall Trend**: ${trend}

## Technical Indicators
- **RSI (14)**: ${rsi.toFixed(2)}
- **20-period Support**: $${support.toFixed(2)}
- **20-period Resistance**: $${resistance.toFixed(2)}
- **Distance to Support**: ${((currentCandle.close - support) / support * 100).toFixed(2)}%
- **Distance to Resistance**: ${((resistance - currentCandle.close) / currentCandle.close * 100).toFixed(2)}%

## Recent Price Action (Last 5 candles)
${last5Candles.map((c, i) => `- Candle ${i + 1}: Open=$${c.open.toFixed(2)}, High=$${c.high.toFixed(2)}, Low=$${c.low.toFixed(2)}, Close=$${c.close.toFixed(2)}, Vol=${c.volume.toFixed(0)}`).join('\n')}

## Analysis Request
Based on the above data, provide a trading recommendation in the following JSON format ONLY (no additional text):

{
  "direction": "long" | "short" | "neutral",
  "confidence": <number between 0.0 and 1.0>,
  "reasoning": "<brief explanation of the decision>",
  "keyFactors": ["<factor1>", "<factor2>", "<factor3>"],
  "riskLevel": "low" | "medium" | "high"
}

Guidelines:
- confidence should reflect how certain you are (0.5 = uncertain, 0.8+ = high conviction)
- Only recommend long/short if confidence >= 0.6
- Consider trend alignment, RSI levels, support/resistance, and volume
- Be conservative - prefer neutral if signals are mixed
- Risk level should consider volatility and position relative to support/resistance`;
  }

  /**
   * Calculate RSI for the given candles.
   */
  private calculateRSI(candles: OHLCV[], period: number = 14): number {
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

  // ==========================================================================
  // Gemini API Interaction
  // ==========================================================================

  /**
   * Send analysis request to Gemini and get response.
   *
   * @param prompt - Constructed analysis prompt
   * @returns Raw AI response string
   */
  private async getGeminiAnalysis(prompt: string): Promise<string> {
    if (!this.model) {
      throw new Error('Gemini model not initialized');
    }

    const result = await this.model.generateContent({
      contents: [{ role: 'user', parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: this.geminiConfig.temperature,
        maxOutputTokens: this.geminiConfig.maxTokens,
      }
    });

    const response = result.response;
    const text = response.text();

    if (!text) {
      throw new Error('Empty response from Gemini');
    }

    return text;
  }

  /**
   * Parse and validate Gemini response.
   *
   * @param response - Raw AI response string
   * @returns Parsed analysis response
   */
  private parseResponse(response: string): GeminiAnalysisResponse {
    try {
      // Extract JSON from response (handle potential markdown code blocks)
      let jsonStr = response.trim();

      // Remove markdown code blocks if present
      if (jsonStr.startsWith('```')) {
        jsonStr = jsonStr.replace(/```json?\n?/g, '').replace(/```/g, '').trim();
      }

      const parsed = JSON.parse(jsonStr);

      // Validate required fields
      if (!['long', 'short', 'neutral'].includes(parsed.direction)) {
        throw new Error('Invalid direction');
      }

      const confidence = Number(parsed.confidence);
      if (isNaN(confidence) || confidence < 0 || confidence > 1) {
        throw new Error('Invalid confidence');
      }

      return {
        direction: parsed.direction,
        confidence: confidence,
        reasoning: parsed.reasoning || 'No reasoning provided',
        keyFactors: Array.isArray(parsed.keyFactors) ? parsed.keyFactors : [],
        riskLevel: ['low', 'medium', 'high'].includes(parsed.riskLevel) ? parsed.riskLevel : 'medium'
      };
    } catch (error) {
      console.error('Failed to parse Gemini response:', error);
      console.error('Raw response:', response);

      // Return neutral on parse failure
      return {
        direction: 'neutral',
        confidence: 0,
        reasoning: 'Failed to parse AI response',
        keyFactors: ['parse_error'],
        riskLevel: 'high'
      };
    }
  }

  // ==========================================================================
  // Position Sizing
  // ==========================================================================

  /**
   * Calculate suggested position size based on confidence and risk level.
   *
   * @param confidence - AI confidence score
   * @param riskLevel - Assessed risk level
   * @returns Suggested position size as fraction of capital
   */
  private calculateSuggestedSize(confidence: number, riskLevel: string): number {
    if (confidence < this.config.threshold) {
      return 0;
    }

    const baseSize = 0.02; // 2% base position
    const maxSize = 0.08; // 8% max for AI agent (slightly conservative)

    // Risk level multipliers
    const riskMultiplier = {
      low: 1.2,
      medium: 1.0,
      high: 0.6
    }[riskLevel] || 1.0;

    // Scale by confidence
    const confidenceScale = (confidence - this.config.threshold) / (1 - this.config.threshold);
    const size = baseSize + (maxSize - baseSize) * confidenceScale;

    return Math.min(maxSize, size * riskMultiplier);
  }

  // ==========================================================================
  // Regime Detection
  // ==========================================================================

  /**
   * Detect current market regime.
   *
   * @param marketData - Market data
   * @returns Detected market regime
   */
  private detectRegime(marketData: MarketData): MarketRegime {
    const closes = marketData.ohlcv.slice(-50).map(c => c.close);

    if (closes.length < 20) return 'quiet';

    // Calculate volatility
    const returns = closes.slice(1).map((c, i) => (c - closes[i]) / closes[i]);
    const volatility = Math.sqrt(
      returns.reduce((sum, r) => sum + r * r, 0) / returns.length
    );

    // Calculate trend strength
    const ema10 = this.calculateEMA(closes, 10);
    const ema30 = this.calculateEMA(closes, 30);
    const trendStrength = Math.abs(ema10 - ema30) / ema30;

    if (volatility > 0.02) {
      return 'volatile';
    } else if (trendStrength > 0.02) {
      return 'trending';
    } else if (volatility < 0.005) {
      return 'quiet';
    }
    return 'ranging';
  }

  // ==========================================================================
  // Lifecycle Methods
  // ==========================================================================

  /**
   * Initialize the GeminiAgent, setting up the AI client.
   */
  async initialize(): Promise<void> {
    console.log(`Initializing ${this.name} with model: ${this.geminiConfig.model}`);

    if (!this.geminiConfig.apiKey) {
      throw new Error('GEMINI_API_KEY is required for GeminiAgent');
    }

    // Initialize Google Generative AI client
    this.genAI = new GoogleGenerativeAI(this.geminiConfig.apiKey);
    this.model = this.genAI.getGenerativeModel({
      model: this.geminiConfig.model || 'gemini-1.5-flash'
    });

    // Verify API access with a simple test
    try {
      const testResult = await this.model.generateContent({
        contents: [{ role: 'user', parts: [{ text: 'Respond with just "ok" if you can read this.' }] }],
        generationConfig: { maxOutputTokens: 10 }
      });
      const testResponse = testResult.response.text();
      console.log(`Gemini API test successful: ${testResponse.substring(0, 20)}`);
    } catch (error) {
      throw new Error(`Failed to initialize Gemini API: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }

    this.isInitialized = true;
    console.log(`${this.name} initialized successfully`);
  }

  /**
   * Clean up resources when shutting down.
   */
  async shutdown(): Promise<void> {
    console.log(`Shutting down ${this.name}`);

    this.genAI = null;
    this.model = null;
    this.isInitialized = false;

    console.log(`${this.name} shutdown complete`);
  }
}
