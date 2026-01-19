/**
 * @fileoverview Agent module exports
 * @module agents
 *
 * This module exports all trading agents and their associated types.
 * Each agent specializes in a different market pattern:
 *
 * - **MomentumAgent**: LSTM-based trend prediction
 * - **MeanReversionAgent**: XGBoost + Bayesian reversion detection
 * - **VolatilityAgent**: GARCH + CNN + RL breakout trading
 * - **GeminiAgent**: Google Gemini AI-powered market analysis
 *
 * The agents work together in an ensemble, with signals combined by
 * the meta-learner (see src/meta-learner/Ensemble.ts).
 */

export { BaseAgent, AgentSignal, AgentConfig } from './BaseAgent';
export { MomentumAgent, MomentumAgentConfig } from './MomentumAgent';
export { MeanReversionAgent, MeanReversionAgentConfig } from './MeanReversionAgent';
export { VolatilityAgent, VolatilityAgentConfig } from './VolatilityAgent';
export { GeminiAgent, GeminiAgentConfig } from './GeminiAgent';

