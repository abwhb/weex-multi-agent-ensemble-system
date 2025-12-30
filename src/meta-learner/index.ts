/**
 * @fileoverview Meta-learner module exports
 * @module meta-learner
 * 
 * This module exports the ensemble meta-learner and performance tracking
 * components that combine signals from individual agents into optimal
 * trading decisions.
 */

export { Ensemble, EnsembleConfig, TradeDecision } from './Ensemble';
export { 
  PerformanceTracker, 
  TradeRecord, 
  AgentPerformanceSummary,
  TrainingDataExport 
} from './PerformanceTracker';

