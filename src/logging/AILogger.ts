/**
 * @fileoverview AI decision logging for competition compliance
 * @module logging/AILogger
 * 
 * ## Overview
 * 
 * Critical component for competition compliance. Logs all AI decision points
 * including model inputs, outputs, reasoning, and execution results.
 * Format matches WEEX AI log submission requirements.
 * 
 * ## Competition Requirements
 * 
 * Every AI-driven trading decision must be logged with:
 * - Timestamp
 * - Model version
 * - Agent name
 * - Inputs (features/signals)
 * - Outputs (predictions/decisions)
 * - Reasoning (natural language explanation)
 * - Order ID (if trade executed)
 * 
 * ## Log Format
 * 
 * Logs are stored as JSON and can be exported for competition submission.
 */

import * as fs from 'fs';
import * as path from 'path';
import { TradeDecision } from '../meta-learner/Ensemble';
import { AgentSignal } from '../agents/BaseAgent';
import { MarketRegime } from '../types';

// ============================================================================
// Interfaces
// ============================================================================

/**
 * Extended AI log entry with additional metadata.
 */
export interface AILogEntry {
  /** ISO timestamp of the decision. */
  timestamp: string;
  /** Unique log entry ID. */
  id: string;
  /** Type of log entry. */
  type: 'prediction' | 'decision' | 'execution';
  /** Versions of all models involved. */
  modelVersions: Record<string, string>;
  /** Agent that generated this entry. */
  agentName: string;
  /** Input data and features. */
  inputs: {
    symbol: string;
    features: number[];
    regime?: MarketRegime;
    rawData?: Record<string, unknown>;
  };
  /** Model/agent outputs. */
  outputs: {
    prediction?: number;
    direction?: string;
    confidence?: number;
    size?: number;
  };
  /** Natural language reasoning. */
  reasoning: string;
  /** Order ID if trade was executed. */
  orderId?: string;
  /** Additional metadata. */
  metadata?: Record<string, unknown>;
}

/**
 * Summary statistics for log export.
 */
interface LogSummary {
  totalEntries: number;
  entriesByType: Record<string, number>;
  entriesByAgent: Record<string, number>;
  timeRange: { start: string; end: string };
  avgConfidence: number;
  successRate: number;
}

// ============================================================================
// AILogger Implementation
// ============================================================================

/**
 * Comprehensive AI decision logger for competition compliance.
 * 
 * ## Key Features
 * 
 * 1. **Full Traceability**: Every AI decision is logged with complete context
 * 2. **Competition Format**: Output format matches WEEX submission requirements
 * 3. **Real-time Logging**: Immediate logging with buffered file writes
 * 4. **Export Capability**: Generate competition-ready log files
 * 
 * @example
 * ```typescript
 * const logger = new AILogger('./logs/ai_decisions.json');
 * 
 * // Log a prediction
 * logger.logPrediction('MomentumAgent', inputs, prediction);
 * 
 * // Log a decision
 * logger.logDecision(tradeDecision);
 * 
 * // Export for submission
 * const logPath = await logger.exportLog('./submission/ai_log.json');
 * ```
 */
export class AILogger {
  /** Path to log file. */
  private logPath: string;
  
  /** In-memory log buffer. */
  private logs: AILogEntry[] = [];
  
  /** Log entry counter. */
  private entryCounter: number = 0;
  
  /** Model versions registry. */
  private modelVersions: Record<string, string> = {};
  
  /** Buffer size before flush. */
  private bufferSize: number = 100;
  
  /** Execution success tracking. */
  private executionResults: { success: boolean; orderId?: string }[] = [];

  /**
   * Create a new AILogger.
   * 
   * @param logPath - Path to log file
   */
  constructor(logPath: string = './logs/ai_decisions.json') {
    this.logPath = logPath;
    
    // Ensure log directory exists
    const dir = path.dirname(logPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    
    // Register default model versions
    this.registerModelVersions({
      'MomentumAgent/LSTM': '1.0.0',
      'MeanReversionAgent/RegimeClassifier': '1.0.0',
      'MeanReversionAgent/ParamOptimizer': '1.0.0',
      'VolatilityAgent/GARCH': '1.0.0',
      'VolatilityAgent/CNN': '1.0.0',
      'VolatilityAgent/DQN': '1.0.0',
      'Ensemble/MetaLearner': '1.0.0'
    });
    
    console.log(`AILogger initialized: ${logPath}`);
  }

  // ==========================================================================
  // Model Version Management
  // ==========================================================================

  /**
   * Register model versions for logging.
   * 
   * @param versions - Map of model names to versions
   */
  registerModelVersions(versions: Record<string, string>): void {
    this.modelVersions = { ...this.modelVersions, ...versions };
  }

  /**
   * Update a specific model version.
   */
  updateModelVersion(modelName: string, version: string): void {
    this.modelVersions[modelName] = version;
  }

  // ==========================================================================
  // Logging Methods
  // ==========================================================================

  /**
   * Log a model prediction.
   * 
   * Use this to log individual agent predictions before they're combined
   * by the ensemble.
   * 
   * @param agentName - Name of the agent making the prediction
   * @param inputs - Input features and data
   * @param signal - Agent's output signal
   * @returns Created log entry
   */
  logPrediction(
    agentName: string,
    inputs: { symbol: string; features: number[]; regime?: MarketRegime },
    signal: AgentSignal
  ): AILogEntry {
    const entry: AILogEntry = {
      timestamp: new Date().toISOString(),
      id: this.generateId('pred'),
      type: 'prediction',
      modelVersions: this.getAgentVersions(agentName),
      agentName,
      inputs,
      outputs: {
        prediction: signal.rawScore,
        direction: signal.direction,
        confidence: signal.confidence,
        size: signal.suggestedSize
      },
      reasoning: signal.reasoning
    };
    
    this.addEntry(entry);
    return entry;
  }

  /**
   * Log a trade decision from the ensemble.
   * 
   * @param decision - Trade decision from ensemble
   * @returns Created log entry
   */
  logDecision(decision: TradeDecision): AILogEntry {
    const entry: AILogEntry = {
      timestamp: new Date().toISOString(),
      id: this.generateId('dec'),
      type: 'decision',
      modelVersions: this.modelVersions,
      agentName: 'Ensemble',
      inputs: {
        symbol: decision.symbol,
        features: [], // Would include actual features in production
        regime: decision.regime
      },
      outputs: {
        direction: decision.action,
        confidence: decision.confidence,
        size: decision.size
      },
      reasoning: decision.reasoning,
      metadata: {
        agentContributions: decision.agentContributions,
        stopLoss: decision.stopLoss,
        takeProfit: decision.takeProfit,
        currentPrice: decision.currentPrice
      }
    };
    
    this.addEntry(entry);
    return entry;
  }

  /**
   * Log trade execution result.
   * 
   * @param decision - Original trade decision
   * @param success - Whether execution was successful
   * @param orderId - Order ID if executed
   * @returns Created log entry
   */
  logExecution(
    decision: TradeDecision,
    success: boolean,
    orderId?: string
  ): AILogEntry {
    const entry: AILogEntry = {
      timestamp: new Date().toISOString(),
      id: this.generateId('exec'),
      type: 'execution',
      modelVersions: this.modelVersions,
      agentName: 'OrderManager',
      inputs: {
        symbol: decision.symbol,
        features: [],
        regime: decision.regime
      },
      outputs: {
        direction: decision.action,
        confidence: decision.confidence,
        size: decision.size
      },
      reasoning: success 
        ? `Order executed successfully. ${decision.reasoning}`
        : `Order execution failed. ${decision.reasoning}`,
      orderId,
      metadata: {
        success,
        agentContributions: decision.agentContributions
      }
    };
    
    // Track execution results
    this.executionResults.push({ success, orderId });
    
    this.addEntry(entry);
    return entry;
  }

  /**
   * Log a custom AI event.
   * 
   * @param agentName - Source agent
   * @param eventType - Type of event
   * @param data - Event data
   * @param reasoning - Explanation
   */
  logEvent(
    agentName: string,
    eventType: string,
    data: Record<string, unknown>,
    reasoning: string
  ): void {
    const entry: AILogEntry = {
      timestamp: new Date().toISOString(),
      id: this.generateId('event'),
      type: 'prediction', // Use prediction type for custom events
      modelVersions: this.getAgentVersions(agentName),
      agentName,
      inputs: {
        symbol: (data.symbol as string) || 'N/A',
        features: [],
        rawData: data
      },
      outputs: {},
      reasoning,
      metadata: { eventType }
    };
    
    this.addEntry(entry);
  }

  // ==========================================================================
  // Entry Management
  // ==========================================================================

  /**
   * Add an entry to the log buffer.
   */
  private addEntry(entry: AILogEntry): void {
    this.logs.push(entry);
    
    // Flush to disk if buffer is full
    if (this.logs.length >= this.bufferSize) {
      this.flush();
    }
  }

  /**
   * Flush buffer to disk.
   */
  flush(): void {
    if (this.logs.length === 0) return;
    
    try {
      // Read existing log if it exists
      let existingLogs: AILogEntry[] = [];
      if (fs.existsSync(this.logPath)) {
        const content = fs.readFileSync(this.logPath, 'utf-8');
        const parsed = JSON.parse(content);
        existingLogs = parsed.entries || [];
      }
      
      // Append new logs
      const allLogs = [...existingLogs, ...this.logs];
      
      // Write back
      const output = {
        version: '1.0.0',
        lastUpdated: new Date().toISOString(),
        entries: allLogs
      };
      
      fs.writeFileSync(this.logPath, JSON.stringify(output, null, 2));
      
      console.log(`Flushed ${this.logs.length} log entries to disk`);
      this.logs = [];
    } catch (error) {
      console.error('Error flushing logs:', error);
    }
  }

  // ==========================================================================
  // Export Methods
  // ==========================================================================

  /**
   * Export complete log for competition submission.
   * 
   * @param outputPath - Path to export file (optional, uses default if not provided)
   * @returns Path to exported file
   */
  async exportLog(outputPath?: string): Promise<string> {
    // Flush any buffered entries first
    this.flush();
    
    const exportPath = outputPath || this.logPath.replace('.json', '_export.json');
    
    try {
      // Read all logs
      let allLogs: AILogEntry[] = [];
      if (fs.existsSync(this.logPath)) {
        const content = fs.readFileSync(this.logPath, 'utf-8');
        const parsed = JSON.parse(content);
        allLogs = parsed.entries || [];
      }
      
      // Generate summary
      const summary = this.generateSummary(allLogs);
      
      // Create export object
      const exportData = {
        version: '1.0.0',
        exportTime: new Date().toISOString(),
        modelVersions: this.modelVersions,
        summary,
        entries: allLogs
      };
      
      // Write export file
      fs.writeFileSync(exportPath, JSON.stringify(exportData, null, 2));
      
      console.log(`Exported ${allLogs.length} log entries to ${exportPath}`);
      return exportPath;
    } catch (error) {
      console.error('Error exporting log:', error);
      throw error;
    }
  }

  /**
   * Generate summary statistics for log export.
   */
  private generateSummary(logs: AILogEntry[]): LogSummary {
    if (logs.length === 0) {
      return {
        totalEntries: 0,
        entriesByType: {},
        entriesByAgent: {},
        timeRange: { start: '', end: '' },
        avgConfidence: 0,
        successRate: 0
      };
    }
    
    const entriesByType: Record<string, number> = {};
    const entriesByAgent: Record<string, number> = {};
    let totalConfidence = 0;
    let confidenceCount = 0;
    
    for (const entry of logs) {
      entriesByType[entry.type] = (entriesByType[entry.type] || 0) + 1;
      entriesByAgent[entry.agentName] = (entriesByAgent[entry.agentName] || 0) + 1;
      
      if (entry.outputs.confidence !== undefined) {
        totalConfidence += entry.outputs.confidence;
        confidenceCount++;
      }
    }
    
    const executionLogs = logs.filter(l => l.type === 'execution');
    const successfulExecutions = executionLogs.filter(
      l => l.metadata?.success === true
    ).length;
    
    return {
      totalEntries: logs.length,
      entriesByType,
      entriesByAgent,
      timeRange: {
        start: logs[0].timestamp,
        end: logs[logs.length - 1].timestamp
      },
      avgConfidence: confidenceCount > 0 ? totalConfidence / confidenceCount : 0,
      successRate: executionLogs.length > 0 
        ? successfulExecutions / executionLogs.length 
        : 0
    };
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  /**
   * Generate unique log entry ID.
   */
  private generateId(prefix: string): string {
    return `${prefix}_${Date.now()}_${++this.entryCounter}`;
  }

  /**
   * Get model versions for a specific agent.
   */
  private getAgentVersions(agentName: string): Record<string, string> {
    const versions: Record<string, string> = {};
    
    for (const [model, version] of Object.entries(this.modelVersions)) {
      if (model.startsWith(agentName)) {
        versions[model] = version;
      }
    }
    
    return Object.keys(versions).length > 0 ? versions : this.modelVersions;
  }

  /**
   * Get all log entries (for debugging).
   */
  getEntries(): AILogEntry[] {
    return [...this.logs];
  }

  /**
   * Get entry count.
   */
  getEntryCount(): number {
    return this.logs.length;
  }

  /**
   * Clear in-memory buffer (does not affect disk).
   */
  clearBuffer(): void {
    this.logs = [];
  }

  /**
   * Get log file path.
   */
  getLogPath(): string {
    return this.logPath;
  }
}

// Export the interface as well
export type { LogSummary };

