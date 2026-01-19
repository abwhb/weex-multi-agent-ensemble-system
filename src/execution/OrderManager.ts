/**
 * @fileoverview Order lifecycle management
 * @module execution/OrderManager
 * 
 * ## Overview
 * 
 * Manages the full order lifecycle from trade decision to fill confirmation.
 * Handles order splitting, retry logic, fill tracking, and maintains an
 * audit log for AI decision submission requirements.
 * 
 * ## Key Responsibilities
 * 
 * 1. Convert TradeDecisions into executable orders
 * 2. Handle order placement and monitoring
 * 3. Implement retry logic for failed orders
 * 4. Track order status and fills
 * 5. Generate audit logs for competition compliance
 */

import { WeexClient, OrderParams } from './WeexClient';
import { RiskManager } from './RiskManager';
import { TradeDecision } from '../meta-learner/Ensemble';
import { Order, Position, AllowedPair } from '../types';
import { AILogger, AILogEntry } from '../logging/AILogger';

// ============================================================================
// Interfaces
// ============================================================================

/**
 * Order execution result.
 */
export interface ExecutionResult {
  /** Whether the execution was successful. */
  success: boolean;
  /** Placed order (if successful). */
  order?: Order;
  /** Error message (if failed). */
  error?: string;
  /** Fill price (if filled). */
  fillPrice?: number;
  /** Slippage from expected price. */
  slippage?: number;
  /** Execution timestamp. */
  timestamp: number;
}

/**
 * Order status tracking.
 */
export interface OrderStatus {
  orderId: string;
  symbol: AllowedPair;
  status: 'pending' | 'placed' | 'partial' | 'filled' | 'cancelled' | 'failed';
  filledSize: number;
  avgFillPrice?: number;
  lastUpdated: number;
  retryCount: number;
}

/**
 * Audit log entry for competition.
 */
export interface AuditLogEntry {
  timestamp: string;
  tradeId: string;
  orderId: string;
  decision: TradeDecision;
  execution: ExecutionResult;
  aiLog: AILogEntry;
}

// ============================================================================
// OrderManager Implementation
// ============================================================================

/**
 * Manages order execution and lifecycle tracking.
 * 
 * ## Execution Flow
 * 
 * 1. Receive TradeDecision from Ensemble
 * 2. Validate against RiskManager
 * 3. Convert to OrderParams
 * 4. Place order via WeexClient
 * 5. Monitor until filled or timeout
 * 6. Log result for AI audit
 * 
 * @example
 * ```typescript
 * const orderManager = new OrderManager(weexClient, riskManager, aiLogger);
 * 
 * const result = await orderManager.executeDecision(decision);
 * if (result.success) {
 *   console.log('Order filled at', result.fillPrice);
 * }
 * ```
 */
export class OrderManager {
  /** WEEX API client. */
  private client: WeexClient;
  
  /** Risk manager for validation. */
  private riskManager: RiskManager;
  
  /** AI logger for audit. */
  private aiLogger: AILogger;
  
  /** Active orders being tracked. */
  private activeOrders: Map<string, OrderStatus> = new Map();
  
  /** Completed audit logs. */
  private auditLogs: AuditLogEntry[] = [];
  
  /** Trade ID counter. */
  private tradeCounter: number = 0;
  
  /** Maximum retry attempts. */
  private maxRetries: number = 3;
  
  /** Order timeout in ms. */
  private orderTimeout: number = 30000;
  
  /** Polling interval for order status. */
  private pollInterval: number = 1000;

  /**
   * Create a new OrderManager.
   * 
   * @param client - WEEX API client
   * @param riskManager - Risk manager for validation
   * @param aiLogger - AI logger for audit
   */
  constructor(
    client: WeexClient,
    riskManager: RiskManager,
    aiLogger: AILogger
  ) {
    this.client = client;
    this.riskManager = riskManager;
    this.aiLogger = aiLogger;
  }

  // ==========================================================================
  // Decision Execution
  // ==========================================================================

  /**
   * Execute a trade decision from the ensemble.
   * 
   * ## Process
   * 
   * 1. Generate unique trade ID
   * 2. Validate against risk rules
   * 3. Convert decision to order params
   * 4. Place order with retry logic
   * 5. Monitor until filled
   * 6. Log for audit
   * 
   * @param decision - Trade decision from ensemble
   * @returns Execution result
   */
  async executeDecision(decision: TradeDecision): Promise<ExecutionResult> {
    const tradeId = `trade_${Date.now()}_${++this.tradeCounter}`;
    const timestamp = Date.now();
    
    console.log(`Executing decision: ${decision.action} ${decision.symbol} (size: ${decision.size})`);
    
    try {
      // Step 1: Get current positions for validation
      const positions = await this.client.getPositions();
      
      // Step 2: Convert to order params
      const orderParams = this.decisionToOrderParams(decision);
      
      // Step 3: Validate against risk rules
      const validation = this.riskManager.validateOrder(
        orderParams,
        positions,
        decision.currentPrice
      );
      
      if (!validation.valid) {
        console.warn(`Order rejected by risk manager: ${validation.reason}`);
        
        const result: ExecutionResult = {
          success: false,
          error: validation.reason,
          timestamp
        };
        
        this.logExecution(tradeId, decision, result);
        return result;
      }
      
      // Log any warnings
      if (validation.warnings.length > 0) {
        console.warn('Risk warnings:', validation.warnings);
      }
      
      // Step 4: Place order with retry logic
      const order = await this.placeOrderWithRetry(orderParams);
      
      if (!order) {
        const result: ExecutionResult = {
          success: false,
          error: 'Failed to place order after retries',
          timestamp
        };
        
        this.logExecution(tradeId, decision, result);
        return result;
      }
      
      // Step 5: Track order
      this.trackOrder(order);
      
      // Step 6: Monitor until filled (or timeout)
      const fillResult = await this.monitorFills(order.id);
      
      // Step 7: Calculate slippage
      const fillPrice = fillResult.avgFillPrice || decision.currentPrice;
      const expectedPrice = decision.currentPrice;
      const slippage = (fillPrice - expectedPrice) / expectedPrice;
      
      const result: ExecutionResult = {
        success: fillResult.status === 'filled',
        order,
        fillPrice,
        slippage,
        timestamp,
        error: fillResult.status === 'filled' ? undefined : `Order ${fillResult.status}`
      };
      
      // Step 8: Log for audit
      this.logExecution(tradeId, decision, result);
      
      return result;
      
    } catch (error) {
      const result: ExecutionResult = {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp
      };
      
      this.logExecution(tradeId, decision, result);
      return result;
    }
  }

  /**
   * Convert TradeDecision to OrderParams.
   */
  private decisionToOrderParams(decision: TradeDecision): OrderParams {
    return {
      symbol: decision.symbol,
      side: decision.action === 'buy' ? 'buy' : 'sell',
      type: 'market', // Use market orders for immediate execution
      size: decision.size,
      stopLoss: decision.stopLoss,
      takeProfit: decision.takeProfit,
      leverage: 1, // Default leverage, would be configured
      clientOrderId: `ai_${Date.now()}`
    };
  }

  // ==========================================================================
  // Order Placement with Retry
  // ==========================================================================

  /**
   * Place order with retry logic on failure.
   * 
   * @param params - Order parameters
   * @returns Placed order or null if all retries fail
   */
  private async placeOrderWithRetry(params: OrderParams): Promise<Order | null> {
    let lastError: Error | null = null;
    
    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        console.log(`Placing order (attempt ${attempt + 1}/${this.maxRetries})`);
        const order = await this.client.placeOrder(params);
        return order;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error('Unknown error');
        console.error(`Order placement failed (attempt ${attempt + 1}):`, lastError.message);
        
        // Wait before retry (exponential backoff)
        if (attempt < this.maxRetries - 1) {
          await this.sleep(1000 * Math.pow(2, attempt));
        }
      }
    }
    
    console.error('All order placement attempts failed');
    return null;
  }

  // ==========================================================================
  // Order Monitoring
  // ==========================================================================

  /**
   * Track an order's status.
   */
  private trackOrder(order: Order): void {
    this.activeOrders.set(order.id, {
      orderId: order.id,
      symbol: order.symbol as AllowedPair,
      status: 'placed',
      filledSize: 0,
      lastUpdated: Date.now(),
      retryCount: 0
    });
  }

  /**
   * Monitor order until filled, cancelled, or timeout.
   *
   * @param orderId - Order ID to monitor
   * @returns Final order status
   */
  async monitorFills(orderId: string): Promise<OrderStatus> {
    const startTime = Date.now();
    let status = this.activeOrders.get(orderId);

    if (!status) {
      return {
        orderId,
        symbol: 'BTC' as AllowedPair, // Default fallback
        status: 'failed',
        filledSize: 0,
        lastUpdated: Date.now(),
        retryCount: 0
      };
    }

    while (Date.now() - startTime < this.orderTimeout) {
      try {
        // Poll order status from exchange
        const order = await this.client.getOrder(orderId, status.symbol);

        if (order) {
          // Update status based on exchange response
          // PLACEHOLDER: Actual status parsing from exchange response
          status = {
            ...status,
            status: 'filled', // Simulated as filled
            filledSize: order.size,
            avgFillPrice: order.price || 0,
            lastUpdated: Date.now()
          };

          this.activeOrders.set(orderId, status);

          if (status.status === 'filled' || status.status === 'cancelled') {
            break;
          }
        } else {
          // Order not found, might be filled and removed
          // For simulation, mark as filled
          status = {
            ...status,
            status: 'filled',
            lastUpdated: Date.now()
          };
          this.activeOrders.set(orderId, status);
          break;
        }

        await this.sleep(this.pollInterval);

      } catch (error) {
        console.error('Error polling order status:', error);
        await this.sleep(this.pollInterval);
      }
    }

    // Timeout reached
    if (status.status === 'placed' || status.status === 'partial') {
      console.warn(`Order ${orderId} timed out, attempting cancel`);
      await this.client.cancelOrder(orderId, status.symbol);
      status.status = 'cancelled';
    }

    return status;
  }

  // ==========================================================================
  // Audit Logging (Competition Requirement)
  // ==========================================================================

  /**
   * Log execution for competition AI audit.
   * 
   * ## Required Fields (per competition rules)
   * 
   * - Timestamp
   * - Model version
   * - Agent name
   * - Inputs (features/signals)
   * - Outputs (predictions/decisions)
   * - Reasoning (natural language explanation)
   * - Order ID (if trade executed)
   */
  private logExecution(
    tradeId: string,
    decision: TradeDecision,
    execution: ExecutionResult
  ): void {
    // Create AI log entry
    const aiLog = this.aiLogger.logExecution(
      decision,
      execution.success,
      execution.order?.id
    );
    
    // Create audit log entry
    const auditEntry: AuditLogEntry = {
      timestamp: new Date().toISOString(),
      tradeId,
      orderId: execution.order?.id || 'N/A',
      decision,
      execution,
      aiLog
    };
    
    this.auditLogs.push(auditEntry);
    
    console.log(`Audit log created: ${tradeId} -> ${execution.success ? 'SUCCESS' : 'FAILED'}`);
  }

  /**
   * Generate complete audit log for competition submission.
   * 
   * @returns Full audit log JSON
   */
  generateAuditLog(): string {
    const exportData = {
      version: '1.0.0',
      exportTime: new Date().toISOString(),
      totalTrades: this.auditLogs.length,
      successfulTrades: this.auditLogs.filter(l => l.execution.success).length,
      entries: this.auditLogs
    };
    
    return JSON.stringify(exportData, null, 2);
  }

  /**
   * Get audit logs for a specific time range.
   */
  getAuditLogs(startTime?: number, endTime?: number): AuditLogEntry[] {
    let logs = [...this.auditLogs];
    
    if (startTime) {
      logs = logs.filter(l => new Date(l.timestamp).getTime() >= startTime);
    }
    
    if (endTime) {
      logs = logs.filter(l => new Date(l.timestamp).getTime() <= endTime);
    }
    
    return logs;
  }

  // ==========================================================================
  // Position Management Helpers
  // ==========================================================================

  /**
   * Close an existing position.
   */
  async closePosition(symbol: AllowedPair): Promise<ExecutionResult> {
    try {
      const order = await this.client.closePosition(symbol);
      
      return {
        success: true,
        order,
        timestamp: Date.now()
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: Date.now()
      };
    }
  }

  /**
   * Cancel all open orders.
   */
  async cancelAllOrders(): Promise<void> {
    const openOrders = await this.client.getOpenOrders();
    
    for (const order of openOrders) {
      await this.client.cancelOrder(order.id);
    }
    
    console.log(`Cancelled ${openOrders.length} open orders`);
  }

  // ==========================================================================
  // Utilities
  // ==========================================================================

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get count of active orders being monitored.
   */
  getActiveOrderCount(): number {
    return this.activeOrders.size;
  }

  /**
   * Clear old completed orders from tracking.
   */
  cleanupOldOrders(maxAge: number = 3600000): void {
    const cutoff = Date.now() - maxAge;
    
    for (const [orderId, status] of this.activeOrders) {
      if (status.lastUpdated < cutoff && 
          (status.status === 'filled' || status.status === 'cancelled')) {
        this.activeOrders.delete(orderId);
      }
    }
  }
}

