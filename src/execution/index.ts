/**
 * @fileoverview Execution module exports
 * @module execution
 * 
 * This module exports all execution-related components:
 * - WeexClient: API wrapper for WEEX exchange
 * - RiskManager: Risk validation and portfolio protection
 * - OrderManager: Order lifecycle and audit logging
 */

export { WeexClient, OrderParams, Balance, Ticker } from './WeexClient';
export { RiskManager, RiskConfig, ValidationResult, PortfolioExposure } from './RiskManager';
export { OrderManager, ExecutionResult, OrderStatus, AuditLogEntry } from './OrderManager';

