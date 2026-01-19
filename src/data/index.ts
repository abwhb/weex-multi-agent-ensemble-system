/**
 * @fileoverview Data module exports
 * @module data
 * 
 * This module exports market data services and feature engineering components.
 */

export { MarketDataService, Indicators } from './MarketDataService';
export { FeatureStore, FeatureDefinition, FeatureSet } from './FeatureStore';
export { BinanceClient } from './BinanceClient';
export { 
  IMarketDataProvider, 
  PriceCallback, 
  CandleCallback, 
  Subscription as DataSubscription,
  TickerInfo,
  formatSymbol,
  formatInterval 
} from './IMarketDataProvider';

