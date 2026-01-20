/**
 * Verify WeexClient API fixes work correctly
 */

import { WeexClient } from '../src/execution/WeexClient';

async function main() {
  console.log('='.repeat(60));
  console.log('WEEX Client API Verification');
  console.log('='.repeat(60));

  const client = new WeexClient();

  try {
    // Step 1: Connect
    console.log('\n--- Step 1: Connect ---');
    await client.connect();
    console.log('✅ Connected to WEEX API');

    // Step 2: Get Ticker
    console.log('\n--- Step 2: Get Ticker ---');
    const ticker = await client.getTicker('BTC');
    console.log(`✅ BTC Price: $${ticker.lastPrice}`);
    console.log(`   Bid: $${ticker.bidPrice}, Ask: $${ticker.askPrice}`);

    // Step 3: Get Balance
    console.log('\n--- Step 3: Get Balance ---');
    const balances = await client.getBalance();
    const usdtBalance = balances.find(b => b.currency === 'USDT');
    if (usdtBalance) {
      console.log(`✅ USDT Available: ${usdtBalance.available}`);
      console.log(`   Total Equity: ${usdtBalance.total}`);
    }

    // Step 4: Set Leverage
    console.log('\n--- Step 4: Set Leverage ---');
    const leverageSet = await client.setLeverage('BTC', 10);
    console.log(`✅ Leverage set: ${leverageSet}`);

    // Step 5: Get Market Data (candles)
    console.log('\n--- Step 5: Get Market Data ---');
    const candles = await client.getMarketData('BTC', '5m', 5);
    console.log(`✅ Fetched ${candles.length} candles`);
    if (candles.length > 0) {
      const latest = candles[candles.length - 1];
      console.log(`   Latest: O=${latest.open} H=${latest.high} L=${latest.low} C=${latest.close}`);
    }

    // Step 6: Get Open Orders
    console.log('\n--- Step 6: Get Open Orders ---');
    const openOrders = await client.getOpenOrders('BTC');
    console.log(`✅ Open orders: ${openOrders.length}`);

    // Step 7: Get Order History
    console.log('\n--- Step 7: Get Order History ---');
    const history = await client.getOrderHistory('BTC', 5);
    console.log(`✅ Order history: ${history.length} orders`);

    // Step 8: Get Positions
    console.log('\n--- Step 8: Get Positions ---');
    const positions = await client.getPositions('BTC');
    console.log(`✅ Positions: ${positions.length}`);
    if (positions.length > 0) {
      const pos = positions[0];
      console.log(`   ${pos.symbol} ${pos.side}: ${pos.size} @ ${pos.entryPrice}`);
    }

    console.log('\n' + '='.repeat(60));
    console.log('✅ All API endpoints verified successfully!');
    console.log('='.repeat(60));

    await client.disconnect();

  } catch (error: any) {
    console.error('\n❌ Error:', error.message);
    if (error.response?.data) {
      console.error('Response:', JSON.stringify(error.response.data, null, 2));
    }
  }
}

main();
