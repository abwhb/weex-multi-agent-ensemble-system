/**
 * WEEX API Test Script for AI Wars Hackathon
 * Tests API connection and places a small test trade (~$4)
 */

import { config } from '../src/config';
import axios from 'axios';
import * as crypto from 'crypto';

const baseUrl = config.weex.baseUrl;
const apiKey = config.weex.apiKey;
const apiSecret = config.weex.apiSecret;
const passphrase = config.weex.passphrase;

// Sign request using HMAC-SHA256
function signRequest(
  method: string,
  requestPath: string,
  timestamp: string,
  queryString: string = '',
  body: string = ''
): string {
  let stringToSign = timestamp + method.toUpperCase() + requestPath;
  if (queryString) {
    stringToSign += '?' + queryString;
  }
  if (body) {
    stringToSign += body;
  }

  return crypto
    .createHmac('sha256', apiSecret)
    .update(stringToSign)
    .digest('base64');
}

// Make authenticated request
async function authRequest<T>(
  method: 'GET' | 'POST',
  path: string,
  params?: Record<string, string>,
  body?: Record<string, unknown>
): Promise<T> {
  const timestamp = Date.now().toString();
  const queryString = params ? new URLSearchParams(params).toString() : '';
  const bodyString = body ? JSON.stringify(body) : '';

  const signature = signRequest(method, path, timestamp, queryString, bodyString);

  const headers = {
    'ACCESS-KEY': apiKey,
    'ACCESS-SIGN': signature,
    'ACCESS-TIMESTAMP': timestamp,
    'ACCESS-PASSPHRASE': passphrase,
    'Content-Type': 'application/json',
    'locale': 'en-US'
  };

  const url = queryString ? `${baseUrl}${path}?${queryString}` : `${baseUrl}${path}`;

  const response = await axios({
    method,
    url,
    headers,
    data: body
  });

  return response.data;
}

async function main() {
  console.log('='.repeat(60));
  console.log('WEEX API Test for AI Wars Hackathon');
  console.log('='.repeat(60));
  console.log(`\nBase URL: ${baseUrl}`);
  console.log(`API Key: ${apiKey.substring(0, 20)}...`);

  try {
    // Step 1: Test connection with public endpoint (ticker)
    console.log('\n--- Step 1: Get BTC Price ---');
    const tickerUrl = `${baseUrl}/capi/v2/market/ticker?symbol=cmt_btcusdt`;
    const tickerResponse = await axios.get(tickerUrl);
    // Ticker API returns data directly, not wrapped
    const ticker = tickerResponse.data;
    console.log('BTC/USDT Ticker:');
    console.log(`  Last Price: $${ticker.last}`);
    console.log(`  Best Bid: $${ticker.best_bid}`);
    console.log(`  Best Ask: $${ticker.best_ask}`);
    console.log(`  24h Volume: ${ticker.base_volume} BTC`);

    const btcPrice = parseFloat(ticker.last);

    // Step 2: Check account balance
    console.log('\n--- Step 2: Check Account Balance ---');
    const balanceResponse = await authRequest<any>(
      'GET',
      '/capi/v2/account/assets'
    );
    console.log('Balance Response:', JSON.stringify(balanceResponse, null, 2));

    if (balanceResponse.code === '00000' && balanceResponse.data) {
      const balances = balanceResponse.data;
      if (Array.isArray(balances)) {
        const usdtBalance = balances.find((b: any) => b.marginCoin === 'USDT');
        if (usdtBalance) {
          console.log(`Available: ${usdtBalance.available} USDT`);
          console.log(`Equity: ${usdtBalance.equity} USDT`);
        }
      }
    }

    // Step 3: Set leverage to 10x
    console.log('\n--- Step 3: Set Leverage (10x) ---');
    const leverageResponse = await authRequest<any>(
      'POST',
      '/capi/v2/account/leverage',
      undefined,
      {
        symbol: 'cmt_btcusdt',
        marginMode: '1', // 1 = Cross mode
        longLeverage: '10',
        shortLeverage: '10'
      }
    );
    console.log('Leverage Response:', JSON.stringify(leverageResponse, null, 2));

    // Step 4: Calculate order size for ~$4
    // With 10x leverage, $4 margin = $40 position
    // Size in BTC = $40 / BTC price
    const marginAmount = 4; // $4 USDT margin
    const leverage = 10;
    const positionValue = marginAmount * leverage; // $40
    const btcSize = positionValue / btcPrice;
    // Round to appropriate precision (0.001 BTC minimum usually)
    const orderSize = Math.max(0.001, Math.floor(btcSize * 1000) / 1000);

    console.log('\n--- Step 4: Place Test Order ---');
    console.log(`BTC Price: $${btcPrice}`);
    console.log(`Margin: $${marginAmount} USDT`);
    console.log(`Leverage: ${leverage}x`);
    console.log(`Position Value: $${positionValue}`);
    console.log(`Order Size: ${orderSize} BTC`);

    const clientOid = `test_${Date.now()}`;
    const orderResponse = await authRequest<any>(
      'POST',
      '/capi/v2/order/placeOrder',
      undefined,
      {
        symbol: 'cmt_btcusdt',
        client_oid: clientOid,
        size: orderSize.toString(),
        type: '1',        // 1 = Open long
        order_type: '0',  // 0 = Normal
        match_price: '1', // 1 = Market order
        marginMode: '1'   // 1 = Cross margin
      }
    );
    console.log('Order Response:', JSON.stringify(orderResponse, null, 2));

    if (orderResponse.order_id) {
      const orderId = orderResponse.order_id;
      console.log(`\n✅ Order placed successfully! Order ID: ${orderId}`);

      // Wait a moment for order to fill
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Step 5: Check open orders
      console.log('\n--- Step 5: Check Current Orders ---');
      const ordersResponse = await authRequest<any>(
        'GET',
        '/capi/v2/order/current',
        { symbol: 'cmt_btcusdt' }
      );
      console.log('Current Orders:', JSON.stringify(ordersResponse, null, 2));

      // Step 6: Check positions
      console.log('\n--- Step 6: Check Positions ---');
      const positionsResponse = await authRequest<any>(
        'GET',
        '/capi/v2/position/allPosition',
        { marginCoin: 'USDT', symbol: 'cmt_btcusdt' }
      );
      console.log('Positions:', JSON.stringify(positionsResponse, null, 2));

      // Step 7: Get order history
      console.log('\n--- Step 7: Order History ---');
      const historyResponse = await authRequest<any>(
        'GET',
        '/capi/v2/order/history',
        { symbol: 'cmt_btcusdt', pageSize: '5' }
      );
      console.log('Order History:', JSON.stringify(historyResponse, null, 2));

      // Step 8: Get trade details
      console.log('\n--- Step 8: Trade Details ---');
      const tradesResponse = await authRequest<any>(
        'GET',
        '/capi/v2/trade/fills',
        { symbol: 'cmt_btcusdt', pageSize: '5' }
      );
      console.log('Trade Details:', JSON.stringify(tradesResponse, null, 2));

      // Step 9: Close the position (exit the trade)
      console.log('\n--- Step 9: Close Position (Exit Trade) ---');
      const closeClientOid = `close_${Date.now()}`;
      const closeResponse = await authRequest<any>(
        'POST',
        '/capi/v2/order/placeOrder',
        undefined,
        {
          symbol: 'cmt_btcusdt',
          client_oid: closeClientOid,
          size: orderSize.toString(),
          type: '3',        // 3 = Close long
          order_type: '0',  // 0 = Normal
          match_price: '1', // 1 = Market order
          marginMode: '1'   // 1 = Cross margin
        }
      );
      console.log('Close Order Response:', JSON.stringify(closeResponse, null, 2));

      if (closeResponse.order_id) {
        console.log(`\n✅ Position closed successfully! Order ID: ${closeResponse.order_id}`);
      } else {
        console.log(`\n⚠️ Close order response: ${JSON.stringify(closeResponse)}`);
      }

      // Wait for close order to fill
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Step 10: Check final balance
      console.log('\n--- Step 10: Final Balance ---');
      const finalBalanceResponse = await authRequest<any>(
        'GET',
        '/capi/v2/account/assets'
      );
      console.log('Final Balance:', JSON.stringify(finalBalanceResponse, null, 2));

    } else {
      console.log(`\n❌ Order failed: ${orderResponse.msg}`);
    }

    console.log('\n' + '='.repeat(60));
    console.log('API Test Complete!');
    console.log('='.repeat(60));

  } catch (error: any) {
    console.error('\n❌ Error:', error.response?.data || error.message);
    if (error.response?.data) {
      console.error('Full error response:', JSON.stringify(error.response.data, null, 2));
    }
  }
}

main();
