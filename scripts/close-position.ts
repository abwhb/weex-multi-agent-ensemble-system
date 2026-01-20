/**
 * Close existing BTC position
 */

import { config } from '../src/config';
import axios from 'axios';
import * as crypto from 'crypto';

const baseUrl = config.weex.baseUrl;
const apiKey = config.weex.apiKey;
const apiSecret = config.weex.apiSecret;
const passphrase = config.weex.passphrase;

function signRequest(
  method: string,
  requestPath: string,
  timestamp: string,
  queryString: string = '',
  body: string = ''
): string {
  let stringToSign = timestamp + method.toUpperCase() + requestPath;
  if (queryString) stringToSign += '?' + queryString;
  if (body) stringToSign += body;

  return crypto
    .createHmac('sha256', apiSecret)
    .update(stringToSign)
    .digest('base64');
}

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
  const response = await axios({ method, url, headers, data: body });
  return response.data;
}

async function main() {
  console.log('=== Closing BTC Position ===\n');

  try {
    // Check balance first
    const balance = await authRequest<any>('GET', '/capi/v2/account/assets');
    console.log('Current Balance:', JSON.stringify(balance, null, 2));

    // Close the long position (0.001 BTC from previous orders)
    // We need to close 0.002 BTC total (2 orders of 0.001)
    console.log('\nClosing position...');
    const closeResponse = await authRequest<any>(
      'POST',
      '/capi/v2/order/placeOrder',
      undefined,
      {
        symbol: 'cmt_btcusdt',
        client_oid: `close_${Date.now()}`,
        size: '0.002',    // Close both positions
        type: '3',        // 3 = Close long
        order_type: '0',  // Normal
        match_price: '1', // Market
        marginMode: '1'   // Cross
      }
    );
    console.log('Close Response:', JSON.stringify(closeResponse, null, 2));

    if (closeResponse.order_id) {
      console.log(`\n✅ Position closed! Order ID: ${closeResponse.order_id}`);
    }

    // Wait and check balance
    await new Promise(r => setTimeout(r, 2000));

    const finalBalance = await authRequest<any>('GET', '/capi/v2/account/assets');
    console.log('\nFinal Balance:', JSON.stringify(finalBalance, null, 2));

  } catch (error: any) {
    console.error('Error:', error.response?.data || error.message);
  }
}

main();
