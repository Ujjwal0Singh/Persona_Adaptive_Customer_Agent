# API Authentication Troubleshooting

## Common Error: 401 Unauthorized

If your API requests return a `401 Unauthorized` error, the most common causes are:

1. **Expired API Key**: API keys expire after 12 months. Check the "Created"
   date in your dashboard under Settings > API Keys.
2. **Missing Authorization Header**: Every request must include:

   ```
   Authorization: Bearer YOUR_API_KEY
   ```

3. **Incorrect Key Scope**: Keys created with "read-only" scope cannot be
   used for write operations such as `POST /v1/orders`.

## Common Error: 429 Too Many Requests

This means you have exceeded your rate limit (default: 100 requests/minute
on the Free tier, 1000 requests/minute on the Pro tier). Implement
exponential backoff and retry after the `Retry-After` header duration.

## Rotating Your API Key

1. Go to Settings > API Keys.
2. Click "Generate New Key."
3. Update your `.env` file or secrets manager with the new key.
4. Revoke the old key only after confirming the new key works in production,
   to avoid downtime.

## Webhook Signature Verification Failing

If incoming webhook signatures fail verification, ensure you are hashing the
**raw request body** (not the parsed JSON object) using HMAC-SHA256 with
your webhook signing secret, found under Settings > Webhooks.
