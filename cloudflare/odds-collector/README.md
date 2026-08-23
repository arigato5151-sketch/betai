# Bet AI Odds Collector

Cloudflare Worker that captures immutable API-Football 1X2 entry and closing
snapshots while the local application is offline.

## Runtime

- `POST /v1/tracked-fixtures` registers an upcoming fixture and optionally
  captures its entry market.
- Cron runs every 15 minutes. It requests a closing market inside the 45-minute
  window and retries once inside the 15-minute window.
- `GET /v1/snapshots` exposes captured snapshots to the authenticated local
  backend.
- `POST /v1/run` provides an authenticated manual health run.
- `GET /health` is public and contains no account or fixture data.

All `/v1/*` requests require a timestamped HMAC-SHA256 signature. The shared
secret is domain-separated from the API-Football key and the raw key is stored
only as a Cloudflare encrypted secret.

## Deploy

```powershell
npm install
npm run check
npm run db:migrate:remote
npm run deploy
```

Required Worker secrets:

- `API_FOOTBALL_KEY`
- `CLIENT_HMAC_SECRET`

The D1 database binding is `DB`. Never put either secret in `wrangler.toml`.
