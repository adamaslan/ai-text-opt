# Seed ETF History

Seed or delta-update permanent ETF price history for all industry tracker ETFs. Populates multi-period returns (1D–10Y) and 52W Hi/Lo.

## Steps

1. Fetch the SCHEDULER_SECRET from GCP Secret Manager
2. Call `POST /admin/seed-etf-history` on the deployed backend
3. Call `POST /refresh/all` to recompute returns from stored history
4. Verify data by calling `GET /industry-tracker` and checking that returns and 52W hi/lo are populated
5. Report results: how many ETFs seeded, rows stored, any failures

## Run

```bash
# 1. Get scheduler token
SCHEDULER_SECRET=$(gcloud secrets versions access latest --secret=SCHEDULER_SECRET)

# 2. Get backend URL
BACKEND_URL=$(grep BACKEND_URL frontend/.env.local | grep -v '#' | cut -d= -f2)

# 3. Seed ETF history (first run = full yfinance history, subsequent = delta)
curl -s -X POST "${BACKEND_URL}/admin/seed-etf-history" \
  -H "x-scheduler-token: ${SCHEDULER_SECRET}" \
  -H "Content-Type: application/json"

# 4. Refresh all caches to recompute returns from stored history
curl -s -X POST "${BACKEND_URL}/refresh/all" \
  -H "x-scheduler-token: ${SCHEDULER_SECRET}" \
  -H "Content-Type: application/json"

# 5. Verify returns are populated
curl -s "${BACKEND_URL}/industry-tracker" | python3 -c "
import json, sys
data = json.load(sys.stdin)
industries = data.get('industries', {})
has_returns = sum(1 for v in industries.values() if v.get('returns'))
has_52w = sum(1 for v in industries.values() if v.get('52w_high') is not None)
missing = [n for n, v in industries.items() if not v.get('returns')]
print(f'Total: {len(industries)} industries')
print(f'With returns: {has_returns}')
print(f'With 52W hi/lo: {has_52w}')
if missing:
    print(f'Missing: {missing}')
else:
    print('All industries fully populated')
"
```

## Rules

- Never hardcode the scheduler token — always read from Secret Manager
- Never hardcode the backend URL — always read from frontend/.env.local
- Seed timeout can be long (5-10 min) on first run — all 50 ETFs fetch full yfinance history
- Delta runs (subsequent) are fast — only appends new trading days
- If an ETF returns 0 rows on delta, that's normal (no new data since last seed)
- Report any ETFs with errors or missing data

## Final Output (required)

```
Seed: X ETFs, Y total rows
Refresh: ok/error
Verification: X/50 with returns, X/50 with 52W hi/lo
Missing: [list any failures]
```
