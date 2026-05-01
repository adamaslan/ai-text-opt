# Backend Debug

Deep diagnostic targeting the two most common production failure modes:

1. **Route drift** — `main.py` was updated but backend was never redeployed, so live routes don't match what the frontend expects (e.g. `/industry-intel` 404 while old `/industry-tracker` still works)
2. **Rate-limit pipeline stall** — morning `/refresh/all` hit Finnhub + yfinance 429s, `industry_cache` was not written, and `/industry-returns` is serving yesterday's stale data

Combines `/debug/status` (live backend snapshot) with Cloud Run log analysis.

## Run

```bash
PROJECT="${GCP_PROJECT_ID:-ttb-lang1}"
REGION="us-central1"
SERVICE="gcp3-backend"
WINDOW="${1:-120}"   # minutes to look back, default 2h (covers a morning refresh run)

PASS=0; FAIL=0; WARN=0
pass() { PASS=$((PASS+1)); echo "  ✅ $1"; }
fail() { FAIL=$((FAIL+1)); echo "  ❌ $1"; }
warn() { WARN=$((WARN+1)); echo "  ⚠️  $1"; }

echo "=== Backend Debug — $(date -u '+%Y-%m-%d %H:%M UTC') ==="
echo "    Window : last ${WINDOW} min"
echo ""

# ─── 0. Resolve backend URL ──────────────────────────────────────────────────
BACKEND_URL=$(gcloud run services describe "$SERVICE" \
  --region "$REGION" --project "$PROJECT" \
  --format="value(status.url)" 2>/dev/null)
if [ -z "$BACKEND_URL" ]; then
  fail "Cannot resolve Cloud Run URL for $SERVICE — check project/region"
  echo "Cannot continue."; exit 1
fi
echo "  Backend: $BACKEND_URL"
echo ""

# ─── 1. /health — basic liveness ─────────────────────────────────────────────
echo "=== 1. Backend health ==="
HEALTH=$(curl -s --max-time 15 "$BACKEND_URL/health" 2>/dev/null)
HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$BACKEND_URL/health" 2>/dev/null)
if [ "$HEALTH_CODE" = "200" ] && echo "$HEALTH" | python3 -c 'import sys,json; d=json.load(sys.stdin); assert d.get("status")=="ok"' 2>/dev/null; then
  VERSION=$(echo "$HEALTH" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("version","unknown"))' 2>/dev/null)
  pass "Backend alive version=$VERSION"
else
  fail "Backend /health returned HTTP $HEALTH_CODE"
  echo "  Cannot continue without a live backend."; exit 1
fi

# ─── 2. Route drift check via /debug/status ───────────────────────────────────
echo ""
echo "=== 2. Route inventory & deployed-code drift ==="
DEBUG_STATUS=$(curl -s --max-time 15 "$BACKEND_URL/debug/status" 2>/dev/null)
DEBUG_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$BACKEND_URL/debug/status" 2>/dev/null)

if [ "$DEBUG_CODE" != "200" ]; then
  warn "/debug/status returned HTTP $DEBUG_CODE — skipping route-drift check (endpoint may not be deployed yet)"
else
  python3 - <<PYEOF
import json, sys

raw = '''$DEBUG_STATUS'''
try:
    d = json.loads(raw)
except Exception as e:
    print(f"  ⚠️  Could not parse /debug/status response: {e}")
    sys.exit(0)

routes = d.get("route_inventory", [])
missing = d.get("missing_expected_routes", [])
ic = d.get("industry_cache", {})
rl = d.get("rate_limits", {}).get("finnhub_429s", {})
cache = d.get("gcp3_cache", {})

# Route check
expected = {"/industry-intel", "/signals", "/industry-returns", "/screener",
            "/market-overview", "/content", "/macro-pulse"}
present = set(routes)
found_expected = sorted(expected & present)
missing_now = sorted(expected - present)

if missing_now:
    print(f"  ❌ ROUTE DRIFT: {len(missing_now)} expected route(s) missing from live backend:")
    for r in missing_now:
        print(f"       {r}  ← 404 for users until backend is redeployed")
    print(f"  ℹ️  Fix: cd backend && gcloud builds submit --config cloudbuild.yaml")
else:
    print(f"  ✅ All {len(expected)} expected routes present in live backend")

if routes:
    print(f"  ℹ️  Registered routes ({len(routes)}): {', '.join(routes)}")

# industry_cache freshness
ic_count = ic.get("doc_count", -1)
ic_fresh_h = ic.get("freshness_hours")
ic_stale = ic.get("stale", False)
newest = ic.get("newest_updated", "unknown")

if ic_count < 0:
    print(f"  ❌ industry_cache: could not read collection")
elif ic_count == 0:
    print(f"  ❌ industry_cache: empty — /industry-returns will have no data")
    print(f"  ℹ️  Fix: trigger POST /refresh/all or POST /admin/seed-etf-history")
elif ic_stale:
    print(f"  ❌ industry_cache: STALE — {ic_count} docs, newest updated {ic_fresh_h:.1f}h ago ({newest})")
    print(f"  ℹ️  This means the last compute_returns or get_industry_data call failed (likely rate-limited)")
    print(f"  ℹ️  /industry-returns is serving yesterday's data with a stale timestamp")
else:
    print(f"  ✅ industry_cache: fresh — {ic_count} docs, newest updated {ic_fresh_h:.1f}h ago")

# Finnhub 429 counter
fh_429s = rl.get("count", 0)
fh_since = rl.get("since")
if fh_429s > 5:
    print(f"  ❌ Finnhub 429s: {fh_429s} since {fh_since} — sustained rate-limiting in progress")
elif fh_429s > 0:
    print(f"  ⚠️  Finnhub 429s: {fh_429s} since {fh_since} — some rate-limiting occurred")
else:
    print(f"  ✅ Finnhub 429 counter: 0 (no active rate-limiting on this instance)")

# Live cache keys
live_keys = cache.get("live_keys", [])
live_count = cache.get("live_doc_count", 0)
critical_keys = ["industry_returns", "industry_data", "technical_signals", "macro_pulse"]
missing_critical = [k for k in critical_keys if not any(k in key for key in live_keys)]
if missing_critical:
    print(f"  ⚠️  gcp3_cache: missing critical keys: {missing_critical}")
    print(f"       (these will trigger a live Finnhub fetch on next request)")
else:
    print(f"  ✅ gcp3_cache: {live_count} live docs, all critical keys present")
PYEOF
fi

# ─── 3. Pull logs once ────────────────────────────────────────────────────────
echo ""
echo "=== 3. Fetching Cloud Run logs (last ${WINDOW} min) ==="
TMPLOG=$(mktemp /tmp/gcp3-backend-debug.XXXXXX)
gcloud run services logs read "$SERVICE" \
  --region "$REGION" --project "$PROJECT" \
  --limit 2000 \
  --format="value(timestamp,severity,textPayload,jsonPayload.message)" \
  2>/dev/null > "$TMPLOG"
TOTAL=$(wc -l < "$TMPLOG")
echo "  Fetched $TOTAL log lines"

# ─── 4. Rate-limit stall detection ───────────────────────────────────────────
echo ""
echo "=== 4. Rate-limit stall detection (Finnhub + yfinance) ==="
python3 - "$TMPLOG" "$WINDOW" <<'PYEOF'
import sys, re
from datetime import datetime, timezone, timedelta

logfile, window = sys.argv[1], int(sys.argv[2])
cutoff = datetime.now(timezone.utc) - timedelta(minutes=window)

fh_429s, yf_429s, all_failed, cache_writes, cache_skips = [], [], [], [], []
for line in open(logfile):
    m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
    if not m: continue
    try: ts = datetime.fromisoformat(m.group(1) + '+00:00')
    except: continue
    if ts < cutoff: continue

    if re.search(r'rate_limited_429|finnhub.*429|429.*finnhub', line, re.I):
        fh_429s.append(line.strip()[:160])
    if re.search(r'yfinance.*429|Too Many Requests.*yfinance|YFRateLimitError', line, re.I):
        yf_429s.append(line.strip()[:160])
    if re.search(r'all_sources_failed|all sources failed', line, re.I):
        all_failed.append(line.strip()[:160])
    if re.search(r'industry_cache: write_complete|cache_written key=industry', line, re.I):
        cache_writes.append(line.strip()[:160])
    if re.search(r'no_valid_quotes.*skipping cache write', line, re.I):
        cache_skips.append(line.strip()[:160])

print(f"  Finnhub 429s   : {len(fh_429s)}")
print(f"  yfinance 429s  : {len(yf_429s)}")
print(f"  All-source fail: {len(all_failed)}")
print(f"  Cache writes   : {len(cache_writes)}")

if fh_429s and yf_429s:
    print(f"  ❌ STALL DETECTED: both Finnhub and yfinance were rate-limited simultaneously")
    print(f"     This prevents industry_cache from being written (returns data will be stale)")
    print(f"     Last Finnhub 429: {fh_429s[-1][:120]}")
    print(f"     Last yfinance 429: {yf_429s[-1][:120]}")
elif fh_429s:
    print(f"  ⚠️  Finnhub was rate-limited ({len(fh_429s)}x) but yfinance fallback may have covered it")
elif yf_429s:
    print(f"  ⚠️  yfinance was rate-limited ({len(yf_429s)}x) — may have missed some ETF quotes")
else:
    print(f"  ✅ No rate-limiting detected in window")

if cache_skips:
    print(f"  ❌ industry_cache write was SKIPPED ({len(cache_skips)}x) — all quotes failed, ranked list empty")
elif cache_writes:
    print(f"  ✅ industry_cache was written {len(cache_writes)} time(s) in window")
    for w in cache_writes[-2:]: print(f"    {w[:120]}")
else:
    print(f"  ⚠️  No industry_cache write log in window (may be a cache hit from a prior run)")

if all_failed:
    print(f"  ⚠️  {len(all_failed)} industry/ETF all-source failures (shown last 3):")
    for e in all_failed[-3:]: print(f"    {e[:140]}")
PYEOF

# ─── 5. industry_cache freshness from logs ───────────────────────────────────
echo ""
echo "=== 5. industry_returns serving-stale-data check ==="
python3 - "$TMPLOG" "$WINDOW" <<'PYEOF'
import sys, re
from datetime import datetime, timezone, timedelta

logfile, window = sys.argv[1], int(sys.argv[2])
cutoff = datetime.now(timezone.utc) - timedelta(minutes=window)

stale_serves, fresh_serves, cache_hits, cache_misses = [], [], [], []
for line in open(logfile):
    m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
    if not m: continue
    try: ts = datetime.fromisoformat(m.group(1) + '+00:00')
    except: continue
    if ts < cutoff: continue

    if re.search(r'serving_stale_data|serving_prev_day_data', line, re.I):
        stale_serves.append(line.strip()[:160])
    if re.search(r'industry_returns: cache_written', line, re.I):
        fresh_serves.append(line.strip()[:160])
    if re.search(r'industry_returns: cache_hit', line, re.I):
        cache_hits.append(line.strip()[:160])
    if re.search(r'industry_returns: cache_miss', line, re.I):
        cache_misses.append(line.strip()[:160])

if stale_serves:
    print(f"  ❌ /industry-returns is serving STALE data ({len(stale_serves)}x in window):")
    for s in stale_serves[-3:]: print(f"    {s[:140]}")
    print(f"  ℹ️  Root cause: industry_cache was not refreshed — likely a rate-limit stall (see section 4)")
    print(f"  ℹ️  Fix: wait for next scheduler run, or manually trigger POST /refresh/all")
elif fresh_serves:
    print(f"  ✅ industry_returns cache was freshly written {len(fresh_serves)}x in window")
else:
    print(f"  ✅ No stale-serve events in window")

print(f"  Cache hits/misses: {len(cache_hits)} hits, {len(cache_misses)} misses")
PYEOF

# ─── 6. Refresh pipeline stage health ────────────────────────────────────────
echo ""
echo "=== 6. Refresh pipeline runs ==="
python3 - "$TMPLOG" "$WINDOW" <<'PYEOF'
import sys, re
from datetime import datetime, timezone, timedelta

logfile, window = sys.argv[1], int(sys.argv[2])
cutoff = datetime.now(timezone.utc) - timedelta(minutes=window)

runs, stage_errors, stage_oks = [], [], []
for line in open(logfile):
    m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
    if not m: continue
    try: ts = datetime.fromisoformat(m.group(1) + '+00:00')
    except: continue
    if ts < cutoff: continue

    if re.search(r'POST /refresh/(all|premarket|intraday).*(started|complete)', line, re.I):
        ms = re.search(r'total_ms=(\d+)', line)
        runs.append((ts.isoformat(), line.strip()[:160], int(ms.group(1)) if ms else None))
    if re.search(r'refresh/.* stage \d+ error', line, re.I):
        stage_errors.append(line.strip()[:160])

if runs:
    print(f"  {len(runs)} refresh run(s) in window:")
    for ts_str, entry, ms in runs[-5:]:
        ms_str = f" ({ms}ms)" if ms else ""
        print(f"    {ts_str}: {entry[:100]}{ms_str}")
else:
    print(f"  ⚠️  No refresh pipeline runs in last {window} min — scheduler may not have fired")

if stage_errors:
    print(f"  ❌ {len(stage_errors)} stage-level failure(s):")
    for e in stage_errors[-5:]: print(f"    {e[:140]}")
else:
    print(f"  ✅ No stage-level errors in window")
PYEOF

# ─── 7. ERROR and Exception summary ──────────────────────────────────────────
echo ""
echo "=== 7. Errors and exceptions ==="
python3 - "$TMPLOG" "$WINDOW" <<'PYEOF'
import sys, re
from datetime import datetime, timezone, timedelta

logfile, window = sys.argv[1], int(sys.argv[2])
cutoff = datetime.now(timezone.utc) - timedelta(minutes=window)

errors = []
for line in open(logfile):
    m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
    if not m: continue
    try: ts = datetime.fromisoformat(m.group(1) + '+00:00')
    except: continue
    if ts < cutoff: continue
    if re.search(r'\bERROR\b|\bCRITICAL\b|Traceback|Exception', line):
        errors.append(line.strip()[:200])

# Deduplicate by first 80 chars (collapses repeated identical errors)
seen = set()
unique_errors = []
for e in errors:
    key = e[:80]
    if key not in seen:
        seen.add(key)
        unique_errors.append(e)

if unique_errors:
    print(f"  ❌ {len(unique_errors)} unique error type(s) (last 5):")
    for e in unique_errors[-5:]: print(f"    {e[:180]}")
else:
    print(f"  ✅ No ERROR/Exception entries in window")
PYEOF

# ─── 8. Deployed revision vs expected routes (openapi cross-check) ────────────
echo ""
echo "=== 8. Live route cross-check vs openapi.json ==="
OPENAPI=$(curl -s --max-time 10 "$BACKEND_URL/openapi.json" 2>/dev/null)
python3 - <<PYEOF
import json, sys

raw = '''$OPENAPI'''
try:
    d = json.loads(raw)
except Exception as e:
    print(f"  ⚠️  Could not fetch openapi.json: {e}")
    sys.exit(0)

live_paths = set(d.get("paths", {}).keys())
expected = {"/industry-intel", "/signals", "/industry-returns", "/screener",
            "/market-overview", "/content", "/macro-pulse", "/health"}

missing = sorted(expected - live_paths)
old_routes = sorted(live_paths & {"/industry-tracker", "/industry-quotes", "/technical-signals",
                                   "/morning-brief", "/ai-summary", "/news-sentiment"})

if missing:
    print(f"  ❌ {len(missing)} expected route(s) NOT in openapi.json (backend not redeployed):")
    for r in missing: print(f"       {r}")
    print(f"  ℹ️  Fix: cd backend && gcloud builds submit --config cloudbuild.yaml")
else:
    print(f"  ✅ All expected routes present in openapi.json")

if old_routes:
    print(f"  ⚠️  Old pre-consolidation routes still registered: {old_routes}")
    print(f"       (harmless if consolidated routes also present, but indicates partially deployed state)")
else:
    print(f"  ✅ No old pre-consolidation routes detected")

print(f"  ℹ️  Total routes registered: {len(live_paths)}")
PYEOF

# ─── 9. Summary ──────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  BACKEND DEBUG: $PASS passed, $FAIL failed, $WARN warnings"
echo "  Window: last ${WINDOW} min"
echo "════════════════════════════════════════════"
rm -f "$TMPLOG"

if [ "$FAIL" -gt 0 ]; then
  echo "  Status: ISSUES FOUND"
  exit 1
elif [ "$WARN" -gt 0 ]; then
  echo "  Status: WARNINGS"
  exit 0
else
  echo "  Status: ALL CLEAR"
  exit 0
fi
```

## What this checks

### Section 1 — Backend liveness
Confirms `/health` returns `{"status":"ok"}` and prints the deployed version string.

### Section 2 — Route inventory & deployed-code drift
Calls the new `/debug/status` endpoint which returns the **live registered route list** from FastAPI's router. Compares against the expected consolidated set (`/industry-intel`, `/signals`, etc.). If routes are missing, the backend has not been redeployed after `main.py` was changed — this was the root cause of the 2026-04-09 outage.

Also shows:
- `industry_cache` doc count and freshness in hours
- Finnhub 429 rolling counter from the live instance
- Which `gcp3_cache` keys are currently live

### Section 3–4 — Rate-limit stall detection
Scans logs for `rate_limited_429`, `YFRateLimitError`, and `all_sources_failed` patterns. If **both** Finnhub and yfinance hit 429 simultaneously, `industry_cache` cannot be written and `/industry-returns` will serve stale data until the next successful refresh run. This is the exact failure mode seen 2026-04-09.

### Section 5 — Stale-data serving log check
Looks for `serving_stale_data` and `serving_prev_day_data` log entries emitted by `industry_returns.py` when it falls back to expired cache. These are now logged at WARNING level so they surface clearly.

### Section 6 — Refresh pipeline
Lists all `/refresh/all`, `/refresh/premarket`, `/refresh/intraday` run events with timestamps and durations. If no runs appear in the 2-hour window, the Cloud Scheduler job likely failed.

### Section 7 — Error summary
Deduplicates repeated error lines (collapses e.g. 50 identical "all sources failed" lines) to give a clean unique-error-type list.

### Section 8 — OpenAPI cross-check
Independently fetches `openapi.json` to verify which routes are registered. Detects old pre-consolidation routes (`/industry-tracker`, `/technical-signals`) that would indicate a partial or rolled-back deploy.

## Usage

```bash
# Default: last 2 hours (covers a morning refresh run)
/backend-debug

# Narrow window for a recent deploy
/backend-debug 30

# Wide window for overnight investigation
/backend-debug 480
```

## Vercel CLI — test a new backend deployment end-to-end

After deploying a new backend revision, use the Vercel CLI to verify the frontend
proxy layer picks up the new backend correctly without waiting for ISR to expire.

```bash
cd /Users/adamaslan/code/gcp3/frontend

# ── 1. Confirm which BACKEND_URL Vercel has on file ──────────────────────────
vercel env ls
# Expect: BACKEND_URL set for Production, Preview, and Development
# If missing from Production: vercel env add BACKEND_URL production

# ── 2. Pull current env vars to a local .env.vercel for inspection ────────────
vercel env pull .env.vercel --environment production
# Shows the decrypted BACKEND_URL value — confirm it matches the new Cloud Run URL:
grep BACKEND_URL .env.vercel
# Expected: https://gcp3-backend-cif7ppahzq-uc.a.run.app
rm .env.vercel   # never commit this file

# ── 3. List recent deployments — confirm the latest is Production ─────────────
vercel ls
# Look for Status=Ready, Environment=Production in the most recent row
# If latest is Preview only: vercel --prod to promote

# ── 4. Probe the production deployment's API proxy routes directly ────────────
PROD_URL="https://sectors.nuwrrrld.com"
for ROUTE in /api/industry-intel /api/signals /api/industry-returns /api/screener /api/market-overview /api/macro; do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$PROD_URL$ROUTE")
  UPDATED=$(curl -s --max-time 15 "$PROD_URL$ROUTE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('updated') or d.get('quotes_as_of') or d.get('date','?'))" 2>/dev/null)
  echo "  $ROUTE → $CODE  updated=$UPDATED"
done

# ── 5. Check Vercel function logs for proxy errors ────────────────────────────
vercel logs --limit 50
# Look for: 500 errors, "BACKEND_URL not set", "fetch failed", timeout messages
# If 500s appear after a backend URL change: update BACKEND_URL in Vercel and redeploy

# ── 6. If BACKEND_URL needs updating after a Cloud Run URL change ─────────────
NEW_URL="https://gcp3-backend-cif7ppahzq-uc.a.run.app"
vercel env rm BACKEND_URL production
echo "$NEW_URL" | vercel env add BACKEND_URL production
vercel --prod   # trigger a new production deployment to pick up the new value
```

### When to run this
- After every `gcloud builds submit` — confirm the new Cloud Run revision is reachable
  through the Vercel proxy, not just directly
- When timestamps are frozen — verify `BACKEND_URL` in Vercel still points to the right URL
- After a Cloud Run service recreate — the URL can change; Vercel won't know automatically
- Before opening a PR — confirm end-to-end data flow from Cloud Run → Vercel → browser

## Fix runbook

| Symptom | Section | Fix |
|---------|---------|-----|
| `/industry-intel` or `/signals` → 404 | 2, 8 | `cd backend && gcloud builds submit --config cloudbuild.yaml` |
| `/industry-returns` shows yesterday's date | 2, 4, 5 | Trigger `POST /refresh/all` or wait for next scheduler run; check Finnhub quota |
| Both Finnhub and yfinance 429 | 4 | Rate limits are IP-based and reset after ~1 min; next scheduler run should succeed |
| No refresh runs in window | 6 | Check `gcloud scheduler jobs describe gcp3-ai-summary-refresh --location us-central1` |
| Refresh runs but all scheduler jobs return code=2 | — | OIDC env vars missing — add `SCHEDULER_EXPECTED_AUDIENCE` + `SCHEDULER_EXPECTED_SA` to Cloud Run and `cloudbuild.yaml` |
| Refresh runs return 401 even with correct OIDC config | — | OIDC audience mismatch — verify `SCHEDULER_EXPECTED_AUDIENCE` matches the exact Cloud Run service URL (no trailing slash) |
| `seed_etf_history: failed for X: Too Many Requests` (all ETFs) | 4, 7 | yfinance per-ticker loop hits rate limit — `seed_etf_history` should use `yf.download(all_tickers)` batch instead of 54 individual `yf.Ticker` calls |
| `refresh/bake: aborted — checkpoint date=YESTERDAY expected=TODAY` | 6, 7 | The fetch phase failed or was skipped, so the checkpoint wasn't written. Fix fetch first, then re-trigger bake |
| stage error in refresh | 6 | Check section 7 for the specific exception; often a transient API timeout |
| Endpoint returns stale/empty data after code change | — | Three-layer cache: in-memory (60s) → Firestore `gcp3_cache` (1–6h) → frontend ISR. See lessons below |
| New backend code not taking effect after redeploy | — | Same-revision reuse keeps in-memory cache alive. Force flush: delete Firestore key + deploy again (new revision = clean memory) |

## Lessons learned

### Three-layer cache pipeline — applies to ALL endpoints
Every backend endpoint goes through the same cache stack. When debugging stale data on
any endpoint, check all three layers in order:

```
Layer 1: In-memory (firestore.py mem_get, 60s TTL)
  ↓ miss
Layer 2: Firestore gcp3_cache (keyed {module}:{date}, TTL 1–6h)
  ↓ miss
Layer 3: Live computation (Finnhub, yfinance, industry_cache, Gemini, etc.)
```

**Cache keys and TTLs by endpoint:**

| Endpoint | Firestore cache key pattern | TTL | Source collection |
|----------|---------------------------|-----|-------------------|
| `/signals` | `technical_signals:{symbol\|all}:YYYY-MM-DD` | 2h | `industry_cache` |
| `/industry-returns` | `industry_returns:YYYY-MM-DD` | 6h | `industry_cache` |
| `/industry-intel` | `industry_data:YYYY-MM-DD` | 6h | Finnhub + yfinance |
| `/screener` | `screener:YYYY-MM-DD` | 1h | Finnhub quotes |
| `/macro-pulse` | `macro_pulse:YYYY-MM-DD` | 2h | Finnhub macro |
| `/market-overview` | `ai_summary:YYYY-MM-DD` | 3h | Gemini API |

**To force-flush any endpoint after a code change:**
```python
# 1. Delete the Firestore cache key
from google.cloud import firestore
db = firestore.Client(project='ttb-lang1')
db.collection('gcp3_cache').document('KEY_FROM_TABLE_ABOVE').delete()
# 2. Deploy again so Cloud Run spins a new revision (fresh in-memory)
#    OR wait 60s for the old in-memory entry to expire
```

Deleting only the Firestore key is **not enough** — `get_cache()` checks in-memory first.
The 60s in-memory layer will keep serving the old value until it expires or the instance restarts.

### Silent data-source failure — endpoint returns empty data, no error
When an endpoint's upstream data source stops producing data (collection empty, external
pipeline not writing, API key expired), the endpoint returns a valid 200 with zero results.
No 503, no log warning. This happened with `/signals` (sourced from empty `analysis` collection)
and can happen with `/industry-returns` if `industry_cache` is never populated.

**Detection heuristic:** if `total` is unexpectedly low or zero, the data source is dry.
The endpoint won't tell you — you have to know what normal looks like:
- `/signals`: expect 54 ETFs, 150+ signals
- `/industry-returns`: expect 54 industries, 13 periods
- `/industry-intel`: expect 54 industries with quotes

**Prevention:** prefer data sources within the gcp3 scheduler pipeline (`industry_cache`,
`gcp3_cache`) over external collections (`analysis`) that depend on separate systems.

### Ticker staleness causes silent lookup failures
Defunct, renamed, or mistyped tickers produce no error — they just return empty results.
This affects any pipeline that looks up quotes or historical data by symbol.

**Known stale tickers (fixed 2026-04-10):** AMEDX (acquired), MAXR (taken private 2023),
MON (acquired 2018), NRZ→RITM (rebranded), ARMOUR→ARR (wrong ticker), LULUR→LULU,
RRL→RL (typos).

**Check before adding tickers:** validate against a live quote source. A ticker that silently
returns nothing will reduce signal count and skew rankings without any error trace.

### Scheduler jobs failing silently — code=2 despite correct HTTP method
Cloud Scheduler jobs can fail with `code=2` even when method=POST and the URL is correct.
The most common hidden cause: `SCHEDULER_EXPECTED_AUDIENCE` and `SCHEDULER_EXPECTED_SA`
env vars are missing from Cloud Run. Without them, `_verify_scheduler` has no audience to
validate the OIDC Bearer token against and returns 401 for every job invocation.

**Symptoms:** all scheduler jobs show `code=2` in `gcloud scheduler jobs list`; backend logs
show `POST /refresh/fetch 401 Unauthorized` at the scheduled times; data timestamps stay
frozen at the last successful run (often midnight UTC from a prior day).

**Fix:** add both vars to Cloud Run and `cloudbuild.yaml`:
```bash
gcloud run services update gcp3-backend --region us-central1 \
  --update-env-vars="SCHEDULER_EXPECTED_AUDIENCE=https://gcp3-backend-cif7ppahzq-uc.a.run.app,\
SCHEDULER_EXPECTED_SA=gcp3-scheduler@ttb-lang1.iam.gserviceaccount.com"
```
Then add them to `cloudbuild.yaml` `--set-env-vars` so they survive the next deploy.

**Verify fix:** `gcloud scheduler jobs run gcp3-premarket-warmup --location us-central1`,
then check logs for `POST /refresh/fetch started` (not 401).

### yfinance per-ticker loop saturates rate limit across the entire pipeline
When `seed_etf_history` (or any function) calls `yf.Ticker(symbol).history()` in a loop
over 50+ symbols, yfinance rate-limits after ~5 requests. Every subsequent call fails with
`Too Many Requests`, returning 0 rows. The fetch phase reports `status=fetch_ok` anyway
(stage F0 errors are non-fatal), so the pipeline appears healthy while writing no history.

**Symptom:** logs show `seed_etf_history: failed for X: Too Many Requests` for nearly all
54 ETFs, followed by `seed_etf_history complete: 54 ETFs, 0 total rows`.

**Fix:** replace per-ticker calls with a single `yf.download(all_tickers_list)` batch call.
One HTTP request fetches all 54 tickers at once, avoiding rate limiting entirely.
Run the batch in an executor (`asyncio.get_event_loop().run_in_executor`) since yf.download
is synchronous.

**General rule:** any yfinance loop over more than ~5 symbols should use `yf.download`
instead of individual `yf.Ticker` calls.

### Fetch–bake checkpoint prevents bake from running on stale fetch data
`/refresh/bake` checks a `refresh_state:fetch` checkpoint in Firestore. If the checkpoint
date is not today's date, bake aborts with:
```
refresh/bake: aborted — checkpoint date=YYYY-MM-DD expected=YYYY-MM-DD
```
This protects against baking Gemini summaries from yesterday's data. But it also means
a failed fetch silently blocks bake indefinitely — the user sees data from the last
successful fetch+bake cycle, which may be multiple days old.

**Debugging order when data is stale:**
1. Check the checkpoint: `gcloud firestore get --project ttb-lang1 'refresh_state/fetch'`
2. If checkpoint date < today → fetch failed or was skipped; fix fetch first
3. Re-trigger fetch: `gcloud scheduler jobs run gcp3-premarket-warmup --location us-central1`
4. Confirm fetch succeeded: look for `POST /refresh/fetch complete status=fetch_ok`
5. Re-trigger bake: `gcloud scheduler jobs run gcp3-ai-summary-refresh --location us-central1`
