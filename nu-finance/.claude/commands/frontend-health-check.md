# Frontend Health Check

Comprehensive health check verifying the full request path: Vercel frontend (nuwrrrld.com), Cloud Run backend, Clerk auth, frontend-to-backend proxy connectivity, individual API routes, fetch/bake pipeline checkpoints, Cloud Scheduler jobs, and response times.

## Run

```bash
PROJECT="${GCP_PROJECT_ID:-ttb-lang1}"
REGION="us-central1"
SERVICE="gcp3-backend"
FRONTEND="https://sectors.nuwrrrld.com"

PASS=0
FAIL=0
WARN=0
pass() { PASS=$((PASS + 1)); echo "  ✅ $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  ❌ $1"; }
warn() { WARN=$((WARN + 1)); echo "  ⚠️  $1"; }

# ─── 1. Backend health ───────────────────────────────────────────────
echo "=== 1. Backend health ==="
BACKEND_URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT" --format="value(status.url)" 2>/dev/null)
if [ -z "$BACKEND_URL" ]; then
  fail "Cloud Run service $SERVICE not found in $REGION"
  echo "Cannot continue without backend URL."
  echo ""; echo "=== RESULTS: $PASS passed, $FAIL failed, $WARN warnings ==="; exit 1
fi
echo "  Backend URL: $BACKEND_URL"

BACKEND_START=$(python3 -c 'import time; print(time.time())')
BACKEND_HEALTH=$(curl -s --max-time 15 "$BACKEND_URL/health" 2>/dev/null)
BACKEND_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$BACKEND_URL/health" 2>/dev/null)
BACKEND_END=$(python3 -c 'import time; print(time.time())')
BACKEND_MS=$(python3 -c "print(int(($BACKEND_END - $BACKEND_START) * 500))")

if [ "$BACKEND_CODE" = "200" ] && echo "$BACKEND_HEALTH" | python3 -c 'import sys,json; d=json.load(sys.stdin); assert d.get("status") == "ok"' 2>/dev/null; then
  BACKEND_VERSION=$(echo "$BACKEND_HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('version','?'))" 2>/dev/null)
  BACKEND_TOOLS=$(echo "$BACKEND_HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tools','?'))" 2>/dev/null)
  pass "Backend /health OK v${BACKEND_VERSION} tools=${BACKEND_TOOLS} (HTTP $BACKEND_CODE, ${BACKEND_MS}ms)"
  if [ "$BACKEND_MS" -gt 5000 ]; then
    warn "Backend response slow (${BACKEND_MS}ms > 5000ms) — possible cold start"
  fi
else
  fail "Backend /health failed (HTTP $BACKEND_CODE)"
  echo "  Response: $BACKEND_HEALTH"
fi

# ─── 2. Backend data endpoints (spot check + latency + body validation) ──
echo ""
echo "=== 2. Backend data endpoints (8 consolidated + 1 internal) ==="
for ENDPOINT in /screener /industry-returns /market-overview /signals /macro-pulse /content /industry-intel /earnings-radar; do
  EP_START=$(python3 -c 'import time; print(time.time())')
  EP_BODY=$(curl -s --max-time 15 "$BACKEND_URL$ENDPOINT" 2>/dev/null)
  EP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$BACKEND_URL$ENDPOINT" 2>/dev/null)
  EP_END=$(python3 -c 'import time; print(time.time())')
  EP_MS=$(python3 -c "print(int(($EP_END - $EP_START) * 500))")

  EP_VALID=$(echo "$EP_BODY" | python3 -c "
import sys, json
try:
  d = json.load(sys.stdin)
  if isinstance(d, dict) and len(d) > 0 and 'error' not in d:
    print('ok')
  else:
    print('bad')
except Exception:
  print('bad')
" 2>/dev/null)

  if [ "$EP_CODE" = "200" ] && [ "$EP_VALID" = "ok" ]; then
    MSG="Backend $ENDPOINT (HTTP $EP_CODE, ${EP_MS}ms)"
    pass "$MSG"
    if [ "$EP_MS" -gt 8000 ]; then
      warn "$ENDPOINT response slow (${EP_MS}ms) — may be cache miss"
    fi
  elif [ "$EP_CODE" = "200" ] && [ "$EP_VALID" = "bad" ]; then
    warn "Backend $ENDPOINT returned 200 but body is empty or contains error field"
    echo "    Body snippet: $(echo "$EP_BODY" | head -c 120)"
  elif [ "$EP_CODE" = "503" ]; then
    warn "Backend $ENDPOINT returned 503 — upstream data source may be unavailable"
  else
    fail "Backend $ENDPOINT (HTTP $EP_CODE)"
  fi
done

# ─── 2b. Backend debug/status endpoint ──────────────────────────────
echo ""
echo "=== 2b. Backend debug/status ==="
DEBUG_BODY=$(curl -s --max-time 15 "$BACKEND_URL/debug/status" 2>/dev/null)
DEBUG_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$BACKEND_URL/debug/status" 2>/dev/null)
if [ "$DEBUG_CODE" = "200" ]; then
  MISSING_ROUTES=$(echo "$DEBUG_BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d.get('missing_expected_routes',[]); print(','.join(m) if m else 'none')" 2>/dev/null)
  IC_STALE=$(echo "$DEBUG_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('industry_cache',{}).get('stale',False))" 2>/dev/null)
  IC_FRESHNESS=$(echo "$DEBUG_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('industry_cache',{}).get('freshness_hours','?'))" 2>/dev/null)
  LIVE_CACHE=$(echo "$DEBUG_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('gcp3_cache',{}).get('live_doc_count',0))" 2>/dev/null)

  if [ "$MISSING_ROUTES" = "none" ]; then
    pass "All expected routes registered"
  else
    fail "Missing routes: $MISSING_ROUTES"
  fi

  if [ "$IC_STALE" = "True" ]; then
    warn "industry_cache is stale (${IC_FRESHNESS}h old)"
  else
    pass "industry_cache fresh (${IC_FRESHNESS}h old)"
  fi

  if [ "$LIVE_CACHE" -gt 0 ] 2>/dev/null; then
    pass "gcp3_cache has $LIVE_CACHE live docs"
  else
    warn "gcp3_cache has 0 live docs — pipeline may not have run today"
  fi
else
  warn "Backend /debug/status unavailable (HTTP $DEBUG_CODE)"
fi

# ─── 3. Frontend root + pages ────────────────────────────────────────
echo ""
echo "=== 3. Frontend pages (8 routes) ==="

FRONTEND_START=$(python3 -c 'import time; print(time.time())')
FRONTEND_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$FRONTEND" 2>/dev/null)
FRONTEND_END=$(python3 -c 'import time; print(time.time())')
FRONTEND_MS=$(python3 -c "print(int(($FRONTEND_END - $FRONTEND_START) * 1000))")

if [ "$FRONTEND_CODE" = "200" ]; then
  pass "Frontend root / (HTTP $FRONTEND_CODE, ${FRONTEND_MS}ms)"
else
  fail "Frontend root / (HTTP $FRONTEND_CODE)"
fi

for PAGE in /market-overview /industry-intel /industry-returns /signals /screener /macro /content; do
  PAGE_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$FRONTEND$PAGE" 2>/dev/null)
  if [ "$PAGE_CODE" = "200" ]; then
    pass "Frontend $PAGE (HTTP $PAGE_CODE)"
  else
    fail "Frontend $PAGE (HTTP $PAGE_CODE)"
  fi
done

# ─── 3b. Clerk auth integration ─────────────────────────────────────
echo ""
echo "=== 3b. Clerk auth ==="
CLERK_JS=$(curl -s --max-time 15 "$FRONTEND" 2>/dev/null | grep -o 'clerk' | head -1)
if [ -n "$CLERK_JS" ]; then
  pass "Clerk JS detected in frontend HTML"
else
  warn "Clerk JS not found in initial HTML — may load async or SSR-only"
fi

# Check ClerkProvider is wrapping the app (layout.tsx imports)
CLERK_LAYOUT=$(grep -c "ClerkProvider" /Users/adamaslan/code/gcp3/frontend/src/app/layout.tsx 2>/dev/null)
if [ "$CLERK_LAYOUT" -gt 0 ]; then
  pass "ClerkProvider wraps app in layout.tsx"
else
  fail "ClerkProvider missing from layout.tsx — auth will not work"
fi

# Verify Clerk env vars exist in .env* or vercel config
CLERK_ENV_LOCAL=$(grep -l "NEXT_PUBLIC_CLERK" /Users/adamaslan/code/gcp3/frontend/.env* 2>/dev/null | head -1)
if [ -n "$CLERK_ENV_LOCAL" ]; then
  pass "Clerk env vars found in $(basename $CLERK_ENV_LOCAL)"
else
  warn "No NEXT_PUBLIC_CLERK_* env vars in local .env files — must be set in Vercel"
fi

# ─── 4. Frontend API proxy routes ────────────────────────────────────
echo ""
echo "=== 4. Frontend API proxy routes (frontend → backend) ==="
for ROUTE in /api/screener /api/industry-returns /api/market-overview /api/signals /api/macro /api/content /api/industry-intel; do
  PROXY_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 "$FRONTEND$ROUTE" 2>/dev/null)
  if [ "$PROXY_CODE" = "200" ]; then
    pass "Proxy $ROUTE (HTTP $PROXY_CODE)"
  elif [ "$PROXY_CODE" = "503" ]; then
    warn "Proxy $ROUTE returned 503 — backend data source may be down"
  else
    fail "Proxy $ROUTE (HTTP $PROXY_CODE) — frontend cannot reach backend via this route"
  fi
done

# ─── 5. SSL/TLS verification ─────────────────────────────────────────
echo ""
echo "=== 5. SSL/TLS ==="
for URL_LABEL in "Frontend:$FRONTEND" "Backend:$BACKEND_URL"; do
  LABEL="${URL_LABEL%%:*}"
  CHECK_URL="${URL_LABEL#*:}"
  DOMAIN=$(echo "$CHECK_URL" | sed 's|https://||' | sed 's|/.*||')
  EXPIRY=$(echo | openssl s_client -servername "$DOMAIN" -connect "$DOMAIN:443" 2>/dev/null | openssl x509 -noout -enddate 2>/dev/null | cut -d= -f2)
  if [ -n "$EXPIRY" ]; then
    EXPIRY_EPOCH=$(date -j -f "%b %d %T %Y %Z" "$EXPIRY" "+%s" 2>/dev/null || date -d "$EXPIRY" "+%s" 2>/dev/null)
    NOW_EPOCH=$(date "+%s")
    if [ -n "$EXPIRY_EPOCH" ]; then
      DAYS_LEFT=$(( (EXPIRY_EPOCH - NOW_EPOCH) / 86400 ))
      if [ "$DAYS_LEFT" -lt 7 ]; then
        fail "$LABEL SSL cert expires in ${DAYS_LEFT} days ($EXPIRY)"
      elif [ "$DAYS_LEFT" -lt 30 ]; then
        warn "$LABEL SSL cert expires in ${DAYS_LEFT} days ($EXPIRY)"
      else
        pass "$LABEL SSL cert valid (${DAYS_LEFT} days remaining)"
      fi
    else
      warn "$LABEL SSL cert expiry could not be parsed"
    fi
  else
    fail "$LABEL SSL cert could not be retrieved for $DOMAIN"
  fi
done

# ─── 6. Frontend ↔ Backend sync ──────────────────────────────────────
echo ""
echo "=== 6. Frontend ↔ Backend sync ==="
if [ "$FRONTEND_CODE" = "200" ] && [ "$BACKEND_CODE" = "200" ]; then
  pass "Both frontend and backend are reachable"
else
  fail "Frontend (HTTP $FRONTEND_CODE) and/or backend (HTTP $BACKEND_CODE) are down"
fi

PROXY_SAMPLE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$FRONTEND/api/screener" 2>/dev/null)
if [ "$PROXY_SAMPLE" = "200" ] || [ "$PROXY_SAMPLE" = "503" ]; then
  pass "Vercel BACKEND_URL appears correctly configured (proxy returns data or 503)"
else
  fail "Vercel BACKEND_URL may be misconfigured — proxy /api/screener returned HTTP $PROXY_SAMPLE"
fi

# ─── 7. Fetch/Bake pipeline checkpoints ─────────────────────────────
echo ""
echo "=== 7. Fetch/Bake pipeline checkpoints ==="
if [ -n "$BACKEND_URL" ]; then
  for PHASE in fetch bake; do
    CP_BODY=$(curl -s --max-time 10 "$BACKEND_URL/debug/status" 2>/dev/null)
    # Checkpoints are in Firestore — read via the gcp3_cache live keys
    CP_KEY=$(echo "$CP_BODY" | python3 -c "
import sys, json
d = json.load(sys.stdin)
keys = d.get('gcp3_cache', {}).get('live_keys', [])
matches = [k for k in keys if 'refresh_state:$PHASE' in k]
print(matches[0] if matches else '')
" 2>/dev/null)
    if [ -n "$CP_KEY" ]; then
      pass "$PHASE checkpoint exists in gcp3_cache ($CP_KEY)"
    else
      warn "No $PHASE checkpoint found in live gcp3_cache — pipeline may not have run today"
    fi
  done
fi

# ─── 8. Cloud Scheduler jobs ─────────────────────────────────────────
echo ""
echo "=== 8. Cloud Scheduler jobs ==="
SCHEDULER_URI_MISMATCHES=""
SCHEDULER_JOBS="gcp3-premarket-warmup gcp3-ai-summary-refresh gcp3-midday-intraday-refresh gcp3-eod-intraday-refresh gcp3-nightly-cache-purge"

for JOB in $SCHEDULER_JOBS; do
  JOB_INFO=$(gcloud scheduler jobs describe "$JOB" --location "$REGION" --project "$PROJECT" --format="value(schedule,latestAttemptTime,httpTarget.uri,state)" 2>/dev/null)
  SCHEDULE=$(echo "$JOB_INFO" | cut -f1)
  LAST_RUN=$(echo "$JOB_INFO" | cut -f2)
  JOB_URI=$(echo "$JOB_INFO" | cut -f3)
  JOB_STATE=$(echo "$JOB_INFO" | cut -f4)

  JOB_OIDC=$(gcloud scheduler jobs describe "$JOB" --location "$REGION" --project "$PROJECT" --format="value(httpTarget.oidcToken.serviceAccountEmail)" 2>/dev/null)

  printf "\n  Job: %s\n" "$JOB"
  printf "    Schedule : %s\n" "${SCHEDULE:-unknown}"
  printf "    URI      : %s\n" "${JOB_URI:-unknown}"
  printf "    State    : %s\n" "${JOB_STATE:-unknown}"
  printf "    Last run : %s\n" "${LAST_RUN:-never}"
  printf "    OIDC SA  : %s\n" "${JOB_OIDC:-NONE — shared secret only}"

  if [ "$JOB_STATE" != "ENABLED" ] && [ -n "$JOB_STATE" ]; then
    warn "$JOB is $JOB_STATE (not ENABLED)"
  fi

  if [ -z "$JOB_OIDC" ]; then
    warn "$JOB has no OIDC service account — using shared secret auth only"
  elif echo "$JOB_OIDC" | grep -q "gcp3-scheduler"; then
    pass "$JOB OIDC auth configured (${JOB_OIDC})"
  else
    warn "$JOB OIDC SA is unexpected: $JOB_OIDC"
  fi

  if [ -n "$JOB_URI" ] && ! echo "$JOB_URI" | grep -q "$BACKEND_URL"; then
    SCHEDULER_URI_MISMATCHES="$SCHEDULER_URI_MISMATCHES $JOB"
  fi

  python3 -c "
from datetime import datetime, timezone, timedelta
import sys
job, last_run = '$JOB', '$LAST_RUN'
if not last_run:
    print('    ❌ No latestAttemptTime available')
    sys.exit(0)
try:
    last_dt = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
except ValueError:
    print('    ❌ Unable to parse latestAttemptTime')
    sys.exit(0)
now = datetime.now(timezone.utc)
threshold_hours = 30
if 'midday' in job or 'eod' in job:
    threshold_hours = 20
elif 'premarket' in job or 'ai-summary' in job:
    threshold_hours = 36
elif 'nightly' in job:
    threshold_hours = 40
delta = now - last_dt
hours_ago = delta.total_seconds() / 3600
if delta > timedelta(hours=threshold_hours):
    print(f'    ❌ Last run {hours_ago:.1f}h ago (threshold: {threshold_hours}h)')
else:
    print(f'    ✅ Last run {hours_ago:.1f}h ago (within {threshold_hours}h window)')
"
done

# ─── 9. Scheduler URI synchronization ────────────────────────────────
echo ""
echo "=== 9. Scheduler URI synchronization ==="
if [ -z "$SCHEDULER_URI_MISMATCHES" ]; then
  pass "All scheduler jobs point to current backend URL"
else
  fail "These jobs do NOT point to $BACKEND_URL:$SCHEDULER_URI_MISMATCHES"
fi

# ─── 10. Recent Cloud Run error logs (last 30 min) ────────────────────
echo ""
echo "=== 10. Recent backend errors (last 30 min) ==="
RECENT_ERRORS=$(gcloud run services logs read "$SERVICE" \
  --region "$REGION" --project "$PROJECT" \
  --limit 200 --format="value(timestamp,textPayload,jsonPayload.message)" 2>/dev/null | \
  python3 -c "
import sys, re
from datetime import datetime, timezone, timedelta

cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
errors = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
    if m:
        try:
            ts = datetime.fromisoformat(m.group(1) + '+00:00')
            if ts >= cutoff and ('ERROR' in line or 'Exception' in line or 'failed' in line.lower() or '401' in line or '503' in line):
                errors.append(line[:180])
        except Exception:
            pass
if errors:
    for e in errors[-5:]:
        print(' ', e)
else:
    print('  none')
" 2>/dev/null)

if [ "$RECENT_ERRORS" = "  none" ] || [ -z "$RECENT_ERRORS" ]; then
  pass "No ERROR/Exception/401/503 logs in last 30 min"
else
  FAIL=$((FAIL + 1))
  echo "  ❌ Recent errors found:"
  echo "$RECENT_ERRORS"
fi

# ─── 11. Summary ─────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  RESULTS: $PASS passed, $FAIL failed, $WARN warnings"
echo "════════════════════════════════════════════"
if [ "$FAIL" -gt 0 ]; then
  echo "  Status: UNHEALTHY"
  exit 1
elif [ "$WARN" -gt 0 ]; then
  echo "  Status: DEGRADED"
  exit 0
else
  echo "  Status: HEALTHY"
  exit 0
fi
```

## What this checks

### Core connectivity (sections 1-2b)
- Backend `/health` responds with `{"status": "ok"}`, version, and tool count
- All 8 backend data endpoints return 200 with valid non-empty JSON bodies and per-endpoint latency
- `/debug/status` validates route inventory, industry_cache freshness, and gcp3_cache live doc count
- Warns if any endpoint body contains an `error` field or is empty (200 with bad body)
- Warns if any endpoint takes >8s (likely a cache miss)

### Frontend & Clerk auth (sections 3-3b)
- Frontend root and all 7 page routes return 200 (market-overview, industry-intel, industry-returns, signals, screener, macro, content)
- Clerk JS presence in frontend HTML
- ClerkProvider wraps the app in layout.tsx
- Clerk env vars present locally or flagged as Vercel-only

### End-to-end proxy (section 4)
- All 7 frontend `/api/*` proxy routes successfully reach the backend
- This catches misconfigured `BACKEND_URL` in Vercel

### Security (section 5)
- SSL/TLS certificate validity for both frontend and backend domains
- Warns at 30 days, fails at 7 days before expiry

### Pipeline & synchronization (sections 6-9)
- Frontend and backend both reachable simultaneously
- Vercel `BACKEND_URL` env var is correctly configured
- Fetch/Bake pipeline checkpoints exist in gcp3_cache (confirms today's pipeline ran)
- All 5 Cloud Scheduler jobs are ENABLED, have recent runs, and point to the current backend URL
- Each job checked against its own freshness threshold (20-40h depending on schedule)
- Each job's OIDC service account email verified (`gcp3-scheduler@...`)

### Recent error log scan (section 10)
- Tails last 30 min of Cloud Run logs for ERROR, Exception, 401, 503, or "failed" entries
- Surfaces the 5 most recent matching lines inline — no need to open Cloud Console

### Summary (section 11)
- Aggregated pass/fail/warn counts
- Exit code 1 if any failures, 0 otherwise (usable in CI)

## If the frontend is not working

- Confirm the backend `/health` endpoint is healthy first.
- If backend reports `ok` but Vercel returns `4xx/5xx`, verify `BACKEND_URL` in Vercel is correct and redeploy.
- If the backend is unhealthy, inspect Cloud Run logs with `gcloud run services logs read gcp3-backend --region us-central1 --limit 50`.
- If proxy routes fail but backend endpoints work directly, the Vercel `BACKEND_URL` is likely wrong — check it in the Vercel dashboard.
- If Clerk auth is broken, check `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` and `CLERK_SECRET_KEY` in Vercel env vars.
- If SSL warnings appear, check Vercel (auto-renews) or Cloud Run (managed by GCP) certificate settings.
- If pipeline checkpoints are missing, run `/refresh/fetch` then `/refresh/bake` manually via the scheduler secret.
- If scheduler jobs are stale or missing, check their schedule and target URI:
  - `gcloud scheduler jobs describe <job> --location=us-central1 --project=$PROJECT`
  - `gcloud scheduler jobs run <job> --location=us-central1 --project=$PROJECT`
- If scheduler job URIs do not match the current backend URL, update the job or redeploy the backend with the correct Cloud Run URL.
- If a scheduler job shows no OIDC SA, it is using shared-secret auth only — migrate it by running the OIDC setup commands in `backend-change-rules.md`.
- If the recent error log scan shows failures, run `gcloud run services logs read gcp3-backend --region us-central1 --limit 50` for full context.
- When fixing pipeline issues, use the existing `/health-check` and `/post-deploy-verify` commands for follow-up validation.
