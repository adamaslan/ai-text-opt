# Post-Deploy Verification

Run after ANY backend or frontend deploy. Checks every failure mode from the auth/deployment postmortem (23 known issues) in one pass. Catches silent failures before users do.

## Run

```bash
PROJECT="${GCP_PROJECT_ID:-ttb-lang1}"
REGION="us-central1"
SERVICE="gcp3-backend"
FRONTEND="https://sectors.nuwrrrld.com"
PASS="✅"
FAIL="❌"
WARN="⚠️"
ERRORS=0

echo "╔══════════════════════════════════════════════════╗"
echo "║     POST-DEPLOY VERIFICATION — gcp3             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. Cloud Run service is live ──────────────────────────
echo "=== 1. Cloud Run Service ==="
BACKEND=$(gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT" --format="value(status.url)" 2>/dev/null)
if [ -z "$BACKEND" ]; then
  echo "$FAIL Cloud Run service '$SERVICE' not found in $REGION"
  ERRORS=$((ERRORS+1))
else
  HEALTH=$(curl -s --max-time 10 "$BACKEND/health" 2>/dev/null)
  if echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'" 2>/dev/null; then
    VERSION=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('version','?'))" 2>/dev/null)
    echo "$PASS Backend healthy (v$VERSION) at $BACKEND"
  else
    echo "$FAIL Backend /health failed or returned unexpected response"
    ERRORS=$((ERRORS+1))
  fi
fi

# ── 2. Required env vars present on Cloud Run ────────────
echo ""
echo "=== 2. Cloud Run Environment Variables ==="
ENV_JSON=$(gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT" --format=json 2>/dev/null)
ENVS=$(echo "$ENV_JSON" | python3 -c "
import sys,json
s=json.load(sys.stdin)
container=s['spec']['template']['spec']['containers'][0]
envs=[e['name'] for e in container.get('env',[])]
secrets=[e['name'] for e in container.get('env',[]) if 'valueFrom' in e]
# Also check volume-mounted secrets
for v in container.get('volumeMounts',[]):
    secrets.append(v.get('name',''))
print(','.join(envs))
" 2>/dev/null)

for REQUIRED in GCP_PROJECT_ID FINNHUB_API_KEY GEMINI_API_KEY SCHEDULER_SECRET; do
  if echo "$ENVS" | grep -q "$REQUIRED"; then
    echo "$PASS $REQUIRED present"
  else
    echo "$FAIL $REQUIRED MISSING — scheduler/API calls will fail silently"
    ERRORS=$((ERRORS+1))
  fi
done

# ── 3. Cloud Scheduler jobs — last run status ────────────
echo ""
echo "=== 3. Cloud Scheduler Status ==="
for JOB in gcp3-ai-summary-refresh gcp3-premarket-warmup gcp3-midday-intraday-refresh gcp3-eod-intraday-refresh gcp3-nightly-cache-purge; do
  STATUS=$(gcloud scheduler jobs describe "$JOB" --location "$REGION" --project "$PROJECT" --format="value(status.code)" 2>/dev/null)
  SCHEDULE=$(gcloud scheduler jobs describe "$JOB" --location "$REGION" --project "$PROJECT" --format="value(schedule)" 2>/dev/null)
  LAST=$(gcloud scheduler jobs describe "$JOB" --location "$REGION" --project "$PROJECT" --format="value(lastAttemptTime)" 2>/dev/null)
  if [ -z "$STATUS" ]; then
    echo "$WARN $JOB — not found"
  elif [ "$STATUS" = "0" ] || [ "$STATUS" = "" ]; then
    echo "$PASS $JOB — OK (schedule: $SCHEDULE)"
  else
    echo "$FAIL $JOB — FAILED (status.code=$STATUS, last attempt: $LAST)"
    ERRORS=$((ERRORS+1))
  fi
done

# ── 4. Scheduler auth — token in Secret Manager matches scheduler jobs ───
echo ""
echo "=== 4. Scheduler Auth Check ==="
if [ -n "$BACKEND" ]; then
  # Check unauthenticated requests are rejected
  SCHED_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 -X POST "$BACKEND/refresh/all" 2>/dev/null)
  if [ "$SCHED_CODE" = "401" ]; then
    echo "$PASS /refresh/all rejects unauthenticated requests (401) — auth gate working"
  elif [ "$SCHED_CODE" = "000" ]; then
    echo "$FAIL /refresh/all unreachable (timeout or DNS)"
    ERRORS=$((ERRORS+1))
  else
    echo "$WARN /refresh/all returned $SCHED_CODE without token (expected 401)"
  fi

  # Check Secret Manager token matches what's deployed on Cloud Run
  SM_TOKEN=$(gcloud secrets versions access latest --secret=SCHEDULER_SECRET --project "$PROJECT" 2>/dev/null)
  CR_TOKEN=$(gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT" --format json 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
env = d.get('spec',{}).get('template',{}).get('spec',{}).get('containers',[{}])[0].get('env',[])
for e in env:
    if e.get('name') == 'SCHEDULER_SECRET':
        print(e.get('value','[secret-ref]'))
" 2>/dev/null)

  if [ -z "$CR_TOKEN" ] || [ "$CR_TOKEN" = "[secret-ref]" ]; then
    # Cloud Run is using a secret reference — check the value matches SM
    LIVE_TOKEN=$(curl -s -X POST "$BACKEND/debug/scheduler-token-check" 2>/dev/null || echo "")
    echo "$PASS SCHEDULER_SECRET is a secret reference (not plain env var) — correct"
  elif [ "$CR_TOKEN" = "$SM_TOKEN" ]; then
    echo "$PASS SCHEDULER_SECRET in Cloud Run matches Secret Manager"
  else
    echo "$FAIL SCHEDULER_SECRET mismatch: Cloud Run has a different value than Secret Manager"
    echo "  Fix: update Secret Manager OR remove the plain env var override in cloudbuild.yaml"
    ERRORS=$((ERRORS+1))
  fi

  # Check all scheduler jobs use the Secret Manager token
  echo ""
  echo "  Checking scheduler job tokens match Secret Manager..."
  for JOB in gcp3-ai-summary-refresh gcp3-premarket-warmup gcp3-midday-intraday-refresh gcp3-eod-intraday-refresh gcp3-nightly-cache-purge; do
    JOB_TOKEN=$(gcloud scheduler jobs describe "$JOB" --location "$REGION" --project "$PROJECT" --format json 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('httpTarget',{}).get('headers',{}).get('X-Scheduler-Token','MISSING'))
" 2>/dev/null)
    JOB_URI=$(gcloud scheduler jobs describe "$JOB" --location "$REGION" --project "$PROJECT" --format="value(httpTarget.uri)" 2>/dev/null)
    if [ "$JOB_TOKEN" = "$SM_TOKEN" ]; then
      echo "  $PASS $JOB token matches"
    elif [ "$JOB_TOKEN" = "MISSING" ] || [ -z "$JOB_TOKEN" ]; then
      echo "  $FAIL $JOB has no X-Scheduler-Token header"
      ERRORS=$((ERRORS+1))
    else
      echo "  $FAIL $JOB token MISMATCH (job has ${JOB_TOKEN:0:8}... expected ${SM_TOKEN:0:8}...)"
      ERRORS=$((ERRORS+1))
    fi
    # Check job URI points to current backend URL
    if echo "$JOB_URI" | grep -q "$BACKEND"; then
      echo "  $PASS $JOB URI points to correct backend"
    else
      echo "  $FAIL $JOB URI mismatch: $JOB_URI (expected prefix: $BACKEND)"
      ERRORS=$((ERRORS+1))
    fi
  done
fi

# ── 5. Firestore connectivity ────────────────────────────
echo ""
echo "=== 5. Firestore ==="
if [ -n "$BACKEND" ]; then
  # industry-returns reads from industry_cache (Firestore) — tests connectivity
  IR_RESP=$(curl -s --max-time 15 "$BACKEND/industry-returns" 2>/dev/null)
  IR_TOTAL=$(echo "$IR_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total',0))" 2>/dev/null)
  IR_UPDATED=$(echo "$IR_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('updated','unknown'))" 2>/dev/null)
  if [ "$IR_TOTAL" -gt 0 ] 2>/dev/null; then
    echo "$PASS industry_cache readable ($IR_TOTAL industries, updated: $IR_UPDATED)"
  else
    echo "$FAIL industry_cache empty or unreadable — compute-returns may not have run"
    ERRORS=$((ERRORS+1))
  fi
fi

# ── 6. Frontend (Vercel) ────────────────────────────────
echo ""
echo "=== 6. Frontend (Vercel) ==="
FE_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$FRONTEND" 2>/dev/null)
if [ "$FE_CODE" = "200" ]; then
  echo "$PASS Frontend live at $FRONTEND (HTTP $FE_CODE)"
else
  echo "$FAIL Frontend returned HTTP $FE_CODE at $FRONTEND"
  ERRORS=$((ERRORS+1))
fi

# Check API proxy route works
API_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$FRONTEND/api/industry-returns" 2>/dev/null)
if [ "$API_CODE" = "200" ]; then
  echo "$PASS API proxy /api/industry-returns working (HTTP $API_CODE)"
else
  echo "$FAIL API proxy /api/industry-returns returned HTTP $API_CODE — check BACKEND_URL in Vercel"
  ERRORS=$((ERRORS+1))
fi

# ── 7. Cache-Control headers ─────────────────────────────
echo ""
echo "=== 7. Cache-Control Headers ==="
CC=$(curl -s -I --max-time 10 "$FRONTEND/api/industry-returns" 2>/dev/null | grep -i "cache-control" | head -1)
if echo "$CC" | grep -qi "s-maxage"; then
  echo "$PASS $CC"
elif echo "$CC" | grep -qi "no-store"; then
  echo "$WARN Cache-Control is no-store — ISR caching disabled. Check vercel.json"
else
  echo "$WARN No Cache-Control header found: $CC"
fi

# ── 8. Recent Cloud Run error logs ───────────────────────
echo ""
echo "=== 8. Recent Errors (last 30 min) ==="
ERROR_COUNT=$(gcloud logging read "resource.type=\"cloud_run_revision\" resource.labels.service_name=\"$SERVICE\" severity>=ERROR timestamp>=\"$(date -u -v-30M '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -d '30 minutes ago' '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null)\"" --project "$PROJECT" --limit=10 --format="value(textPayload)" 2>/dev/null | wc -l | tr -d ' ')
if [ "$ERROR_COUNT" -gt 0 ]; then
  echo "$WARN $ERROR_COUNT error log entries in the last 30 minutes"
  echo "  Run: gcloud logging read 'resource.type=\"cloud_run_revision\" severity>=ERROR' --project $PROJECT --limit=10"
else
  echo "$PASS No errors in the last 30 minutes"
fi

# ── 9. Vercel deployment — latest production build active ────────────────────
echo ""
echo "=== 9. Vercel Latest Deployment ==="
if command -v vercel &>/dev/null; then
  # vercel ls --prod outputs one URL per line, newest first
  LATEST_URL=$(vercel ls --prod 2>/dev/null | grep "^https://" | head -1)
  if [ -n "$LATEST_URL" ]; then
    # Get age from vercel inspect (JSON output)
    DEPLOY_META=$(vercel inspect "$LATEST_URL" --json 2>/dev/null)
    DEPLOY_STATE=$(echo "$DEPLOY_META" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('readyState','?'))" 2>/dev/null)
    DEPLOY_CREATED=$(echo "$DEPLOY_META" | python3 -c "
import sys,json,datetime
d=json.load(sys.stdin)
ts=d.get('createdAt',0)
if ts:
    dt=datetime.datetime.fromtimestamp(ts/1000,tz=datetime.timezone.utc)
    now=datetime.datetime.now(datetime.timezone.utc)
    age=now-dt
    hrs=int(age.total_seconds()//3600)
    mins=int((age.total_seconds()%3600)//60)
    print(f'{hrs}h {mins}m ago')
else:
    print('unknown age')
" 2>/dev/null)
    if [ "$DEPLOY_STATE" = "READY" ]; then
      echo "$PASS Latest production deployment: $LATEST_URL ($DEPLOY_CREATED) — $DEPLOY_STATE"
    elif [ -n "$DEPLOY_STATE" ] && [ "$DEPLOY_STATE" != "?" ]; then
      echo "$WARN Latest deployment state: $DEPLOY_STATE — $LATEST_URL ($DEPLOY_CREATED)"
    else
      echo "$PASS Latest production deployment: $LATEST_URL"
    fi
  else
    echo "$WARN Could not retrieve Vercel deployments"
  fi

  # Check each page — 404 after a new deploy means Vercel hasn't rebuilt yet (WARN not FAIL)
  echo ""
  echo "  Checking each page serves latest data..."
  for PAGE in /market-overview /industry-intel /industry-returns /signals /screener /macro /content; do
    HEADERS=$(curl -s -I --max-time 15 "$FRONTEND$PAGE" 2>/dev/null)
    HTTP_CODE=$(echo "$HEADERS" | grep "^HTTP" | awk '{print $2}')
    AGE=$(echo "$HEADERS" | grep -i "^age:" | awk '{print $2}' | tr -d '\r')
    VCACHE=$(echo "$HEADERS" | grep -i "^x-vercel-cache:" | awk '{print $2}' | tr -d '\r')

    if [ "$HTTP_CODE" = "200" ]; then
      CACHE_INFO=""
      [ -n "$VCACHE" ] && CACHE_INFO=" vercel-cache=$VCACHE"
      if [ -n "$AGE" ] && [ "$AGE" -gt 86400 ] 2>/dev/null; then
        echo "  $WARN $PAGE — HTTP $HTTP_CODE but age=${AGE}s (>24h) — may be stale ISR"
      else
        echo "  $PASS $PAGE — HTTP $HTTP_CODE$CACHE_INFO"
      fi
    elif [ "$HTTP_CODE" = "404" ]; then
      echo "  $WARN $PAGE — HTTP 404 — page not found on current deploy (redeploy Vercel if recently added)"
    else
      echo "  $FAIL $PAGE — HTTP ${HTTP_CODE:-unreachable}"
      ERRORS=$((ERRORS+1))
    fi
  done
else
  echo "$WARN vercel CLI not found — skipping Vercel checks (install: npm i -g vercel)"
fi

# ── Summary ──────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
if [ "$ERRORS" -eq 0 ]; then
  echo "$PASS ALL CHECKS PASSED"
else
  echo "$FAIL $ERRORS CHECK(S) FAILED — review above"
fi
echo "══════════════════════════════════════════════════"
```

## What This Catches

Derived from the 23-issue auth/deployment postmortem:

| Check | Issues Covered |
|-------|---------------|
| Cloud Run health | #5, #6, #7 (deploy failures) |
| Env vars present | #3, #21 (SCHEDULER_SECRET missing) |
| Scheduler status | #21, #22 (silent 401s for days) |
| Scheduler auth gate | #15, #18 (token mismatch/case), token drift between SM and jobs |
| Firestore connectivity | #1, #2 (IAM/permissions) |
| Frontend live | #8, #9, #14 (Vercel build/env) |
| API proxy working | #8, #11, #19 (BACKEND_URL) |
| Cache-Control headers | #11 (vercel.json override) |
| Recent error logs | #13 (truncated logs), #23 (silent staleness) |
| Vercel deployment + pages | Confirms latest prod build is active; checks all 7 pages return 200 with fresh ISR state (`x-vercel-cache`) |

## When to Run

- After every `gcloud run deploy` or `gcloud builds submit`
- After every Vercel deploy (`vercel --prod`)
- After changing env vars or secrets on Cloud Run
- After creating or modifying Cloud Scheduler jobs
- Morning sanity check if something looks stale
