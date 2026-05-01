# Pipeline Diagnostics (Last 5 Days)

Inspects the fetch/bake pipeline, Cloud Scheduler jobs, and Cloud Run logs over the last 5 days. For each day, reports whether the pipeline ran, what failed, and explains likely root causes.

## Run

```bash
PROJECT="${GCP_PROJECT_ID:-ttb-lang1}"
REGION="us-central1"
SERVICE="gcp3-backend"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     PIPELINE DIAGNOSTICS — Last 5 Trading Days          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

BACKEND_URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT" --format="value(status.url)" 2>/dev/null)
if [ -z "$BACKEND_URL" ]; then
  echo "❌ Cloud Run service $SERVICE not found — cannot proceed."
  exit 1
fi
echo "Backend: $BACKEND_URL"
echo ""

# ─── 1. Scheduler job execution history ────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "  1. SCHEDULER JOB STATUS & HISTORY"
echo "═══════════════════════════════════════════════════════════"

# gcp3-premarket-warmup calls /refresh/fetch (9:30 AM ET)
# gcp3-ai-summary-refresh calls /refresh/bake (9:45 AM ET)
# gcp3-midday-intraday-refresh calls /refresh/intraday (12:00 PM ET)
# gcp3-eod-intraday-refresh calls /refresh/intraday?skip_gemini=true (4:15 PM ET)
# gcp3-nightly-cache-purge calls /admin/purge-cache (6:00 AM UTC daily)
SCHEDULER_JOBS="gcp3-premarket-warmup gcp3-ai-summary-refresh gcp3-midday-intraday-refresh gcp3-eod-intraday-refresh gcp3-nightly-cache-purge"

for JOB in $SCHEDULER_JOBS; do
  JOB_INFO=$(gcloud scheduler jobs describe "$JOB" --location "$REGION" --project "$PROJECT" \
    --format="value(schedule,state,httpTarget.uri,lastAttemptTime)" 2>/dev/null)
  SCHEDULE=$(echo "$JOB_INFO" | cut -f1)
  STATE=$(echo "$JOB_INFO" | cut -f2)
  URI=$(echo "$JOB_INFO" | cut -f3)
  LAST_ATTEMPT=$(echo "$JOB_INFO" | cut -f4)

  echo ""
  echo "  ── $JOB ──"
  echo "    State    : ${STATE:-unknown}"
  echo "    Schedule : ${SCHEDULE:-unknown}"
  echo "    URI      : ${URI:-unknown}"
  echo "    Last attempt: ${LAST_ATTEMPT:-never}"

  if [ "$STATE" != "ENABLED" ] && [ -n "$STATE" ]; then
    echo "    ⚠️  Job is $STATE — not running!"
  fi

  if [ -n "$URI" ] && ! echo "$URI" | grep -q "$(echo "$BACKEND_URL" | sed 's|https://||')"; then
    echo "    ❌ URI MISMATCH — job points to a different backend than $BACKEND_URL"
    echo "       Fix: gcloud scheduler jobs update http $JOB --location $REGION --uri=${BACKEND_URL}/..."
  fi

  python3 -c "
from datetime import datetime, timezone, timedelta
job, last = '$JOB', '$LAST_ATTEMPT'
if not last:
    print('    ❌ Never executed')
else:
    try:
        dt = datetime.fromisoformat(last.replace('Z','+00:00'))
        now = datetime.now(timezone.utc)
        age_h = (now - dt).total_seconds() / 3600
        if age_h > 48:
            print(f'    ❌ Last attempt {age_h:.0f}h ago — stale (>48h)')
        elif age_h > 25:
            print(f'    ⚠️  Last attempt {age_h:.0f}h ago — may have missed today')
        else:
            print(f'    ✅ Last attempt {age_h:.1f}h ago')
    except Exception as e:
        print(f'    ⚠️  Cannot parse last attempt time: {last}')
" 2>/dev/null
done

# ─── 2. Cloud Run logs — pipeline runs last 5 days ──────────────────
echo ""
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  2. PIPELINE EXECUTION LOG (Last 5 Days)"
echo "═══════════════════════════════════════════════════════════"

FIVE_DAYS_AGO=$(python3 -c "
from datetime import datetime, timezone, timedelta
print((datetime.now(timezone.utc) - timedelta(days=5)).strftime('%Y-%m-%dT%H:%M:%SZ'))
")

gcloud logging read "
  resource.type=\"cloud_run_revision\"
  resource.labels.service_name=\"$SERVICE\"
  timestamp>=\"$FIVE_DAYS_AGO\"
  (textPayload=~\"refresh/fetch\" OR textPayload=~\"refresh/bake\" OR
   textPayload=~\"refresh/intraday\" OR textPayload=~\"refresh/all\" OR
   textPayload=~\"purge-cache\" OR textPayload=~\"seed-etf-history\" OR
   textPayload=~\"compute-returns\" OR jsonPayload.message=~\"refresh\" OR
   jsonPayload.message=~\"purge\")
" --project "$PROJECT" --limit=300 --format="value(timestamp,textPayload,jsonPayload.message)" 2>/dev/null | \
python3 -c "
import sys, re
from collections import defaultdict

days = defaultdict(lambda: defaultdict(list))
# /refresh/premarket removed — that job calls /refresh/fetch, not a separate endpoint
pipelines = ['refresh/fetch', 'refresh/bake', 'refresh/intraday', 'refresh/all', 'purge-cache', 'seed-etf-history', 'compute-returns']

# Completion markers per pipeline (checked before generic error keywords to avoid false positives)
COMPLETION_MARKERS = {
    'refresh/fetch': 'status=fetch_',     # fetch_ok or fetch_partial — not an error
    'refresh/bake':  'status=bake_',      # bake_ok or bake_partial
    'refresh/intraday': 'status=intraday_',
    'purge-cache': 'purge',
}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    m = re.match(r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})', line)
    if not m:
        continue
    day, time_str = m.group(1), m.group(2)

    for p in pipelines:
        if p not in line:
            continue
        lower = line.lower()

        # Check completion markers first — these contain 'failed' but aren't errors
        marker = COMPLETION_MARKERS.get(p, '')
        if marker and marker in lower:
            status = 'OK'
        elif 'stage' in lower and ('error' in lower or 'failed' in lower):
            # Stage-level warning inside an otherwise completing run
            status = 'STAGE_WARN'
        elif 'aborted' in lower or ('503' in line) or ('401' in line and 'Unauthorized' in line):
            status = 'ABORT'
        elif 'DEPRECATED' in line:
            status = 'DEPRECATED'
        elif 'skipping' in lower or ('skip' in lower and 'not a trading day' in lower):
            status = 'SKIP'
        elif 'error' in lower or 'exception' in lower:
            status = 'ERROR'
        else:
            status = 'INFO'

        days[day][p].append((time_str, status, line[:180].replace('\t', ' ')))
        break

if not days:
    print('  No pipeline log entries found in the last 5 days.')
    print('  Possible causes:')
    print('    - Cloud Scheduler jobs are disabled or misconfigured')
    print('    - Backend was not deployed in this period')
    print('    - Logs have been purged or retention is < 5 days')
else:
    for day in sorted(days.keys(), reverse=True):
        print(f'\n  ── {day} ──')
        found_any = False
        for p in pipelines:
            if p not in days[day]:
                continue
            found_any = True
            events = days[day][p]
            errors = [e for e in events if e[1] in ('ERROR', 'ABORT')]
            stage_warns = [e for e in events if e[1] == 'STAGE_WARN']
            oks = [e for e in events if e[1] == 'OK']
            skips = [e for e in events if e[1] == 'SKIP']
            deprecated = [e for e in events if e[1] == 'DEPRECATED']

            if errors:
                icon = '❌'
                detail = f'{len(errors)} hard error(s)'
            elif stage_warns and oks:
                icon = '⚠️ '
                detail = f'completed with {len(stage_warns)} stage warning(s)'
            elif stage_warns:
                icon = '⚠️ '
                detail = f'{len(stage_warns)} stage warning(s) — no completion marker'
            elif oks:
                icon = '✅'
                detail = 'completed'
            elif skips:
                icon = '⏭️ '
                detail = 'skipped (non-trading day)'
            elif deprecated:
                icon = '⚠️ '
                detail = 'DEPRECATED pipeline still running'
            else:
                icon = '🔵'
                detail = f'{len(events)} info event(s)'

            times = ', '.join(e[0] for e in events[:3])
            print(f'    {icon} {p}: {detail} [{times}]')

            for t, st, snippet in errors:
                print(f'       └─ {snippet[:140]}')
            for t, st, snippet in stage_warns:
                print(f'       └─ {snippet[:140]}')
            if deprecated:
                print(f'       └─ WARNING: /refresh/all doubles API usage. Remove gcp3-ai-summary-refresh scheduler job.')

        # Trading day detection: if fetch or bake ran, it was a trading day
        is_trading_day = any(p in days[day] for p in ['refresh/fetch', 'refresh/bake', 'refresh/intraday'])
        expected = ['refresh/fetch', 'refresh/bake', 'refresh/intraday']
        missing = [p for p in expected if p not in days[day]]
        if missing and is_trading_day:
            print(f'    ⚠️  Missing: {\", \".join(missing)}')
" 2>/dev/null

# ─── 3. Error classification — last 5 days ──────────────────────────
echo ""
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  3. ERROR CLASSIFICATION (Last 5 Days)"
echo "═══════════════════════════════════════════════════════════"

gcloud logging read "
  resource.type=\"cloud_run_revision\"
  resource.labels.service_name=\"$SERVICE\"
  severity>=ERROR
  timestamp>=\"$FIVE_DAYS_AGO\"
" --project "$PROJECT" --limit=200 --format="value(timestamp,textPayload,jsonPayload.message)" 2>/dev/null | \
python3 -c "
import sys, re
from collections import Counter

categories = Counter()
examples = {}

patterns = {
    'gemini_429':    (r'429.*generativelanguage|generativelanguage.*429', 'Gemini rate limit (429)'),
    'finnhub_429':   (r'429.*finnhub|finnhub.*429|rate.?limit.*finnhub', 'Finnhub rate limit (429)'),
    'polygon_403':   (r'403.*polygon|polygon.*403|Forbidden.*polygon', 'Polygon.io 403 (plan/key issue)'),
    'gemini_error':  (r'generativelanguage\.googleapis\.com', 'Gemini API error'),
    'firestore_error':(r'firestore|google\.cloud\.firestore', 'Firestore error'),
    'auth_401':      (r'401|Unauthorized|OIDC|scheduler.*token', 'Auth/scheduler 401'),
    'timeout':       (r'timeout|timed out|TimeoutError|deadline exceeded', 'Timeout'),
    'yfinance':      (r'yfinance|No data found', 'yfinance data error'),
    'import_error':  (r'ImportError|ModuleNotFoundError', 'Missing module'),
    'memory':        (r'MemoryError|OOM|killed', 'Memory/OOM'),
    'connection':    (r'ConnectionError|ConnectionRefused|ECONNREFUSED', 'Connection error'),
}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    matched = False
    for cat, (pattern, label) in patterns.items():
        if re.search(pattern, line, re.IGNORECASE):
            categories[cat] += 1
            if cat not in examples:
                examples[cat] = line[:160]
            matched = True
            break
    if not matched:
        categories['other'] += 1
        if 'other' not in examples:
            examples['other'] = line[:160]

if not categories:
    print('  ✅ No ERROR-level logs in the last 5 days — pipeline is clean.')
else:
    total = sum(categories.values())
    print(f'  Total errors: {total}')
    print()
    for cat, count in categories.most_common():
        label = patterns[cat][1] if cat in patterns else 'Uncategorized'
        pct = count / total * 100
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f'  {bar} {label}: {count} ({pct:.0f}%)')
        if cat in examples:
            print(f'    Example: {examples[cat]}')
        print()

    advice = {
        'gemini_429':    '→ Add asyncio.sleep(4) between bake stages B2-B6 in main.py refresh_bake(). Fixed in 2026-04-21 deploy.',
        'finnhub_429':   '→ Check Firestore cache TTLs. Add jitter to concurrent fetches. Consider Finnhub premium plan.',
        'polygon_403':   '→ Check MASSIVE_API_KEY in Secret Manager. Verify Polygon.io subscription tier supports /v2/snapshot.',
        'gemini_error':  '→ Check GEMINI_API_KEY in Secret Manager. Verify quota at console.cloud.google.com.',
        'firestore_error':'→ Check IAM roles (roles/datastore.user). Run /iam-check.',
        'auth_401':      '→ Scheduler token mismatch. Run /post-deploy-verify to check all 5 jobs.',
        'timeout':       '→ Increase Cloud Run timeout or reduce concurrent fetches.',
        'yfinance':      '→ yfinance may be rate-limited or Yahoo changed their API.',
        'import_error':  '→ Missing dependency in Docker image. Check backend/requirements.txt and rebuild.',
        'memory':        '→ Cloud Run instance needs more memory. Check --memory flag in cloudbuild.yaml.',
        'connection':    '→ Transient network issue. If recurring, check DNS and firewall rules.',
    }
    print('  ── Recommended Actions ──')
    for cat in categories:
        if cat in advice:
            label = patterns[cat][1] if cat in patterns else 'Uncategorized'
            print(f'  {label}: {advice[cat]}')
" 2>/dev/null

# ─── 4. Firestore cache freshness ────────────────────────────────────
echo ""
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  4. CACHE FRESHNESS"
echo "═══════════════════════════════════════════════════════════"

DEBUG_RESP=$(curl -s --max-time 15 "$BACKEND_URL/debug/status" 2>/dev/null)
echo "$DEBUG_RESP" | python3 -c "
import sys, json
from datetime import datetime, timezone

try:
    d = json.load(sys.stdin)
except Exception:
    print('  ⚠️  Could not reach /debug/status')
    sys.exit(0)

ic = d.get('industry_cache', {})
print(f'  industry_cache:')
print(f'    Docs      : {ic.get(\"doc_count\", \"?\")}')
print(f'    Newest    : {ic.get(\"newest_updated\", \"unknown\")}')
print(f'    Freshness : {ic.get(\"freshness_hours\", \"?\")}h')
print(f'    {\"❌ STALE — industry data is >25h old\" if ic.get(\"stale\") else \"✅ Fresh\"}')

gc = d.get('gcp3_cache', {})
keys = gc.get('live_keys', [])
print()
print(f'  gcp3_cache: {gc.get(\"live_doc_count\", 0)} live doc(s)')
if keys:
    prefixes = {}
    for k in keys:
        prefix = k.split(':')[0] if ':' in k else k
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    for prefix, count in sorted(prefixes.items()):
        print(f'    {prefix}: {count}')
else:
    print('    ⚠️  No live cache docs — all data may be expired')

routes = d.get('route_inventory', [])
missing = d.get('missing_expected_routes', [])
print()
print(f'  Routes: {len(routes)} registered')
if missing:
    print(f'    ❌ Missing expected: {\", \".join(missing)}')
else:
    print(f'    ✅ All expected routes present')

rl = d.get('rate_limits', {})
fh429 = rl.get('finnhub_429s', {})
if isinstance(fh429, dict) and fh429.get('total', 0) > 0:
    print()
    print(f'  Finnhub 429s this instance: {fh429.get(\"total\", 0)}')
    print('    ⚠️  Rate limiting active — cache TTLs may need extending')
" 2>/dev/null

# ─── 5. Summary & Quick Fixes ────────────────────────────────────────
echo ""
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  5. QUICK FIXES"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  If pipeline didn't run:"
echo "    1. List jobs: gcloud scheduler jobs list --location=$REGION --project=$PROJECT"
echo "    2. Force bake: gcloud scheduler jobs run gcp3-ai-summary-refresh --location=$REGION --project=$PROJECT"
echo "    3. Manual: POST $BACKEND_URL/refresh/fetch  then  POST $BACKEND_URL/refresh/bake"
echo ""
echo "  If Gemini 429 on bake stages:"
echo "    → asyncio.sleep(4) gaps between B2-B6 in main.py (fix deployed 2026-04-21)"
echo "    → If still failing, check GEMINI_API_KEY quota at console.cloud.google.com"
echo ""
echo "  If Polygon 403:"
echo "    1. Verify MASSIVE_API_KEY in Secret Manager is valid"
echo "    2. Check polygon.io dashboard — /v2/snapshot requires Starter tier or above"
echo ""
echo "  If auth errors (401):"
echo "    1. Run /post-deploy-verify"
echo "    2. Verify SCHEDULER_SECRET in Secret Manager matches all 5 scheduler jobs"
echo ""
echo "  If data is stale:"
echo "    1. /cache-list — see what's live"
echo "    2. POST $BACKEND_URL/admin/purge-cache — remove expired"
echo "    3. POST $BACKEND_URL/admin/compute-returns — recompute returns"
echo ""
echo "  Other commands: /frontend-health-check  /post-deploy-verify  /iam-check"
```

## What this checks

### 1. Scheduler Job Status & History
- All 5 jobs: state, schedule, URI, last attempt time
- Note: `gcp3-premarket-warmup` calls `/refresh/fetch` (not a separate `/refresh/premarket` endpoint)
- Flags disabled jobs, URI mismatches vs current backend URL, and stale last-attempt times (>48h)

### 2. Pipeline Execution Log (Last 5 Days)
- Reads Cloud Run logs for all pipeline endpoints: refresh/fetch, refresh/bake, refresh/intraday, purge-cache, seed-etf-history, compute-returns
- **Fixed classification logic:** completion markers (`status=fetch_ok`, `status=bake_partial`, etc.) are checked before generic error keywords to avoid false positives from `stages_failed=[]`
- Distinguishes hard errors (ABORT/ERROR) from stage-level warnings within an otherwise completing run
- Trading-day detection based on whether fetch/bake/intraday ran, not premarket (which was a false signal)

### 3. Error Classification (Last 5 Days)
- **Added:** Gemini 429 as separate category (distinct from generic Gemini errors)
- **Added:** Polygon 403 as separate category with specific fix advice
- Categories: Gemini 429, Finnhub 429, Polygon 403, Gemini API errors, Firestore, Auth 401, Timeouts, yfinance, missing modules, OOM, connection errors
- Bar chart with percentages and per-category fix advice

### 4. Cache Freshness
- industry_cache doc count, newest update, freshness hours, stale flag
- gcp3_cache live docs grouped by key prefix
- Route inventory check
- Finnhub 429 counter from current instance

### 5. Quick Fixes
- Remediation steps for each common failure mode
- References specific fix locations (main.py bake stages, Secret Manager, Polygon dashboard)

## When to use

- Morning: "Did the pipeline run overnight and this morning?"
- After deploy: "Did the new code break any pipeline stages?"
- Debugging staleness: "Why is market data from yesterday?"
- Weekly review: "How reliable has the pipeline been?"
