# Vercel + GCP Full-Stack Deploy + PR

Deploys **both** the Python backend (`backend/`) to GCP Cloud Run **and** the Next.js frontend (`frontend/`) to Vercel, then creates a single branch, commits all changes, and opens one PR against `main`.

> Use this when a change touches both `backend/` and `frontend/`. For backend-only, use `/gpr`. For frontend-only, use `/verpr`.

## Pipeline Overview

The backend runs a **fetch/bake** scheduler pipeline:
- `13:30 UTC` → `POST /refresh/fetch` — data ingestion
- `13:45 UTC` → `POST /refresh/bake` — AI synthesis
- `16:00 UTC` → `POST /refresh/intraday` — midday refresh (**must have `?skip_gemini=true`** to avoid Gemini quota drain before next morning's bake)
- `20:15 UTC` → `POST /refresh/intraday?skip_gemini=true` — EOD refresh
- `06:00 UTC` → `POST /admin/purge-cache` — nightly Firestore TTL sweep

Every deploy must keep this pipeline intact. The midday scheduler job **must** include `?skip_gemini=true` or it pre-depletes the free-tier Gemini quota before the 9:45 AM bake runs.

## Pre-Deploy Rules

**Backend (must verify before touching `cloudbuild.yaml` or env vars):**

**Rule 1** — Never `--set-env-vars` for secrets. Always:
```
--set-secrets=FINNHUB_API_KEY=FINNHUB_API_KEY:latest,GEMINI_API_KEY=GEMINI_API_KEY:latest,SCHEDULER_SECRET=SCHEDULER_SECRET:latest
```

**Rule 2** — `cloudbuild.yaml` must list all 3 secrets. A missing secret is silently dropped and breaks the scheduler pipeline.

**Rule 3** — If rotating `SCHEDULER_SECRET`: update Secret Manager first → update all 5 scheduler jobs → redeploy. Never partial.

**Rule 4** — If backend URL changes: update all 5 scheduler job URIs (gcp3-premarket-warmup, gcp3-ai-summary-refresh, gcp3-midday-intraday-refresh, gcp3-eod-intraday-refresh, gcp3-nightly-cache-purge).

**Rule 5 (NEW)** — Midday scheduler job URI must end with `?skip_gemini=true`. Verify before every deploy. Without this, Gemini quota is drained before the morning bake cycle runs.

**Frontend:**
- Never commit `.env`, credential files, or files matching secret patterns
- Only create the PR after both deployments are confirmed ready

## Steps

1. Checkout a fresh branch off `main` — never reuse the current branch
2. Secret scan ALL changed files — hard stop if any secrets found
3. Stage only changed files by name — never `git add .`
4. Run tests in `fin-ai1` mamba environment
5. Verify `cloudbuild.yaml` lists all 3 secrets
6. Deploy backend via Cloud Build and confirm `/health` returns OK
7. Run `/post-deploy-verify` to validate backend change rules 1–4
8. Run `vercel build` from repo root to confirm frontend build passes
9. Deploy frontend to Vercel preview and confirm status is ● Ready
10. Run post-deploy verification checklist (see below)
11. Commit, push, open PR

## Execute

```bash
# 0. Repo root
cd /Users/adamaslan/code/gcp3

# 1. Fresh branch off main
git checkout main && git pull origin main
git checkout -b <feature|fix|refactor>/<short-description>

# 2. Secret scan before staging
git diff --name-only HEAD | xargs -I{} sh -c \
  'grep -lE "AIzaSy[A-Za-z0-9_-]{35}|GOCSPX-[A-Za-z0-9_-]{24,}|ya29\.[A-Za-z0-9_-]{100,}|\"private_key\"" {} 2>/dev/null && echo "SECRET FOUND: {}" || true'
bash .git/hooks/pre-commit 2>&1 || { echo "SECRETS DETECTED — fix before staging"; exit 1; }

# 3. Stage by name only
git add backend/<file1> backend/<file2> frontend/src/...
git diff --cached --name-only  # verify what's staged

# 4. Tests (fin-ai1 environment)
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && \
  source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/mamba.sh && \
  mamba activate fin-ai1 && \
  pytest backend/ -v

# 5. Verify cloudbuild.yaml has all 3 secrets
grep "FINNHUB_API_KEY\|GEMINI_API_KEY\|SCHEDULER_SECRET" backend/cloudbuild.yaml
# All 3 must appear — add any missing before proceeding

# 6. Deploy backend via Cloud Build
gcloud builds submit --config backend/cloudbuild.yaml \
  --project ttb-lang1 \
  backend/

# 7. Confirm backend is up
BACKEND_URL=$(gcloud run services describe gcp3-backend \
  --project ttb-lang1 --region us-central1 \
  --format="value(status.url)")
curl -sf "$BACKEND_URL/health" | python3 -m json.tool
curl -sf "$BACKEND_URL/debug/status" | python3 -m json.tool

# 8. Post-deploy verify (backend change rules 1–4)
# Run /post-deploy-verify or check debug/status manually

# 9. Pull Vercel project settings if needed
vercel pull --yes

# 10. Build frontend
vercel build

# 11. If build fails — fix, re-stage, rebuild before continuing

# 12. Deploy frontend preview
vercel --prebuilt

# 13. Confirm frontend deployment is ● Ready
vercel inspect <deployment-url>

# 14. If deploy fails — check logs, fix, rebuild, redeploy, confirm before continuing

# 15. Commit (pre-commit hook re-scans automatically)
git commit -m "$(cat <<'EOF'
type(scope): description

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"

# 16. Push and open PR — capture URL immediately
git push -u origin HEAD
PR_URL=$(gh pr create \
  --repo adamaslan/gcp3 \
  --title "short title under 70 chars" \
  --body "$(cat <<'EOF'
## Summary
- bullet points of what changed and why

## Backend Change Rules Verified
- [ ] No `--set-env-vars` used for secrets (Secret Manager refs only)
- [ ] `cloudbuild.yaml` lists all 3 secrets: FINNHUB_API_KEY, GEMINI_API_KEY, SCHEDULER_SECRET
- [ ] SCHEDULER_SECRET rotation: updated Secret Manager + all 5 scheduler jobs (if rotated)
- [ ] Backend URL unchanged — scheduler job URIs still valid (or all 5 updated)
- [ ] Midday scheduler job URI ends with `?skip_gemini=true`

## Test Plan
- [ ] Secret scan passed (pre-commit hook did not block)
- [ ] `pytest backend/` passed in fin-ai1 environment
- [ ] Cloud Run deployed and `/health` returns `{"status": "ok"}`
- [ ] `/debug/status` shows no missing expected routes
- [ ] `/post-deploy-verify` passed (all 4 backend change rules green)
- [ ] Vercel build passed locally
- [ ] Vercel preview deployment status ● Ready
- [ ] Post-deploy verification checklist passed (see below)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)")
echo "✅ PR created: $PR_URL"
```

## GCP Pipeline Health Checks (NEW — run before stale-data checks)

These gcloud commands audit the live GCP infrastructure for pipeline issues that cause staleness. Run these first — they catch scheduler misconfiguration, quota exhaustion, and Cloud Run anomalies that no amount of endpoint polling will reveal.

### Step 0A — Scheduler job audit (URIs + skip_gemini + state)

```bash
# List all 5 jobs with full URI so you can verify skip_gemini and state
gcloud scheduler jobs list \
  --location=us-central1 --project=ttb-lang1 \
  --format="table(name,schedule,state,lastAttemptTime)"

# Verify midday job has ?skip_gemini=true in its URI
MIDDAY_URI=$(gcloud scheduler jobs describe gcp3-midday-intraday-refresh \
  --location=us-central1 --project=ttb-lang1 \
  --format="value(httpTarget.uri)")
echo "Midday URI: $MIDDAY_URI"
python3 -c "
import sys
uri = '$MIDDAY_URI'
if 'skip_gemini=true' in uri:
    print('✅ midday job has ?skip_gemini=true')
else:
    print('❌ CRITICAL: midday job missing ?skip_gemini=true — will drain Gemini quota before morning bake')
    print('   Fix: gcloud scheduler jobs update http gcp3-midday-intraday-refresh --uri=\"' + uri.split(\"?\")[0] + '?skip_gemini=true\" --location=us-central1 --project=ttb-lang1')
    sys.exit(1)
"

# Verify all 5 jobs are ENABLED and last attempt was OK
python3 -c "
import subprocess, json
result = subprocess.run(
    ['gcloud','scheduler','jobs','list','--location=us-central1','--project=ttb-lang1','--format=json'],
    capture_output=True, text=True
)
jobs = json.loads(result.stdout)
expected = {'gcp3-premarket-warmup','gcp3-ai-summary-refresh','gcp3-midday-intraday-refresh','gcp3-eod-intraday-refresh','gcp3-nightly-cache-purge'}
found = {j['name'].split('/')[-1] for j in jobs}
missing = expected - found
for name in missing:
    print(f'❌ MISSING job: {name}')
for j in jobs:
    name = j['name'].split('/')[-1]
    if name not in expected:
        continue
    state = j.get('state','')
    last_status = j.get('status',{}).get('code','')
    state_ok = state == 'ENABLED'
    status_ok = last_status in ('', 'OK', None)
    print(f'{\"✅\" if state_ok else \"❌\"} {name}: state={state}  last_status={last_status or \"OK\"}')
"
```
**Expected:** All 5 jobs `ENABLED`. Midday URI contains `?skip_gemini=true`. Last attempt status `OK`.

### Step 0B — Cloud Run revision health

```bash
# Check current serving revision traffic allocation
gcloud run services describe gcp3-backend \
  --project ttb-lang1 --region us-central1 \
  --format="table(status.traffic[].revisionName,status.traffic[].percent,status.traffic[].latestRevision)"

# Check last 3 revisions for ready status
gcloud run revisions list \
  --service=gcp3-backend \
  --project=ttb-lang1 --region=us-central1 \
  --limit=3 \
  --format="table(metadata.name,status.conditions[0].status,status.conditions[0].reason,metadata.creationTimestamp)"
```
**Expected:** Latest revision serving 100% traffic. `status.conditions[0].status = True` (Ready). No `ContainerFailed` or `HealthCheckFailed` reason.

### Step 0C — Cloud Build recent build status

```bash
# Last 3 builds for the backend — confirm latest succeeded
gcloud builds list \
  --project=ttb-lang1 \
  --filter="substitutions.REPO_NAME=gcp3 OR tags=gcp3-backend" \
  --limit=3 \
  --format="table(id,status,createTime,finishTime,duration)"
```
**Expected:** Most recent build `SUCCESS`. If `FAILURE` or `TIMEOUT`, inspect before proceeding.

### Step 0D — Gemini quota check via Cloud Run logs

```bash
# Scan last 200 backend log lines for Gemini 429s or quota exhaustion
gcloud logging read \
  'resource.type="cloud_run_revision" resource.labels.service_name="gcp3-backend" (textPayload=~"429" OR textPayload=~"quota" OR textPayload=~"RESOURCE_EXHAUSTED" OR textPayload=~"Gemini")' \
  --project=ttb-lang1 \
  --limit=20 \
  --freshness=24h \
  --format="table(timestamp,textPayload)"

# Count 429s in last 24h to gauge severity
GEMINI_429_COUNT=$(gcloud logging read \
  'resource.type="cloud_run_revision" resource.labels.service_name="gcp3-backend" textPayload=~"429"' \
  --project=ttb-lang1 --limit=50 --freshness=24h \
  --format="value(timestamp)" 2>/dev/null | wc -l | tr -d ' ')
echo "Gemini 429s in last 24h: $GEMINI_429_COUNT"
python3 -c "
count = int('$GEMINI_429_COUNT')
if count == 0:
    print('✅ No Gemini 429s in last 24h')
elif count < 5:
    print(f'⚠️  {count} Gemini 429s — monitor but proceed')
else:
    print(f'❌ {count} Gemini 429s — quota likely exhausted, bake will fail tonight')
"
```
**Expected:** 0 Gemini 429s (or <5 as warning). Any `RESOURCE_EXHAUSTED` log is a hard blocker.

### Step 0E — Firestore bake checkpoint inspection

```bash
# Read the refresh_state:bake Firestore doc to check for partial bake
python3 -c "
import subprocess, json
result = subprocess.run(
    ['gcloud','firestore','documents','get',
     'projects/ttb-lang1/databases/(default)/documents/gcp3_cache/refresh_state:bake',
     '--project=ttb-lang1','--format=json'],
    capture_output=True, text=True
)
if result.returncode != 0 or not result.stdout.strip():
    print('⚠️  refresh_state:bake doc not found (first run or purged) — OK if system is fresh')
else:
    try:
        doc = json.loads(result.stdout)
        fields = doc.get('fields', {})
        stages_ok = fields.get('stages_completed',{}).get('arrayValue',{}).get('values',[])
        stages_fail = fields.get('stages_failed',{}).get('arrayValue',{}).get('values',[])
        last_run = fields.get('last_run',{}).get('stringValue','?')
        ok_names = [s.get('stringValue','') for s in stages_ok]
        fail_names = [s.get('stringValue','') for s in stages_fail]
        print(f'Last bake run: {last_run}')
        print(f'Stages completed: {ok_names}')
        if fail_names:
            print(f'❌ Stages FAILED: {fail_names}')
            print('   Re-run POST /refresh/bake once Gemini quota resets')
        else:
            print('✅ No failed stages')
    except Exception as e:
        print(f'⚠️  Could not parse bake checkpoint: {e}')
        print(result.stdout[:500])
"
```
**Expected:** `stages_failed` empty. If populated, content articles will be stale until bake reruns.

### Step 0F — Finnhub rate limit check

```bash
# Check for Finnhub 429s or connection errors in last 6h
gcloud logging read \
  'resource.type="cloud_run_revision" resource.labels.service_name="gcp3-backend" (textPayload=~"finnhub" OR textPayload=~"Finnhub") textPayload=~"429|timeout|connection"' \
  --project=ttb-lang1 \
  --limit=10 \
  --freshness=6h \
  --format="table(timestamp,textPayload)"
```
**Expected:** No Finnhub 429s or timeouts in last 6h. If present, screener/signals data may be stale.

---

## Stale-Data Detection

Run these checks after GCP pipeline checks pass. "Stale" means the cache key exists but `date` field is not today's date (`YYYY-MM-DD`), OR the `updated` / `stale_as_of` field shows yesterday's date, OR the response contains `"stale": true`.

### Step 1 — Pull live data for all 7 endpoints

```bash
BACKEND_URL=$(gcloud run services describe gcp3-backend \
  --project ttb-lang1 --region us-central1 \
  --format="value(status.url)")

# Fetch all 7 backend endpoints in parallel and save to temp files
curl -sf "$BACKEND_URL/screener"          -o /tmp/r_screener.json &
curl -sf "$BACKEND_URL/signals"           -o /tmp/r_signals.json &
curl -sf "$BACKEND_URL/industry-intel"    -o /tmp/r_industry_intel.json &
curl -sf "$BACKEND_URL/industry-returns"  -o /tmp/r_industry_returns.json &
curl -sf "$BACKEND_URL/market-overview"   -o /tmp/r_market_overview.json &
curl -sf "$BACKEND_URL/content"           -o /tmp/r_content.json &
curl -sf "$BACKEND_URL/macro-pulse"       -o /tmp/r_macro.json &
curl -sf "$BACKEND_URL/earnings-radar"    -o /tmp/r_earnings.json &
wait

TODAY=$(date +%Y-%m-%d)
echo "Checking against today: $TODAY"
```

### Step 2 — Per-endpoint stale checks

Run each block. Any `❌` is a stale-data failure — do NOT create the PR until resolved.

#### `/screener` → cache key `screener:{today}` · TTL 1h · refreshed by `/refresh/fetch` (F2)
```bash
python3 -c "
import json, sys
d = json.load(open('/tmp/r_screener.json'))
today = '$TODAY'
date_ok  = d.get('date') == today
total_ok = (d.get('total_screened') or 0) > 100
has_err  = 'error' in d
print('date matches today:', '✅' if date_ok  else '❌ got ' + str(d.get('date')))
print('total_screened>100:', '✅' if total_ok else '❌ got ' + str(d.get('total_screened')))
print('quotes present:    ', '✅' if d.get('quotes') else '❌ missing quotes dict')
print('no error key:      ', '❌ error=' + str(d.get('error')) if has_err else '✅')
"
```
**Expected:** `date == today`, `total_screened >= 100`, `quotes` dict non-empty, no `error` key.

#### `/signals` → cache key `technical_signals:all:{today}` · TTL 2h · refreshed by `/refresh/bake` (reads `industry_returns`)
```bash
python3 -c "
import json
d = json.load(open('/tmp/r_signals.json'))
today = '$TODAY'
date_ok  = d.get('date') == today
total_ok = (d.get('total') or 0) >= 40
buys_ok  = d.get('buys') is not None
stale    = d.get('stale', False)
has_err  = 'error' in d
print('date matches today:', '✅' if date_ok  else '❌ got ' + str(d.get('date')))
print('total ETFs >= 40:  ', '✅' if total_ok else '❌ got ' + str(d.get('total')))
print('buys list present: ', '✅' if buys_ok  else '❌ missing buys')
print('not stale:         ', '✅' if not stale else '❌ stale=True stale_as_of=' + str(d.get('stale_as_of')))
print('no error key:      ', '❌ error=' + str(d.get('error')) if has_err else '✅')
"
```
**Expected:** `date == today`, `total >= 40`, `buys`/`sells`/`holds` lists present, `stale` absent or false, no `error` key.

#### `/industry-intel` → cache key `industry_data:{today}` · TTL 24h · refreshed by `/refresh/fetch` (F3)
```bash
python3 -c "
import json
d = json.load(open('/tmp/r_industry_intel.json'))
today = '$TODAY'
date_ok  = d.get('date') == today
inds_ok  = len(d.get('industries') or {}) >= 40
has_rank = bool(d.get('rankings') or d.get('leaders'))
has_err  = 'error' in d
print('date matches today:    ', '✅' if date_ok  else '❌ got ' + str(d.get('date')))
print('industries >= 40:      ', '✅' if inds_ok  else '❌ got ' + str(len(d.get('industries') or {})))
print('rankings/leaders present:', '✅' if has_rank else '❌ missing rankings')
print('no error key:          ', '❌ error=' + str(d.get('error')) if has_err else '✅')
"
```
**Expected:** `date == today`, `industries` dict with ≥40 keys, `rankings` or `leaders` non-empty, no `error` key.

#### `/industry-returns` → cache key `industry_returns:{today}` · TTL 6h · refreshed by `/refresh/bake` (B1)
```bash
python3 -c "
import json
d = json.load(open('/tmp/r_industry_returns.json'))
today = '$TODAY'
date_ok  = d.get('date') == today
total_ok = (d.get('total') or 0) >= 40
stale    = d.get('stale', False)
periods  = d.get('periods_available') or []
has_err  = 'error' in d
print('date matches today:     ', '✅' if date_ok  else '❌ got ' + str(d.get('date')))
print('total industries >= 40: ', '✅' if total_ok else '❌ got ' + str(d.get('total')))
print('periods_available:      ', '✅ ' + str(periods) if periods else '❌ empty')
print('not stale:              ', '✅' if not stale else '❌ stale=True stale_as_of=' + str(d.get('stale_as_of')))
print('no error key:           ', '❌ error=' + str(d.get('error')) if has_err else '✅')
"
```
**Expected:** `date == today`, `total >= 40`, `periods_available` includes `1d`/`1w`/`1m`/`1y`, `stale` false, no `error` key.

#### `/market-overview` → aggregates 4 sub-caches
Sub-cache TTLs: `morning_brief` 8h · `ai_summary` to-midnight · `news_sentiment` 8h · `market_summary` 2h
```bash
python3 -c "
import json
d = json.load(open('/tmp/r_market_overview.json'))
today = '$TODAY'
sections = d.get('sections_included', [])
print('sections returned:', sections)
for section in ['brief', 'ai_summary', 'sentiment', 'history']:
    s = d.get(section, {})
    has_err = 'error' in s
    has_date = s.get('date') == today if s.get('date') else None
    if has_err:
        print(f'  {section}: ❌ error = ' + s['error'])
    elif has_date is False:
        print(f'  {section}: ❌ date mismatch — got ' + str(s.get('date')))
    elif s:
        print(f'  {section}: ✅')
    else:
        print(f'  {section}: ❌ missing or empty')
"
```
**Expected:** All 4 sections present without `error` key. `brief.date == today`, `ai_summary.date == today`.

#### `/content` → aggregates 4 sub-caches
Sub-cache TTLs: `daily_blog` to-midnight · `blog_review` to-midnight · `daily_correlation` to-midnight · `daily_story` to-midnight
```bash
python3 -c "
import json
d = json.load(open('/tmp/r_content.json'))
today = '$TODAY'
for key in ['blog', 'review', 'correlation', 'story']:
    s = d.get(key, {})
    has_err = 'error' in s
    date_val = s.get('date') or s.get('generated_at', '')[:10]
    date_ok  = date_val == today
    if has_err:
        print(f'  {key}: ❌ error = ' + s['error'])
    elif not date_ok and date_val:
        print(f'  {key}: ❌ stale — date={date_val} (expected {today})')
    elif not s:
        print(f'  {key}: ❌ empty')
    else:
        print(f'  {key}: ✅ date={date_val}')
"
```
**Expected:** All 4 keys present. Each has `date == today` (or `generated_at` starting with today). No `error` key.

#### `/macro-pulse` + `/earnings-radar` → consumed by frontend `/api/macro`
Sub-cache TTLs: `macro_pulse` 2h · `earnings_radar` 6h
```bash
python3 -c "
import json
d = json.load(open('/tmp/r_macro.json'))
today = '$TODAY'
date_ok  = d.get('date') == today
tickers  = d.get('tickers') or d.get('indicators') or {}
regime   = d.get('regime') or d.get('ai_regime')
has_err  = 'error' in d
print('macro-pulse date == today:', '✅' if date_ok  else '❌ got ' + str(d.get('date')))
print('indicators present:       ', '✅' if tickers  else '❌ empty tickers/indicators')
print('regime signal present:    ', '✅' if regime   else '❌ missing ai_regime')
print('no error key:             ', '❌ error=' + str(d.get('error')) if has_err else '✅')
"
python3 -c "
import json
d = json.load(open('/tmp/r_earnings.json'))
today = '$TODAY'
date_ok  = d.get('date') == today
count_ok = isinstance(d.get('upcoming') or d.get('earnings'), list)
has_err  = 'error' in d
print('earnings-radar date == today:', '✅' if date_ok  else '❌ got ' + str(d.get('date')))
print('earnings list present:       ', '✅' if count_ok else '❌ missing upcoming/earnings list')
print('no error key:                ', '❌ error=' + str(d.get('error')) if has_err else '✅')
"
```
**Expected:** `macro_pulse.date == today`, tickers dict populated. `earnings_radar.date == today`, earnings list present. No `error` key on either.

### Step 3 — Firestore cache inventory check

```bash
curl -sf "$BACKEND_URL/debug/status" | python3 -c "
import json, sys
d = json.load(sys.stdin)
today = d.get('today')
cache = d.get('gcp3_cache', {})
ic    = d.get('industry_cache', {})

# Expected live keys that must exist after a successful bake
required_keys = [
    f'screener:{today}',
    f'technical_signals:all:{today}',
    f'industry_returns:{today}',
    f'morning:{today}',
    f'macro_pulse:{today}',
    f'news_sentiment:{today}',
    f'ai_summary:{today}',
    f'daily_blog:{today}',
    f'blog_review:{today}',
    f'daily_correlation:{today}',
    f'daily_story:{today}',
    f'earnings_radar:{today}',
]

live_keys = set(cache.get('live_keys', []))
print(f'Live cache docs: {cache.get(\"live_doc_count\", 0)}')
print()
for k in required_keys:
    print(f'  {k}: {\"✅\" if k in live_keys else \"❌ MISSING\"}')

# industry_cache freshness
fresh_h = ic.get('freshness_hours')
stale   = ic.get('stale', False)
print()
print(f'industry_cache: {ic.get(\"doc_count\",0)} docs, newest_updated={ic.get(\"newest_updated\")}, freshness={fresh_h}h')
print(f'industry_cache stale (>25h): {\"❌ YES\" if stale else \"✅ no\"}')
missing = d.get('missing_expected_routes', [])
print(f'missing_expected_routes: {\"❌ \" + str(missing) if missing else \"✅ none\"}')

# refresh_state checkpoint
rs = d.get('refresh_state', {})
if rs:
    stages_fail = rs.get('stages_failed', [])
    if stages_fail:
        print(f'❌ bake checkpoint stages_failed: {stages_fail}')
    else:
        print('✅ bake checkpoint: no failed stages')
"
```
**Expected:** All 12 required cache keys present (`✅`). `industry_cache` freshness < 25h. No missing routes. No `stages_failed` in bake checkpoint.

### Step 4 — Scheduler pipeline integrity

```bash
gcloud scheduler jobs list --location=us-central1 --project=ttb-lang1 \
  --format="table(name,schedule,state,lastAttemptTime,status.code)"
```
**Expected:** All 5 jobs in `ENABLED` state: `gcp3-premarket-warmup`, `gcp3-ai-summary-refresh`, `gcp3-midday-intraday-refresh`, `gcp3-eod-intraday-refresh`, `gcp3-nightly-cache-purge`. No `PAUSED` or `DISABLED`. Last attempt `status.code` should be `OK` (not `DEADLINE_EXCEEDED` or `UNAVAILABLE`).

### Step 5 — Cloud Run recent error rate (NEW)

```bash
# Count 5xx errors in the last 2h — any non-zero count for /refresh/* is a blocker
gcloud logging read \
  'resource.type="cloud_run_revision" resource.labels.service_name="gcp3-backend" httpRequest.status>=500' \
  --project=ttb-lang1 \
  --limit=20 \
  --freshness=2h \
  --format="table(timestamp,httpRequest.status,httpRequest.requestUrl,textPayload)"

# Count 5xx on scheduler-triggered routes specifically
ERR_5XX=$(gcloud logging read \
  'resource.type="cloud_run_revision" resource.labels.service_name="gcp3-backend" httpRequest.status>=500 (httpRequest.requestUrl=~"/refresh/" OR httpRequest.requestUrl=~"/admin/")' \
  --project=ttb-lang1 --limit=10 --freshness=2h \
  --format="value(timestamp)" 2>/dev/null | wc -l | tr -d ' ')
python3 -c "
count = int('$ERR_5XX')
if count == 0:
    print('✅ No 5xx errors on scheduler routes in last 2h')
else:
    print(f'❌ {count} 5xx errors on /refresh/ or /admin/ routes — bake may have failed')
"
```
**Expected:** 0 5xx errors on `/refresh/` and `/admin/` routes in last 2h.

---

## Post-Deploy UI Checklist

After stale-data checks pass, verify these routes against the Vercel preview URL:

### Screener (`/screener`)
- [ ] Page loads without error
- [ ] Subtitle shows "X of 216 symbols screened · YYYY-MM-DD" with **today's date**
- [ ] "All" tab renders the full table without crashing
- [ ] Columns sort correctly (click Price, Chg %, Signal headers)
- [ ] Rows with missing price/change show "—" instead of crashing

### Signals (`/signals`)
- [ ] Page loads without error
- [ ] Top-10 Buys confluence panel visible above the view toggle
- [ ] Top-10 Sells confluence panel visible alongside Buys
- [ ] Each row in the panels shows: rank, symbol, bar, confluence score (+X.XX), 1d change %
- [ ] Cards show ConfluenceBar with colored fill and score label (HIGH/MEDIUM/LOW)
- [ ] Expanding signals on a card shows `detail` text (not just the label)
- [ ] `ai_outlook` paragraph renders below the summary on each card

### Correlation (`/content?tab=correlation`)
- [ ] Breadth note mentions "pairs 3, 7"
- [ ] Screener breadth note references "216-stock cross-sector watchlist"
- [ ] Correlation snapshot badges show counts
- [ ] Article `generated_at` / `date` shows today

### Backend health
- [ ] `/health` returns `{"status": "ok"}`
- [ ] `/debug/status` shows no `missing_expected_routes`
- [ ] Scheduler pipeline: all 5 jobs `ENABLED`, last run `OK`
- [ ] Midday scheduler job URI contains `?skip_gemini=true`

Run each check and report pass/fail. If any check fails, fix before creating the PR.

## Diagnosing Stale Data

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| `date` field is yesterday | Scheduler fetch/bake didn't fire today | Check scheduler job state; trigger `/refresh/fetch` then `/refresh/bake` manually with `SCHEDULER_SECRET` header |
| `stale: true` in response | Firestore cache read failed, serving prior-day fallback | Check `/debug/status` for `industry_cache.stale`; check Cloud Run logs for Firestore 429/timeout |
| Cache key `❌ MISSING` in Step 3 | That stage failed during bake — partial bake | Inspect `refresh_state:bake` checkpoint (Step 0E); re-trigger the specific stage manually |
| `industry_cache` freshness > 25h | `compute_returns` stage failed or ETF history seed missed | Manually trigger `POST /admin/compute-returns` then `POST /admin/refresh-industry-cache` |
| `macro_pulse.date` stale | Finnhub rate limit hit during premarket warmup | Check `rate_limits.finnhub_429s` in `/debug/status`; wait for next warmup cycle |
| `ai_summary` / content articles stale | Gemini call failed during bake (B2–B6) | Check bake checkpoint `stages_failed` (Step 0E); re-run `POST /refresh/bake` once Gemini quota resets |
| Gemini 429s > 5 in Step 0D | Midday job missing `?skip_gemini=true` | Fix midday job URI (Step 0A fix command), redeploy — do not wait for quota to reset |
| Scheduler job `DISABLED` or last run error | Cloud Run URL changed, or secret rotated without updating jobs | Follow Rule 3/4 in Pre-Deploy Rules above |
| 5xx errors on `/refresh/*` in Step 5 | Cold start timeout or missing secret on new revision | Check `/debug/status`, verify all 3 secrets in cloudbuild.yaml |

## Diagnosing Other Failures

| Symptom | Command |
|---------|---------|
| Backend deploy failed | `gcloud logging read 'resource.type="cloud_run_revision" resource.labels.service_name="gcp3-backend"' --project ttb-lang1 --limit 50 --freshness=10m` |
| Frontend build failed | `vercel build` output — check for TS errors or missing env vars |
| Vercel deploy not ● Ready | `vercel inspect <url> --logs` |
| Bake checkpoint state | Step 0E above — or `curl -sf "$BACKEND_URL/debug/status" \| python3 -m json.tool \| grep -A5 refresh_state` |
| Cloud Run revision not serving | `gcloud run revisions list --service=gcp3-backend --project=ttb-lang1 --region=us-central1 --limit=3` |
| Gemini quota exhausted | Step 0D above; also check GCP Console → APIs & Services → Gemini API → Quotas |
| Scheduler job fired but pipeline failed silently | `gcloud logging read 'resource.type="cloud_scheduler_job"' --project=ttb-lang1 --limit=20 --freshness=12h` |

## Final Output (required)

**The PR URL must always be printed as the last thing in your response — no exceptions.**

Capture the PR URL immediately after `gh pr create`. Print it even if verification checks fail or cannot run (e.g. preview is behind SSO). If `gh pr create` fails, print the branch URL instead.

After the PR is created, output this exact block:

```
✅ Done

- Backend revision: <cloud-run-revision-name>
- Preview: <vercel-preview-url>
- PR: <github-pr-url>

GCP Pipeline checks:
- Scheduler jobs all ENABLED:        ✅/❌
- Midday URI has ?skip_gemini=true:  ✅/❌
- Latest Cloud Run revision Ready:   ✅/❌
- Latest Cloud Build SUCCESS:        ✅/❌
- Gemini 429s last 24h (<5):         ✅/❌
- Bake checkpoint stages_failed=0:   ✅/❌
- No 5xx on /refresh/ routes (2h):   ✅/❌
- No Finnhub errors last 6h:         ✅/❌

Stale-data check:
- /screener date==today:         ✅/❌
- /signals date==today:          ✅/❌
- /industry-intel date==today:   ✅/❌
- /industry-returns date==today: ✅/❌
- /market-overview all sections: ✅/❌
- /content all 4 articles today: ✅/❌
- /macro-pulse + /earnings date: ✅/❌
- All 12 cache keys present:     ✅/❌
- industry_cache freshness <25h: ✅/❌
- Scheduler 5 jobs ENABLED/OK:   ✅/❌

UI check:
- Screener date subtitle today:        ✅/❌
- Signals top-10 panels:               ✅/❌
- Confluence bars + detail text:       ✅/❌
- Correlation article date today:      ✅/❌
- Backend /health + no missing routes: ✅/❌
```

All results must be reported. If the Vercel preview is behind SSO and cannot be curl-fetched, verify against source code and mark each item with `✅ (source)` or `❌`. Never end the command without printing the PR URL.
