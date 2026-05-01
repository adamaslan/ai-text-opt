# Cloud Run Log Debug

Deep inspection of Cloud Run logs to verify every backend feature is working correctly. Goes beyond HTTP status codes — checks for stage-level failures, cache hit/miss rates, Finnhub/Gemini errors, slow modules, and OIDC auth rejections.

## Run

```bash
PROJECT="${GCP_PROJECT_ID:-ttb-lang1}"
REGION="us-central1"
SERVICE="gcp3-backend"
WINDOW="${1:-60}"   # minutes to look back, default 60, override: /log-debug 120

PASS=0; FAIL=0; WARN=0
pass() { PASS=$((PASS+1)); echo "  ✅ $1"; }
fail() { FAIL=$((FAIL+1)); echo "  ❌ $1"; }
warn() { WARN=$((WARN+1)); echo "  ⚠️  $1"; }

echo "=== Cloud Run Log Debug — last ${WINDOW} min ==="
echo "    Project : $PROJECT"
echo "    Service : $SERVICE"
echo "    Window  : ${WINDOW}m"
echo ""

# Pull all logs once into a temp file to avoid repeated gcloud calls
TMPLOG=$(mktemp /tmp/gcp3-logdebug.XXXXXX)
gcloud run services logs read "$SERVICE" \
  --region "$REGION" --project "$PROJECT" \
  --limit 1000 \
  --format="value(timestamp,severity,textPayload,jsonPayload.message)" \
  2>/dev/null > "$TMPLOG"

TOTAL_LINES=$(wc -l < "$TMPLOG")
echo "  Fetched $TOTAL_LINES log lines"
echo ""

# ─── 1. ERROR / CRITICAL count ───────────────────────────────────────
echo "=== 1. Error-level log count ==="
python3 - "$TMPLOG" "$WINDOW" <<'EOF'
import sys, re
from datetime import datetime, timezone, timedelta

logfile, window = sys.argv[1], int(sys.argv[2])
cutoff = datetime.now(timezone.utc) - timedelta(minutes=window)

errors, criticals, exceptions = [], [], []
for line in open(logfile):
    m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
    if not m:
        continue
    try:
        ts = datetime.fromisoformat(m.group(1) + '+00:00')
    except ValueError:
        continue
    if ts < cutoff:
        continue
    if 'CRITICAL' in line:
        criticals.append(line.strip()[:200])
    elif 'ERROR' in line:
        errors.append(line.strip()[:200])
    elif 'Traceback' in line or 'Exception' in line:
        exceptions.append(line.strip()[:200])

if criticals:
    print(f"  ❌ {len(criticals)} CRITICAL entries:")
    for e in criticals[-3:]: print(f"    {e}")
elif errors:
    print(f"  ❌ {len(errors)} ERROR entries (last 3):")
    for e in errors[-3:]: print(f"    {e}")
elif exceptions:
    print(f"  ⚠️  {len(exceptions)} unhandled Exception traces:")
    for e in exceptions[-3:]: print(f"    {e}")
else:
    print("  ✅ No ERROR/CRITICAL/Exception log entries in window")
EOF

# ─── 2. Refresh pipeline stage health ────────────────────────────────
echo ""
echo "=== 2. Refresh pipeline stage health ==="
python3 - "$TMPLOG" "$WINDOW" <<'EOF'
import sys, re, json
from datetime import datetime, timezone, timedelta

logfile, window = sys.argv[1], int(sys.argv[2])
cutoff = datetime.now(timezone.utc) - timedelta(minutes=window)

stage_errors = []
stage_oks = []
refresh_runs = []

for line in open(logfile):
    m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
    if not m:
        continue
    try:
        ts = datetime.fromisoformat(m.group(1) + '+00:00')
    except ValueError:
        continue
    if ts < cutoff:
        continue
    if re.search(r'refresh/(all|premarket|intraday) stage \d+ error', line, re.I):
        stage_errors.append(line.strip()[:200])
    if re.search(r'refresh/(all|premarket|intraday).*(complete|started)', line, re.I):
        refresh_runs.append(line.strip()[:160])
    if re.search(r'"status": "ok"', line) and 'stage' in line.lower():
        stage_oks.append(line.strip()[:160])

if refresh_runs:
    print(f"  ✅ {len(refresh_runs)} refresh run(s) logged in window")
    for r in refresh_runs[-3:]: print(f"    {r}")
else:
    print("  ⚠️  No refresh pipeline runs found in window (scheduled jobs may not have fired yet)")

if stage_errors:
    print(f"  ❌ {len(stage_errors)} stage-level failure(s):")
    for e in stage_errors[-5:]: print(f"    {e}")
else:
    print("  ✅ No stage-level failures found")
EOF

# ─── 3. Per-module feature check ─────────────────────────────────────
echo ""
echo "=== 3. Per-module feature log presence ==="
python3 - "$TMPLOG" "$WINDOW" <<'EOF'
import sys, re
from datetime import datetime, timezone, timedelta

logfile, window = sys.argv[1], int(sys.argv[2])
cutoff = datetime.now(timezone.utc) - timedelta(minutes=window)

lines_in_window = []
for line in open(logfile):
    m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
    if not m:
        continue
    try:
        ts = datetime.fromisoformat(m.group(1) + '+00:00')
    except ValueError:
        continue
    if ts >= cutoff:
        lines_in_window.append(line)

combined = '\n'.join(lines_in_window)

# Each tuple: (feature_label, log_signal, is_critical)
features = [
    ("morning_brief",         r'GET /morning-brief|morning.brief|get_morning_brief',       False),
    ("macro_pulse",           r'macro.pulse|get_macro_pulse',                               False),
    ("screener",              r'GET /screener|get_screener_data',                           False),
    ("sector_rotation",       r'sector.rotation|get_sector_rotation',                       False),
    ("earnings_radar",        r'earnings.radar|get_earnings_radar',                         False),
    ("news_sentiment",        r'news.sentiment|get_news_sentiment',                         False),
    ("industry_data",         r'GET /industry-intel|get_industry_data',                     True),
    ("industry_returns",      r'GET /industry-returns|get_industry_returns',                True),
    ("technical_signals",     r'GET /signals|get_technical_signals',                        False),
    ("ai_summary",            r'refresh.ai.summary|get_ai_summary',                         True),
    ("daily_blog",            r'daily.blog|refresh_daily_blog|get_daily_blog',              False),
    ("blog_review",           r'blog.review|refresh_blog_review',                           False),
    ("correlation_article",   r'correlation.article|refresh_correlation_article',           False),
    ("market_summary",        r'GET /market-overview|get_market_summary',                   False),
    ("purge_cache",           r'purge.cache|POST /admin/purge-cache',                       False),
    ("seed_etf_history",      r'seed.etf.history|seed_etf_history',                        False),
]

seen, not_seen_critical, not_seen_optional = [], [], []
for label, pattern, critical in features:
    if re.search(pattern, combined, re.I):
        seen.append(label)
    elif critical:
        not_seen_critical.append(label)
    else:
        not_seen_optional.append(label)

print(f"  Active ({len(seen)}): {', '.join(seen) if seen else 'none'}")
if not_seen_critical:
    print(f"  ❌ Critical modules with no log activity: {', '.join(not_seen_critical)}")
if not_seen_optional:
    print(f"  ⚠️  Optional modules with no log activity (may not have been called): {', '.join(not_seen_optional)}")
if not not_seen_critical:
    print("  ✅ All critical modules have log activity in window")
EOF

# ─── 4. Finnhub error detection ──────────────────────────────────────
echo ""
echo "=== 4. Finnhub / upstream API errors ==="
python3 - "$TMPLOG" "$WINDOW" <<'EOF'
import sys, re
from datetime import datetime, timezone, timedelta

logfile, window = sys.argv[1], int(sys.argv[2])
cutoff = datetime.now(timezone.utc) - timedelta(minutes=window)

finnhub_errors, av_errors, gemini_errors, httpx_errors = [], [], [], []
for line in open(logfile):
    m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
    if not m:
        continue
    try:
        ts = datetime.fromisoformat(m.group(1) + '+00:00')
    except ValueError:
        continue
    if ts < cutoff:
        continue
    ll = line.lower()
    if 'finnhub' in ll and ('error' in ll or '429' in ll or 'timeout' in ll or 'unavailable' in ll):
        finnhub_errors.append(line.strip()[:200])
    if ('alpha vantage' in ll or 'alphavantage' in ll) and ('error' in ll or 'limit' in ll):
        av_errors.append(line.strip()[:200])
    if 'gemini' in ll and ('error' in ll or 'quota' in ll or 'blocked' in ll):
        gemini_errors.append(line.strip()[:200])
    if 'httpx' in ll and ('connecterror' in ll or 'timeout' in ll or 'readtimeout' in ll):
        httpx_errors.append(line.strip()[:200])

for label, entries in [("Finnhub", finnhub_errors), ("Alpha Vantage", av_errors),
                        ("Gemini", gemini_errors), ("httpx transport", httpx_errors)]:
    if entries:
        print(f"  ❌ {label} errors ({len(entries)}):")
        for e in entries[-3:]: print(f"    {e}")
    else:
        print(f"  ✅ No {label} errors")
EOF

# ─── 5. Firestore cache hit/miss ratio ───────────────────────────────
echo ""
echo "=== 5. Firestore cache behaviour ==="
python3 - "$TMPLOG" "$WINDOW" <<'EOF'
import sys, re
from datetime import datetime, timezone, timedelta

logfile, window = sys.argv[1], int(sys.argv[2])
cutoff = datetime.now(timezone.utc) - timedelta(minutes=window)

hits, misses, writes, errors = 0, 0, 0, 0
for line in open(logfile):
    m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
    if not m:
        continue
    try:
        ts = datetime.fromisoformat(m.group(1) + '+00:00')
    except ValueError:
        continue
    if ts < cutoff:
        continue
    ll = line.lower()
    if 'cache hit' in ll or 'serving from cache' in ll or 'cache_hit' in ll:
        hits += 1
    elif 'cache miss' in ll or 'cache_miss' in ll or 'fetching fresh' in ll:
        misses += 1
    if 'cache written' in ll or 'stored in firestore' in ll or 'cache_write' in ll:
        writes += 1
    if 'firestore' in ll and 'error' in ll:
        errors += 1

total = hits + misses
ratio = f"{hits}/{total} ({100*hits//total}% hit rate)" if total else "no cache activity logged"
print(f"  Cache hits   : {hits}")
print(f"  Cache misses : {misses}")
print(f"  Cache writes : {writes}")
print(f"  Hit ratio    : {ratio}")
if errors:
    print(f"  ❌ Firestore errors : {errors}")
else:
    print(f"  ✅ No Firestore errors")
EOF

# ─── 6. OIDC / auth rejection check ──────────────────────────────────
echo ""
echo "=== 6. Auth / OIDC rejection check ==="
python3 - "$TMPLOG" "$WINDOW" <<'EOF'
import sys, re
from datetime import datetime, timezone, timedelta

logfile, window = sys.argv[1], int(sys.argv[2])
cutoff = datetime.now(timezone.utc) - timedelta(minutes=window)

auth_401s, oidc_failures = [], []
for line in open(logfile):
    m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
    if not m:
        continue
    try:
        ts = datetime.fromisoformat(m.group(1) + '+00:00')
    except ValueError:
        continue
    if ts < cutoff:
        continue
    if re.search(r'401|Unauthorized', line):
        auth_401s.append(line.strip()[:200])
    if re.search(r'oidc.token.verification.failed|token.*mismatch', line, re.I):
        oidc_failures.append(line.strip()[:200])

if oidc_failures:
    print(f"  ❌ OIDC verification failures ({len(oidc_failures)}):")
    for e in oidc_failures[-3:]: print(f"    {e}")
else:
    print("  ✅ No OIDC verification failures")

if auth_401s:
    print(f"  ⚠️  {len(auth_401s)} 401/Unauthorized entries (could be probes or OIDC rejections):")
    for e in auth_401s[-3:]: print(f"    {e}")
else:
    print("  ✅ No 401/Unauthorized log entries")
EOF

# ─── 7. Slow request detection (>10s) ────────────────────────────────
echo ""
echo "=== 7. Slow requests (>10 000ms) ==="
python3 - "$TMPLOG" "$WINDOW" <<'EOF'
import sys, re
from datetime import datetime, timezone, timedelta

logfile, window = sys.argv[1], int(sys.argv[2])
cutoff = datetime.now(timezone.utc) - timedelta(minutes=window)

slow = []
for line in open(logfile):
    m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
    if not m:
        continue
    try:
        ts = datetime.fromisoformat(m.group(1) + '+00:00')
    except ValueError:
        continue
    if ts < cutoff:
        continue
    # Match "total_ms=NNNNN" patterns logged by refresh endpoints
    tm = re.search(r'total_ms=(\d+)', line)
    if tm and int(tm.group(1)) > 10000:
        slow.append((int(tm.group(1)), line.strip()[:200]))

if slow:
    slow.sort(reverse=True)
    print(f"  ⚠️  {len(slow)} request(s) exceeded 10 000ms:")
    for ms, entry in slow[:5]: print(f"    {ms}ms — {entry}")
else:
    print("  ✅ No requests exceeded 10 000ms in window")
EOF

# ─── 8. 503 / upstream unavailable count ─────────────────────────────
echo ""
echo "=== 8. 503 / upstream unavailability ==="
python3 - "$TMPLOG" "$WINDOW" <<'EOF'
import sys, re
from datetime import datetime, timezone, timedelta

logfile, window = sys.argv[1], int(sys.argv[2])
cutoff = datetime.now(timezone.utc) - timedelta(minutes=window)

unavailable = []
for line in open(logfile):
    m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
    if not m:
        continue
    try:
        ts = datetime.fromisoformat(m.group(1) + '+00:00')
    except ValueError:
        continue
    if ts < cutoff:
        continue
    if re.search(r'503|unavailable|Service temporarily unavailable', line, re.I):
        unavailable.append(line.strip()[:200])

if unavailable:
    print(f"  ❌ {len(unavailable)} 503/unavailable entries:")
    for e in unavailable[-5:]: print(f"    {e}")
else:
    print("  ✅ No 503/unavailable log entries")
EOF

# ─── 9. Cold start detection ─────────────────────────────────────────
echo ""
echo "=== 9. Cold start detection ==="
python3 - "$TMPLOG" "$WINDOW" <<'EOF'
import sys, re
from datetime import datetime, timezone, timedelta

logfile, window = sys.argv[1], int(sys.argv[2])
cutoff = datetime.now(timezone.utc) - timedelta(minutes=window)

cold_starts = []
for line in open(logfile):
    m = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', line)
    if not m:
        continue
    try:
        ts = datetime.fromisoformat(m.group(1) + '+00:00')
    except ValueError:
        continue
    if ts < cutoff:
        continue
    # Cloud Run logs "Starting container" on cold starts
    if re.search(r'Starting container|container started|cold start', line, re.I):
        cold_starts.append(line.strip()[:160])

if cold_starts:
    print(f"  ⚠️  {len(cold_starts)} cold start(s) detected:")
    for e in cold_starts: print(f"    {e}")
else:
    print("  ✅ No cold starts detected in window")
EOF

# ─── 10. Summary ─────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  LOG DEBUG: $PASS passed, $FAIL failed, $WARN warnings"
echo "  Window: last ${WINDOW} min | Lines scanned: $TOTAL_LINES"
echo "════════════════════════════════════════════"
rm -f "$TMPLOG"

if [ "$FAIL" -gt 0 ]; then
  echo "  Status: ISSUES FOUND — review ❌ sections above"
  exit 1
elif [ "$WARN" -gt 0 ]; then
  echo "  Status: WARNINGS — review ⚠️  sections above"
  exit 0
else
  echo "  Status: ALL CLEAR"
  exit 0
fi
```

## What this checks

### Section 1 — Error-level log count
Counts ERROR, CRITICAL, and unhandled Exception/Traceback entries in the window. Shows the 3 most recent for immediate triage.

### Section 2 — Refresh pipeline stage health
Scans for stage-level failures emitted by `/refresh/all`, `/refresh/premarket`, and `/refresh/intraday`. Also confirms refresh runs were logged at all (catches scheduler jobs that silently stopped firing).

### Section 3 — Per-module feature activity
Checks log presence for all 16 backend modules: `morning_brief`, `macro_pulse`, `screener`, `sector_rotation`, `earnings_radar`, `news_sentiment`, `industry_data`, `industry_returns`, `technical_signals`, `ai_summary`, `daily_blog`, `blog_review`, `correlation_article`, `market_summary`, `purge_cache`, `seed_etf_history`. Critical modules (industry, ai_summary) flag as ❌ if absent; others as ⚠️.

### Section 4 — Upstream API error detection
Pattern-matches log lines for Finnhub 429s/timeouts, Alpha Vantage rate-limit errors, Gemini quota/blocked errors, and httpx transport failures.

### Section 5 — Firestore cache hit/miss ratio
Counts cache hits, misses, and writes using log signal patterns. A high miss rate during non-refresh hours indicates TTL misconfiguration.

### Section 6 — OIDC / auth rejection check
Detects failed OIDC token verifications and 401 responses, which would mean scheduler jobs are silently failing to authenticate.

### Section 7 — Slow request detection
Scans `total_ms=NNNNN` patterns from refresh endpoint logs. Anything over 10 000ms is surfaced with the exact duration.

### Section 8 — 503 / upstream unavailability
Catches any 503 responses or "Service temporarily unavailable" log entries, indicating Finnhub or another upstream was down.

### Section 9 — Cold start detection
Identifies Cloud Run cold starts in the window, which can explain latency spikes on the first request after a scheduled job fires.

## Usage

```bash
# Default: last 60 minutes
/log-debug

# Custom window: last 2 hours
/log-debug 120

# Post-deploy verification (tight window)
/log-debug 15
```

## Interpreting results

| Pattern | Likely cause |
|---------|-------------|
| Section 2: no refresh runs found | Scheduler job didn't fire — check `gcloud scheduler jobs describe` |
| Section 3: `ai_summary` absent | Gemini call failed or `/refresh/all` stage 5 errored |
| Section 4: Finnhub 429 | Hit 60 req/min cap — stagger scheduler jobs or reduce concurrent ETF fetches |
| Section 4: Gemini quota | Daily Gemini quota exhausted — check GCP quotas console |
| Section 5: 0 cache hits | Cache TTL too short or Firestore read errors |
| Section 6: OIDC failures | `gcp3-scheduler` SA missing `roles/run.invoker` or token audience mismatch |
| Section 7: >30 000ms | Stage 3 (50 ETF Finnhub calls) running slowly — check Finnhub latency |
| Section 9: cold starts | Minimum instance count is 0 — consider setting `--min-instances=1` |
