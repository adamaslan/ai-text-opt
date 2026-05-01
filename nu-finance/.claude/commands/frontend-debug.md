# Frontend Debug

Targeted diagnostic for frontend failures on `sectors.nuwrrrld.com`:

1. **Pages not loading / blank skeleton** — proxy route returning 4xx/5xx, or `BACKEND_URL` misconfigured in Vercel
2. **Stale data / old timestamps** — page renders but data is from a prior day
3. **Vercel env var drift** — `BACKEND_URL` points to wrong or stale Cloud Run URL

## Run

```bash
FRONTEND="${FRONTEND_URL:-https://sectors.nuwrrrld.com}"
PROJECT="${GCP_PROJECT_ID:-ttb-lang1}"
REGION="us-central1"
SERVICE="gcp3-backend"

PASS=0; FAIL=0; WARN=0
pass() { PASS=$((PASS+1)); echo "  ✅ $1"; }
fail() { FAIL=$((FAIL+1)); echo "  ❌ $1"; }
warn() { WARN=$((WARN+1)); echo "  ⚠️  $1"; }

echo "=== Frontend Debug — $(date -u '+%Y-%m-%d %H:%M UTC') ==="
echo "    Frontend : $FRONTEND"

BACKEND_URL=$(gcloud run services describe "$SERVICE" \
  --region "$REGION" --project "$PROJECT" \
  --format="value(status.url)" 2>/dev/null)
echo "    Backend  : ${BACKEND_URL:-unknown}"
echo ""

# ─── 1. Page load checks ─────────────────────────────────────────────────────
echo "=== 1. Page load status ==="
for PAGE in / /industry-intel /signals /industry-returns /screener /market-overview /macro /content; do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 "$FRONTEND$PAGE" 2>/dev/null)
  if   [ "$CODE" = "200" ]; then pass "Page $PAGE → $CODE"
  elif [ "$CODE" = "404" ]; then fail "Page $PAGE → 404 (Next.js route file missing or not deployed)"
  elif [ "$CODE" = "500" ]; then fail "Page $PAGE → 500 (server component crashed — check Vercel logs)"
  else                           fail "Page $PAGE → $CODE"
  fi
done

# ─── 2. API proxy routes ─────────────────────────────────────────────────────
echo ""
echo "=== 2. API proxy routes (frontend → backend) ==="
python3 - <<PYEOF
import subprocess, json

FRONTEND = "$FRONTEND"
routes = [
    ("/api/industry-intel",   "industry-intel page"),
    ("/api/signals",          "signals page"),
    ("/api/industry-returns", "industry-returns page"),
    ("/api/screener",         "screener page"),
    ("/api/market-overview",  "market-overview page"),
    ("/api/macro",            "macro page"),
    ("/api/content",          "content page"),
]

for path, label in routes:
    url = FRONTEND + path
    try:
        code = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "20", url],
            capture_output=True, text=True, timeout=25,
        ).stdout.strip()
        body = subprocess.run(
            ["curl", "-s", "--max-time", "20", url],
            capture_output=True, text=True, timeout=25,
        ).stdout[:400]
        try:
            parsed = json.loads(body)
            has_error = "error" in parsed
        except Exception:
            has_error = False

        if code == "200" and not has_error:
            print(f"  ✅ {path} → {code}")
        elif code == "200" and has_error:
            print(f"  ❌ {path} → {code} but body has error field: {body[:200]}")
            print(f"       Likely: backend returned error, or BACKEND_URL wrong")
        elif code == "503":
            print(f"  ⚠️  {path} → {code} (backend data source unavailable)")
        elif code == "404":
            print(f"  ❌ {path} → 404 — Next.js API route missing: src/app/api/{path.split('/')[-1]}/route.ts")
        elif code == "500":
            print(f"  ❌ {path} → 500 — server error (BACKEND_URL missing or network failure)")
            print(f"       Body: {body[:200]}")
        else:
            print(f"  ❌ {path} → {code}")
    except Exception as e:
        print(f"  ❌ {path} → request failed: {e}")
PYEOF

# ─── 3. Backend proxy connectivity ───────────────────────────────────────────
echo ""
echo "=== 3. Backend proxy connectivity ==="
if [ -z "$BACKEND_URL" ]; then
  warn "Could not resolve backend URL — skipping"
else
  DIRECT=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$BACKEND_URL/health" 2>/dev/null)
  PROXY=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 "$FRONTEND/api/screener" 2>/dev/null)

  if [ "$DIRECT" = "200" ] && { [ "$PROXY" = "200" ] || [ "$PROXY" = "503" ]; }; then
    pass "Backend direct=$DIRECT proxy=$PROXY — both reachable"
  elif [ "$DIRECT" = "200" ] && [ "$PROXY" = "500" ]; then
    fail "Backend alive (direct=$DIRECT) but proxy returns 500 — BACKEND_URL in Vercel is wrong or missing"
    echo "    Expected BACKEND_URL: $BACKEND_URL"
    echo "    Fix: Vercel dashboard → Project → Environment Variables → BACKEND_URL"
  elif [ "$DIRECT" = "200" ] && [ "$PROXY" = "404" ]; then
    fail "Proxy /api/screener → 404 — Next.js API route file missing"
  else
    fail "Backend direct=$DIRECT proxy=$PROXY"
  fi
fi

# ─── 4. Stale data detection ─────────────────────────────────────────────────
echo ""
echo "=== 4. Stale data detection ==="
python3 - <<PYEOF
import subprocess, json
from datetime import datetime, timezone

FRONTEND = "$FRONTEND"
today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
STALE_H = 26

checks = [
    ("/api/industry-returns", ["updated", "date"],        "industry-returns"),
    ("/api/signals",          ["date"],                   "signals"),
    ("/api/industry-intel",   ["date", "quotes_as_of"],   "industry-intel"),
]

for path, fields, label in checks:
    url = FRONTEND + path
    try:
        body = subprocess.run(
            ["curl", "-s", "--max-time", "20", url],
            capture_output=True, text=True, timeout=25,
        ).stdout
        d = json.loads(body)
    except Exception as e:
        print(f"  ⚠️  {label}: fetch failed ({e})")
        continue

    if "error" in d:
        print(f"  ❌ {label}: response has error — {d.get('error')}")
        continue

    data_date = d.get("date")
    if data_date and data_date != today:
        print(f"  ⚠️  {label}: date={data_date} (today={today}) — data from prior day")
    elif data_date:
        print(f"  ✅ {label}: date={data_date} current")

    for field in fields:
        val = d.get(field)
        if not val or field == "date":
            continue
        try:
            ts = datetime.fromisoformat(val.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            if age_h > STALE_H:
                print(f"  ❌ {label}.{field} is {age_h:.1f}h old — STALE (threshold {STALE_H}h)")
            elif age_h > 2:
                print(f"  ⚠️  {label}.{field} is {age_h:.1f}h old")
            else:
                print(f"  ✅ {label}.{field} is {age_h:.1f}h old — fresh")
        except Exception:
            pass

    if d.get("stale"):
        print(f"  ❌ {label}: backend flagged data as stale (stale_as_of={d.get('stale_as_of')})")
        print(f"       industry_cache not refreshed — run /backend-debug for root cause")
PYEOF

# ─── 5. Cache-Control / ISR headers ──────────────────────────────────────────
echo ""
echo "=== 5. Cache-Control / ISR headers ==="
for ROUTE in /api/industry-intel /api/signals /api/industry-returns /api/screener; do
  HEADERS=$(curl -sI --max-time 15 "$FRONTEND$ROUTE" 2>/dev/null)
  CC=$(echo "$HEADERS" | grep -i "^cache-control:" | head -1)
  VC=$(echo "$HEADERS" | grep -i "x-vercel-cache" | head -1)
  if echo "$CC" | grep -qi "s-maxage\|max-age"; then
    pass "$ROUTE  $CC"
  else
    warn "$ROUTE missing s-maxage — not ISR-cached: ${CC:-no Cache-Control header}"
  fi
  [ -n "$VC" ] && echo "    $VC"
done

# ─── 6. BACKEND_URL env var sanity ───────────────────────────────────────────
echo ""
echo "=== 6. BACKEND_URL env var sanity ==="
PROBE=$(curl -s --max-time 10 "$FRONTEND/api/screener" 2>/dev/null)
python3 - <<PEOF
import json

probe = '''$PROBE'''
backend = "$BACKEND_URL"

try:
    d = json.loads(probe)
    err = d.get("error", "") + " " + d.get("detail", "")
    if "BACKEND_URL not configured" in err:
        print(f"  ❌ Proxy reports BACKEND_URL not configured — env var missing in Vercel")
        print(f"       Fix: Vercel dashboard → Project → Env Vars → add BACKEND_URL={backend}")
    elif err.strip():
        print(f"  ⚠️  Proxy returned error: {err.strip()[:200]}")
    else:
        print(f"  ✅ Proxy returned data — BACKEND_URL appears configured")
except Exception:
    print(f"  ✅ Proxy returned non-error response — BACKEND_URL likely configured")
PEOF

# ─── 7. Summary ──────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "  FRONTEND DEBUG: $PASS passed, $FAIL failed, $WARN warnings"
echo "════════════════════════════════════════════"

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

### Section 1 — Page load status
Hits every page route and confirms HTTP 200. A 404 means the Next.js route file is missing; 500 means the server component crashed (usually `BACKEND_URL` missing or an unhandled null from the backend).

### Section 2 — API proxy routes
Tests each `/api/*` proxy and checks both status code and response body. A 200 with `{"error":"..."}` is treated as a failure — this is what happens when the backend returns 404 or 503 and the proxy forwards the error JSON, causing the page to render blank.

### Section 3 — Backend proxy connectivity
Compares direct backend response vs frontend proxy response. If the backend is alive but the proxy returns 500, `BACKEND_URL` in Vercel is misconfigured or missing.

### Section 4 — Stale data detection
Fetches each API route and inspects `date`, `updated`, and `quotes_as_of` fields. Flags data older than 26 hours as stale. Also detects the `"stale": true` field that `industry_returns.py` sets when serving expired fallback data — this is the exact signal for a rate-limit pipeline stall.

### Section 5 — Cache-Control / ISR headers
Verifies each API route includes `s-maxage` in its `Cache-Control` header. Missing headers mean every browser request hits the backend directly with no Vercel edge caching.

### Section 6 — BACKEND_URL env var sanity
Checks whether the proxy error messages indicate `BACKEND_URL` is unset in Vercel.

## Usage

```bash
# Default: checks sectors.nuwrrrld.com
/frontend-debug

# Check a different domain
FRONTEND_URL=https://staging.nuwrrrld.com /frontend-debug
```

## Fix runbook

| Symptom | Section | Fix |
|---------|---------|-----|
| Page returns 404 | 1 | Verify `src/app/<page>/page.tsx` exists and frontend is deployed |
| Proxy `/api/*` returns 500 | 2, 3 | `BACKEND_URL` not set in Vercel — add it in Vercel dashboard |
| Proxy returns `{"error":"Backend unavailable"}` | 2 | Backend is down or 404 — run `/backend-debug` |
| Data date is yesterday | 4 | Backend scheduler jobs failing silently — run `/backend-debug` and check section 6 for `code=2` jobs |
| All timestamps frozen at midnight UTC | 4 | Scheduler OIDC auth broken — missing `SCHEDULER_EXPECTED_AUDIENCE`/`SCHEDULER_EXPECTED_SA` on Cloud Run. See `/backend-debug` lessons |
| `"stale": true` in response | 4 | `industry_cache` not refreshed (rate-limit stall or fetch–bake checkpoint mismatch) — see `/backend-debug` section 4 |
| No `s-maxage` header | 5 | Add `Cache-Control: public, s-maxage=N` in the API route handler |
| `BACKEND_URL not configured` | 6 | Add `BACKEND_URL=<cloud-run-url>` in Vercel env vars and redeploy |
| Page shows old data after backend redeploy | 4 | ISR cache at Vercel edge. Check `/api/*` proxy directly — if proxy is fresh, the page just hasn't revalidated yet |
| Proxy `/api/*` also shows old data after backend redeploy | 4 | Backend three-layer cache (in-memory → Firestore → live). See `/backend-debug` lessons. Also: some `/api/*` routes have their own `revalidate` (see table below) |

## Lessons learned

### Frontend cache stack — three more layers on top of the backend
After the backend's own three-layer cache (in-memory → Firestore → live compute), the
frontend adds up to three more caching layers before the user sees data:

```
Backend cache (see /backend-debug lessons)
  ↓
Layer 4: Next.js API route fetch cache (next: { revalidate: N })
  ↓
Layer 5: Vercel API route CDN (Cache-Control: s-maxage=N)
  ↓
Layer 6: Next.js ISR page cache (export const revalidate = N)
  ↓
User sees data
```

**ISR and API cache values by page:**

| Page | Page ISR (s) | API route `revalidate` (s) | API `s-maxage` (s) |
|------|-------------|---------------------------|-------------------|
| `/industry-intel` | 60 | 60 | 60 |
| `/signals` | 3600 | 3600 | 3600 |
| `/market-overview` | 900 | 900 | 900 |
| `/macro` | 900 | 900 | no-cache |
| `/screener` | — | 3600 | 1800 |
| `/industry-returns` | — | — | no-cache |
| `/content` | 14400 | 14400 | 14400 |

### Diagnosing which cache layer is stale
When a page shows old data after a backend change, check each layer:

1. **Backend API directly** — `curl https://gcp3-backend-....run.app/signals`
   If stale → backend cache issue (see `/backend-debug`)
2. **Frontend `/api/*` proxy** — `curl https://sectors.nuwrrrld.com/api/signals`
   If stale but backend is fresh → Next.js API route `revalidate` hasn't expired
3. **Page route** — visit `https://sectors.nuwrrrld.com/signals`
   If stale but `/api/signals` is fresh → page ISR hasn't expired

**Fastest fix:** `vercel --prod` triggers a production redeploy that clears all ISR caches.

### Vercel preview vs production use different BACKEND_URL scopes
`BACKEND_URL` is scoped per Vercel environment (Preview vs Production).
If set in one scope but not the other, previews work but production doesn't (or vice versa).
**Check**: Vercel dashboard → Project → Settings → Environment Variables → verify both scopes.

### Backend response shape changes require a fresh frontend build
If the backend adds/removes/renames fields in its JSON response, `vercel --prebuilt` re-uses
the last local build output. The page's server component may render with missing data.
Always run `vercel build` before `vercel --prebuilt` when the backend API shape changes.
This applies to all endpoints — `/industry-returns` adding a new field, `/signals` changing
from `analysis`-shaped to `industry_cache`-shaped, etc.

### Timestamps frozen at midnight UTC ≠ stale data — check the scheduler first
When all `/api/*` endpoints report timestamps like `00:02 UTC` and the date is today, the
data is technically current but the scheduler has not run since midnight. Before concluding
the frontend is broken, verify that Cloud Scheduler jobs are actually succeeding:

```bash
gcloud scheduler jobs list --project ttb-lang1 --location us-central1 \
  --format="table(name,lastAttemptTime,status.code)"
```

`code=2` on every job means the scheduled refreshes are silently failing (usually OIDC auth).
The frontend shows today's date so section 4 stale detection passes — but the data is from
the morning seed, not an intraday refresh. Run `/backend-debug` to find the root cause.

### Distinguishing "data is old" from "data is missing"
Section 4 flags a timestamp as stale only if it exceeds 26 hours. This misses the case
where data is 8–20 hours old because intraday refreshes failed. For intraday pages
(`/market-overview`, `/industry-intel`, `/macro`), fresh data should be < 4 hours old
during market hours (9:30 AM – 4 PM ET). Outside market hours, midnight UTC timestamps
are expected and normal.

**Expected freshness by time of day (ET):**

| Time window | Expected `updated` age | If older |
|-------------|------------------------|----------|
| Before 9:30 AM ET | Up to ~16h (since last night's run) | Normal |
| 9:30–10:00 AM ET | ≤ 30 min after premarket-warmup | Scheduler may have failed |
| 10:00 AM – 4 PM ET | ≤ 4h (midday refresh at noon, EOD at 4:15) | Check scheduler jobs |
| After 4:30 PM ET | Same-day timestamp from EOD refresh | Normal |

### `/market-overview` shows blank sections or missing AI summary
`/market-overview` aggregates: `brief`, `ai_summary`, `sentiment`, `history`. If the
Gemini API call in `/refresh/bake` failed, `ai_summary` will be absent and the page
renders that section blank with no error.

**Check:** `curl https://sectors.nuwrrrld.com/api/market-overview | python3 -m json.tool | grep sections_included`
Expected: `["brief", "ai_summary", "sentiment", "history"]`
If `ai_summary` is missing from the list → bake failed. Check backend logs for `GEMINI` errors.

### `/content` page — 4h ISR means stale blog after a bake
`/content` has a 14400s (4h) ISR revalidation. After `/refresh/bake` writes a new daily
blog post, the page continues serving the old post for up to 4 hours via Vercel's edge
cache. This is intentional — content doesn't need real-time freshness.

If you need to force an update immediately: `vercel --prod` redeploy clears all ISR caches.

### `/macro` page — `no-cache` on the API route but 900s ISR on the page
The `/api/macro` route sets `Cache-Control: no-cache`, so the proxy always hits the backend.
But the `/macro` page itself has a 900s ISR revalidation. This means the page can be up to
15 minutes stale even when the API proxy is returning fresh data. Both layers are intentional
(macro data changes frequently; page rendering is expensive). If the page looks stale but
`/api/macro` is fresh, this is normal ISR lag — not a bug.
