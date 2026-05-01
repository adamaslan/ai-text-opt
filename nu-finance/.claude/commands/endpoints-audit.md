# Endpoints & Data Freshness Audit

Comprehensive diagnostic audit of all public endpoints checking for:
1. **Stale data** — cache TTLs, Firestore timestamps, last-updated fields
2. **Missing features** — 52w high/low, open/close prices, calculated fields
3. **Query param coverage** — view, sections, scope, type, days, symbol filters
4. **Frontend text** — outdated descriptions, missing field renderers
5. **Industry returns calculations** — verify multi-period return computation

## Endpoints Audited

| # | Endpoint | Purpose | Cache TTL | Query Params |
|---|----------|---------|-----------|-------------|
| 1 | `/industry-intel` | Live quotes + sector heatmap | 1h | `?view=compact` |
| 2 | `/signals` | Technical signals + aggregation | 6h | `?symbol=TICKER`, `?scope=industries` |
| 3 | `/signals/{ticker}` | Multi-timeframe signal matrix | live | — |
| 4 | `/industry-returns` | Multi-period ETF returns (1D→10Y) | 6h | — |
| 5 | `/screener` | 40+ large-cap momentum | 4h | — |
| 6 | `/market-overview` | Morning brief + AI + sentiment + history | 15m–24h | `?sections=brief,ai_summary,sentiment,history`, `?days=1–30` |
| 7 | `/content` | Daily blog, review, correlation, story | 6h–24h | `?type=blog\|review\|correlation\|story` |
| 8 | `/macro-pulse` | 11 macro indicators | 6h | — |
| 9 | `/earnings-radar` | Earnings beats/misses radar | 4h | — |
| D | `/health` | Liveness probe | live | — |
| D | `/debug/status` | Route inventory + cache freshness snapshot | live | — |
| D | `/debug/calibration` | Calibration model metadata from GCS | live | — |
| D | `/debug/costs` | Daily LLM cost breakdown by endpoint | live | — |
| D | `/debug/evals` | Baseline eval over last 20 screener signals | live | — |

> **D** = debug/internal endpoint; audited for availability only, not data freshness.

## Run

```bash
PROJECT="${GCP_PROJECT_ID:-ttb-lang1}"
REGION="us-central1"
SERVICE="gcp3-backend"
BACKEND_URL=$(gcloud run services describe "$SERVICE" \
  --region "$REGION" --project "$PROJECT" \
  --format="value(status.url)" 2>/dev/null)

echo "=== Endpoints & Data Freshness Audit — $(date -u '+%Y-%m-%d %H:%M UTC') ==="
echo "    Backend: $BACKEND_URL"
echo ""

# ─── 0. /health — liveness ──────────────────────────────────────────────────
echo "=== 0. /health (Liveness probe) ==="
HEALTH=$(curl -s "$BACKEND_URL/health" 2>/dev/null)
HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/health" 2>/dev/null)

python3 - <<PYEOF
import json
try:
    d = json.loads('''$HEALTH''')
except:
    print("  ❌ Failed to parse response")
    exit(1)
print(f"  HTTP $HEALTH_CODE — status={d.get('status','?')}  version={d.get('version','?')}  tools={d.get('tools','?')}")
PYEOF

echo ""

# ─── 1. /industry-intel ──────────────────────────────────────────────────────
echo "=== 1. /industry-intel (Live quotes + heatmap) ==="
INTEL=$(curl -s "$BACKEND_URL/industry-intel" 2>/dev/null)
INTEL_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/industry-intel" 2>/dev/null)
INTEL_COMPACT_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/industry-intel?view=compact" 2>/dev/null)

python3 - <<PYEOF
import json, sys
from datetime import datetime, timezone

try:
    d = json.loads('''$INTEL''')
except:
    print("  ❌ Failed to parse response")
    sys.exit(1)

print(f"  HTTP $INTEL_CODE (full)  HTTP $INTEL_COMPACT_CODE (compact)")

date_val = d.get("date", "?")
quotes_ts = d.get("quotes_as_of", "?")
total = d.get("total", 0)
print(f"  Date: {date_val}  Total industries: {total}")

if quotes_ts != "?":
    try:
        ts_dt = datetime.fromisoformat(quotes_ts)
        now_utc = datetime.now(timezone.utc)
        age_min = (now_utc - ts_dt).total_seconds() / 60
        flag = "⚠️ " if age_min > 60 else "✅"
        print(f"  {flag} Quotes updated {age_min:.0f}m ago (TTL 60m)")
    except:
        print(f"  ⓘ  Could not parse quotes_as_of: {quotes_ts}")

industries = d.get("industries", {})
sample_ind = next(iter(industries.values())) if industries else {}
required = ["price", "change_pct", "source"]
missing = [f for f in required if f not in sample_ind]
print(f"  Quote fields: {'✅ all present' if not missing else '⚠️ missing ' + str(missing)}")

# Heatmap / leaders-laggards presence
has_heatmap = "heatmap" in d or "sector_heatmap" in d
has_leaders = "leaders" in d or "laggards" in d
print(f"  Heatmap: {'✅' if has_heatmap else 'ⓘ  not in payload'}")
print(f"  Leaders/laggards: {'✅' if has_leaders else 'ⓘ  not in payload'}")
PYEOF

echo ""

# ─── 2. /signals ────────────────────────────────────────────────────────────
echo "=== 2. /signals (Technical signals) ==="
SIGNALS=$(curl -s "$BACKEND_URL/signals" 2>/dev/null)
SIGNALS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/signals" 2>/dev/null)
SIGNALS_IND_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/signals?scope=industries" 2>/dev/null)

python3 - <<PYEOF
import json
from datetime import datetime, timezone

try:
    d = json.loads('''$SIGNALS''')
except:
    print("  ❌ Failed to parse response")
    exit(1)

print(f"  HTTP $SIGNALS_CODE (default)  HTTP $SIGNALS_IND_CODE (?scope=industries)")

timestamp = d.get("timestamp", "?")
if timestamp != "?":
    try:
        ts_dt = datetime.fromisoformat(timestamp)
        now_utc = datetime.now(timezone.utc)
        age_h = (now_utc - ts_dt).total_seconds() / 3600
        flag = "⚠️ " if age_h > 6 else "✅"
        print(f"  {flag} Data updated {age_h:.2f}h ago (TTL 6h)")
    except:
        pass

signals = d.get("signals", [])
print(f"  Total signals: {len(signals)}")

if signals:
    sample = signals[0]
    has_ma = "ma_signal" in sample or "moving_average" in sample
    has_rsi = "rsi_signal" in sample or "rsi" in sample
    has_direction = "direction" in sample or "signal" in sample
    print(f"  MA signal: {'✅' if has_ma else '⚠️ missing'}")
    print(f"  RSI signal: {'✅' if has_rsi else '⚠️ missing'}")
    print(f"  Direction/signal: {'✅' if has_direction else '⚠️ missing'}")
else:
    print("  ⚠️  No signals returned")
PYEOF

echo ""

# ─── 2b. /signals/{ticker} ──────────────────────────────────────────────────
echo "=== 2b. /signals/{ticker} (Per-ticker multi-timeframe matrix) ==="
TICKER_SIG_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/signals/AAPL" 2>/dev/null)
TICKER_SIG=$(curl -s "$BACKEND_URL/signals/AAPL" 2>/dev/null)

python3 - <<PYEOF
import json
try:
    d = json.loads('''$TICKER_SIG''')
except:
    print("  HTTP $TICKER_SIG_CODE  ❌ Failed to parse response")
    exit(1)

print(f"  HTTP $TICKER_SIG_CODE (AAPL probe)")
if "$TICKER_SIG_CODE" not in ("200",):
    print(f"  ⚠️  Non-200 response — check yfinance / feature_store availability")
    exit(0)

timeframes = d.get("timeframes", {})
alignment = d.get("alignment_score")
divergence = d.get("divergence_pattern")
expected_tfs = {"1D", "5D", "1M", "3M", "6M", "1Y"}
found_tfs = set(timeframes.keys()) if isinstance(timeframes, dict) else set()
missing_tfs = expected_tfs - found_tfs
print(f"  Timeframes: {', '.join(sorted(found_tfs)) or 'none'}")
print(f"  Missing timeframes: {'none ✅' if not missing_tfs else '⚠️ ' + str(missing_tfs)}")
print(f"  Alignment score: {'✅ ' + str(alignment) if alignment is not None else '⚠️ missing'}")
print(f"  Divergence pattern: {'✅ ' + str(divergence) if divergence is not None else '⚠️ missing'}")
PYEOF

echo ""

# ─── 3. /industry-returns ───────────────────────────────────────────────────
echo "=== 3. /industry-returns (Multi-period returns) ==="
RETURNS=$(curl -s "$BACKEND_URL/industry-returns" 2>/dev/null)
RETURNS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/industry-returns" 2>/dev/null)

python3 - <<PYEOF
import json
from datetime import datetime, timezone

try:
    d = json.loads('''$RETURNS''')
except:
    print("  ❌ Failed to parse response")
    exit(1)

print(f"  HTTP $RETURNS_CODE")

date_val = d.get("date", "?")
updated = d.get("updated", "?")
total = d.get("total", 0)
stale = d.get("stale", False)

print(f"  Date: {date_val}  Total industries: {total}")
print(f"  Stale flag: {'⚠️ YES — ' + str(d.get(\"stale_as_of\", \"?\")) if stale else '✅ NO'}")

# Check periods
periods_available = d.get("periods_available", [])
expected_periods = ["1d", "1w", "1m", "1y"]
missing = set(expected_periods) - set(periods_available)
print(f"  Periods: {periods_available}")
print(f"  Missing periods: {'none ✅' if not missing else '⚠️ ' + str(missing)}")

# Check 52w + open/close on a sample industry
industries = d.get("industries", [])
if industries:
    sample = industries[0]
    has_52w_high = "52w_high" in sample and sample["52w_high"] is not None
    has_52w_low = "52w_low" in sample and sample["52w_low"] is not None
    has_open = "open" in sample
    has_close = "close" in sample
    print(f"  52w_high: {'✅' if has_52w_high else '⚠️ missing/null'}")
    print(f"  52w_low:  {'✅' if has_52w_low else '⚠️ missing/null'}")
    print(f"  open:  {'✅' if has_open else 'ⓘ  not included (by design)'}")
    print(f"  close: {'✅' if has_close else 'ⓘ  not included (by design)'}")
    with_returns = [i for i in industries if i.get("returns")]
    pct = (len(with_returns) / len(industries)) * 100
    print(f"  Industries with returns: {len(with_returns)}/{len(industries)} ({pct:.0f}%)")
else:
    print("  ⚠️  No industries returned")
PYEOF

echo ""

# ─── 4. /screener ───────────────────────────────────────────────────────────
echo "=== 4. /screener (Momentum signals) ==="
SCREENER=$(curl -s "$BACKEND_URL/screener" 2>/dev/null)
SCREENER_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/screener" 2>/dev/null)

python3 - <<PYEOF
import json
from datetime import datetime, timezone

try:
    d = json.loads('''$SCREENER''')
except:
    print("  ❌ Failed to parse response")
    exit(1)

print(f"  HTTP $SCREENER_CODE")

timestamp = d.get("timestamp", "?")
stocks = d.get("stocks", d.get("quotes", {}))
stock_count = len(stocks) if isinstance(stocks, (list, dict)) else 0
print(f"  Total stocks: {stock_count}")

if isinstance(stocks, dict) and stocks:
    sample = next(iter(stocks.values()))
elif isinstance(stocks, list) and stocks:
    sample = stocks[0]
else:
    sample = {}

if sample:
    has_signal = "signal" in sample or "momentum" in sample
    has_breadth = "breadth_pct" in sample
    has_score = "score" in sample
    print(f"  Signal field: {'✅' if has_signal else '⚠️ missing'}")
    print(f"  Breadth %:    {'✅' if has_breadth else '⚠️ missing'}")
    print(f"  Score:        {'✅' if has_score else '⚠️ missing'}")

if timestamp != "?":
    try:
        ts_dt = datetime.fromisoformat(timestamp)
        now_utc = datetime.now(timezone.utc)
        age_h = (now_utc - ts_dt).total_seconds() / 3600
        flag = "⚠️ " if age_h > 4 else "✅"
        print(f"  {flag} Data updated {age_h:.2f}h ago (TTL 4h)")
    except:
        pass
PYEOF

echo ""

# ─── 5. /market-overview ────────────────────────────────────────────────────
echo "=== 5. /market-overview (Composite — all sections) ==="
OVERVIEW=$(curl -s "$BACKEND_URL/market-overview" 2>/dev/null)
OVERVIEW_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/market-overview" 2>/dev/null)
# Test section filter param
OVERVIEW_PARTIAL_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/market-overview?sections=brief,sentiment" 2>/dev/null)

python3 - <<PYEOF
import json

try:
    d = json.loads('''$OVERVIEW''')
except:
    print("  ❌ Failed to parse response")
    exit(1)

print(f"  HTTP $OVERVIEW_CODE (full)  HTTP $OVERVIEW_PARTIAL_CODE (?sections=brief,sentiment)")

sections = d.get("sections_included", [])
print(f"  Sections returned: {', '.join(sections)}")

for section in ["brief", "ai_summary", "sentiment", "history"]:
    data = d.get(section)
    if data is None:
        print(f"  ⚠️  {section}: missing from payload")
    elif isinstance(data, dict) and "error" in data:
        print(f"  ❌ {section}: error — {data['error'][:120]}")
    else:
        print(f"  ✅ {section}: present")

# Verify ?days param presence (structural check only)
brief = d.get("brief") or {}
has_open = "open" in brief or "prev_close" in brief
print(f"  brief.open/prev_close: {'✅' if has_open else 'ⓘ  check if present in brief schema'}")
PYEOF

echo ""

# ─── 6. /content ────────────────────────────────────────────────────────────
echo "=== 6. /content (AI articles — all types) ==="
CONTENT=$(curl -s "$BACKEND_URL/content" 2>/dev/null)
CONTENT_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/content" 2>/dev/null)
# Test ?type filter
CONTENT_BLOG_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/content?type=blog" 2>/dev/null)

python3 - <<PYEOF
import json
from datetime import datetime, timezone

try:
    d = json.loads('''$CONTENT''')
except:
    print("  ❌ Failed to parse response")
    exit(1)

print(f"  HTTP $CONTENT_CODE (all)  HTTP $CONTENT_BLOG_CODE (?type=blog)")

for article_type in ["blog", "review", "correlation", "story"]:
    article = d.get(article_type)
    if article is None:
        print(f"  ⚠️  {article_type}: missing")
    elif isinstance(article, dict):
        if "error" in article:
            print(f"  ❌ {article_type}: {article['error'][:120]}")
        else:
            timestamp = article.get("timestamp") or article.get("generated_at")
            if timestamp:
                try:
                    ts_dt = datetime.fromisoformat(timestamp)
                    now_utc = datetime.now(timezone.utc)
                    age_h = (now_utc - ts_dt).total_seconds() / 3600
                    flag = "⚠️ " if age_h > 24 else "✅"
                    print(f"  {flag} {article_type}: {age_h:.1f}h old")
                except:
                    print(f"  ✅ {article_type}: present (timestamp unparseable)")
            else:
                print(f"  ✅ {article_type}: present (no timestamp)")
PYEOF

echo ""

# ─── 7. /macro-pulse ────────────────────────────────────────────────────────
echo "=== 7. /macro-pulse (Macro indicators) ==="
MACRO=$(curl -s "$BACKEND_URL/macro-pulse" 2>/dev/null)
MACRO_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/macro-pulse" 2>/dev/null)

python3 - <<PYEOF
import json
from datetime import datetime, timezone

try:
    d = json.loads('''$MACRO''')
except:
    print("  ❌ Failed to parse response")
    exit(1)

print(f"  HTTP $MACRO_CODE")

indicators = d.get("indicators", [])
print(f"  Macro indicators: {len(indicators)}/11 {'✅' if len(indicators) >= 11 else '⚠️'}")
if indicators:
    sample = indicators[0]
    has_value = "value" in sample
    has_signal = "signal" in sample or "direction" in sample
    has_name = "name" in sample or "label" in sample
    print(f"  value field:  {'✅' if has_value else '⚠️ missing'}")
    print(f"  signal field: {'✅' if has_signal else '⚠️ missing'}")
    print(f"  name/label:   {'✅' if has_name else '⚠️ missing'}")

earnings = d.get("earnings", {})
if earnings:
    total = earnings.get("total", 0)
    beats = earnings.get("beats", 0)
    print(f"  Earnings: {total} reports, {beats} beats")
else:
    print(f"  ⚠️  Earnings section missing from macro-pulse")

timestamp = d.get("timestamp", d.get("updated", "?"))
if timestamp != "?":
    try:
        ts_dt = datetime.fromisoformat(timestamp)
        now_utc = datetime.now(timezone.utc)
        age_h = (now_utc - ts_dt).total_seconds() / 3600
        flag = "⚠️ " if age_h > 6 else "✅"
        print(f"  {flag} Data updated {age_h:.2f}h ago (TTL 6h)")
    except:
        pass
PYEOF

echo ""

# ─── 8. /earnings-radar ─────────────────────────────────────────────────────
echo "=== 8. /earnings-radar (Earnings beats/misses) ==="
EARNINGS=$(curl -s "$BACKEND_URL/earnings-radar" 2>/dev/null)
EARNINGS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/earnings-radar" 2>/dev/null)

python3 - <<PYEOF
import json
from datetime import datetime, timezone

try:
    d = json.loads('''$EARNINGS''')
except:
    print("  ❌ Failed to parse response")
    exit(1)

print(f"  HTTP $EARNINGS_CODE")

reports = d.get("reports", d.get("earnings", []))
total = len(reports) if isinstance(reports, list) else d.get("total", 0)
beats = sum(1 for r in reports if isinstance(r, dict) and r.get("beat")) if isinstance(reports, list) else d.get("beats", "?")
print(f"  Reports: {total}  Beats: {beats}")

timestamp = d.get("timestamp", d.get("updated", "?"))
if timestamp != "?":
    try:
        ts_dt = datetime.fromisoformat(timestamp)
        now_utc = datetime.now(timezone.utc)
        age_h = (now_utc - ts_dt).total_seconds() / 3600
        flag = "⚠️ " if age_h > 4 else "✅"
        print(f"  {flag} Data updated {age_h:.2f}h ago (TTL 4h)")
    except:
        pass

if isinstance(reports, list) and reports:
    sample = reports[0]
    for field in ["ticker", "eps_actual", "eps_estimate", "surprise_pct"]:
        present = field in sample
        print(f"  {field}: {'✅' if present else '⚠️ missing'}")
PYEOF

echo ""

# ─── Debug endpoints — availability only ────────────────────────────────────
echo "=== Debug endpoints (availability check) ==="
for path in /debug/status /debug/calibration /debug/costs /debug/evals; do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL$path" 2>/dev/null)
    if [ "$CODE" = "200" ]; then
        echo "  ✅ $path → $CODE"
    else
        echo "  ⚠️  $path → $CODE"
    fi
done

echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "=== Summary ==="
echo ""
echo "Common issues to investigate:"
echo "  • /industry-returns missing 52w_high/52w_low → check etf_store Firestore docs; run /admin/seed-etf-history"
echo "  • /signals/{ticker} non-200 → check yfinance availability + feature_store import path"
echo "  • Stale timestamps → check Cloud Scheduler jobs (gcp3-premarket-warmup, gcp3-midday-intraday-refresh, etc.)"
echo "  • Missing periods in /industry-returns → compute_returns() stage may have failed; run /admin/compute-returns"
echo "  • /content articles >24h → check /refresh/bake pipeline or Gemini quota"
echo "  • /earnings-radar missing fields → check earnings module data shape vs expected schema"
echo ""
echo "To deep-dive:"
echo "  • \`gcloud run services logs read gcp3-backend --limit 100 --region us-central1\`"
echo "  • \`/backend-debug\` — route drift + rate-limit diagnostics"
echo "  • \`/firestore-csv\` — dump gcp3_cache and industry_cache contents"
echo "  • \`curl \$BACKEND_URL/debug/status\` — live route inventory + cache freshness snapshot"
echo "  • \`curl \$BACKEND_URL/debug/costs\` — daily LLM cost breakdown"
echo ""
```

## Frontend Audit

Check TypeScript files for outdated text and missing field renderers:

```bash
cd frontend/src

# 1. Scan for hardcoded legacy endpoint references (should all be gone)
echo "=== Legacy Endpoint References (should be empty) ==="
grep -r "industry-tracker\|technical-signals\|daily-blog\|blog-review\|industry-quotes\|morning-brief\|ai-summary\|news-sentiment\|market-summary\|correlation-article" \
  . --include="*.tsx" --include="*.ts" 2>/dev/null | head -20

# 2. Verify new consolidated endpoints are used
echo ""
echo "=== Consolidated Endpoint Usage ==="
for ep in industry-intel signals industry-returns screener market-overview content macro-pulse earnings-radar; do
  count=$(grep -r "/$ep" . --include="*.tsx" --include="*.ts" 2>/dev/null | wc -l | tr -d ' ')
  echo "  $ep: $count references"
done

# 3. Check query param usage
echo ""
echo "=== Query Param Usage ==="
grep -r "view=compact\|scope=industries\|sections=\|type=blog\|type=review\|type=correlation\|type=story" \
  . --include="*.tsx" --include="*.ts" 2>/dev/null | head -20

# 4. Check for missing 52w high/low renderers
echo ""
echo "=== 52w High/Low Renderers ==="
grep -r "52w_high\|52w_low\|52w\|high_52\|low_52" . --include="*.tsx" --include="*.ts" 2>/dev/null

# 5. Check open/close price usage
echo ""
echo "=== Open/Close Price Usage ==="
grep -r "\.open\|\.close\|prev_close\|opening\|closing" components --include="*.tsx" 2>/dev/null | head -10

# 6. Check for /signals/{ticker} (multi-timeframe matrix) usage
echo ""
echo "=== Per-Ticker Signal Matrix Usage ==="
grep -r "signals/\$\|signals/\`\|/signals/\${" . --include="*.tsx" --include="*.ts" 2>/dev/null | head -10

# 7. Scan page descriptions for stale copy
echo ""
echo "=== Page Descriptions (check for stale text) ==="
grep -r "description=" . --include="*.tsx" -A1 2>/dev/null | head -30
```

## Industry Returns Calculation Check

Verify the multi-period return calculations in `backend/`:

```bash
# Check if compute_returns() is registered
grep -n "compute.returns\|admin/compute-returns" backend/main.py | head -10

# Verify the calculation periods
grep -n "RETURN_PERIODS\|1d.*1w.*1m\|periods_available" backend/industry_returns.py 2>/dev/null | head -15

# Check _attach_stored_returns wiring
python3 - <<EOF
with open("backend/industry.py") as f:
    content = f.read()
for fn in ["_attach_stored_returns", "seed_etf_history", "get_industry_returns"]:
    found = fn in content
    print(f"{'✅' if found else '⚠️ '} {fn}")
EOF
```

## Expected Findings

### Known Issues (Track Until Resolved)

- [ ] **52w high/low in /industry-returns** — Populated from etf_store; may be null if `/admin/seed-etf-history` hasn't run
- [ ] **`/signals/{ticker}` latency** — Live yfinance + feature_store fetch; may be slow or error under rate limits
- [ ] **Open/close prices** — Not included in any endpoint; requires intraday snapshot (design decision)
- [ ] **Industry returns stale outside market hours** — TTL 6h; expected behaviour

### Known Limitations (By Design)

- Open/close prices require intraday snapshot; currently using daily quotes
- 52w data requires etf_store to have full history (seed via `/admin/seed-etf-history`)
- `/signals/{ticker}` is live (no cache) — not suitable for bulk polling
- `/debug/*` endpoints are internal; non-200 in production may indicate missing GCS model or cost tracker

## Actions

1. **If data is stale:** Run `/backend-debug`, check Cloud Scheduler job last-run times
2. **If 52w fields are null:** Check Firestore `etf_store` collection; run `POST /admin/seed-etf-history`
3. **If returns are missing periods:** Run `POST /admin/compute-returns` manually
4. **If `/signals/{ticker}` errors:** Check `feature_store.py` import path and yfinance network access from Cloud Run
5. **If /content articles are stale:** Check `POST /refresh/bake` and Gemini quota in Cloud Run logs
6. **If frontend shows errors:** Cross-reference component TypeScript interfaces with actual endpoint schemas using `/debug/status` route inventory
