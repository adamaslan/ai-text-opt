# Health Check Command

Check the backend (Cloud Run) and frontend (Vercel) are live and Firestore cache is populated.

## Run

```bash
export GCP_PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
BACKEND=$(gcloud run services describe gcp3-backend --region us-central1 --format="value(status.url)" 2>/dev/null)
FRONTEND="https://gcp3-frontend-adam-aslans-projects.vercel.app"

echo "=== Backend health ==="
curl -s "$BACKEND/health" | python3 -m json.tool || echo "❌ Backend unreachable"

echo "=== Industry Intel ==="
curl -s "$BACKEND/industry-intel?view=compact" | python3 -c '
import sys, json
d = json.load(sys.stdin)
print("Date:", d.get("date"))
print("Industries:", len(d.get("industries", {})))
leaders = sorted(
    [name for name, info in d.get("industries", {}).items() if info.get("change_pct") is not None],
    key=lambda name: d["industries"][name].get("change_pct", 0),
    reverse=True,
)
print("Leaders:", leaders[:3])
' 2>/dev/null || echo "❌ Industry intel failed"

echo "=== Market Overview (brief only) ==="
curl -s "$BACKEND/market-overview?sections=brief" | python3 -c '
import sys, json
d = json.load(sys.stdin)
brief = d.get("brief", {})
print("Tone:", brief.get("market_tone"))
print("Avg change:", f"{brief.get('avg_change_pct')}%")
' 2>/dev/null || echo "❌ Market overview failed"

echo "=== Frontend (Vercel) ==="
curl -s -o /dev/null -w "Status: %{http_code}\n" "$FRONTEND" || echo "❌ Frontend unreachable"

echo "=== Scheduler jobs ==="
gcloud scheduler jobs list --location=us-central1 --filter='name:gcp3-*' --format='table(name,schedule,httpTarget.uri,latestRunTime)'
echo
for job in gcp3-premarket-warmup gcp3-ai-summary-refresh gcp3-midday-intraday-refresh gcp3-eod-intraday-refresh gcp3-nightly-cache-purge; do
  echo "=== $job last 10 scheduler log entries ==="
  gcloud logging read \
    "resource.type=\"cloud_scheduler_job\" AND resource.labels.job_id=\"$job\"" \
    --project="$GCP_PROJECT_ID" \
    --limit=10 \
    --order=desc \
    --format='table(timestamp,severity,logName,textPayload)'
  echo
 done
```

## Expected Output

```
=== Backend health ===
{"status": "ok"}

=== Industry Intel ===
Date: 2026-04-08
Industries: 50
Leaders: ['Semiconductors', 'Biotech', 'Energy']

=== Market Overview (brief only) ===
Tone: neutral
Avg change: -0.48%

=== Frontend (Vercel) ===
Status: 200

=== Scheduler jobs ===
NAME                        SCHEDULE     HTTP_TARGET.URI                       LATEST_RUN_TIME
...                         ...          https://.../refresh/premarket      2026-04-08T12:30:00Z

=== gcp3-premarket-warmup last 10 scheduler log entries ===
... table output ...
```

## If Something Is Down

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Backend 503 | Missing `FINNHUB_API_KEY` | `gcloud run services update gcp3-backend --set-secrets=...` |
| Backend 500 | Missing `GCP_PROJECT_ID` | `gcloud run services update gcp3-backend --set-env-vars=GCP_PROJECT_ID=...` |
| Frontend 4xx/5xx | `BACKEND_URL` env var not set in Vercel | Set in Vercel dashboard → Project → Settings → Environment Variables |
| Frontend not found | Not deployed yet | Run `/verpr` skill |
