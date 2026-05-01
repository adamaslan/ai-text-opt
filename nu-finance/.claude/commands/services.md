# List Cloud Run Services

Show all Cloud Run services in the project with their URLs and status.

## Run

```bash
PROJECT="${GCP_PROJECT_ID:-ttb-lang1}"

echo "=== Cloud Run Services ==="
gcloud run services list --region us-central1 --project "$PROJECT" \
  --format="table(name,status.url,status.conditions[0].lastTransitionTime)"

echo ""
echo "=== Backend URL ==="
gcloud run services describe gcp3-backend --region us-central1 --project "$PROJECT" \
  --format="value(status.url)" 2>/dev/null || echo "gcp3-backend not found"

echo ""
echo "=== Frontend URL ==="
gcloud run services describe gcp3-frontend --region us-central1 --project "$PROJECT" \
  --format="value(status.url)" 2>/dev/null || echo "gcp3-frontend not found"
```
