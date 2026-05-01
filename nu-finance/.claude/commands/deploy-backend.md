# Deploy Backend to Cloud Run

Runs Cloud Build to build and deploy the gcp3 backend to Cloud Run, then verifies the deployment is healthy.

## Run

```bash
cd /Users/adamaslan/code/gcp3/backend
PROJECT="${GCP_PROJECT_ID:-ttb-lang1}"
REGION="us-central1"
SERVICE="gcp3-backend"

echo "Deploying $SERVICE via Cloud Build..."
gcloud builds submit --config cloudbuild.yaml --project "$PROJECT"

echo ""
echo "=== Deployment Status ==="
URL=$(gcloud run services describe "$SERVICE" --region "$REGION" \
  --format="value(status.url)" --project "$PROJECT")
REVISION=$(gcloud run services describe "$SERVICE" --region "$REGION" \
  --format="value(status.latestReadyRevisionName)" --project "$PROJECT")
echo "Service URL : $URL"
echo "Revision    : $REVISION"

echo ""
echo "=== Health Check ==="
curl -sf "$URL/health" && echo " /health OK" || echo " /health FAILED"

echo ""
echo "=== Debug Status ==="
curl -sf "$URL/debug/status" | python3 -m json.tool 2>/dev/null || echo "(debug/status unavailable)"
```
