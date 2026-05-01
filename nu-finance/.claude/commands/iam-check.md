# IAM & Permissions Check

Audit GCP IAM bindings, active identity, Cloud Run invoker policies, service accounts, and Secret Manager for gcp3.

## Run

```bash
PROJECT="${GCP_PROJECT_ID:-ttb-lang1}"

echo "=== Active Identity ==="
gcloud auth list --filter=status:ACTIVE --format="value(account)"
gcloud auth application-default print-access-token > /dev/null 2>&1 && echo "✓ ADC valid" || echo "❌ Run: gcloud auth application-default login"

echo ""
echo "=== Project IAM Bindings ==="
gcloud projects get-iam-policy "$PROJECT" \
  --format="table(bindings.role,bindings.members)" 2>/dev/null | head -40

echo ""
echo "=== Cloud Run Invoker Policies ==="
for svc in gcp3-backend gcp3-frontend; do
  echo "--- $svc ---"
  gcloud run services get-iam-policy "$svc" --region us-central1 --project "$PROJECT" 2>/dev/null || echo "  (not found)"
done

echo ""
echo "=== Service Accounts ==="
gcloud iam service-accounts list --project "$PROJECT" \
  --format="table(email,displayName,disabled)"

echo ""
echo "=== Secret Manager Secrets ==="
gcloud secrets list --project "$PROJECT" \
  --format="table(name,createTime)"
```
