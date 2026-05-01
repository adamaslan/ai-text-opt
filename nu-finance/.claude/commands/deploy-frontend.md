# Deploy Frontend

Runs Cloud Build to build and deploy the gcp3 frontend.

## Run

```bash
cd /Users/adamaslan/code/gcp3/frontend
echo "🚀 Deploying gcp3-frontend via Cloud Build..."
gcloud builds submit --config cloudbuild.yaml --project "${GCP_PROJECT_ID:-ttb-lang1}"
```
