# Backend Change Rules

Enforces the 4 secrets/scheduler rules on every backend change. Run mentally before touching `cloudbuild.yaml`, Cloud Run env vars, `SCHEDULER_SECRET`, or scheduler jobs.

## Rules

**Rule 1 — Never set secrets as plain `--set-env-vars`**

If you see or are about to write:
```
--set-env-vars=SCHEDULER_SECRET=<value>
```
Stop. That bypasses Secret Manager and creates token drift. Always use:
```
--set-secrets=SCHEDULER_SECRET=SCHEDULER_SECRET:latest
```

**Rule 2 — `cloudbuild.yaml` must list every secret the service needs**

Before deploying, confirm `cloudbuild.yaml` has all three secrets in `--set-secrets`:
```
FINNHUB_API_KEY=FINNHUB_API_KEY:latest,GEMINI_API_KEY=GEMINI_API_KEY:latest,SCHEDULER_SECRET=SCHEDULER_SECRET:latest
```
If a secret is missing here, Cloud Build silently drops it from the new revision.

**Rule 3 — When rotating `SCHEDULER_SECRET`, update Secret Manager first, then all 5 scheduler jobs, then redeploy**

Order matters. Never update one without the others or jobs will 401 silently for hours.

Jobs to update:
- `gcp3-ai-summary-refresh`
- `gcp3-premarket-warmup`
- `gcp3-midday-intraday-refresh`
- `gcp3-eod-intraday-refresh`
- `gcp3-nightly-cache-purge`

**Rule 4 — When the backend URL changes, update all 5 scheduler job URIs**

Cloud Run URLs are stable unless the service is deleted and recreated. If the URL ever changes, update every job's `--uri` flag or they silently fire at a dead endpoint.

## Check

Run after any backend change:

```bash
/post-deploy-verify
```

It validates all 4 rules automatically.

## Reference

Full explanation with rotation script and architecture options in `nudocs/scheduler-secrets-simplification.md`.
