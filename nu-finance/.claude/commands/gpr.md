# GCP Deploy + PR

Deploys the Python backend (`backend/`) to GCP Cloud Run via Cloud Build, then creates a branch, commits, and opens a PR against `main`.

> **Scope**: Backend only — `backend/`. For frontend (Vercel), use `/verpr`.

## Pipeline Overview

The backend runs a **fetch/bake** scheduler pipeline:
- `13:30 UTC` → `POST /refresh/fetch` — data ingestion (ETF seed, Finnhub, AV, industry)
- `13:45 UTC` → `POST /refresh/bake` — AI synthesis (compute_returns, industry_returns, Gemini content)
- `16:00 UTC` → `POST /refresh/intraday` — midday refresh (macro, news, screener, story)
- `20:15 UTC` → `POST /refresh/intraday?skip_gemini=true` — EOD refresh (no Gemini)
- `06:00 UTC` → `POST /admin/purge-cache` — nightly Firestore TTL sweep

Every deploy must keep this pipeline intact. Verify secrets and scheduler jobs are not disrupted.

## Pre-Deploy Checklist

Run these checks before touching `cloudbuild.yaml` or env vars:

**Rule 1** — Never `--set-env-vars` for secrets. Always:
```
--set-secrets=FINNHUB_API_KEY=FINNHUB_API_KEY:latest,GEMINI_API_KEY=GEMINI_API_KEY:latest,SCHEDULER_SECRET=SCHEDULER_SECRET:latest
```

**Rule 2** — `cloudbuild.yaml` must list all 3 secrets in `--set-secrets`. A missing secret is silently dropped on deploy and breaks the scheduler pipeline.

**Rule 3** — If rotating `SCHEDULER_SECRET`: update Secret Manager first → update all 5 scheduler jobs → redeploy. Never partial.

**Rule 4** — If backend URL changes: update all 5 scheduler job URIs (gcp3-premarket-warmup, gcp3-ai-summary-refresh, gcp3-midday-intraday-refresh, gcp3-eod-intraday-refresh, gcp3-nightly-cache-purge).

## Steps

1. Checkout a fresh branch off `main` — never reuse the current branch
2. Secret scan ALL changed files — hard stop if any secrets found
3. Stage only changed files by name — never `git add .`
4. Run tests in `fin-ai1` mamba environment
5. Build Docker image locally to confirm it compiles
6. Verify `cloudbuild.yaml` lists all 3 secrets before submitting
7. Deploy via Cloud Build
8. Run `/post-deploy-verify` to validate rules 1–4 automatically
9. Commit, push, open PR

## Secret Scanning Rules

**Hard blockers — stop immediately if any are found.**

Never stage or commit:
- `.env`, `.env.*`, `*.key`, `*.pem`, `*.p12`, `*.pfx`
- `credentials.json`, `service-account*.json`, `*-key.json`
- Any file containing: `AIzaSy`, `GOCSPX-`, `ya29.`, `"private_key"`, `sk_live_`

If found: remove the secret → use `os.getenv()` → store in `.env` (gitignored) or Secret Manager → re-scan → then stage.

Never use `--no-verify` to bypass the pre-commit hook.

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
git add backend/<file1> backend/<file2> ...
git diff --cached --name-only  # verify what's staged

# 4. Tests (fin-ai1 environment)
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && \
  source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/mamba.sh && \
  mamba activate fin-ai1 && \
  pytest backend/ -v

# 5. Docker build (compile check)
docker build -t gcp3-backend:preview backend/

# 6. Verify cloudbuild.yaml has all 3 secrets before deploying
grep "FINNHUB_API_KEY\|GEMINI_API_KEY\|SCHEDULER_SECRET" backend/cloudbuild.yaml
# All 3 must appear. If any is missing — add it before proceeding.

# 7. Deploy via Cloud Build
gcloud builds submit --config backend/cloudbuild.yaml \
  --project ttb-lang1 \
  backend/

# 8. Confirm service URL is up
gcloud run services describe gcp3-backend \
  --project ttb-lang1 \
  --region us-central1 \
  --format="value(status.conditions[0].status,status.url)"

# 9. Post-deploy verification (validates all 4 backend change rules)
# Run /post-deploy-verify skill or manually:
BACKEND_URL=$(gcloud run services describe gcp3-backend \
  --project ttb-lang1 --region us-central1 \
  --format="value(status.url)")
curl -sf "$BACKEND_URL/health" | python3 -m json.tool
curl -sf "$BACKEND_URL/debug/status" | python3 -m json.tool

# 10. If deploy fails — read logs before retrying
gcloud logging read \
  'resource.type="cloud_run_revision" resource.labels.service_name="gcp3-backend"' \
  --project ttb-lang1 --limit 50 \
  --format="value(timestamp, textPayload)" --freshness=10m

# 11. Commit (pre-commit hook re-scans automatically)
git commit -m "$(cat <<'EOF'
type(scope): description

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"

# 12. Push and open PR
git push -u origin HEAD
gh pr create \
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

## Test Plan
- [ ] Secret scan passed (pre-commit hook did not block)
- [ ] `pytest backend/` passed in fin-ai1 environment
- [ ] Docker image built successfully
- [ ] Cloud Run deployed and `/health` returns `{"status": "ok"}`
- [ ] `/debug/status` shows no missing expected routes
- [ ] `/post-deploy-verify` passed (all 4 backend change rules green)
- [ ] Scheduler pipeline unaffected: fetch/bake/intraday/EOD/purge jobs still point to correct URL

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

## Diagnosing Failures

| Symptom | Command |
|---------|---------|
| Deploy failed | `gcloud logging read ... --freshness=10m` |
| Scheduler jobs firing at dead endpoint | `gcloud scheduler jobs list --location=us-central1 --project=ttb-lang1` |
| Cache not refreshing after deploy | `curl $BACKEND_URL/debug/status` — check `gcp3_cache.live_doc_count` |
| Missing routes after deploy | `/debug/status` → `missing_expected_routes` field |
| yfinance 429s in seed | Don't re-trigger `/admin/seed-etf-history` within 2h of scheduled fetch (13:30 UTC) |
| Gemini key in logs | Ensure `logging.getLogger("httpx").setLevel(logging.WARNING)` is in `main.py` |
