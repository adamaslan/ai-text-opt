# Vercel MCP Deploy + PR Workflow

**Goal:** Ensure absolute production stability by locally validating the build, deploying a preview via Vercel, verifying its `READY` state via the Vercel MCP, and opening a context-rich Pull Request.

## 🛠️ Execution Steps (Agent Instructions)

### 1. Branch & Sync (Start Clean)
Never reuse the current branch. Always branch off the latest production-ready code.
```bash
git checkout main
git pull origin main
git checkout -b <type>/<short-description>
```
*Types:* `feature`, `fix`, `refactor`, `chore`.

### 2. Security & Staging
Stage all relevant files, explicitly excluding secrets. Leverage existing pre-commit hooks to prevent credential leakage before proceeding.
```bash
git add <specific files>
# Run pre-commit hooks if configured
pre-commit run --all-files 
```
*If the security scan fails, halt and fix the leaked secrets immediately.*

### 3. Production-Parity Local Build
Confirm the build passes locally. Run the build simulating the production environment to catch strict type/linting errors before they hit Vercel.
```bash
cd frontend && vercel build --prod
```
*If the build fails, diagnose the error, modify the code, re-stage, and rebuild until it passes cleanly.*

### 4. Deploy & MCP Verification (Critical)
Trigger the deployment, then immediately switch to **Vercel MCP Tools** to monitor the state. Terminal scraping is unreliable; use the API.
```bash
# Push the prebuilt assets to Vercel
vercel deploy --prebuilt > vercel_output.txt
```
**MCP Agent Action:** * Extract the deployment URL from `vercel_output.txt`.
* Call the Vercel MCP tool `get_deployment(url)` and poll until the state is exactly `READY`.
* If the state is `ERROR`, call the Vercel MCP tool `get_runtime_logs(projectId)` or `get_build_logs()`, analyze the failure, fix the code, and return to Step 2. Do NOT proceed to PR creation.

### 5. Commit
Once the Vercel MCP confirms a `READY` state, commit the code.
```bash
git commit -m "type(scope): description"
git push -u origin HEAD
```

### 6. Create Context-Rich PR
Create the PR and inject the verified Vercel preview URL directly into the body so reviewers can immediately test the visual changes.
```bash
# Capture the URL
PR_URL=$(gh pr create \
  --title "type(scope): description" \
  --body "### Summary
[Brief description of changes]

### Vercel Preview
🟢 Verified Ready: <Insert Vercel Preview URL here>

### Test Plan
- [ ] UI verified on preview link
- [ ] Build passed with zero errors")
```

***

## 🛑 Immutable Rules for AI Agent

1. **No Secrets:** Never commit `.env`, `.pem`, or files matching secret patterns.
2. **Hard Gate:** Step 6 (`gh pr create`) MUST NOT be executed unless Step 4 (`get_deployment`) returns a `READY` status.
3. **No Ghosting:** If an error occurs, explicitly state what failed and propose the fix. Do not silently skip steps.

## 🏁 Final Output (Required Format)

Once the `gh` CLI returns success, output the following exactly:

```text
✅ Deployment & PR Complete

- 🔍 Preview: <vercel-preview-url>
- 🔀 PR: <github-pr-url>
```
*(Both links must be present, accurate, and clickable).*
