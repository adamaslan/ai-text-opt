# Vercel Deploy + PR

Verifies the frontend builds and deploys successfully to Vercel, then creates a **brand new branch**, commits all changes, and opens a PR.

## Steps

1. Checkout a fresh branch off `main` — always start clean, never reuse the current branch
2. Stage all changed files (excluding secrets)
3. Run `vercel build` in `frontend/` to confirm the build passes locally
4. If build fails, diagnose and fix the error before continuing
5. Deploy to Vercel preview with `vercel --prebuilt` and confirm status is ● Ready
6. If deployment fails, check logs with `vercel inspect <url> --logs` and fix
7. Run the post-deploy verification checklist (see below)
8. Commit with a descriptive message
9. Push to remote
10. Create a PR against main

## Rules

- Never commit `.env`, credential files, or files matching secret patterns
- Scan all new/changed files for secrets before staging
- **Always checkout a brand new branch off main — never commit to the current branch or reuse an existing feature branch**
- Branch naming: `feature/`, `fix/`, or `refactor/` prefix
- Commit format: `type(scope): description`
- PR body includes summary and test plan
- **Only create the PR after the Vercel deployment is confirmed ● Ready**

## Execute

Run these steps now for the current working directory:

```bash
# 1. Checkout a brand new branch off main
git checkout main
git pull origin main
git checkout -b <feature|fix|refactor>/<short-description>

# 2. Stage relevant files (not secrets)
git add <specific files>

# 3. Build
cd frontend && vercel build

# 4. If build fails — fix, re-stage, then rebuild before continuing

# 5. Deploy preview
vercel --prebuilt

# 6. Confirm deployment is ● Ready
vercel inspect <deployment-url> --logs

# 7. If deploy fails — fix, re-stage, rebuild, redeploy, confirm before continuing

# 8. Commit
git commit -m "type(scope): description

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

# 9. Push
git push -u origin HEAD

# 10. Create PR and capture the URL
PR_URL=$(gh pr create --title "..." --body "...")

# 11. Output clickable PR link
echo "✅ PR created: $PR_URL"
```

Always start from a fresh branch off main. Diagnose and fix all build/deploy failures before creating the PR.

## Post-Deploy Verification Checklist

After the preview URL is live, fetch and verify these routes against the preview URL:

### Screener (`/screener`)
- [ ] Page loads without error
- [ ] Subtitle shows "X of 216 symbols screened · YYYY-MM-DD" (not a raw count like 201)
- [ ] "All" tab renders the full table without crashing — confirm no JS errors in console
- [ ] Columns sort correctly (click Price, Chg %, Signal headers)
- [ ] Rows with missing price/change show "—" instead of crashing

### Signals (`/signals`)
- [ ] Page loads without error
- [ ] Top-10 Buys confluence panel visible above the view toggle
- [ ] Top-10 Sells confluence panel visible alongside Buys
- [ ] Each row in the panels shows: rank, symbol, bar, confluence score (+X.XX), 1d change %
- [ ] Avg confluence score shown under Buy/Sell count boxes
- [ ] Cards show ConfluenceBar with colored fill and score label (HIGH/MEDIUM/LOW)
- [ ] Expanding signals on a card shows `detail` text (the one-sentence explanation) not just the label
- [ ] `ai_outlook` paragraph renders below the summary on each card
- [ ] Buys tab shows only BUY cards ranked by confluence_score descending
- [ ] Sells tab shows only SELL cards ranked by confluence_score descending

### Correlation (`/content?tab=correlation`)
- [ ] Breadth note mentions "pairs 3, 7" (not 9 and 12, which now use ETF-breadth)
- [ ] Screener breadth note references "216-stock cross-sector watchlist"
- [ ] Correlation snapshot badges show counts

### Screener note update
- [ ] AI Regime box description mentions "80 hand-picked" (not "tech-heavy")

Run each check and report pass/fail. If any check fails, fix before creating the PR.

## Final Output (required)

**The PR URL must always be printed as the last thing in your response — no exceptions.**

Capture the PR URL immediately after `gh pr create` and store it in a variable. Print it even if verification checks fail or cannot run (e.g. preview is behind SSO). If `gh pr create` fails, print the branch URL instead.

After the PR is created, output this exact block:

```
✅ Done

- Preview: <vercel-preview-url>
- PR: <github-pr-url>

Verification:
- Screener all tab: ✅/❌
- Signals top-10 panels: ✅/❌
- Confluence bars + detail text: ✅/❌
- Correlation breadth note: ✅/❌
```

All four verification results must be reported. If the preview is behind SSO and cannot be curl-fetched, verify against source code and mark each item with `✅ (source)` or `❌`. Never end the command without printing the PR URL.
