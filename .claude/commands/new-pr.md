# new-pr

Create a new branch, commit changes securely (checking for secrets), and open a pull request.

## Usage

```
/new-pr <branch_name> [--title "PR Title"] [--description "PR Description"]
```

If no arguments given, interactively prompt for: branch name, PR title, and description.

---

## Pipeline

### Step 0 — Validate inputs and check git status (you do this)

1. Run `git status` to check current state
2. Ask user to confirm:
   - **Which untracked/modified files should be committed?** (multiselect)
   - What should we **exclude from the commit?** (e.g., `node_modules/`, `.env`, build artifacts)
3. Prompt for:
   - **Branch name** (if not provided): suggest `feature/`, `fix/`, `add/`, `docs/` prefixes
   - **PR title** (if not provided): ask user
   - **PR description** (if not provided): ask user

### Step 1 — Security scan (you do this, do NOT delegate)

Before creating the branch, scan staged changes for secrets:

```bash
# Check for common secret patterns
git diff --cached | grep -iE "api[_-]?key|secret|password|token|credential|bearer|authorization" | head -20
```

**If secrets detected:**
- List the matches for the user
- Ask: "Do you want to continue? These may be real secrets."
- If user says NO, abort
- If user says YES (e.g., they're in .example files), proceed

**If no secrets found:**
- Print: "✅ No obvious secrets detected"

### Step 2 — Create branch and commit (you do this)

```bash
git checkout -b <branch_name>
git add <selected_files>
git commit -m "<commit_message_from_user_or_generated>"
git push -u origin <branch_name>
```

**Commit message format:**
- If user provided description, use that
- Otherwise, auto-generate from branch name + file changes
- Always end with: `Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>`

### Step 3 — Create PR (you do this)

```bash
gh pr create --title "<PR_title>" --body "<PR_body_with_test_plan>"
```

**PR body template:**

```
## Summary

<PR description from user or inferred from changes>

## Changes

- <file_category_1>: <what changed>
- <file_category_2>: <what changed>

## Test plan

- [ ] Changes reviewed for security (no secrets exposed)
- [ ] All files properly staged and committed
- [ ] PR body complete with context

## Links

Closes: <issue_link_if_applicable>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

### Step 4 — Verify and report (you do this)

1. Verify PR was created:
   ```bash
   gh pr view <branch_name> --json number,title,url
   ```

2. Report to user:
   ```
   ✅ PR #N created: <title>
   🔗 <url>
   
   Summary:
   - Branch: add/your-feature
   - Commits: 1
   - Files changed: 28
   - Status: Ready for review
   ```

3. Remind user:
   - To review the PR description for accuracy
   - To fill in any checklist items
   - To request reviewers if applicable

---

## Security Checklist

Before proceeding with commit, verify:

- [ ] No `.env` files committed (only `.env.example` or templates)
- [ ] No private keys (`.pem`, `.key`) committed
- [ ] No plaintext credentials in code
- [ ] No API keys hardcoded (should use environment variables)
- [ ] `node_modules/` excluded from git
- [ ] Build artifacts (dist/, build/) excluded
- [ ] Sensitive directories have proper `.gitignore` rules

---

## Examples

### Example 1: Interactive mode (no arguments)

```
/new-pr
→ What is your branch name? add/user-auth
→ What files should be committed? [select: src/, docs/, CHANGELOG.md]
→ What should we exclude? [node_modules/, .env]
→ PR title? Add user authentication system
→ PR description? Implements JWT-based auth with refresh tokens...
✅ PR #42 created: Add user authentication system
```

### Example 2: With arguments

```
/new-pr feature/rag-integration --title "Add RAG pipeline" --description "Integrates vector database with LLM inference"
✅ PR #43 created: Add RAG pipeline
```

### Example 3: Security warning

```
/new-pr fix/payment-bug
🚨 WARNING: Detected potential secrets:
  - STRIPE_API_KEY=sk_live_...
  - DATABASE_PASSWORD=...

Continue anyway? (y/n)
```

---

## Why this works

- **Interactive prompts** ensure user intent is clear before destructive git operations
- **Security scanning** prevents accidental secret leakage
- **Single commit per PR** keeps history clean
- **gh CLI integration** creates PR directly from CLI with proper formatting
- **Validation at each step** catches errors early
