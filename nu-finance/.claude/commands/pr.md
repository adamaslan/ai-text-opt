# Create Branch, Commit, and PR

Creates a new branch from current changes, commits everything, and opens a pull request.

## Steps

1. Check git status to see what's changed
2. Create a new branch if not already on a feature branch (branch name derived from changes)
3. Stage all changed/untracked files (excluding secrets)
4. Commit with a descriptive message
5. Push to remote
6. Create a PR against main

## Rules

- Never commit `.env`, credential files, or files matching secret patterns
- do a scan of all new files and changes to existing to make sure no secrects or sensitive data is shown

- Branch naming: `feature/`, `fix/`, or `refactor/` prefix
- Commit format: `type(scope): description`
- PR body includes summary and test plan

## Execute

Run these steps now for the current working directory:

```bash
# 1. Show current state
git status
git diff
git checkout -b <relevant branch name>

# 2. Stage relevant files (not secrets)
git add <specific files>

# 3. Commit
git commit -m "type(scope): description

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

# 4. Push
git push -u origin HEAD

# 5. Create PR and output the URL
PR_URL=$(gh pr create --title "..." --body "...")
echo "$PR_URL"
```

Analyze the actual changes and generate an appropriate branch name, commit message, and PR description automatically.

## Output Requirement

After creating the PR, you MUST output the URL as a clickable markdown link in your response:

**PR:** [owner/repo#NNN](URL)

Capture the URL from `gh pr create` output and always render it as a markdown link so the user can click it directly.
