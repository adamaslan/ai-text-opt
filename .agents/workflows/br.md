---
description: Create Branch, Commit, and PR
---

// turbo-all

# Create Branch, Commit, and PR

This workflow automates the creation of a new branch from current changes, staging, committing, and opening a pull request. As an Antigravity agent, you will analyze the changes and autonomously generate appropriate names and messages.

## Rules
- **SECURITY FIRST:** NEVER commit `.env`, credential files, `lambda/bundle/`, `lambda/*.zip`, or files matching secret patterns.
- **Branch Naming:** Prefix branches with `feature/`, `fix/`, or `refactor/` as appropriate.
- **Commit Format:** Use semantic commits format: `type(scope): description`.
- **Co-Author:** Always append the Antigravity co-author credit at the end of the commit message body:
  `Co-Authored-By: Antigravity <noreply@google.com>`
- **PR Content:** The PR body should include a clear summary of changes and a test plan.

## Execution Steps

Execute these commands sequentially using your `run_command` tool. Due to the `// turbo-all` annotation, these steps are pre-approved to auto-run when you set `SafeToAutoRun: true`.

### 1. Analyze State
Run `git status` and `git diff` to inspect changes.
```bash
git status
git diff
```

### 2. Create Branch
Derive a branch name from the analyzed changes and check it out.
```bash
git checkout -b <derived_branch_name>
```

### 3. Stage Files
Carefully stage the appropriate files. Ensure you do not add sensitive files.
```bash
git add <specific_files>
```

### 4. Commit Changes
Create the commit with a robust message and the co-author tag.
```bash
git commit -m "<type>(<scope>): <description>

Co-Authored-By: Antigravity <noreply@google.com>"
```

### 5. Push Branch
Push the uniquely named branch to the origin.
```bash
git push -u origin HEAD
```

### 6. Create PR
Automatically generate a well-formatted PR against the target base branch (e.g. `main`, `master`, or `update` depending on the repo).
```bash
gh pr create --base <target_branch> --title "<Generated Title>" --body "<Generated Summary and Test Plan>"
```
