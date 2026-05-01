# Secret Scan

Scan the codebase for hardcoded credentials, API keys, and sensitive values that should never be committed.

## Run

```bash
echo "=== Scanning for hardcoded secrets ==="
grep -r "AIzaSy\|GOCSPX-\|ya29\.\|private_key" \
  /Users/adamaslan/code/gcp3 \
  --include="*.py" --include="*.ts" --include="*.js" --include="*.json" \
  --exclude-dir=".git" --exclude-dir="node_modules" --exclude-dir=".next" \
  2>/dev/null | grep -v "os\.environ\|process\.env\|example\|placeholder\|your-" | head -20

echo ""
echo "=== Staged changes secret scan ==="
git -C /Users/adamaslan/code/gcp3 diff --cached | \
  grep -E "AIzaSy|GOCSPX-|ya29\.|private_key" && \
  echo "❌ POTENTIAL SECRET IN STAGED CHANGES" || echo "✓ Staged changes clean"

echo ""
echo "=== Sensitive files accidentally staged ==="
git -C /Users/adamaslan/code/gcp3 diff --cached --name-only | \
  grep -E "\.env|credentials|service-account" && \
  echo "❌ Do not commit these files!" || echo "✓ No sensitive files staged"

echo ""
echo "=== Untracked sensitive files ==="
git -C /Users/adamaslan/code/gcp3 ls-files --others --exclude-standard | \
  grep -E "\.env|\.key|credentials|service-account" || echo "✓ None found"
```
