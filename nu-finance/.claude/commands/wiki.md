# Wiki — LLM-Maintained Knowledge Base

Implements the Karpathy LLM Wiki pattern. Three immutable layers:

- `docs/wiki-gcp3/raw/` — source documents (user drops files; LLM reads, never writes)
- `docs/wiki-gcp3/` — LLM-written pages: entities, concepts, decisions, incidents, synthesis
- `docs/wiki-gcp3/SCHEMA.md` — conventions and workflow (co-evolved by user + LLM)

## Usage

```
/wiki ingest <path>   — integrate a new source into the wiki (reads SCHEMA.md for workflow)
/wiki query <q>       — answer a question from the wiki with citations
/wiki lint            — health-check: orphans, contradictions, stale claims, missing entities
/wiki init            — bootstrap wiki structure for this project
```

## Page Types

The wiki uses typed pages — the type prefix governs structure:

| Prefix | Purpose | Required Sections |
|--------|---------|-------------------|
| `entity-*.md` | One page per named system component | What it is, Where used, Known failures, Open questions |
| `concept-*.md` | Cross-cutting patterns / design philosophy | The pattern, Where it appears, Contradictions/tensions |
| `incident-*.md` | One page per production incident | Timeline, Root cause, Resolution, Impact on design |
| `decision-*.md` | Recorded design decisions | Decision, Context, Alternatives rejected, Validated by |
| `overview.md` | System map and synthesis | Stack, data flow, entity map, current health, open issues |

A single ingest typically creates or updates 3–10 pages. If it touches 1, not enough integration happened.

## Secret Scan (runs before every operation)

```bash
echo "=== Secret scan before wiki operation ==="
SECRET_COUNT=$(grep -rn \
  "AIzaSy\|GOCSPX-\|ya29\.\|private_key\|sk_live\|sk_test" \
  /Users/adamaslan/code/gcp3/docs/ \
  --include="*.md" --include="*.txt" \
  --exclude-dir=".git" 2>/dev/null \
  | grep -v "os\.environ\|process\.env\|your-.*-here\|placeholder\|example\|REDACTED" \
  | wc -l | tr -d ' ')

if [ "$SECRET_COUNT" -gt "0" ]; then
  echo "❌ BLOCKED: $SECRET_COUNT potential secret(s) found in docs/. Fix before continuing."
  exit 1
fi
echo "✓ Secret scan passed"
```

**Secret placeholders to use in wiki pages:** `{gcp-project-id}`, `{backend-url}`, `{secret-name}`, "stored in Secret Manager"

## Operation: init

Bootstrap the wiki directory structure:

```bash
mkdir -p /Users/adamaslan/code/gcp3/docs/wiki-gcp3/raw
```

Create these files:

**`docs/wiki-gcp3/SCHEMA.md`** — the authoritative conventions doc. Read it before every ingest. It defines page types, required sections, frontmatter format, contradiction/open-question markers, log format, and secret policy. Never rely on memory of SCHEMA.md — always read the current file.

**`docs/wiki-gcp3/index.md`** — content-oriented catalog, organized by page type:
```
# Wiki Index
_Last updated: {date}_

## Overview
## Entities
## Concepts
## Architecture
## Incidents
## Decisions
## Sources (raw/)
## Stubs Needed
```

**`docs/wiki-gcp3/log.md`** — append-only operation record:
```
# Wiki Log
_Format: `## [{date}] {operation} | {detail} | pages touched: N`_
Parse with: grep "^## \[" log.md | tail -10
```

## Operation: ingest

When user runs `/wiki ingest <path>`:

1. **Read `docs/wiki-gcp3/SCHEMA.md`** — refresh conventions before doing anything
2. **Secret scan** on the source file — abort if secrets found
3. **Read the source** — extract: key facts, decisions made, failures recorded, open questions raised
4. **Identify pages to create or update**:
   - New named component → create `entity-*.md`
   - New failure → create `incident-*.md` AND update every entity page it touches
   - Design choice revealed → create or update `decision-*.md`
   - Cross-cutting pattern → create or update `concept-*.md`
   - Contradiction with existing page → mark inline on both pages with `> ⚠️ Contradiction:`
5. **Never copy verbatim** — synthesize, integrate, link. Extract the *why*, not just the *what*.
6. **Update `index.md`** — add new pages; move any resolved stubs out of the Stubs section
7. **Append to `log.md`**: `## [{date}] ingest | {source title} | pages touched: N`

Source secret check:
```bash
SOURCE_SECRETS=$(grep -n \
  "AIzaSy\|GOCSPX-\|ya29\.\|private_key\|sk_live\|sk_test" \
  "<path>" 2>/dev/null \
  | grep -v "os\.environ\|process\.env\|your-.*-here\|placeholder\|REDACTED" \
  | wc -l | tr -d ' ')

if [ "$SOURCE_SECRETS" -gt "0" ]; then
  echo "❌ Source contains potential secrets — sanitize before ingesting"
  exit 1
fi
```

## Operation: query

When user runs `/wiki query <question>`:

1. **Read `index.md`** — identify relevant pages by type and topic
2. **Read those pages** — note any `⚠️ Contradiction` or `❓ Open question` markers relevant to the query
3. **Synthesize answer** with citations: `[entity-gemini-client.md](entity-gemini-client.md#known-failures)`
4. **Surface contradictions** — if the answer touches an unresolved contradiction, say so
5. **Offer to file** — if the answer is worth keeping: "Want me to add this as a wiki page?"
6. **Append to `log.md`**: `## [{date}] query | {question summary}`

## Operation: lint

When user runs `/wiki lint`:

```bash
echo "=== Wiki health check ==="

echo "--- Orphan pages (no inbound links) ---"
for f in /Users/adamaslan/code/gcp3/docs/wiki-gcp3/*.md; do
  name=$(basename "$f")
  [[ "$name" == "index.md" || "$name" == "log.md" || "$name" == "SCHEMA.md" ]] && continue
  links=$(grep -rl "$name" /Users/adamaslan/code/gcp3/docs/wiki-gcp3/ --include="*.md" | grep -v "^$f$" | wc -l | tr -d ' ')
  [ "$links" -eq "0" ] && echo "  ⚠ $name — no inbound links"
done

echo "--- Pages missing from index ---"
for f in /Users/adamaslan/code/gcp3/docs/wiki-gcp3/*.md; do
  name=$(basename "$f")
  [[ "$name" == "index.md" || "$name" == "log.md" || "$name" == "SCHEMA.md" ]] && continue
  grep -q "$name" /Users/adamaslan/code/gcp3/docs/wiki-gcp3/index.md 2>/dev/null \
    || echo "  ⚠ $name — not in index.md"
done

echo "--- Unresolved contradictions ---"
grep -rn "⚠️ Contradiction" /Users/adamaslan/code/gcp3/docs/wiki-gcp3/ --include="*.md" | grep -v "SCHEMA.md"

echo "--- Open questions ---"
grep -rn "❓ Open question" /Users/adamaslan/code/gcp3/docs/wiki-gcp3/ --include="*.md" | grep -v "SCHEMA.md"

echo "--- Secret scan ---"
grep -rn "AIzaSy\|GOCSPX-\|ya29\.\|private_key\|sk_live\|sk_test" \
  /Users/adamaslan/code/gcp3/docs/wiki-gcp3/ --include="*.md" 2>/dev/null \
  | grep -v "your-.*-here\|placeholder\|REDACTED" \
  && echo "❌ Secrets found — redact immediately" \
  || echo "✓ No secrets"

echo "--- Page count ---"
ls /Users/adamaslan/code/gcp3/docs/wiki-gcp3/*.md 2>/dev/null | wc -l | xargs echo "pages in wiki"
```

Then Claude-side review:
- Entity pages missing "Known Failures" entries that should be populated from incidents
- Incident pages not linked from the entity pages they affected
- Contradictions marked `⚠️` for more than 2 ingest cycles (resolve or document as intentional)
- Open questions `❓` that a web search or code grep could answer
- Concepts mentioned inline across 3+ pages that deserve their own `concept-*.md`
- Stubs in `index.md` that are now needed (new sources may fill them)

Append to `log.md`: `## [{date}] lint | {N issues found} | {summary}`

## Notes

- **SCHEMA.md is authoritative** — read it at the start of every ingest, not from memory
- Wiki lives in `docs/wiki-gcp3/` — tracked in the private `gcp3-wiki` repo, not the public `gcp3` repo
- Raw sources in `docs/wiki-gcp3/raw/` are never modified
- Obsidian: open `docs/wiki-gcp3/` as a vault for graph view of entity relationships
- `grep "^## \[" log.md | tail -10` — last 10 wiki operations at a glance
