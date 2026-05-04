# ai-text-opt-1024 — ChromaDB-Central Fork of nu-finance

New project at `/Users/adamaslan/code/ai-text-opt-1024` that mirrors nu-finance's
backend/frontend/ingest structure but replaces Zilliz + LlamaParse + VoyageAI with
ChromaDB (free local `PersistentClient` → upgradeable to Chroma Cloud) and local
1024D embeddings (no per-token cost).

---

## Cost Mapping Against Chroma Cloud Tiers

| Tier | Rate | Our approach |
|---|---|---|
| Written | $2.50 / GB | Embed once locally; upsert compressed float32 — ~4 KB/record for 1024D |
| Stored | $0.33 / GB / mo | 1024D × 4 bytes × N records; 10k records ≈ 40 MB ≈ $0.013/mo |
| Queried | $0.0075 / TB + $0.09 / GB returned | `include=["ids","distances"]` only by default; documents fetched on demand |
| Fork | $0.03 / call | Never used — local `PersistentClient` has no fork concept |
| Sync | $0.04 / GiB processed | Not used — local disk is source of truth |
| Doc extract | $0.01 / page | Plain-text markdown loader replaces LlamaParse ($0) |
| Web scrape | $0.01 / page | No web scraping in pipeline |

### Cost Ceiling & Alerting

A monthly budget guardrail prevents runaway spend on the cloud upgrade path:

- **Budget cap**: $5/mo soft cap, $20/mo hard cap (env vars `CHROMA_BUDGET_SOFT`, `CHROMA_BUDGET_HARD`)
- **Pre-flight check**: `ingest.py` estimates write cost from `len(chunks) × 4 KB` before any cloud upsert; aborts if projected write exceeds soft cap
- **Query metering**: `lib/rag.ts` logs estimated GB-returned per query to enable cost attribution per endpoint
- **Worst-case scenario documented**: 100k records × 1024D × float32 = 400 MB stored = $0.13/mo; 1M queries returning 10 docs each ≈ 9 GB returned = $0.81/query batch

---

## Project Structure

```
/Users/adamaslan/code/ai-text-opt-1024/
├── ingest.py               # replaces nu-finance/ingest.py
│                           #   reads docs/trader-qa/*.md as plain text (no LlamaParse)
│                           #   chunks with SentenceSplitter (llama_index.core)
│                           #   embeds locally with intfloat/e5-large-v2 (1024D)
│                           #   upserts into ChromaDB PersistentClient
├── backend/
│   ├── lib/
│   │   ├── chroma.ts       # replaces zilliz.ts — ChromaDB HTTP client singleton
│   │   ├── rag.ts          # replaces rag.ts — query via chromadb-js, no VoyageAI
│   │   ├── llm.ts          # copy from nu-finance (Gemini/Mistral, unchanged)
│   │   └── workflow.ts     # copy from nu-finance (unchanged)
│   ├── app/api/            # copy route structure from nu-finance
│   ├── scripts/
│   │   └── ingest.ts       # thin TS wrapper that calls ingest.py via child_process
│   ├── package.json        # drop @llamaindex/milvus, @zilliz/*; add chromadb
│   ├── .env.example        # CHROMA_* vars instead of ZILLIZ_*
│   └── next.config.js
├── frontend/               # copy from nu-finance frontend as-is
├── config/
│   └── pipeline.yaml       # embedding + chroma + io settings
├── environment.yml         # mamba env: sentence-transformers, chromadb, llama-index-core
└── .env.example            # CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE (cloud upgrade path)
```

---

## Build Steps

### Step 1 — Scaffold directories
Copy frontend verbatim. Copy `lib/llm.ts`, `lib/workflow.ts`, `app/` verbatim from
nu-finance. Copy route handlers. Create `config/` and `data/` directories.

### Step 2 — `ingest.py`
Plain-text markdown loader → sentence splitter → 1024D local embed → ChromaDB upsert.
- No LlamaParse — plain `open()` reads each `.md` file directly
- `SentenceSplitter(chunk_size=400, chunk_overlap=50)` from `llama_index.core`
- `intfloat/e5-large-v2` via SentenceTransformer for 1024D vectors
- Batched, idempotent upsert with `{file_hash}_{chunk_idx}` IDs (sha1 of source path)
- Metadata per chunk: `source_file`, `chunk_index`, `char_len`, `text_preview`, `content_hash`, `ingested_at`
- **Failure modes handled**:
  - Empty/whitespace-only chunks skipped with warning (not silently embedded as zero vectors)
  - Embedding dimension verified == 1024 before any upsert (fail fast on model mismatch)
  - SHA1 content hash on each chunk; unchanged chunks skip re-embed on rerun (saves write cost)
  - Per-batch retry with exponential backoff (3 attempts) for transient Chroma errors
  - Pre-flight cost check — aborts cloud upsert if projected write > `CHROMA_BUDGET_SOFT`
  - Checkpoint every N chunks to `data/ingest.checkpoint.json` for resume-after-crash
  - Atomic collection swap: ingest into `{name}_staging`, validate counts, then alias-swap (no half-ingested state visible to readers)

### Step 3 — `backend/lib/chroma.ts`
ChromaDB HTTP client pointing at local server (started alongside ingest) or Chroma
Cloud via env vars. `CHROMA_MODE=local` uses `ChromaClient({ path: "http://localhost:8000" })`;
`CHROMA_MODE=cloud` uses `CloudClient({ apiKey, tenant, database })`.
- **Singleton with health check**: `getChromaClient()` lazily initialises and pings `/api/v1/heartbeat` on first call; throws clearly-named `ChromaUnavailableError` if down
- **Connection pooling**: keep-alive HTTP agent for cloud mode to avoid TLS handshake on every query
- **Env validation at boot**: missing/malformed `CHROMA_MODE` or cloud creds throws on import, not on first request
- **Graceful degradation**: if Chroma is unreachable, `/api/health` endpoint returns 503 with cached last-known status

### Step 4 — `backend/lib/rag.ts`
Query ChromaDB with pre-embedded query vector from the Python embed service.
- Simple top-K cosine retrieval; no VoyageAI reranker (saves cost)
- `include: ["documents", "metadatas", "distances"]` configurable per call
- Returns top-N docs to LLM context window
- Trader filter via `where: { source_file: { $in: [...] } }` metadata filter
- **Embed service**: separate `embed-service.py` (FastAPI on :8001) loads e5-large-v2 once and exposes `POST /embed`; backend calls this for query embedding (avoids loading 1.47 GB model in Node)
- **Query budget guard**: per-request `n_results` capped at `RAG_MAX_TOP_K` (default 50) to prevent accidental large fetches
- **Score threshold filtering**: results with `distance > RAG_SCORE_THRESHOLD` (default 0.75) dropped before LLM; reduces hallucination on weak matches
- **Empty-result handling**: explicit "no relevant context" response path; LLM never gets empty `[]` context that would prompt fabrication
- **Query logging**: structured JSON log per query with `{query_hash, top_k, n_returned, max_distance, latency_ms, gb_returned_estimate}` for cost attribution

### Step 5 — `backend/package.json`
Drop `@llamaindex/milvus`, `@zilliz/milvus2-sdk-node`. Add `chromadb` (official JS
client). Keep `llamaindex` core only if needed for query engine; otherwise remove
entirely and use raw ChromaDB query + direct Gemini/Mistral call.

### Step 6 — `environment.yml`
Mamba env with `chromadb`, `sentence-transformers`, `llama-index-core`.
No LlamaParse, no pymilvus.

```yaml
name: ai-text-opt-1024
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy
  - pandas
  - sentence-transformers
  - pip
  - pip:
    - chromadb
    - llama-index-core
    - python-dotenv
    - tqdm
```

### Step 7 — `.env.example`
Document the Chroma Cloud upgrade path: swap `CHROMA_MODE=local` to
`CHROMA_MODE=cloud` and set three additional vars.

```
# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_MODE=local                      # "local" | "cloud"

# Cloud upgrade — set these three when CHROMA_MODE=cloud
# CHROMA_API_KEY=<your-chroma-api-key>
# CHROMA_TENANT=<your-tenant-name>
# CHROMA_DATABASE=<your-database-name>

CHROMA_COLLECTION=ideas_1024d

# ── Embedding (Python ingest side) ───────────────────────────────────────────
EMBEDDING_MODEL=intfloat/e5-large-v2
EMBEDDING_BATCH_SIZE=32

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER=gemini                    # "gemini" | "mistral"
GEMINI_API_KEY=<your-gemini-api-key>
# MISTRAL_API_KEY=<your-mistral-api-key>

# ── RAG tuning ───────────────────────────────────────────────────────────────
RAG_TOP_K=10
RAG_SCORE_THRESHOLD=0.75
```

### Step 8 — `config/pipeline.yaml`
Central config with all cost-aware defaults:
- `query_include: ["ids", "distances"]` as minimum (expand per endpoint)
- `default_n_results: 10`
- `distance_metric: cosine`
- `hnsw_ef_construction: 200 / hnsw_M: 32` for recall/speed balance

### Step 9 — Verification & smoke tests
After ingest, run an automated verification suite before declaring success:
- `scripts/verify-ingest.py` — asserts `collection.count() == expected_chunks`
- Random sample of 5 chunks: query their text, confirm self-match is rank #1 with distance < 0.05
- Cross-file query test: 10 hand-picked queries with known expected source files
- Collection metadata check: confirms `embedding_dimension == 1024` and `distance_metric == cosine`
- Backend `/api/health` endpoint returns 200 with `{chroma: "ok", embed_service: "ok", llm: "ok"}`

### Step 10 — Observability & rollback
- **Structured logging**: all components emit JSON logs to `logs/{component}.log` (rotated daily, 7-day retention)
- **Metrics endpoint**: `/api/metrics` exposes `{ingest_count, query_count_24h, avg_latency_ms, error_rate}` for at-a-glance health
- **Rollback procedure**: ingest writes to `{collection}_v{N}` versioned collections; `CHROMA_COLLECTION_VERSION` env var picks active version; rollback = decrement env var + restart backend (no data loss)
- **Backup**: nightly `tar -czf chroma_db_$(date +%Y%m%d).tar.gz chroma_db/` for local mode; cloud mode relies on Chroma Cloud durability

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| No LlamaParse | Saves $0.01/page on every ingest run; markdown files parse cleanly as plain text |
| No VoyageAI rerank | ChromaDB cosine on 1024D vectors has sufficient recall at this scale; add later if needed |
| `embedding_function=None` | We pass pre-computed vectors — Chroma never calls an external embedding API |
| `include=["ids","distances"]` default | Minimises GB-returned cost when running on Chroma Cloud |
| `PersistentClient` → `CloudClient` via one env var | `CHROMA_MODE=cloud` triggers the HTTP client path; zero code changes to switch |
| 1024D over 768D | Higher dimensionality captures more semantic nuance; storage cost difference is ~33% more per record but recall improvement justifies it |
| Idempotent upsert IDs | `{file_hash}_{chunk_idx}` IDs mean reruns overwrite, never duplicate |
| Separate embed service | Avoids loading 1.47 GB model in Node; Python service can be GPU-accelerated independently |
| Versioned collections | Atomic swap + rollback path without re-ingestion |
| Content-hash dedupe | Unchanged chunks skip re-embed and re-upsert on rerun; cuts incremental ingest cost ~80% |

---

## Security & Secrets

- `.env` files excluded via `.gitignore`; only `.env.example` committed
- All API keys read at boot — fail-fast on missing creds rather than runtime
- Chroma Cloud API key scoped to single database (least privilege)
- Backend `/api/embed` and `/api/query` rate-limited (default 60 req/min/IP) to prevent abuse-driven cloud bills
- No PII in chunk text; if input data contains it, add a redaction pass in `ingest.py` before embedding
- Embed service bound to `127.0.0.1:8001` only (not exposed externally)

---

## Upgrade Path to Chroma Cloud

When ready to move off local disk:

```bash
# 1. Create a Chroma Cloud database
chroma login
chroma db create ai-text-opt-1024

# 2. Set env vars
echo "CHROMA_MODE=cloud" >> .env
echo "CHROMA_API_KEY=..." >> .env
echo "CHROMA_TENANT=..." >> .env
echo "CHROMA_DATABASE=ai-text-opt-1024" >> .env

# 3. Re-run ingest — upserts to cloud, same command
python ingest.py
```

No other code changes required.
