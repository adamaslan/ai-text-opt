# Ingesting Trader-QA Docs into Zilliz Cloud via Terminal

Step-by-step guide for populating the `trader_qa_v2` Zilliz collection using the existing `nu-finance/backend` pipeline.

---

## Prerequisites

| Requirement | How to get it |
|---|---|
| **Node.js 18+** | `node -v` to check; install via `nvm install 18` or `brew install node` |
| **Zilliz Cloud cluster** | Create a free serverless cluster at [cloud.zilliz.com](https://cloud.zilliz.com) — copy the **Public Endpoint** and **API Key** |
| **LlamaCloud API key** | Sign up at [cloud.llamaindex.ai](https://cloud.llamaindex.ai) — free tier covers LlamaParse |
| **Gemini API key** | Get one at [aistudio.google.com](https://aistudio.google.com) (free tier) |
| **Voyage AI API key** | Get one at [dash.voyageai.com](https://dash.voyageai.com) (200M free rerank tokens) |

---

## 1. Fill in the `.env` file

The env file lives at `nu-finance/.env`. Open it and replace the placeholder values:

```bash
cd ~/code/ai-text-opt/nu-finance
nano .env    # or use any editor
```

Fill in these required values:

```
ZILLIZ_CLOUD_URI=https://<your-cluster-id>.serverless.gcp-us-west1.cloud.zilliz.com
ZILLIZ_API_KEY=<your-zilliz-api-key>
LLAMA_CLOUD_API_KEY=<your-llama-cloud-api-key>
GEMINI_API_KEY=<your-gemini-api-key>
VOYAGE_API_KEY=<your-voyage-api-key>
```

The rest of the values (`ZILLIZ_COLLECTION`, `LLM_PROVIDER`, `RAG_TOP_K`, etc.) have working defaults and don't need to change.

---

## 2. Install dependencies

```bash
cd ~/code/ai-text-opt/nu-finance/backend
npm install
```

This installs `llamaindex`, `@llamaindex/milvus`, `@zilliz/milvus2-sdk-node`, `tsx`, and everything else listed in `package.json`.

---

## 3. Verify the docs directory

The ingest script reads markdown files from `docs/trader-qa/` (relative to the repo root). Confirm the files exist:

```bash
ls ~/code/ai-text-opt/docs/trader-qa/
```

The script ingests these 7 files (defined in `backend/scripts/ingest.ts`):

| File | Content |
|---|---|
| `trader-profiles-updated.md` | Trader personality profiles |
| `doc-1-options-and-swing-trading.md` | Options & swing trading Q&A |
| `doc-2-industries-and-sectors.md` | Industry/sector analysis Q&A |
| `doc-3-recent-news-part1.md` | Recent news Q&A (part 1) |
| `doc-4-recent-news-part2.md` | Recent news Q&A (part 2) |
| `REMAINING-QA-OUTLINE.md` | Remaining Q&A outline |
| `vde-xle-war-end-qa.md` | VDE/XLE war-end scenario Q&A |

---

## 4. Run the ingestion

```bash
cd ~/code/ai-text-opt/nu-finance/backend
npm run ingest
```

This executes `npx tsx scripts/ingest.ts`, which does the following in order:

1. **LlamaParse** — sends each `.md` file to LlamaCloud for intelligent markdown parsing (preserves Q&A structure)
2. **IngestionPipeline** — chunks with `SentenceSplitter` (400 tokens, 50 overlap), then extracts `SummaryExtractor` and `TitleExtractor` metadata using Gemini
3. **Collection creation** — creates `trader_qa_v2` in Zilliz Cloud if it doesn't exist, with a built-in `TEXTEMBEDDING` function (`BAAI/bge-base-en-v1.5`, 768-dim COSINE, AUTOINDEX). No external embedding API needed — Zilliz generates embeddings server-side from `text_content`.
4. **Insert** — writes chunked rows (text + metadata only, no embedding field) into the collection. Zilliz auto-generates the 768-dim embedding for each row.

Expected terminal output:

```
📄 Parsing trader-profiles-updated.md…
📄 Parsing doc-1-options-and-swing-trading.md…
📄 Parsing doc-2-industries-and-sectors.md…
📄 Parsing doc-3-recent-news-part1.md…
📄 Parsing doc-4-recent-news-part2.md…
📄 Parsing REMAINING-QA-OUTLINE.md…
📄 Parsing vde-xle-war-end-qa.md…
✅ Parsed 7 documents
✅ Produced N nodes
✅ Created collection "trader_qa_v2"
✅ Ingested N chunks into "trader_qa_v2"
```

> **Note:** The LlamaParse + metadata extraction steps call external APIs, so ingestion may take 2-5 minutes depending on document size and API response times.

---

## 5. Verify in Zilliz Cloud

Log into [cloud.zilliz.com](https://cloud.zilliz.com), open your cluster, and check:

- Collection `trader_qa_v2` exists
- Row count matches the "Ingested N chunks" output
- Schema has fields: `text_content`, `chunk_summary`, `document_title`, `source_file`, `embedding` (768-dim), etc.

---

## 6. Run the full stack (optional)

Once ingestion is complete, start the backend and frontend:

```bash
# Terminal 1 — backend (port 3001)
cd ~/code/ai-text-opt/nu-finance/backend
npm run dev

# Terminal 2 — frontend (port 3000)
cd ~/code/ai-text-opt/nu-finance/frontend
npm install   # first time only
npm run dev
```

Open `http://localhost:3000` to chat with the trader Q&A RAG system.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ZILLIZ_CLOUD_URI is not set` | Check that `nu-finance/.env` has the correct URI and the dotenv path in `ingest.ts` resolves correctly |
| `LlamaParse` 401 error | Verify `LLAMA_CLOUD_API_KEY` is valid at [cloud.llamaindex.ai](https://cloud.llamaindex.ai) |
| `Collection already exists` | This is informational, not an error — the script skips creation and inserts into the existing collection |
| Timeout during parsing | Large docs can take time; retry or split the `FILES` array into smaller batches |
| `Cannot find module 'llamaindex'` | Run `npm install` in `backend/` first |

---

## How it works (architecture)

```
docs/trader-qa/*.md
        │
        ▼
   LlamaParse (LlamaCloud API)
        │  markdown → structured documents
        ▼
   IngestionPipeline
   ├─ SentenceSplitter (400 tokens, 50 overlap)
   ├─ SummaryExtractor (Gemini — LLM only, not embeddings)
   └─ TitleExtractor   (Gemini — LLM only, not embeddings)
        │  documents → enriched nodes (text + metadata, no vectors)
        ▼
   Zilliz Cloud (trader_qa_v2)
   ├─ Built-in BAAI/bge-base-en-v1.5 embedding function
   │  (auto-embeds text_content → 768-dim vector server-side)
   ├─ AUTOINDEX, COSINE metric
   └─ metadata: source_file, theme, trader, summary, title
```

**No OpenAI dependency.** Gemini is used only as the LLM for metadata extraction (summaries, titles). Embeddings are handled entirely by Zilliz's built-in pipeline.

At query time (`lib/rag.ts`), the backend uses:
- **MilvusVectorStore** from `@llamaindex/milvus` to search the collection
- **Voyage AI reranker** (`rerank-2.5`) to distill top-10 → top-5
- **TraderQueryWorkflow** to route queries to T1, T2, comparative, or broad engines
