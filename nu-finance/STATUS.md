# nu-finance — Project Status

**Pipeline:** LlamaIndex + Zilliz Cloud RAG — Trader Q&A Chatbot  
**Spec:** `/docs/trader-qa/llamaindex-zilliz-rag-pipeline.md`  
**Stack:** Next.js · LlamaIndex.TS · Zilliz Cloud · LlamaParse · Voyage AI Rerank · Gemini 2.5 Flash *(swap: Mistral)*

---

## ✅ Done

### Project Scaffolding
- [x] Created `/nu-finance/` root folder
- [x] Created `/nu-finance/backend/` with subdirs: `lib/`, `scripts/`, `app/api/chat/`, `app/components/`
- [x] Created `/nu-finance/frontend/src/` with subdirs: `components/`, `pages/`, `styles/`

### Backend Config Files
- [x] `backend/package.json` — all deps: `llamaindex`, `@llamaindex/google`, `@llamaindex/mistral`, `@llamaindex/voyageai`, `@zilliz/milvus2-sdk-node`, `next`, `react`, `dotenv`
- [x] `backend/tsconfig.json` — TypeScript config with `experimentalDecorators` + `emitDecoratorMetadata`
- [x] `backend/.env.example` — full env var template with all required keys documented
- [x] `backend/next.config.js` — loads root `../../.env` via dotenv; CORS headers for frontend `:3000`

### Backend — Library Files
- [x] `backend/lib/llm.ts` — LLM abstraction; returns Gemini or Mistral based on `LLM_PROVIDER` env var
- [x] `backend/lib/rag.ts` — Zilliz vector store, Voyage AI reranker, `traderEngine()`, `subQuestionEngine`, `broadEngine`
- [x] `backend/lib/workflow.ts` — `TraderQueryWorkflow` router: T1 / T2 / comparative / broad paths
- [x] `backend/lib/zilliz.ts` — `MilvusClient` singleton + field name constants

### Backend — Scripts & API
- [x] `backend/scripts/ingest.ts` — LlamaParse → `IngestionPipeline` (SentenceSplitter + SummaryExtractor + TitleExtractor) → creates + inserts into Zilliz `trader_qa_v2`
- [x] `backend/app/api/chat/route.ts` — Next.js POST `/api/chat` wired to `traderWorkflow`

### Frontend — Config
- [x] `frontend/package.json` — React + Vite + TypeScript deps
- [x] `frontend/tsconfig.json` — TS config
- [x] `frontend/vite.config.ts` — Vite config with proxy to backend `:3001`
- [x] `frontend/index.html` — Vite HTML entry point (Inter + JetBrains Mono fonts, SEO meta)

### Frontend — App & Components
- [x] `frontend/src/main.tsx` — React entry point
- [x] `frontend/src/App.tsx` — root component
- [x] `frontend/src/pages/ChatPage.tsx` — main chat UI with T1 / T2 / Both toggle, starter prompts, loading state
- [x] `frontend/src/components/ChatMessage.tsx` — message bubble (user vs assistant, LLM badge, source count)
- [x] `frontend/src/components/SourcesPanel.tsx` — collapsible sources sidebar (file, theme, rerank score colour-coded)
- [x] `frontend/src/styles/index.css` — global dark-mode styles, glassmorphism, micro-animations

---

## 🔲 Still To Do

| Phase | File | Status |
|---|---|---|
| 7 | Validate LLM swap (Gemini → Mistral) end-to-end | 🔲 |
| — | Fill in `ZILLIZ_CLOUD_URI` + `ZILLIZ_API_KEY` in root `.env` | 🔲 |
| — | Run `npm install` in `backend/` and `frontend/` | 🔲 |
| — | Run `npm run ingest` in `backend/` (one-time) | 🔲 |

---

## Delivery Order (Phased per Spec § 12)

| Phase | File | Status |
|---|---|---|
| 1 | `lib/llm.ts` — Gemini/Mistral abstraction | ✅ |
| 2 | `scripts/ingest.ts` — LlamaParse → Zilliz | ✅ |
| 3 | `lib/rag.ts` — vector store + reranker + engines | ✅ |
| 4 | `lib/workflow.ts` — router workflow | ✅ |
| 5 | `app/api/chat/route.ts` — API route | ✅ |
| 6 | `frontend/` — full chat UI | ✅ |
| 7 | Validate LLM swap (Gemini → Mistral) end-to-end | 🔲 |

---

## Key Notes

- **Env file:** All services read from root `/ai-text-opt/.env` — no backend-specific `.env.local` needed
- **Zilliz collection:** `trader_qa_v2` — auto-created on first `npm run ingest`
- **LLM swap:** change `LLM_PROVIDER=mistral` in root `.env` — no code edits required
- **Reranker:** Voyage AI `rerank-2.5`, pulls top-10 from Zilliz → distills to top-5
- **Frontend port:** `:3000` · **Backend port:** `:3001`
- **Ingest command:** `cd backend && npm run ingest` (one-time, after filling Zilliz keys in `.env`)
