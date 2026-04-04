# LlamaIndex + Zilliz Cloud RAG Pipeline — Trader Q&A Chatbot

**Supersedes:** `zilliz-rag-pipeline-outline.md`
**Stack:** Next.js · LlamaIndex.TS · Zilliz Cloud (Milvus) · LlamaParse · Voyage AI Rerank · Gemini 2.5 Flash *(swap: Mistral)*
**Source corpus:** `/docs/trader-qa/` — 10 markdown files, ~470 KB, dual-trader Q&A format

> **Free-tier budget at a glance**
>
> | Service | Free Allowance | Our Usage |
> |---|---|---|
> | **Zilliz Cloud** | 1 M vectors × 768 dim, 2 collections | ~950 chunks, 1 collection |
> | **LlamaParse** | 1,000 pages / month (or 10 K credits on 2026 promo) | ~10 files ≈ 10 pages |
> | **Voyage AI Rerank** | **200 M free tokens** — models `rerank-2.5`, `rerank-2.5-lite` | ~1 K tokens/query |
> | **Gemini API** | Free tier via Google AI Studio — `gemini-2.5-flash` | Prototype/dev |
> | **Mistral API** | Free tier (`mistral-small-latest`) | Swap-in alternative |
> | **LlamaIndex.TS** | Open-source (MIT) | Unlimited |
>
> **Note on Gemini free tier:** Intended for lab/dev use. Rate limits apply; switch to paid Vertex AI for production. Data may be used for model improvement on free tier.

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            INGESTION (one-time)                            │
│                                                                            │
│  /docs/trader-qa/*.md                                                      │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────┐   structured   ┌─────────────────────┐                    │
│  │  LlamaParse  │ ────────────▶ │   LlamaIndex Nodes   │                   │
│  │  (markdown   │   markdown    │  + SummaryExtractor  │                   │
│  │   mode)      │               │  + TitleExtractor    │                   │
│  └─────────────┘               └─────────┬───────────┘                    │
│                                           │                                │
│                                           ▼                                │
│                                ┌──────────────────┐                        │
│                                │   Zilliz Cloud    │                        │
│                                │   trader_qa_v2   │                        │
│                                │   (AUTOINDEX)     │                        │
│                                └──────────────────┘                        │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                          QUERY (per user message)                          │
│                                                                            │
│  User query                                                                │
│       │                                                                    │
│       ▼                                                                    │
│  ┌─────────────────────┐                                                   │
│  │  LlamaIndex Workflow │                                                   │
│  │  (Router Step)       │                                                   │
│  └───────┬─────────────┘                                                   │
│          │                                                                  │
│    ┌─────┴──────┐                                                          │
│    │            │                                                           │
│    ▼            ▼                                                           │
│  Single-     Sub-Question                                                  │
│  Trader      Engine                                                        │
│  Filter      (T1 view + T2 view)                                           │
│    │            │                                                           │
│    └─────┬──────┘                                                          │
│          ▼                                                                  │
│  ┌──────────────┐  top-10  ┌──────────────────┐  top-5  ┌──────────────┐  │
│  │ Zilliz Cloud  │ ───────▶│ Voyage AI Rerank  │ ───────▶│ Gemini Flash │  │
│  │ vector search │         │  rerank-2.5       │  best 5 │  (or Mistral)│  │
│  └──────────────┘          └──────────────────┘          └──────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Source Document Inventory

*(Unchanged from original pipeline — see `zilliz-rag-pipeline-outline.md § 1`)*

| File | Content | Chunks (est.) |
|---|---|---|
| `trader-profiles-updated.md` | Full profiles: Tactical Opportunist & Structured Growth Investor | ~20 |
| `00-theme-outline.md` | 50-theme outline, Docs 1–4 | ~50 |
| `doc-1-options-and-swing-trading.md` | Themes 1–17: options mechanics, swing trading, earnings plays, hedging | ~170 |
| `doc-2-industries-and-sectors.md` | Themes 18–22: AI/GPU, Cloud, Semiconductors, Quantum, Nuclear | ~200 |
| `doc-2-industries-and-sectors-PREVIEW.md` | Preview/draft of Doc 2 | ~50 |
| `doc-3-recent-news-part1.md` | Market commentary, recent events Part 1 | ~150 |
| `doc-4-recent-news-part2.md` | Market commentary, recent events Part 2 | ~180 |
| `REMAINING-QA-OUTLINE.md` | 150 additional Q&A outline across themes 18–50 | ~80 |
| `vde-xle-war-end-outline.md` | Energy sector: VDE/XLE analysis outline | ~20 |
| `vde-xle-war-end-qa.md` | Energy sector: VDE/XLE comparative Q&A | ~30 |

**Total estimated chunks:** ~950 (at ~400 token chunk size with 50-token overlap)

---

## 2. Data Ingestion & Vectorization

### 2.1 LlamaParse: Structured Parsing

```typescript
// scripts/ingest.ts
import { LlamaParse } from "llamaindex";

const parser = new LlamaParse({
  apiKey: process.env.LLAMA_CLOUD_API_KEY,
  resultType: "markdown",
  parsingInstruction:
    "This file contains Q&A pairs between two traders. " +
    "Each question starts with a bold number like **Q1.** or **Q2.** " +
    "Keep each question and both trader answers together as one logical unit. " +
    "Preserve Trader 1 and Trader 2 headers.",
});

const documents = await parser.loadData("./docs/trader-qa/doc-1-options-and-swing-trading.md");
```

> **Why LlamaParse over regex?** Handles inconsistent formatting, keeps Q + T1 answer + T2 answer as one semantic unit, and costs nothing on the free tier for our ~10 files.

### 2.2 Semantic Chunking + Metadata Extraction

```typescript
import {
  SentenceSplitter,
  TitleExtractor,
  SummaryExtractor,
  IngestionPipeline,
} from "llamaindex";
import { getLLM } from "@/lib/llm";           // ← unified LLM abstraction (§ 3.1)

const llm = getLLM();

const pipeline = new IngestionPipeline({
  transformations: [
    new SentenceSplitter({ chunkSize: 400, chunkOverlap: 50 }),

    // Auto-generates a 1-sentence summary per chunk → stored in Zilliz metadata
    new SummaryExtractor({ llm, summaries: ["self"] }),

    // Pulls the nearest heading as a "title" field
    new TitleExtractor({ llm, nodes: 3 }),
  ],
});

const nodes = await pipeline.run({ documents });
// Each node now has: node.metadata.section_summary, node.metadata.document_title
```

### 2.3 Zilliz Cloud Collection Schema (v2)

**Collection name:** `trader_qa_v2` (1 of 2 free-tier collections)

| Field | Type | Description |
|---|---|---|
| `id` | INT64 (PK, auto) | Unique chunk identifier |
| `text_content` | VARCHAR(4096) | Full chunk text (Q + both trader answers) |
| `chunk_summary` | VARCHAR(1024) | LlamaIndex-generated 1-sentence summary |
| `document_title` | VARCHAR(256) | Auto-extracted document/section title |
| `question_text` | VARCHAR(512) | Extracted question for display in sources panel |
| `source_file` | VARCHAR(128) | Origin markdown filename |
| `theme_number` | INT64 | Theme 1–50 (0 for non-Q&A docs) |
| `theme_name` | VARCHAR(256) | Human-readable theme label |
| `doc_section` | VARCHAR(128) | e.g. "Options & Swing Trading" |
| `trader_mentioned` | VARCHAR(64) | "T1", "T2", "both", or "none" |
| `embedding` | FLOAT_VECTOR(768) | Built-in Zilliz BAAI/bge embedding from `text_content` |

**Index:** AUTOINDEX on `embedding` · **Metric:** COSINE

### 2.4 Full Ingestion Script

```typescript
// scripts/ingest.ts  (run with: npx tsx scripts/ingest.ts)
import { LlamaParse, SentenceSplitter, SummaryExtractor, TitleExtractor, IngestionPipeline } from "llamaindex";
import { MilvusClient } from "@zilliz/milvus2-sdk-node";
import { getLLM } from "../lib/llm";

async function ingest() {
  const llm = getLLM();

  const parser = new LlamaParse({
    apiKey: process.env.LLAMA_CLOUD_API_KEY,
    resultType: "markdown",
    parsingInstruction: "...", // see § 2.1
  });

  const files = [
    "trader-profiles-updated.md",
    "doc-1-options-and-swing-trading.md",
    "doc-2-industries-and-sectors.md",
    "doc-3-recent-news-part1.md",
    "doc-4-recent-news-part2.md",
    "REMAINING-QA-OUTLINE.md",
    "vde-xle-war-end-qa.md",
  ];

  const allDocuments = [];
  for (const file of files) {
    const docs = await parser.loadData(`./docs/trader-qa/${file}`);
    docs.forEach((d) => (d.metadata.source_file = file));
    allDocuments.push(...docs);
  }

  const pipeline = new IngestionPipeline({
    transformations: [
      new SentenceSplitter({ chunkSize: 400, chunkOverlap: 50 }),
      new SummaryExtractor({ llm, summaries: ["self"] }),
      new TitleExtractor({ llm, nodes: 3 }),
    ],
  });

  const nodes = await pipeline.run({ documents: allDocuments });

  const client = new MilvusClient({
    address: process.env.ZILLIZ_CLOUD_URI!,
    token: process.env.ZILLIZ_API_KEY!,
  });

  await client.createCollection({ collection_name: "trader_qa_v2" /* schema from § 2.3 */ });

  const rows = nodes.map((node) => ({
    text_content: node.text,
    chunk_summary: node.metadata.section_summary || "",
    document_title: node.metadata.document_title || "",
    question_text: node.metadata.question_text || "",
    source_file: node.metadata.source_file,
    theme_number: node.metadata.theme_number || 0,
    theme_name: node.metadata.theme_name || "",
    doc_section: node.metadata.doc_section || "",
    trader_mentioned: node.metadata.trader_mentioned || "both",
  }));

  await client.insert({ collection_name: "trader_qa_v2", data: rows });
  console.log(`✅ Ingested ${rows.length} chunks into trader_qa_v2`);
}

ingest();
```

---

## 3. LLM Abstraction — Gemini (default) / Mistral (swap)

### 3.1 `lib/llm.ts` — Single File to Swap LLMs

The entire pipeline uses `getLLM()`. To switch from Gemini to Mistral, change **one env var** (`LLM_PROVIDER=mistral`) — no code changes.

```typescript
// lib/llm.ts
import { Gemini, GEMINI_MODEL, MistralAI } from "llamaindex";

export type LLMProvider = "gemini" | "mistral";

export function getLLM(provider?: LLMProvider) {
  const active = (provider ?? process.env.LLM_PROVIDER ?? "gemini") as LLMProvider;

  switch (active) {
    // ── Default: Google Gemini 2.5 Flash (free tier via Google AI Studio) ──
    case "gemini":
      return new Gemini({
        apiKey: process.env.GEMINI_API_KEY!,
        model: GEMINI_MODEL.GEMINI_2_5_FLASH,  // or "gemini-2.5-flash"
      });

    // ── Swap: Mistral (free tier: mistral-small-latest) ────────────────────
    case "mistral":
      return new MistralAI({
        apiKey: process.env.MISTRAL_API_KEY!,
        model: process.env.MISTRAL_MODEL ?? "mistral-small-latest",
      });

    default:
      throw new Error(`Unknown LLM_PROVIDER: "${active}". Use "gemini" or "mistral".`);
  }
}
```

> **How to swap:**
> - **Use Gemini:** `LLM_PROVIDER=gemini` (or omit — it's the default)
> - **Use Mistral:** `LLM_PROVIDER=mistral`
> - Both providers are natively supported in LlamaIndex.TS — no custom wrappers needed.

### 3.2 LLM Comparison

| | Gemini 2.5 Flash | Mistral Small |
|:---|:---|:---|
| **Free tier** | Yes — Google AI Studio | Yes — La Plateforme |
| **Context window** | 1 M tokens | 32 K tokens |
| **Speed** | Fast | Fast |
| **Best for** | Large context, long Q&A threads | Cost-efficiency, European data residency |
| **Production** | Upgrade to Vertex AI | Upgrade to Mistral paid |

---

## 4. Query Pipeline — Voyage AI Rerank + LlamaIndex Workflows

### 4.1 `lib/rag.ts` — Vector Store + Reranker + Query Engines

```typescript
// lib/rag.ts
import {
  VectorStoreIndex,
  MilvusVectorStore,
  SubQuestionQueryEngine,
  QueryEngineTool,
  VoyageAIRerank,
} from "llamaindex";
import { getLLM } from "./llm";

const llm = getLLM();

// ── Zilliz Vector Store ──────────────────────────────────────────────
const vectorStore = new MilvusVectorStore({
  address: process.env.ZILLIZ_CLOUD_URI!,
  token: process.env.ZILLIZ_API_KEY!,
  collectionName: "trader_qa_v2",
  dim: 768,
});

const index = await VectorStoreIndex.fromVectorStore(vectorStore, { llm });

// ── Voyage AI Reranker ───────────────────────────────────────────────
// Free tier: 200M tokens across rerank-2.5 / rerank-2.5-lite / rerank-2 / rerank-2-lite
// Pricing after free: (query_tokens × doc_count) + sum(doc_tokens)
const reranker = new VoyageAIRerank({
  apiKey: process.env.VOYAGE_API_KEY!,
  model: "rerank-2.5",       // latest; swap to "rerank-2.5-lite" to preserve tokens
  topN: 5,                   // distill top-10 from Zilliz down to best 5
});

// ── Single-Trader Query Engine (metadata-filtered + reranked) ────────
function traderEngine(traderTag: "T1" | "T2") {
  return index.asQueryEngine({
    llm,
    similarityTopK: 10,                      // pull 10 from Zilliz...
    nodePostprocessors: [reranker],          // ...rerank to 5
    preFilters: {
      filters: [{ key: "trader_mentioned", value: traderTag, operator: "==" }],
    },
  });
}

// ── Comparative (Sub-Question) Query Engine ──────────────────────────
const t1Engine = traderEngine("T1");
const t2Engine = traderEngine("T2");

const subQuestionEngine = new SubQuestionQueryEngine({
  queryEngineTools: [
    new QueryEngineTool({
      queryEngine: t1Engine,
      metadata: {
        name: "tactical_opportunist",
        description:
          "Answers from the Tactical Opportunist (T1): options-heavy, short-dated, narrative-driven speculator",
      },
    }),
    new QueryEngineTool({
      queryEngine: t2Engine,
      metadata: {
        name: "structured_growth_investor",
        description:
          "Answers from the Structured Growth Investor (T2): equity-primary, long-hold, quality-biased thematic investor",
      },
    }),
  ],
  llm,
});

// ── Default / Broad Query Engine ─────────────────────────────────────
const broadEngine = index.asQueryEngine({
  llm,
  similarityTopK: 10,
  nodePostprocessors: [reranker],
});

export { broadEngine, subQuestionEngine, traderEngine };
```

### 4.2 `lib/workflow.ts` — Router

```typescript
// lib/workflow.ts
import { Workflow, StartEvent, StopEvent, step } from "llamaindex";
import { broadEngine, subQuestionEngine, traderEngine } from "./rag";

class TraderQueryWorkflow extends Workflow {
  @step()
  async route(ev: StartEvent): Promise<StopEvent> {
    const query: string = ev.data.query;
    const traderFilter: string | null = ev.data.traderFilter;

    // 1. Explicit trader filter from UI toggle
    if (traderFilter === "T1" || traderFilter === "T2") {
      const response = await traderEngine(traderFilter).query({ query });
      return new StopEvent({ result: response });
    }

    // 2. Heuristic: detect trader mention in query text
    const mentionsT1 = /tactical|opportunist|\bt1\b/i.test(query);
    const mentionsT2 = /structured|growth|investor|\bt2\b/i.test(query);

    if (mentionsT1 && !mentionsT2) {
      const response = await traderEngine("T1").query({ query });
      return new StopEvent({ result: response });
    }
    if (mentionsT2 && !mentionsT1) {
      const response = await traderEngine("T2").query({ query });
      return new StopEvent({ result: response });
    }

    // 3. Comparative → sub-question decomposition
    if (/both|compare|differ|vs\.?|versus/i.test(query)) {
      const response = await subQuestionEngine.query({ query });
      return new StopEvent({ result: response });
    }

    // 4. Default: broad search + rerank
    const response = await broadEngine.query({ query });
    return new StopEvent({ result: response });
  }
}

export const traderWorkflow = new TraderQueryWorkflow();
```

---

## 5. Backend API Route

### 5.1 `app/api/chat/route.ts`

```typescript
import { NextRequest, NextResponse } from "next/server";
import { traderWorkflow } from "@/lib/workflow";

export async function POST(req: NextRequest) {
  const { message, trader_filter } = await req.json();

  const result = await traderWorkflow.run({
    query: message,
    traderFilter: trader_filter || null,
  });

  const response = result.data.result;

  return NextResponse.json({
    answer: response.response,
    llm_provider: process.env.LLM_PROVIDER ?? "gemini",   // surfaced for debugging
    sources: response.sourceNodes?.map((node: any) => ({
      text_preview: node.node.text.slice(0, 200),
      chunk_summary: node.node.metadata.chunk_summary,
      source_file: node.node.metadata.source_file,
      theme_name: node.node.metadata.theme_name,
      rerank_score: node.score,
    })),
  });
}
```

### 5.2 System Prompt

LlamaIndex passes this automatically via its `ResponseSynthesizer`:

```
You are a trading education assistant grounded in the Trader Q&A knowledge base.

The knowledge base features two trader perspectives:
- Tactical Opportunist (T1): options-heavy, short-dated, narrative-driven speculator
- Structured Growth Investor (T2): equity-primary, long-hold, quality-biased thematic investor

Use only the retrieved context to answer. When both trader perspectives appear,
present both views clearly labeled. If the context doesn't cover the question, say so.
```

---

## 6. Environment Variables

```env
# ─── Zilliz Cloud ──────────────────────────────────────────────────────────
ZILLIZ_CLOUD_URI=https://<cluster-id>.serverless.gcp-us-west1.cloud.zilliz.com
ZILLIZ_API_KEY=<your-api-key>
ZILLIZ_COLLECTION=trader_qa_v2

# ─── LlamaIndex / LlamaParse ───────────────────────────────────────────────
LLAMA_CLOUD_API_KEY=<your-llama-cloud-api-key>

# ─── LLM Selection ─────────────────────────────────────────────────────────
# Change LLM_PROVIDER to "mistral" to swap with zero code changes
LLM_PROVIDER=gemini                  # "gemini" | "mistral"

# Gemini (default) — get key at aistudio.google.com, free tier available
GEMINI_API_KEY=<your-gemini-api-key>
# GEMINI_MODEL=gemini-2.5-flash      # optional override

# Mistral (swap) — get key at console.mistral.ai, free tier available
# MISTRAL_API_KEY=<your-mistral-api-key>
# MISTRAL_MODEL=mistral-small-latest # or open-mistral-7b for fully free OSS

# ─── Voyage AI Rerank ──────────────────────────────────────────────────────
# 200M free tokens — get key at dash.voyageai.com
VOYAGE_API_KEY=<your-voyage-api-key>
# VOYAGE_RERANK_MODEL=rerank-2.5     # or rerank-2.5-lite to conserve tokens

# ─── RAG Tuning ────────────────────────────────────────────────────────────
RAG_TOP_K=10           # pull 10 from Zilliz, Voyage reranks to 5
RAG_RERANK_TOP_N=5
RAG_SCORE_THRESHOLD=0.75
```

---

## 7. File & Folder Structure

```
/
├── app/
│   ├── page.tsx                          # Chat UI
│   ├── api/
│   │   └── chat/
│   │       └── route.ts                  # POST /api/chat (LlamaIndex Workflow)
│   └── components/
│       ├── ChatMessage.tsx
│       └── SourcesPanel.tsx
├── scripts/
│   └── ingest.ts                         # LlamaParse + IngestionPipeline → Zilliz
├── lib/
│   ├── llm.ts                            # ← LLM abstraction (Gemini / Mistral swap)
│   ├── rag.ts                            # VectorStoreIndex, Voyage reranker, engines
│   ├── workflow.ts                       # LlamaIndex Workflow (Router)
│   └── zilliz.ts                         # MilvusClient singleton (if needed standalone)
├── docs/
│   └── trader-qa/                        # Source markdown corpus
├── .env.local
├── package.json
└── tsconfig.json
```

---

## 8. Dependencies

```bash
# Core
npm install llamaindex @zilliz/milvus2-sdk-node

# LLM providers (install both; only the active one is called at runtime)
npm install @llamaindex/google     # Gemini
npm install @llamaindex/mistral    # Mistral

# Reranker
npm install @llamaindex/voyageai   # Voyage AI Rerank

# Dev
npm install -D tsx
```

```json
{
  "dependencies": {
    "llamaindex": "^0.8.x",
    "@llamaindex/google": "^0.x",
    "@llamaindex/mistral": "^0.x",
    "@llamaindex/voyageai": "^0.x",
    "@zilliz/milvus2-sdk-node": "^2.4.x",
    "next": "^14.x",
    "react": "^18.x"
  },
  "devDependencies": {
    "tsx": "^4.x"
  }
}
```

> **Note:** Removed `openai` dependency entirely. If you later want GPT-4o back, add `@llamaindex/openai` and a new `case "openai"` branch in `lib/llm.ts`.

---

## 9. Comparison: Original DIY vs. This Pipeline

| Feature | Original DIY | This Pipeline |
|:---|:---|:---|
| **Parsing** | Manual `**Q\d+.**` regex | LlamaParse (markdown mode) |
| **Chunking** | Fixed 400-token window | `SentenceSplitter` with semantic boundaries |
| **Metadata** | Manual `theme_number` | Auto: `chunk_summary`, `document_title`, `trader_mentioned` |
| **Search** | Basic top-k vector match | Router → filtered or sub-question decomposition |
| **Reranking** | None | Voyage AI `rerank-2.5` (200M free tokens) |
| **LLM** | GPT-4o (pay-as-you-go) | Gemini 2.5 Flash (free tier) |
| **LLM swapping** | Code rewrite | `LLM_PROVIDER=mistral` in `.env` |
| **Language** | Python + TypeScript | Full TypeScript |
| **Cost** | OpenAI charges per query | Free across all services |

---

## 10. Example Queries & Expected Routing

| User Question | Router Path | Engine |
|---|---|---|
| "How do I use covered calls?" | Broad | `broadEngine` → Voyage rerank → Gemini |
| "What does the Tactical Opportunist think about NVDA?" | T1 detected | `traderEngine("T1")` → rerank → Gemini |
| "Compare both traders on earnings plays" | Comparative | `subQuestionEngine` → merge → Gemini |
| "How should I size my position?" *(UI toggle: T2)* | Explicit filter | `traderEngine("T2")` → rerank → Gemini |
| "What is IV crush?" | Broad | `broadEngine` → Voyage rerank → Gemini |

---

## 11. Build & Run

```bash
# 1. Install dependencies
npm install

# 2. Configure environment
cp .env.example .env.local
# Fill in: ZILLIZ_*, LLAMA_CLOUD_API_KEY, GEMINI_API_KEY, VOYAGE_API_KEY

# 3. Ingest docs into Zilliz (one-time)
npx tsx scripts/ingest.ts

# 4. Start dev server
npm run dev
# → http://localhost:3000

# ── To switch to Mistral ──────────────────────────────────────────────
# In .env.local:
#   LLM_PROVIDER=mistral
#   MISTRAL_API_KEY=<your-key>
# Then restart: npm run dev
```

---

## 12. Phased Delivery

| Phase | Deliverable | Status |
|---|---|---|
| **1** | `lib/llm.ts` — Gemini/Mistral abstraction | 🔲 |
| **2** | `scripts/ingest.ts` — LlamaParse + IngestionPipeline → `trader_qa_v2` | 🔲 |
| **3** | `lib/rag.ts` — VectorStoreIndex, Voyage reranker, query engines | 🔲 |
| **4** | `lib/workflow.ts` — Router workflow | 🔲 |
| **5** | `app/api/chat/route.ts` — API route wired to workflow | 🔲 |
| **6** | `app/page.tsx` + components — Chat UI | 🔲 |
| **7** | Validate LLM swap (Gemini → Mistral) end-to-end | 🔲 |

---

## 13. Future Enhancements

- **Streaming:** `StreamingResponse` + Next.js `ReadableStream` for token-by-token output
- **Chat memory:** `ChatMemoryBuffer` for multi-turn conversations
- **Evaluation:** LlamaIndex `FaithfulnessEvaluator` or `ragas` to score answer quality
- **Hybrid search:** Zilliz BM25 + vector hybrid mode (currently in beta on free tier)
- **Add GPT-4o:** One new `case "openai"` in `lib/llm.ts` + `npm install @llamaindex/openai`
