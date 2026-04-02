// lib/rag.ts
// Vector store, Voyage AI reranker, and query engines.
// Uses top-level await — Next.js App Router supports this natively.

import {
  VectorStoreIndex,
  SubQuestionQueryEngine,
  QueryEngineTool,
} from "llamaindex";
import { MilvusVectorStore } from "@llamaindex/milvus";
import { VoyageAIRerank } from "@llamaindex/voyage-ai";
import { getLLM } from "./llm";

const llm = getLLM();

// ── Zilliz / Milvus Vector Store ─────────────────────────────────────────────
const vectorStore = new MilvusVectorStore({
  collection: process.env.ZILLIZ_COLLECTION ?? "trader_qa_v2",
  params: {
    configOrAddress: process.env.ZILLIZ_CLOUD_URI!,
    token: process.env.ZILLIZ_API_KEY!,
  },
});

// Build index from the already-ingested Zilliz collection (no document loading here)
export const index = await VectorStoreIndex.fromVectorStore(vectorStore, {
  llm,
});

// ── Voyage AI Reranker ────────────────────────────────────────────────────────
// Free tier: 200 M tokens across rerank-2.5 / rerank-2.5-lite / rerank-2 / rerank-2-lite
// Pricing after free: (query_tokens × doc_count) + sum(doc_tokens)
const reranker = new VoyageAIRerank({
  apiKey: process.env.VOYAGE_API_KEY!,
  model: (process.env.VOYAGE_RERANK_MODEL as any) ?? "rerank-2.5",
  topN: Number(process.env.RAG_RERANK_TOP_N ?? 5),
});

// ── Single-Trader Query Engine (metadata-filtered + reranked) ─────────────────
// Pulls top-K from Zilliz, then reranks to top-N for the chosen trader.
export function traderEngine(traderTag: "T1" | "T2") {
  return index.asQueryEngine({
    llm,
    similarityTopK: Number(process.env.RAG_TOP_K ?? 10),
    nodePostprocessors: [reranker],
    preFilters: {
      filters: [
        { key: "trader_mentioned", value: traderTag, operator: "==" },
      ],
    },
  });
}

// ── Comparative (Sub-Question) Query Engine ───────────────────────────────────
// Decomposes a comparative question into a T1 sub-query and a T2 sub-query,
// runs them concurrently, then merges the answers.
const t1Engine = traderEngine("T1");
const t2Engine = traderEngine("T2");

export const subQuestionEngine = new SubQuestionQueryEngine({
  queryEngineTools: [
    new QueryEngineTool({
      queryEngine: t1Engine,
      metadata: {
        name: "tactical_opportunist",
        description:
          "Answers from the Tactical Opportunist (T1): " +
          "options-heavy, short-dated, narrative-driven speculator.",
      },
    }),
    new QueryEngineTool({
      queryEngine: t2Engine,
      metadata: {
        name: "structured_growth_investor",
        description:
          "Answers from the Structured Growth Investor (T2): " +
          "equity-primary, long-hold, quality-biased thematic investor.",
      },
    }),
  ],
  llm,
});

// ── Default / Broad Query Engine ──────────────────────────────────────────────
// No trader filter — searches all chunks, then Voyage reranks to top-N.
export const broadEngine = index.asQueryEngine({
  llm,
  similarityTopK: Number(process.env.RAG_TOP_K ?? 10),
  nodePostprocessors: [reranker],
});
