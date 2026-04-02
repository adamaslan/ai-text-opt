// lib/zilliz.ts
// Singleton MilvusClient for standalone use (e.g. in scripts, health checks).
// The RAG pipeline (lib/rag.ts) uses MilvusVectorStore from llamaindex instead.

import { MilvusClient } from "@zilliz/milvus2-sdk-node";

let _client: MilvusClient | null = null;

export function getMilvusClient(): MilvusClient {
  if (!_client) {
    if (!process.env.ZILLIZ_CLOUD_URI) {
      throw new Error("ZILLIZ_CLOUD_URI is not set in environment variables.");
    }
    if (!process.env.ZILLIZ_API_KEY) {
      throw new Error("ZILLIZ_API_KEY is not set in environment variables.");
    }
    _client = new MilvusClient({
      address: process.env.ZILLIZ_CLOUD_URI,
      token: process.env.ZILLIZ_API_KEY,
    });
  }
  return _client;
}

/** The collection name used across all pipeline components */
export const COLLECTION_NAME =
  process.env.ZILLIZ_COLLECTION ?? "trader_qa_v2";

/** Schema field names (kept in one place to avoid magic strings) */
export const FIELDS = {
  PK: "id",
  TEXT: "text_content",
  SUMMARY: "chunk_summary",
  TITLE: "document_title",
  QUESTION: "question_text",
  SOURCE_FILE: "source_file",
  THEME_NUMBER: "theme_number",
  THEME_NAME: "theme_name",
  DOC_SECTION: "doc_section",
  TRADER: "trader_mentioned",
  EMBEDDING: "embedding",
} as const;
