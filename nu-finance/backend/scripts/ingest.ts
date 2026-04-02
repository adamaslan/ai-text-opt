// scripts/ingest.ts
// Run with: npx tsx scripts/ingest.ts
// One-time ingestion: LlamaParse → IngestionPipeline → Zilliz Cloud (trader_qa_v2)

import { config } from "dotenv";
import { resolve } from "path";
// Load nu-finance/.env (one level above backend/)
config({ path: resolve(__dirname, "../.env") });
import path from "path";
import {
  LlamaParse,
  SentenceSplitter,
  SummaryExtractor,
  TitleExtractor,
  IngestionPipeline,
} from "llamaindex";
import { getMilvusClient, COLLECTION_NAME, FIELDS } from "../lib/zilliz";
import { getLLM } from "../lib/llm";
import { DataType, FunctionType } from "@zilliz/milvus2-sdk-node";

const DOCS_ROOT = path.resolve(__dirname, "../../../docs/trader-qa");

const FILES = [
  "trader-profiles-updated.md",
  "doc-1-options-and-swing-trading.md",
  "doc-2-industries-and-sectors.md",
  "doc-3-recent-news-part1.md",
  "doc-4-recent-news-part2.md",
  "REMAINING-QA-OUTLINE.md",
  "vde-xle-war-end-qa.md",
];

async function ensureCollection() {
  const client = getMilvusClient();

  const exists = await client.hasCollection({ collection_name: COLLECTION_NAME });
  if (exists.value) {
    console.log(`ℹ️  Collection "${COLLECTION_NAME}" already exists — skipping creation.`);
    return;
  }

  await client.createCollection({
    collection_name: COLLECTION_NAME,
    fields: [
      { name: FIELDS.PK,           data_type: DataType.Int64,   is_primary_key: true, autoID: true },
      { name: FIELDS.TEXT,         data_type: DataType.VarChar, max_length: 4096, is_function_output: false },
      { name: FIELDS.SUMMARY,      data_type: DataType.VarChar, max_length: 1024 },
      { name: FIELDS.TITLE,        data_type: DataType.VarChar, max_length: 256  },
      { name: FIELDS.QUESTION,     data_type: DataType.VarChar, max_length: 512  },
      { name: FIELDS.SOURCE_FILE,  data_type: DataType.VarChar, max_length: 128  },
      { name: FIELDS.THEME_NUMBER, data_type: DataType.Int64   },
      { name: FIELDS.THEME_NAME,   data_type: DataType.VarChar, max_length: 256  },
      { name: FIELDS.DOC_SECTION,  data_type: DataType.VarChar, max_length: 128  },
      { name: FIELDS.TRADER,       data_type: DataType.VarChar, max_length: 64   },
      {
        name: FIELDS.EMBEDDING,
        data_type: DataType.FloatVector,
        dim: 768,
        is_function_output: true,
      },
    ],
    functions: [
      {
        name: "text_to_embedding",
        type: FunctionType.TEXTEMBEDDING,
        input_field_names: [FIELDS.TEXT],
        output_field_names: [FIELDS.EMBEDDING],
        params: {
          provider: "openai",
          model_name: "BAAI/bge-base-en-v1.5",
        },
      },
    ],
    index_params: [
      {
        field_name: FIELDS.EMBEDDING,
        index_type: "AUTOINDEX",
        metric_type: "COSINE",
      },
    ],
  });

  console.log(`✅ Created collection "${COLLECTION_NAME}"`);
}

async function ingest() {
  const llm = getLLM();

  // ── 1. Parse all documents with LlamaParse ──────────────────────────────
  const parser = new LlamaParse({
    apiKey: process.env.LLAMA_CLOUD_API_KEY!,
    resultType: "markdown",
    parsingInstruction:
      "This file contains Q&A pairs between two traders. " +
      "Each question starts with a bold number like **Q1.** or **Q2.** " +
      "Keep each question and both trader answers together as one logical unit. " +
      "Preserve Trader 1 and Trader 2 headers.",
  });

  const allDocuments: any[] = [];
  for (const file of FILES) {
    const filePath = path.join(DOCS_ROOT, file);
    console.log(`📄 Parsing ${file}…`);
    const docs = await parser.loadData(filePath);
    docs.forEach((d: any) => (d.metadata.source_file = file));
    allDocuments.push(...docs);
  }
  console.log(`✅ Parsed ${allDocuments.length} documents`);

  // ── 2. Chunk + extract metadata ─────────────────────────────────────────
  const pipeline = new IngestionPipeline({
    transformations: [
      new SentenceSplitter({ chunkSize: 400, chunkOverlap: 50 }),
      new SummaryExtractor({ llm, summaries: ["self"] }),
      new TitleExtractor({ llm, nodes: 3 }),
    ],
  });

  const nodes = await pipeline.run({ documents: allDocuments });
  console.log(`✅ Produced ${nodes.length} nodes`);

  // ── 3. Create Zilliz collection if needed ───────────────────────────────
  await ensureCollection();

  // ── 4. Insert rows ───────────────────────────────────────────────────────
  const client = getMilvusClient();

  // Omit the embedding field — Zilliz's built-in BAAI/bge-base-en-v1.5
  // generates it server-side from text_content via the collection function.
  const rows = nodes.map((node: any) => ({
    [FIELDS.TEXT]:         node.text ?? "",
    [FIELDS.SUMMARY]:      node.metadata?.section_summary ?? "",
    [FIELDS.TITLE]:        node.metadata?.document_title ?? "",
    [FIELDS.QUESTION]:     node.metadata?.question_text ?? "",
    [FIELDS.SOURCE_FILE]:  node.metadata?.source_file ?? "",
    [FIELDS.THEME_NUMBER]: Number(node.metadata?.theme_number ?? 0),
    [FIELDS.THEME_NAME]:   node.metadata?.theme_name ?? "",
    [FIELDS.DOC_SECTION]:  node.metadata?.doc_section ?? "",
    [FIELDS.TRADER]:       node.metadata?.trader_mentioned ?? "both",
  }));

  await client.insert({ collection_name: COLLECTION_NAME, data: rows });
  console.log(`✅ Ingested ${rows.length} chunks into "${COLLECTION_NAME}"`);
}

ingest().catch((err) => {
  console.error("❌ Ingestion failed:", err);
  process.exit(1);
});
