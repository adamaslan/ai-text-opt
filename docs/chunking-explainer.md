# How Chunking Works in ai-text-opt-1024

## Overview

The pipeline converts raw markdown files into fixed-size text chunks, embeds each
chunk as a 1024D vector, and stores them in ChromaDB. Chunking is the step between
"whole document" and "embeddable unit."

This doc covers the current behaviour, the limits it bumps into, and how to tune
it for the **ChromaDB free tier** (local PersistentClient, or Chroma Cloud free
tier with $0 monthly credit) without losing recall quality.

---

## Step-by-Step Flow

```
.md file on disk
      │
      ▼
load_markdown_docs()          plain open() read — no parsing API ($0)
      │
      ▼
LlamaIndex Document object    text= full file content, metadata= {source_file}
      │
      ▼
SentenceSplitter              chunk_size=400 tokens, chunk_overlap=50 tokens
      │  splits on sentence boundaries within the token window
      ▼
List[Node]                    one Node per chunk
      │
      ▼
build_chunks()                wraps each Node into a typed Chunk dataclass
      │  assigns chunk_id, content_hash, metadata
      ▼
List[Chunk]                   ready for embedding
      │
      ▼
content-hash dedupe           skip chunks whose text didn't change since last run
      │
      ▼
EmbeddingModel.encode_batch() local e5-large-v2 → 1024D float32 vectors ($0)
      │
      ▼
collection.upsert()           ChromaDB write (free locally; metered in cloud)
```

---

## SentenceSplitter Behaviour

`SentenceSplitter` is from `llama_index.core.node_parser`. It works like this:

1. **Tokenises** the document text using a token counter (tiktoken by default).
2. **Tries to cut at sentence boundaries** — periods, newlines, paragraph breaks —
   rather than mid-sentence.
3. **Fills a window** up to `chunk_size` tokens, then emits a Node.
4. **Slides forward by `chunk_size - chunk_overlap`** tokens, so the next chunk
   starts 50 tokens back into the previous chunk.

### Current settings

| Parameter | Value | What it means |
|---|---|---|
| `chunk_size` | 400 tokens | ~300–350 words per chunk |
| `chunk_overlap` | 50 tokens | ~37–45 words shared between adjacent chunks |
| Splitter | `SentenceSplitter` | Respects sentence boundaries |

### What "tokens" means here

LlamaIndex counts tokens using OpenAI's `cl100k_base` tokeniser by default
(same as GPT-4). One token ≈ 0.75 English words, so:

- 400 tokens ≈ 300 words ≈ ~1,800 characters
- 50 tokens ≈ 37 words ≈ ~225 characters
- One full markdown file at ~100 KB ≈ ~25,000 tokens ≈ ~63 chunks at the
  current 400-token setting (overlap-adjusted).

> **Note:** the `e5-large-v2` embedding model itself has a 512-token input cap.
> Anything longer is silently truncated by SentenceTransformer. Keeping
> `chunk_size <= 512` is therefore both a quality and a correctness constraint —
> larger chunks waste characters that the model never sees.

---

## Chunk Identity and Deduplication

Each chunk gets two identifiers computed in `build_chunks()`
([ingest.py:168-195](../../ai-text-opt-1024/ingest.py#L168-L195)):

```
chunk_id      = sha1(source_file_path) + "_" + zero-padded chunk index
                e.g. "a3f9c1d2_00003"

content_hash  = sha1(chunk text)
```

On re-runs, the checkpoint file (`data/ingest.checkpoint.json`) stores
`{chunk_id: content_hash}`. A chunk is **skipped** if its `content_hash`
hasn't changed — meaning unchanged content is never re-embedded or re-upserted,
saving both compute time and Chroma Cloud write cost.

> **Free-tier impact:** dedupe is the single biggest cost lever. A 1,000-chunk
> collection re-ingested daily without dedupe = 30,000 writes/month. With
> dedupe on a stable corpus, that drops to <100 writes/month — well inside
> any free-tier write quota.

---

## What Each Chunk Contains

After `build_chunks()`, every `Chunk` dataclass holds:

| Field | Example | Stored in ChromaDB as |
|---|---|---|
| `chunk_id` | `"a3f9c1_00003"` | document ID |
| `text` | `"Covered calls are..."` | document body |
| `content_hash` | `"b8f2c1..."` | metadata |
| `source_file` | `"t1-tactical-opportunist-100-questions.md"` | metadata |
| `chunk_index` | `3` | metadata |
| `char_len` | `1742` | metadata |
| `text_preview` | first 200 chars | metadata |
| `ingested_at` | `"2026-05-01T14:00:00Z"` | metadata |

The embedding vector (1024 floats = 4 KB raw) is stored separately as the
ChromaDB vector, not in metadata. Including the document body and metadata,
each row averages **~6–8 KB on disk**.

---

## Concrete Example

Given this passage from `doc-1-options-and-swing-trading.md`:

```
A covered call involves selling a call option against shares you already own.
The premium collected reduces your cost basis. If the stock stays below the
strike, you keep the premium and the shares. If it rises above the strike,
your shares get called away at the strike price.
```

With `chunk_size=400`, this entire paragraph (≈65 tokens) fits comfortably in
one chunk alongside surrounding sentences until the 400-token window fills.
The splitter will not cut mid-sentence to hit the limit.

---

## Known Limitations of Current Settings

| Issue | Detail | Impact |
|---|---|---|
| **400 tokens is small for prose** | The four large doc files (82–101 KB each) are dense trading content with multi-sentence concepts. A 400-token window often cuts an idea in half. | Query recall suffers — the answer spans two chunks, neither has full context |
| **50-token overlap is thin** | At 400 tokens, 50-token overlap is 12.5%. Standard recommendation for dense technical text is 15–20%. | Boundary chunks miss context from the previous chunk |
| **No semantic splitting** | `SentenceSplitter` is purely token-count-based. It doesn't know that a paragraph about "IV crush" is semantically different from the next one about "delta hedging". | Chunks mix topics, reducing embedding precision |
| **Uniform settings across all files** | `t1-100-questions.md` is a Q&A file (short atomic units) and benefits from small chunks. `doc-1-options.md` is long-form prose and benefits from larger chunks. Both get the same settings. | Sub-optimal for both file types |
| **Empty/near-empty chunks emitted** | `SentenceSplitter` can occasionally produce chunks with only a heading or a single short line. The current loop skips empty chunks but still embeds 20-token "stub" chunks. | Wastes a write and pollutes top-k results |

### Recommended improvements

- Raise `chunk_size` to **600–768 tokens** for the four large `doc-*.md` files
  (768 tokens stays inside e5-large-v2's 512-token effective window after the
  `"Passage: "` prefix is added — the prefix consumes 2 tokens, so practical
  ceiling is ~510)
- **Correction:** because of the e5-large-v2 512-token cap, keep `chunk_size`
  at or below **480 tokens** for prose files. The recall gain comes from a
  larger overlap and better boundary detection, not bigger windows
- Raise `chunk_overlap` to **80–96 tokens** (≈20% of 480)
- Keep `chunk_size=300, chunk_overlap=40` for Q&A files (`t1-*`, `t2-*`) which
  have natural question-answer atomic units that should not be merged
- Consider `MarkdownNodeParser` (also in `llama_index.core`) which respects
  `##` heading boundaries — better for files structured with headers
- Add a **min-chunk filter**: drop any chunk under 80 tokens (≈60 words) to
  avoid embedding stubs

---

## ChromaDB Free-Tier Optimisations

The pipeline runs in two modes (`CHROMA_MODE=local` or `cloud`). Free-tier
guidance is similar for both — minimise stored bytes and write count.

### Storage budget per chunk (1024D float32 + metadata)

| Component | Size |
|---|---|
| Embedding vector (1024 × 4 bytes) | 4,096 B |
| Document body (avg chunk text) | ~1,800 B |
| Metadata (8 fields, JSON-encoded) | ~400 B |
| HNSW index overhead (M=32) | ~600 B |
| **Total per chunk on disk** | **~6.9 KB** |

### Free-tier targets

- **Local PersistentClient**: free, capped only by your disk. A typical laptop
  SSD comfortably holds 1M+ chunks (~7 GB).
- **Chroma Cloud free tier (current)**: confirm the live quota at
  trychroma.com/pricing — historically a small monthly write/storage credit.
  At ~7 KB/chunk and $2.50/GB written, **a 10K-chunk corpus = ~$0.18 to write**
  and ~$0.02/month to store. The pipeline's `BUDGET_SOFT=$5.00` guardrail
  ([ingest.py:69](../../ai-text-opt-1024/ingest.py#L69)) gives you ~280K chunks
  of headroom before pre-flight aborts.

### Levers, ranked by free-tier impact

1. **Content-hash dedupe** (already enabled) — eliminates re-writes on rerun.
2. **Smaller chunks** — 300-token Q&A chunks are ~25% smaller than 400-token
   prose chunks. Use the file-type-aware config (below).
3. **Drop unused metadata fields** — `text_preview` duplicates `documents` for
   the first 200 chars. Removing it saves ~200 B/chunk × 10K = 2 MB.
4. **Restrict `query_include`** — currently returns documents+metadatas+
   distances. For ranked-only flows, `["distances"]` cuts the GB-returned
   charge by ~95%.
5. **Compress before storage** (advanced) — `np.float16` halves vector bytes
   (4 KB → 2 KB) at ~0.5% recall cost. ChromaDB stores whatever you upsert,
   so cast embeddings to fp16 before `.tolist()`. Verify recall first.

### Free-tier anti-patterns to avoid

- Re-running ingest without the checkpoint file (forces full re-write).
- `chunk_overlap > 30%` of `chunk_size` — duplicates content across chunks
  and inflates storage with no recall benefit past ~20%.
- Calling `collection.query(include=["embeddings"])` from the backend — pulls
  4 KB × top_k bytes back over the wire on every search.

---

## File-Type-Aware Chunking (Recommended Workflow)

Currently `ingest.py` uses one `SentenceSplitter` for all files. The optimised
workflow detects file type by filename pattern and applies different settings:

```python
QA_PATTERNS = ("t1-", "t2-", "-qa.md", "-100-questions")
PROSE_DEFAULT = {"chunk_size": 480, "chunk_overlap": 96}   # ~20% overlap, fits e5
QA_DEFAULT    = {"chunk_size": 300, "chunk_overlap": 40}   # ~13% overlap, atomic units


def chunker_for(filename: str) -> SentenceSplitter:
    is_qa = any(p in filename for p in QA_PATTERNS)
    settings = QA_DEFAULT if is_qa else PROSE_DEFAULT
    return SentenceSplitter(**settings)
```

Build chunks per-document instead of in one batch so each file gets the right
splitter:

```python
def build_chunks(documents: List[Document]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in documents:
        source_file = doc.metadata.get("source_file", "unknown")
        splitter = chunker_for(source_file)
        nodes = splitter.get_nodes_from_documents([doc])
        # ... existing per-node logic ...
    return chunks
```

Expected impact on the trader-qa corpus (4 prose files + 2 Q&A files):

| Metric | Current (uniform 400/50) | File-aware (480/96 prose, 300/40 Q&A) |
|---|---|---|
| Total chunks | ~430 | ~360 (-16%) |
| Storage on disk | ~3.0 MB | ~2.5 MB (-17%) |
| Recall@5 (estimated) | baseline | +5–10% on prose, neutral on Q&A |
| Free-tier writes/rerun (no dedupe) | 430 | 360 |

---

## Configuration

Chunk settings are constants in [ingest.py](../../ai-text-opt-1024/ingest.py):

```python
CHUNK_SIZE    = 400   # line 64
CHUNK_OVERLAP = 50    # line 65
```

And mirrored in [config/pipeline.yaml](../../ai-text-opt-1024/config/pipeline.yaml):

```yaml
embedding:
  chunk_size: 400
  chunk_overlap: 50
```

To change them, update both and re-run `python ingest.py`. The content-hash
dedupe means only chunks whose text actually changed will be re-embedded.

> **Tip:** when you change `chunk_size`, **every** chunk's text changes, so
> dedupe will not save you on the next run. Bump
> `CHROMA_COLLECTION_VERSION` first so the new chunks land in a fresh
> versioned collection (e.g. `ideas_1024d_v2`) and you can roll back if recall
> regresses.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: Model produces 768D vectors; expected 1024D` | Wrong `EMBEDDING_MODEL` env var | Set to `intfloat/e5-large-v2` |
| Recall feels worse after tuning | Chunks crossed the 512-token e5 cap | Drop `chunk_size` back below 480 |
| `Embed attempt N/3 failed` loops | OOM on GPU with `BATCH_SIZE=32` | Drop `EMBEDDING_BATCH_SIZE` to 16 or 8 |
| Self-match test fails ([scripts/verify_ingest.py](../../ai-text-opt-1024/scripts/verify_ingest.py)) | Query-time prefix mismatch | Confirm embed service uses `"Query: "` and ingest uses `"Passage: "` |
| Pre-flight cost abort | Too many new chunks at once | Raise `CHROMA_BUDGET_SOFT` or split ingest by file glob |
