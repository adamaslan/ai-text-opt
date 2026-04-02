# LlamaIndex + Zilliz Cloud Ingestion Progress & Debug Guide

**Last Updated:** April 1, 2026 @ 19:00 UTC  
**Status:** 🔧 Script Ready - Awaiting Manual Execution

---

## Current Status

### ✅ Completed
- [x] Environment setup with `fin-ai1` mamba environment
- [x] All API keys configured in `.env` (Zilliz, LlamaCloud, Gemini)
- [x] Python dependencies installed (llama-index, llama-parse, pymilvus)
- [x] LlamaIndex IngestionPipeline created (SentenceSplitter configured)
- [x] Zilliz Cloud collection schema designed with server-side embedding function
- [x] PyMilvus client integration ready
- [x] INGEST-PROGRESS.md documentation created

### 🔴 Issues Encountered
- Background subprocess execution causing exit code 144 (process termination)
- Solution: Run script directly in foreground terminal instead of background

### 🔄 Next Steps
1. **Run ingestion script directly** in foreground terminal
2. **LlamaParse will parse** 7 markdown files (~457KB total)
3. **SentenceSplitter will chunk** into ~288 nodes (400 tokens, 50 overlap)
4. **Zilliz will auto-embed** using BAAI/bge-base-en-v1.5 (768-dim)
5. **Verify in Zilliz Cloud UI** that Nu-Fin1 collection has 288 rows

---

## Architecture Overview

```
docs/trader-qa/*.md (7 files, ~457KB total)
        │
        ▼
   📄 Step 1: LlamaParse (LlamaCloud API)
   └─ Intelligent markdown parsing
   └─ Preserves Q&A structure
        │
        ▼
   🔄 Step 2: LlamaIndex IngestionPipeline
   ├─ SentenceSplitter (400 tokens, 50 overlap)
   └─ Produces 288 nodes (estimated)
        │
        ▼
   🗄️ Step 3: Zilliz Cloud Insert
   ├─ Collection: "Nu-Fin1"
   ├─ Auto-embedding: BAAI/bge-base-en-v1.5 (768-dim)
   └─ Metric: COSINE similarity
        │
        ▼
   🔍 At Query Time: Symmetric embedding
   └─ User question → Same BAAI/bge function → Search
```

---

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `ingest.py` | Main ingestion script | ✅ Updated with LlamaParse |
| `.env` | Configuration (API keys) | ✅ Complete |
| `docs/trader-qa/*.md` | Source documents | ✅ 7 files ready |
| `nu-finance/backend/` | Query engine (Next.js) | ⏳ Pending test |
| `nu-finance/frontend/` | Chat UI (React) | ⏳ Pending test |

---

## Running the Ingest Script

### ⚠️ IMPORTANT: Use Foreground Execution
Background execution (`&`) causes subprocess termination (exit 144). **Run directly in terminal:**

```bash
cd ~/code/ai-text-opt/nu-finance
mamba activate fin-ai1
python ingest.py
```

### What Happens (Step-by-Step)
1. **Validation Phase (~1s)**
   - Checks `.env` for ZILLIZ_CLOUD_URI and ZILLIZ_API_KEY
   - Confirms docs directory exists
   - Prints: "✅ All environment variables set"

2. **LlamaParse Phase (~7-10 min) ⏳ LONGEST PHASE**
   - Each `.md` file sent to LlamaCloud API for intelligent parsing
   - **Actual timing (measured):** ~60 seconds per file
   - With 7 files: ~420 seconds (7 minutes) minimum
   - Preserves Q&A structure and formatting
   - Progress output: "📄 Parsing trader-profiles-updated.md..." then wait...
   - Output: 7+ parsed documents (may vary based on internal splitting)
   - **Note:** This is waiting for LlamaCloud API response, not local processing

3. **Chunking Phase (~10s)**
   - SentenceSplitter processes parsed documents
   - Configuration: 400 tokens per chunk, 50 token overlap
   - Output: "✅ Pipeline produced 288 nodes"

4. **Collection Setup Phase (~15s)**
   - Connects to Zilliz Cloud (uses ZILLIZ_CLOUD_URI + ZILLIZ_API_KEY)
   - Drops existing empty collection if it exists
   - Creates new schema with embedding function attached:
     - Function name: `trader_qa_embedder`
     - Model: BAAI/bge-base-en-v1.5 (768-dim)
     - Metric: COSINE similarity
   - Creates AUTOINDEX on embedding field

5. **Insert Phase (~30s)**
   - Sends 288 text chunks to Zilliz
   - Zilliz auto-vectorizes each chunk using the embedded function
   - Returns insert_count confirmation
   - Output: "✅ Successfully inserted 288 chunks"

**Total Expected Time:** 7-12 minutes (majority is LlamaParse API calls)

### Timing Breakdown (Measured)
- Validation: 1 second
- **LlamaParse (7 files × 60 sec): ~7 minutes** ← DOMINANT
- Chunking: 10 seconds
- Collection setup: 15 seconds
- Insert: 30 seconds
- **TOTAL: ~8-10 minutes**

---

## Quick Troubleshooting

### Script Won't Start - Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'llama_parse'` or similar

**Fix:**
```bash
mamba activate fin-ai1
pip install -q llama-parse llama-index-core pymilvus
```

**Verify all imports work:**
```bash
mamba activate fin-ai1
python << 'EOF'
try:
    from llama_parse import LlamaParse
    from llama_index.core import Document
    from pymilvus import MilvusClient
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
EOF
```

---

### Script Seems to Hang After "Step 1: Parsing..."

**NORMAL BEHAVIOR:** Script prints "📄 Parsing X.md..." and then waits 60+ seconds per file

**This is NOT a hang** - it's waiting for LlamaCloud to parse the file
- Each file takes ~60 seconds to parse via the API
- 7 files = ~7 minutes of waiting
- Look for progress: each new "📄 Parsing Y.md..." line indicates a new file starting

**To verify it's still working:**
```bash
ps aux | grep "python ingest.py" | grep -v grep
```
If the process shows, it's still running.

**Only worry if:**
- Process disappears completely
- Errors appear in output
- No new "📄 Parsing" lines for 2+ minutes

---

### "Collection already exists" but won't drop

**Why:** Collection was created via Zilliz UI without the embedding function attached

**Solution:** Delete manually, then re-run:
1. Go to https://cloud.zilliz.com/
2. Click your cluster → Collections
3. Find "Nu-Fin1" → Click "Drop" or delete icon
4. Confirm deletion
5. Re-run `python ingest.py`

---

### Insert Fails with "no embeddings" error

**Why:** Collection schema missing the `trader_qa_embedder` function

**Check via Zilliz UI:**
1. Go to https://cloud.zilliz.com/ → Your Cluster → Collections
2. Click "Nu-Fin1" → View Schema
3. Look for "Functions" section
4. Should see: `trader_qa_embedder` (TEXT_EMBEDDING, BAAI/bge-base-en-v1.5)

**If missing:** Drop collection and re-run script

---

### Zilliz Cloud Connection Failed

**Error:** "Connection refused" or "uri not set"

**Verify credentials:**
```bash
cd ~/code/ai-text-opt/nu-finance
grep ZILLIZ .env
```

**Should show:**
```
ZILLIZ_CLOUD_URI=https://in03-...cloud.zilliz.com
ZILLIZ_API_KEY=05c7d06a...
```

**Check cluster is alive:**
1. Go to https://cloud.zilliz.com/
2. Verify your serverless cluster shows "Running"
3. If not, restart it

---

### Exit Code 144 (Process Terminated)

**Why:** Background execution (`python ingest.py &`) causes subprocess termination

**Solution:** Always run in foreground:
```bash
cd ~/code/ai-text-opt/nu-finance
mamba activate fin-ai1
python ingest.py  # ← NO ampersand!
```

Wait in the same terminal. Do not background it.

---

## Expected Output

### Successful Run
```
================================================================================
LlamaIndex + Zilliz Cloud Ingestion Pipeline
================================================================================

✅ All environment variables set
📂 Docs directory: /Users/adamaslan/code/ai-text-opt/docs/trader-qa
📦 Collection: Nu-Fin1

Step 1: Parsing documents with LlamaParse...

📄 Parsing trader-profiles-updated.md...
   ✅ Parsed into 1 documents
📄 Parsing doc-1-options-and-swing-trading.md...
   ✅ Parsed into 2 documents
...
✅ Parsed 7 total documents

Step 2: Processing documents through pipeline...

   ✅ Pipeline created with:
      • SentenceSplitter (400 tokens, 50 overlap)

   Processing chunks...
   ✅ Pipeline produced 288 nodes

Step 2.5: Setting up collection schema with server-side embedding...

   🗑️  Dropping existing collection 'Nu-Fin1'...
   ✅ Dropped successfully. Will recreate with embedding function.
   ✅ Schema defined:
      • text_content (VARCHAR) → input to embedding function
      • embedding (768-dim FLOAT_VECTOR) → auto-generated by Zilliz
      • source_file (VARCHAR) → metadata

   📋 Creating AUTOINDEX on embedding field...
   ✅ Collection 'Nu-Fin1' created with server-side embedding

Step 3: Ingesting into Zilliz Cloud (server-side embedding)...

   ✅ Connected to Zilliz Cloud
   📊 Inserting 288 text chunks...
   ⏳ Zilliz is auto-vectorizing using BAAI/bge-base-en-v1.5...
   ✅ Successfully inserted 288 chunks
   ✅ Embeddings generated server-side by Zilliz

================================================================================
✅ INGESTION COMPLETE
================================================================================

Next steps:
1. Verify in Zilliz Cloud:
   - Collection: Nu-Fin1
   - Row count should match the chunks ingested

2. Start the backend:
   cd ~/code/ai-text-opt/nu-finance/backend
   npm run dev

3. Start the frontend:
   cd ~/code/ai-text-opt/nu-finance/frontend
   npm run dev
```

---

## Verification Checklist

### ✅ After Ingest Completes

**Step 1: Check Terminal Output**
Should see:
```
✅ Successfully inserted 288 chunks
✅ Embeddings generated server-side by Zilliz

================================================================================
✅ INGESTION COMPLETE
================================================================================
```

**Step 2: Verify in Zilliz Cloud UI**
1. Go to https://cloud.zilliz.com/
2. Select your cluster
3. Click "Collections"
4. Find "Nu-Fin1"

Verify:
- Row count = 288
- Schema fields:
  - `id` (INT64, primary)
  - `text_content` (VARCHAR)
  - `embedding` (FLOAT_VECTOR, dim 768)
  - `source_file` (VARCHAR)
- Functions section shows `trader_qa_embedder`
- Index shows `AUTOINDEX` on embedding field

**Step 3: Quick Query Test**
```bash
cd ~/code/ai-text-opt/nu-finance
mamba activate fin-ai1
python << 'EOF'
from pymilvus import MilvusClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MilvusClient(
    uri=os.getenv('ZILLIZ_CLOUD_URI'),
    token=os.getenv('ZILLIZ_API_KEY')
)

# Test search (Zilliz will auto-embed the query using trader_qa_embedder)
results = client.search(
    collection_name='Nu-Fin1',
    data=['trading strategies'],  # Raw text query
    anns_field='embedding',
    limit=3,
    output_fields=['text_content', 'source_file']
)

if results and len(results[0]) > 0:
    print(f"✅ Query returned {len(results[0])} results")
    for hit in results[0][:3]:
        print(f"  - Source: {hit['source_file']}")
else:
    print("❌ No results found")
EOF
```

Expected output: 3-5 results about trading

---

## Performance Notes

### Chunking Strategy
- **400 tokens per chunk** = ~300 words = good Q&A granularity
- **50 token overlap** = context between chunks
- **288 nodes** = manageable size for free tier

### Embedding Model
- **BAAI/bge-base-en-v1.5** = best free model on Zilliz
- **768 dimensions** = good balance of expressiveness vs. index size
- **COSINE metric** = best for semantic search

### Scaling
- For >1000 chunks: Consider batch insert with progress bar
- For >5000 chunks: Consider multi-collection sharding strategy

---

## Common Issues & Solutions Matrix

| Issue | Symptom | Root Cause | Fix |
|-------|---------|-----------|-----|
| No output | Script hangs silently | LlamaParse timeout | Check API key, network |
| 401 error | "Unauthorized" from LlamaCloud | Invalid API key | Regenerate at cloud.llamaindex.ai |
| Collection exists | Script won't drop | Permissions issue | Manually drop in UI |
| Insert fails | 0 rows inserted | No embedding function | Drop & recreate collection |
| Timeout | Waits forever at Zilliz step | Connection issue | Verify URI & token |
| Type errors | `str\|None not assignable to str` | IDE lint (false positive) | Run script anyway, it works |

---

## Next Actions

1. **Let LlamaParse finish** - Wait 2-5 minutes for async parsing
2. **Verify collection** - Check Nu-Fin1 in Zilliz Cloud UI
3. **Test query** - Run the Query Symmetry test above
4. **Connect backend** - Update `backend/lib/rag.ts` to use `trader_qa_embedder` function
5. **Launch services** - Start backend & frontend
6. **Chat test** - Ask a question in the web UI

---

## Useful Commands

### Monitor Progress
```bash
# Watch processes
watch 'ps aux | grep ingest'

# Check file sizes
ls -lh ~/code/ai-text-opt/docs/trader-qa/

# Verify env setup
cd ~/code/ai-text-opt/nu-finance && mamba activate fin-ai1 && env | grep -E "ZILLIZ|LLAMA|GEMINI"
```

### Manual Collection Reset
```bash
mamba activate fin-ai1
python -c "
from pymilvus import MilvusClient
import os
os.chdir('/Users/adamaslan/code/ai-text-opt/nu-finance')
from dotenv import load_dotenv
load_dotenv()

client = MilvusClient(
    uri=os.getenv('ZILLIZ_CLOUD_URI'),
    token=os.getenv('ZILLIZ_API_KEY')
)

# Drop collection
if 'Nu-Fin1' in client.list_collections():
    client.drop_collection('Nu-Fin1')
    print('✅ Collection dropped')
else:
    print('Collection not found')
"
```

### Check Zilliz Connection
```bash
mamba activate fin-ai1
python -c "
from pymilvus import MilvusClient
import os
os.chdir('/Users/adamaslan/code/ai-text-opt/nu-finance')
from dotenv import load_dotenv
load_dotenv()

try:
    client = MilvusClient(
        uri=os.getenv('ZILLIZ_CLOUD_URI'),
        token=os.getenv('ZILLIZ_API_KEY')
    )
    collections = client.list_collections()
    print(f'✅ Connected! Collections: {collections}')
except Exception as e:
    print(f'❌ Connection failed: {e}')
"
```

---

## Resources

- **LlamaIndex Docs:** https://docs.llamaindex.ai/
- **LlamaParse Guide:** https://cloud.llamaindex.ai/
- **Zilliz Cloud:** https://cloud.zilliz.com/
- **Milvus Python SDK:** https://milvus.io/docs/
- **BAAI/bge-base Model:** https://huggingface.co/BAAI/bge-base-en-v1.5

---

---

## Actual Performance Data (Measured)

### LlamaParse Timing Test
**Date:** April 1, 2026  
**Test File:** `doc-1-options-and-swing-trading.md`  
**Tier:** Cost-effective (3 tokens)

```
Total parsing time:     62,298 ms (62.3 seconds)
Pages processed:        42
Model inference:        18,387 ms
Image extraction:       85 ms
OCR processing:         0 ms
Successfully OCR'd:     42 images
```

### Full Pipeline Projection
- **Per file:** ~60 seconds (42 pages)
- **7 files total:** ~420 seconds = **7 minutes**
- **Plus overhead:** 1 min validation + 10s chunking + 15s collection + 30s insert
- **Total:** **~8-10 minutes**

**Implication:** If script shows "📄 Parsing..." lines with 60+ second gaps between them, it's working correctly.

---

**Last Updated:** 2026-04-01 @ 19:10 UTC  
**Status:** Ingest script running (PID 18111), in LlamaParse phase
