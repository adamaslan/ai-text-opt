# 8 Most Professional Pipelines in ai-text-opt

**Repository:** `/Users/adamaslan/code/ai-text-opt`  
**Date:** May 1, 2026  
**Status:** Production-grade and experimental implementations

---

## 1. **Multi-Phase Therapeutic Agent Framework** (`ma5.ipynb`)
**Category:** Local LLM + Multi-Agent Orchestration  
**Tech Stack:** Ollama (local inference), Python dataclasses, ThreadPoolExecutor, Pickle serialization

### Architecture
- **OllamaClient** — Robust HTTP client with retry logic, model verification, auto-pulling
- **BaseAgent** — Timeout-aware base class with `safe_generate()` for production safety
- **Specialized Agents:**
  - IntimacyContextAnalyzer (analyzes desires/communication patterns)
  - IntimacyActionGenerator (creates action plans)
  - IntimacyCustomizer (refines suggestions for individuals)
  - IntensitySpecialist (creates hyper-intense variants)

### Key Features
- **Timeout Management**: ThreadPoolExecutor with 300s max wait per generation
- **Retry Logic**: Exponential backoff on failures, max 5 retries per request
- **Error Handling**: Graceful degradation, comprehensive logging, structured response objects
- **Pickle Checkpointing**: Each agent saves outputs for pipeline resumption
- **JSON Parsing Fallback**: Safe extraction of JSON from malformed LLM responses

### Production Readiness
✅ Type hints throughout  
✅ Structured dataclass responses (TherapeuticResponse)  
✅ Comprehensive logging  
✅ Checkpoint-based recovery  
⚠️ Local model dependency (Ollama must be running)

---

## 2. **Combined Embeddings Generator** (`roberta-emb.ipynb` / `create_embeddings.py`)
**Category:** Dual-Model Embeddings (Ollama + SentenceTransformer)  
**Tech Stack:** AsyncIO, httpx, SentenceTransformers, PyTorch, Pickle batching

### Architecture
- **EmbeddingConfig** — YAML + environment variable configuration
- **EmbeddingCache** — Dual-cache system (in-memory + disk persistence)
- **EmbeddingGenerator** — Async batch processor with GPU acceleration

### Key Features
- **Async Processing**: Full asyncio pipeline for parallel embeddings
- **GPU Acceleration**: Automatic CUDA detection and device placement
- **Dual Models:**
  - Ollama embeddings (via HTTP API)
  - SentenceTransformer (local GPU acceleration)
  - Combined via concatenation
- **Checkpoint System**: Save every N=100 items for recovery
- **Exponential Backoff**: Smart retry on API failures
- **Combined Embeddings**: Concatenates Ollama (768d) + ST (384d) → 1152d vectors

### Production Readiness
✅ Async/await for I/O efficiency  
✅ GPU detection and auto-placement  
✅ Checkpoint every 100 items  
✅ Comprehensive error logging  
✅ Configuration via YAML or environment  
⚠️ Memory-intensive for large datasets (pickle batching helps)

---

## 3. **Fast CSV Embedding Pipeline** (`create_embeddings.py` variant)
**Category:** Batch CSV Processing  
**Tech Stack:** Pandas, HuggingFaceEmbeddings, CSV I/O, Progress bars

### Architecture
- **EmbeddingConfig** — Input/output files, model name, batch size
- **EmbeddingGenerator** — Synchronous batch processing with HuggingFace
- **Progress Tracking** — tqdm progress bars with checkpointing

### Key Features
- **CSV In/Out**: Loads `151_ideas_updated2.csv`, outputs embeddings to new CSV
- **Text Preprocessing**: Cleaning, lowercasing, punctuation removal
- **Batch Processing**: Configurable batch size (default 32)
- **Checkpoint Recovery**: Resumes from partial completion
- **Model Support**: RoBERTa, sentence-transformers, any HuggingFace model

### Production Readiness
✅ Pandas integration  
✅ Checkpoint/recovery  
✅ GPU auto-detection  
✅ Clean CSV output  
⚠️ Synchronous (not async) — slower for many requests

---

## 4. **RoBERTa Batch Embeddings** (`roberta-emb.ipynb` cell with transformers)
**Category:** Direct Transformer Embeddings  
**Tech Stack:** PyTorch, transformers library, HuggingFace models, NumPy

### Architecture
```python
Tokenization → RoBERTa Encoding → Mean Pooling → NumPy Arrays → CSV Output
```

### Key Features
- **RoBERTa-Base**: Industry-standard transformer (355M parameters)
- **Batch Processing**: GPU-accelerated with configurable batch size (default 32)
- **Mean Pooling**: Token embeddings → sentence embeddings (384d)
- **Device Auto-Selection**: CUDA if available, CPU fallback
- **Output Format**: CSV with original text + embeddings column

### Production Readiness
✅ Minimal dependencies  
✅ Standard model (RoBERTa)  
✅ GPU acceleration  
✅ Direct CSV output  
⚠️ No checkpointing in this variant

---

## 5. **Knowledge Graph Builder** (`knowledge_graph.py`)
**Category:** Semantic Graph Construction  
**Tech Stack:** ChromaDB, Cosine similarity math, JSON output, NetworkX (optional)

### Architecture
- **Node** — Document wrapper with embedding, metadata, collection info
- **Graph Construction** — All-pairs cosine similarity computation
- **Edge Definition** — Similarity threshold (default 0.75)
- **Adjacency List** — JSON export with cross-collection bridges

### Key Features
- **Multi-Collection**: Bridges across multiple ChromaDB collections
- **Cosine Similarity**: Optimized vector math (norm pre-computation)
- **Configurable Threshold**: `--threshold 0.70` command-line override
- **Sampling**: Limits graph to N nodes per collection (default 100)
- **Output Formats:**
  - JSON adjacency list
  - Console summary (most connected nodes)
  - Optional NetworkX visualization

### Production Readiness
✅ Command-line interface  
✅ Configurable parameters  
✅ Modular node/edge classes  
✅ JSON export  
⚠️ O(n²) complexity — not scalable to 1M+ nodes

---

## 6. **Weaviate Vector DB Migration** (`weaviate_migrate.py`)
**Category:** Vector Database Pipeline  
**Tech Stack:** Weaviate Python client, schema definitions, bulk import

### Architecture
- **Schema Definition** — Weaviate class/property setup
- **Bulk Import** — Batch object creation
- **Health Checks** — Verify connectivity before operations

### Key Features
- **Schema Migration**: Define custom properties (vectors, metadata)
- **Batch Operations**: Efficient bulk creation
- **Vector Integration**: Direct vector storage in Weaviate
- **Health Monitoring**: Connection verification

### Production Readiness
✅ Standard Weaviate client  
✅ Schema-aware  
✅ Batch operations  
⚠️ Limited error recovery  
⚠️ No pagination for large migrations

---

## 7. **Chroma RAG Chatbot** (`chroma_rag_chatbot.py`)
**Category:** RAG (Retrieval-Augmented Generation)  
**Tech Stack:** LangChain, ChromaDB, HuggingFace LLM, PromptTemplates

### Architecture
```
Query → Retrieval (ChromaDB) → Context Assembly → LLM Generation → Response
```

### Key Features
- **Retrieval**: Semantic search in ChromaDB vector store
- **Prompt Template**: Custom instruction format for LLM
- **Chain Type**: "Stuff" (simple document concatenation into context)
- **Source Tracking**: Returns relevant documents alongside answer
- **HTML UI** (in Marimo version): Interactive chatbot interface

### Production Readiness
✅ LangChain framework  
✅ Retriever-based  
✅ Source attribution  
✅ HTML UI rendering  
⚠️ No rate limiting  
⚠️ Single-threaded

---

## 8. **Gemini RAG Chatbot** (`gemini_rag_chatbot.py`)
**Category:** Cloud LLM Integration (Google Gemini)  
**Tech Stack:** LangChain, Google Gemini API, PromptTemplates, FAISS

### Architecture
```
CSV Embeddings → FAISS Vector Store → Retrieval → Gemini LLM → Response
```

### Key Features
- **FAISS Vector Store**: Fast in-memory similarity search
- **Google Gemini API**: Cloud-based LLM integration
- **Batch Embedding Loading**: Pre-computed embeddings from CSV
- **Custom Prompt**: Instruction format for Gemini
- **Source Documents**: Retrieval context attribution

### Production Readiness
✅ Cloud LLM integration  
✅ FAISS for fast search  
✅ API key management  
⚠️ Depends on Google API availability  
⚠️ Cost per API call  

---

## Comparison Matrix

| Pipeline | Type | LLM | Embeddings | I/O | GPU | Async | Checkpoints | Scale |
|----------|------|-----|-----------|-----|-----|-------|-------------|-------|
| **ma5** | Agent | Ollama | N/A | Pickle | ✗ | ✗ | ✅ | <1K |
| **roberta-emb** | Embedding | N/A | RoBERTa | CSV | ✅ | ✅ | ✅ | 10K-100K |
| **CSV Embed** | Embedding | N/A | HF | CSV | ✅ | ✗ | ✅ | 10K-100K |
| **Knowledge Graph** | Graph | N/A | Vector math | JSON | ✗ | ✗ | ✗ | <100K |
| **Weaviate** | VectorDB | N/A | External | HTTP | N/A | ✗ | ✗ | 100K-1M |
| **Chroma RAG** | RAG | HF/Local | ChromaDB | Chat | ✅ | ✗ | ✗ | <100K |
| **Gemini RAG** | RAG | Google API | FAISS | Chat | ✗ | ✗ | ✗ | <1M |

---

## Recommended Next Steps

### 1. **Production Hardening**
- [ ] Add request rate limiting to RAG chatbots
- [ ] Implement multi-threading for concurrent Ollama requests
- [ ] Add database persistence for chat history
- [ ] Implement authentication for APIs

### 2. **Scalability Improvements**
- [ ] Migrate knowledge graph to Neo4j for 100K+ nodes
- [ ] Add pagination to Weaviate migration
- [ ] Implement streaming responses in RAG pipelines
- [ ] Use batch APIs instead of sequential for Gemini

### 3. **Integration with Nu-Finance**
- [ ] Use `ma5` framework as base for **trader profiling agent**
- [ ] Use `roberta-emb` for **swing prediction embeddings**
- [ ] Use `chroma_rag_chatbot` template for **trading advice chatbot**
- [ ] Use `knowledge_graph` for **cross-trader strategy analysis**

### 4. **Monitoring & Observability**
- [ ] Add structured logging to all pipelines (already in most)
- [ ] Add metrics collection (latency, embeddings/sec, cache hit rate)
- [ ] Add health check endpoints for each pipeline
- [ ] Create dashboard for pipeline performance

### 5. **Testing & Validation**
- [ ] Unit tests for agent framework (BaseAgent, specialized agents)
- [ ] Integration tests for embedding pipelines with real CSV
- [ ] Load tests for RAG chatbots (concurrent requests)
- [ ] Validation of embedding quality (similarity searches)

---

## File Locations Quick Reference

```
/Users/adamaslan/code/ai-text-opt/
├── ma5.ipynb                          # [1] Therapeutic Agent Framework
├── roberta-emb.ipynb                  # [2] RoBERTa Embeddings (interactive)
├── create_embeddings.py                # [2/3] Async Embedding Generator
├── knowledge_graph.py                  # [5] Knowledge Graph Builder
├── weaviate_migrate.py                 # [6] Weaviate Migration
├── weaviate_import.py                  # [6] Weaviate Bulk Import
├── chroma_rag_chatbot.py               # [7] Chroma RAG Chatbot
├── gemini_rag_chatbot.py               # [8] Gemini RAG Chatbot
├── chromadb_storage/                   # [7] ChromaDB vector store
├── 151_ideas_updated2.csv              # Input data for embeddings
└── docs/
    ├── trader-qa/                      # [New] Trading profiler docs
    │   ├── t1-tactical-opportunist-100-questions.md
    │   └── t2-structured-growth-investor-100-questions.md
    └── PROFESSIONAL-PIPELINES-SUMMARY.md  # [This file]
```

---

## Success Metrics by Pipeline

1. **ma5** — Agent framework response quality, timeout rate <5%, pickle load success >99%
2. **roberta-emb** — Embedding quality (semantic search recall >0.85), GPU utilization >80%
3. **Knowledge Graph** — Cross-collection edge density, most-connected node count
4. **Weaviate** — Import speed (docs/sec), query latency <50ms
5. **Chroma RAG** — Retrieval relevance (BM25 >0.7), answer quality (human eval)
6. **Gemini RAG** — API cost per query, latency <2s, answer coherence

---

**Version:** 1.0  
**Last Updated:** May 1, 2026  
**Author:** Claude Code Agent
