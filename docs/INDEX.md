# Documentation Index - ai-text-opt

**Repository:** `/Users/adamaslan/code/ai-text-opt`  
**Last Updated:** May 1, 2026  
**Total Docs:** 15+ comprehensive guides

---

## 📚 Core Documentation

### **New Additions (May 2026)**

#### 1. **[END-TO-END-PIPELINE-CHAINS.md](END-TO-END-PIPELINE-CHAINS.md)** ⭐
   - **Purpose:** Complete data flow documentation for 8 production pipelines
   - **Scope:** CSV → Processing → Storage → Frontend
   - **Best For:** Understanding full systems (not just individual components)
   - **Size:** 28KB, ~300 lines
   - **Contains:**
     - Pipeline #1: CSV → FAISS → Gemini RAG
     - Pipeline #2: CSV → ChromaDB → Local LLM (RECOMMENDED MVP)
     - Pipeline #3: Ollama Multi-Agent Framework
     - Pipeline #4: Knowledge Graph Builder
     - Pipeline #5: Ollama → Weaviate
     - Pipeline #6: Form → Agent → Email (Personalization)
     - Pipeline #7: Live Market → Embeddings → Alerts
     - Pipeline #8: Fine-Tuning → Inference API
   - **Key Outputs:** Decision matrix, deployment paths, 4-phase rollout plan

#### 2. **[PROFESSIONAL-PIPELINES-SUMMARY.md](PROFESSIONAL-PIPELINES-SUMMARY.md)**
   - **Purpose:** Survey of 8 most professional components in the codebase
   - **Scope:** Individual pipeline components (not full chains)
   - **Best For:** Understanding what exists and code quality assessment
   - **Size:** 12KB, ~250 lines
   - **Contains:**
     - ma5.ipynb (Multi-Phase Therapeutic Agent)
     - roberta-emb.ipynb (Combined Embeddings)
     - CSV Embedding Pipeline
     - Knowledge Graph Builder
     - Weaviate Migration
     - Chroma RAG Chatbot
     - Gemini RAG Chatbot
   - **Key Features:** Comparison matrix, production readiness checklist

#### 3. **[README_1024D_EMBEDDINGS.md](README_1024D_EMBEDDINGS.md)** ✨
   - **Purpose:** Complete guide to 1024-dimensional embeddings
   - **Scope:** Model selection, usage, integration, benchmarks
   - **Best For:** Building semantic search with high precision
   - **Size:** 11KB, ~280 lines
   - **Contains:**
     - 3 recommended 1024D models (E5-Large-V2, BGE-Large, Jina-Large)
     - Quick start (3 options)
     - Integration examples (ChromaDB, FAISS, Weaviate)
     - Performance metrics
     - Troubleshooting guide
     - Migration from 384D/768D
   - **Models Covered:**
     - intfloat/e5-large-v2 (State-of-the-art)
     - BAAI/bge-large-en-v1.5 (Production baseline)
     - jinaai/jina-embeddings-v2-large-en (8K context)

#### 4. **[create_embeddings_1024d.py](create_embeddings_1024d.py)** 🔧
   - **Purpose:** Production-ready script for 1024D embedding generation
   - **Features:** Async batch processing, GPU acceleration, checkpointing
   - **Size:** 11KB, ~350 lines
   - **Usage:** `python create_embeddings_1024d.py`
   - **Output:** CSV + Pickle with 1024D vectors
   - **Supports:** All 3 recommended models via config

#### 5. **[embedding_config_1024d.yaml](embedding_config_1024d.yaml)** ⚙️
   - **Purpose:** Configuration template for 1024D embeddings
   - **Sections:**
     - Model selection (with recommendations)
     - Batch size tuning for different GPUs
     - Input/output file paths
     - Text column specification
   - **Usage:** Edit and run script with `python create_embeddings_1024d.py`

---

### **Trader Profiling (May 2026)**

#### 6. **[trader-qa/t1-tactical-opportunist-100-questions.md](trader-qa/t1-tactical-opportunist-100-questions.md)**
   - **Purpose:** Deep-dive profiling questions for T1 traders
   - **Profile:** Tactical Opportunists (1-3 month holding periods)
   - **Size:** 18KB, ~750 lines
   - **Sections:**
     - Holding period & mechanics (20Q)
     - Mean-reversion strategy (20Q)
     - Options selling & Axiom IV (20Q)
     - Tax-aware trading (20Q)
     - Risk management & sizing (20Q)
     - Behavioral & execution (10Q)
   - **Use Case:** Form-based trader profiling, personalization

#### 7. **[trader-qa/t2-structured-growth-investor-100-questions.md](trader-qa/t2-structured-growth-investor-100-questions.md)**
   - **Purpose:** Deep-dive profiling questions for T2 traders
   - **Profile:** Structured Growth Investors (1-10 year holding periods)
   - **Size:** 18KB, ~750 lines
   - **Sections:**
     - Multi-year thesis development (20Q)
     - Trend-following & momentum (20Q)
     - Structural fundamentals (20Q)
     - Tax optimization for long-holds (20Q)
     - Portfolio construction (20Q)
     - Behavioral & conviction (10Q)
   - **Use Case:** Form-based trader profiling, personalization

#### 8. **[trader-qa/swing-vs-trade-optimized-agents.md](trader-qa/swing-vs-trade-optimized-agents.md)**
   - **Purpose:** Foundational trader taxonomy and logic framework
   - **Content:**
     - 50 tax-optimized trading proofs
     - Swing trading logic (50 proofs)
     - Real-world examples with $AAPL
     - Mathematical formulations
   - **Use Case:** Backing logic for T1/T2 profiling

---

### **Legacy Documentation (Reference)**

#### 9. **[ma5-notebook-summary.md](ma5-notebook-summary.md)**
   - **Purpose:** Overview of ma5.ipynb (multi-agent framework)
   - **Topics:** Agent orchestration, Ollama integration, checkpoint system

#### 10. **[roberta-rag1-notebook-summary.md](roberta-rag1-notebook-summary.md)**
   - **Purpose:** Overview of RoBERTa embedding pipeline
   - **Topics:** Batch processing, GPU acceleration, output formats

#### 11. **[chromadb-pipeline-explanation.md](chromadb-pipeline-explanation.md)**
   - **Purpose:** ChromaDB setup and usage guide
   - **Topics:** Vector store initialization, collection management, queries

#### 12. **[gemini-rag-setup.md](gemini-rag-setup.md)**
   - **Purpose:** Google Gemini API integration for RAG
   - **Topics:** API setup, prompt templates, response handling

#### 13. **[embeddings-generation-summary.md](embeddings-generation-summary.md)**
   - **Purpose:** Overview of embedding generation approaches
   - **Compares:** Ollama, RoBERTa, SentenceTransformer methods

#### 14. **[csv-files-comparison.md](csv-files-comparison.md)**
   - **Purpose:** Documentation of input/output CSV formats
   - **Lists:** Column definitions, data types, examples

#### 15. **[additional-pipelines-summary.md](additional-pipelines-summary.md)**
   - **Purpose:** Notes on experimental pipeline variations
   - **Covers:** Alternative implementations and branches

---

## 🎯 Quick Reference by Use Case

### For Building Swing Trade Recommendation Engine
1. Start: [END-TO-END-PIPELINE-CHAINS.md](END-TO-END-PIPELINE-CHAINS.md) - Pipeline #2 (ChromaDB RAG)
2. Embeddings: [README_1024D_EMBEDDINGS.md](README_1024D_EMBEDDINGS.md)
3. Run: [create_embeddings_1024d.py](create_embeddings_1024d.py)
4. Reference: [chromadb-pipeline-explanation.md](chromadb-pipeline-explanation.md)

### For Personalizing to Trader Type
1. Start: [trader-qa/swing-vs-trade-optimized-agents.md](trader-qa/swing-vs-trade-optimized-agents.md)
2. T1 Profile: [trader-qa/t1-tactical-opportunist-100-questions.md](trader-qa/t1-tactical-opportunist-100-questions.md)
3. T2 Profile: [trader-qa/t2-structured-growth-investor-100-questions.md](trader-qa/t2-structured-growth-investor-100-questions.md)
4. Pipeline: [END-TO-END-PIPELINE-CHAINS.md](END-TO-END-PIPELINE-CHAINS.md) - Pipeline #6 (Form → Agent → Email)

### For Building Real-Time Alerts
1. Start: [END-TO-END-PIPELINE-CHAINS.md](END-TO-END-PIPELINE-CHAINS.md) - Pipeline #7 (Live Market → Alerts)
2. Embeddings: [README_1024D_EMBEDDINGS.md](README_1024D_EMBEDDINGS.md)
3. Pattern Matching: [PROFESSIONAL-PIPELINES-SUMMARY.md](PROFESSIONAL-PIPELINES-SUMMARY.md) - Knowledge Graph

### For Production Deployment
1. Overview: [PROFESSIONAL-PIPELINES-SUMMARY.md](PROFESSIONAL-PIPELINES-SUMMARY.md)
2. Architecture: [END-TO-END-PIPELINE-CHAINS.md](END-TO-END-PIPELINE-CHAINS.md) - Recommended MVP
3. Implementation: [create_embeddings_1024d.py](create_embeddings_1024d.py) + [chromadb-pipeline-explanation.md](chromadb-pipeline-explanation.md)

### For Understanding Code Quality
1. Components: [PROFESSIONAL-PIPELINES-SUMMARY.md](PROFESSIONAL-PIPELINES-SUMMARY.md)
2. Detailed Comparison Matrix for each pipeline
3. Production readiness checklist

---

## 📊 Documentation Matrix

| Doc | Type | Pages | Focus | Audience | Status |
|-----|------|-------|-------|----------|--------|
| END-TO-END-PIPELINE-CHAINS.md | Guide | 28KB | Full systems | Architects | ✅ |
| PROFESSIONAL-PIPELINES-SUMMARY.md | Reference | 12KB | Components | Developers | ✅ |
| README_1024D_EMBEDDINGS.md | Guide | 11KB | Embeddings | Engineers | ✅ |
| t1-tactical-opportunist-100q | Reference | 18KB | T1 profile | Product | ✅ |
| t2-structured-growth-100q | Reference | 18KB | T2 profile | Product | ✅ |
| swing-vs-trade-optimized | Reference | 25KB | Logic | Product | ✅ |
| create_embeddings_1024d.py | Code | 11KB | Script | Engineers | ✅ |
| embedding_config_1024d.yaml | Config | 1.2KB | Setup | Operators | ✅ |
| Legacy docs (7 files) | Reference | 45KB | Components | Context | ✅ |

---

## 🚀 Recommended Reading Order

### For First-Time Users (30 minutes)
```
1. END-TO-END-PIPELINE-CHAINS.md (intro + Pipeline #2 section)
2. README_1024D_EMBEDDINGS.md (quick start)
3. Skim PROFESSIONAL-PIPELINES-SUMMARY.md (comparison matrix)
```

### For Builders (2-3 hours)
```
1. END-TO-END-PIPELINE-CHAINS.md (full read)
2. PROFESSIONAL-PIPELINES-SUMMARY.md (full read)
3. README_1024D_EMBEDDINGS.md (full read)
4. create_embeddings_1024d.py (code review)
5. chromadb-pipeline-explanation.md (integration)
```

### For Product Managers (1-2 hours)
```
1. swing-vs-trade-optimized-agents.md (trading logic)
2. t1-tactical-opportunist-100-questions.md (T1 profile)
3. t2-structured-growth-100-questions.md (T2 profile)
4. END-TO-END-PIPELINE-CHAINS.md (Pipeline #6 - personalization)
```

### For Architects (4-5 hours)
```
1. All documentation (comprehensive read)
2. Code review: create_embeddings_1024d.py
3. Code review: ma5.ipynb (in parent dir)
4. Code review: chroma_rag_chatbot.py (in parent dir)
```

---

## 🔄 File Organization

```
/Users/adamaslan/code/ai-text-opt/
│
├── docs/                          # Main documentation folder
│   ├── INDEX.md                   # ← You are here
│   │
│   ├── 📖 CORE GUIDES (New May 2026)
│   ├── END-TO-END-PIPELINE-CHAINS.md
│   ├── PROFESSIONAL-PIPELINES-SUMMARY.md
│   ├── README_1024D_EMBEDDINGS.md
│   │
│   ├── 🔧 IMPLEMENTATIONS
│   ├── create_embeddings_1024d.py
│   ├── embedding_config_1024d.yaml
│   │
│   ├── 📊 TRADER PROFILING
│   └── trader-qa/
│       ├── t1-tactical-opportunist-100-questions.md
│       ├── t2-structured-growth-investor-100-questions.md
│       └── swing-vs-trade-optimized-agents.md
│
├── 📱 Root files (legacy)
├── ma5.ipynb
├── roberta-emb.ipynb
├── chroma_rag_chatbot.py
├── gemini_rag_chatbot.py
└── ...
```

---

## 📈 Statistics

- **Total Documentation:** 15 files
- **Total Size:** 180+ KB
- **Code Examples:** 40+
- **Deployment Paths:** 8
- **Trading Profiles:** 2 (T1, T2)
- **Profiling Questions:** 200 (100 per profile)
- **Vector Dimensions Covered:** 384D, 768D, 1024D
- **Models Documented:** 10+
- **Pipelines Detailed:** 8 end-to-end

---

## ✅ Checklist for Nu-Finance Implementation

### Phase 1: MVP (2 weeks)
- [ ] Read: END-TO-END-PIPELINE-CHAINS.md (Pipeline #2)
- [ ] Run: create_embeddings_1024d.py on swing trade data
- [ ] Implement: ChromaDB RAG chatbot
- [ ] Deploy: Web interface with Vercel

### Phase 2: Personalization (3 weeks)
- [ ] Read: All trader-qa/ documents
- [ ] Build: Trader profile form (T1 vs T2)
- [ ] Implement: Pipeline #6 (Form → Agent → Email)
- [ ] Deploy: Email notification system

### Phase 3: Real-Time (4 weeks)
- [ ] Implement: Pipeline #7 (Live Market → Alerts)
- [ ] Build: Market data ingestion
- [ ] Deploy: Slack/email alerts

### Phase 4: Specialized Model (6 weeks)
- [ ] Curate: Fine-tuning dataset
- [ ] Train: Llama-3.2-3B (fine-tune)
- [ ] Deploy: Pipeline #8 (Inference API)

---

## 🔗 Cross-References

**Related Files in Repository:**
- Marimo notebooks: `ma5.ipynb`, `roberta-emb.ipynb`, etc.
- Python scripts: `chroma_rag_chatbot.py`, `gemini_rag_chatbot.py`, etc.
- Data: `151_ideas_updated2.csv`, `ideas_with_embeddings.csv`, etc.
- Database: `chromadb_storage/` (ChromaDB persistent storage)

**Related Projects:**
- Nu-Finance backend: `/Users/adamaslan/code/gcp-app-w-mcp1/`
- Swing predictions endpoint: `backend/app/api/swing-predictions/`

---

## 📝 Document Maintenance

**Last Updated:** May 1, 2026  
**Next Review:** June 1, 2026  
**Maintained By:** Claude Code Agent  
**Version:** 1.0

---

**Start here:** [END-TO-END-PIPELINE-CHAINS.md](END-TO-END-PIPELINE-CHAINS.md) for overview  
**Implementation:** [create_embeddings_1024d.py](create_embeddings_1024d.py) to get started  
**Questions?** Refer to specific guide based on use case above
