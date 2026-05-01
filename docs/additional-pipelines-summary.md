# Multi-Pipeline Engineering Overview

In addition to the core ChromaDB pipeline, this repository contains five specialized pipelines designed for production RAG (Retrieval-Augmented Generation), experimental embedding strategies, and semantic data visualization.

---

## 1. LlamaIndex + Zilliz RAG Pipeline (TypeScript)
This is the most sophisticated "production-grade" pipeline in the repo, powering the **Trader Q&A Chatbot** features within the Nu-Finance application.

- **Source Code**: `nu-finance/backend/lib/workflow.ts`
- **Stack**: LlamaIndex.TS, Zilliz Cloud (Serverless Milvus), Voyage AI Reranker, LlamaParse.
- **Workflow Architecture**:
    - **Router Step**: Dynamically detects the user's intent.
    - **Filtered Retrieval**: Routes queries to specific "Trader Profiles" based on keyword heuristics or explicit UI toggles.
    - **Sub-Question Engine**: Decomposes complex comparative questions (e.g., "How do T1 and T2 differ on NVDA?") into granular tasks.

## 2. Weaviate + Ollama Transition Pipeline (Python)
A local-first pipeline designed as a migration path for existing data from ChromaDB to a full-featured vector database.

- **Source Code**: `weaviate_pipeline.py`
- **Stack**: Weaviate (Local), Ollama (`nomic-embed-text:latest`).
- **Core Functionality**:
    - Automates the transition from CSV data to Weaviate collections.
    - **In-Database Vectorization**: Implements the `text2vec-ollama` module to handle embeddings server-side.
    - Provides a standard interface for semantic search during the migration testing phase.

## 3. Gemini + FAISS RAG Pipeline (Python)
A high-speed, LangChain-based utility for rapid search and conversational response generation.

- **Source Code**: `gemini_rag_chatbot.py`
- **Stack**: FAISS (In-memory), HuggingFace MiniLM (384D), Google Gemini 2.0 Flash.
- **Key Features**:
    - **Modern Orchestration**: Uses the `Runnable` API for clear, modular RAG logic.
    - **Normalized Retrieval**: Pre-normalizes embeddings for unit-sphere cosine similarity inside FAISS.
    - Optimized for low-latency retrieval from medium-sized CSV knowledge bases.

## 4. RoBERTa + Ollama Hybrid Embedding Pipeline (Python)
A specialized ingestion tool designed for creating high-performance, high-dimensional semantic vectors.

- **Source Code**: `roberta-emb.py`
- **Stack**: SentenceTransformers (RoBERTa), Ollama API, PyTorch, Marimo Notebooks.
- **Innovation**:
    - **Vector Concatenation**: Combines local model embeddings (fine-grained) with large-scale API embeddings (broad context).
    - **Async/Parallel Processing**: Uses `httpx` and `asyncio` for high-throughput embedding generation with GPU acceleration.

## 5. Semantic Knowledge Graph Pipeline (Python)
A meta-analysis tool used to understand the structure and overlap between different data collections.

- **Source Code**: `knowledge_graph.py` & `visualize_graph.py`
- **Stack**: ChromaDB, NetworkX, Matplotlib, JSON Adjacency Lists.
- **Analytical Output**:
    - **Cross-Collection Bridges**: Identifies documents in different collections that share the same semantic root.
    - **Clustering**: Measures intra-collection density versus inter-collection connectivity.
    - **Degree Analysis**: Surfaces "hub" documents that are highly connected to multiple concepts.

---

## Comparative Stack Analysis

| Feature | LlamaIndex (Zilliz) | Weaviate | Gemini (FAISS) |
| :--- | :--- | :--- | :--- |
| **Language** | TypeScript | Python | Python |
| **Vector DB** | Zilliz Cloud | Weaviate (Local) | FAISS (In-memory) |
| **Embedder** | BAAI/bge (Auto) | Ollama | HuggingFace (Local) |
| **Reranker** | Voyage AI 2.5 | None | None |
| **LLM** | Gemini / Mistral | dolphin-phi (Ollama) | Gemini 2.0 Flash |
| **Ideal Use Case** | Production Web App | Transitory/Scaling | CLI / Protoyping |
