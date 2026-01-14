# RAG System Revamp Plan

## Overview

Revamp the existing Gemini RAG chatbot to add:
1. **PostgreSQL + pgvector** (Supabase free tier) for vector storage
2. **Knowledge Graph** (NetworkX + JSON persistence) for context enrichment
3. **Federated Search** across multiple embedding files (384D + 768D)
4. **Mistral API** as alternative LLM alongside Gemini

## Architecture

```
User Query
    │
    ├──► [LLM Router] ──► Gemini API / Mistral API
    │
    ├──► [Federated Search Engine]
    │         │
    │    ┌────┴────┐
    │    ▼         ▼
    │  FAISS    pgvector
    │  (384D)   (768D)
    │    │         │
    │    └────┬────┘
    │         ▼
    │    [RRF Fusion]
    │
    └──► [Knowledge Graph] ──► Context Enrichment
              │
              ▼
         [RAG Chain] ──► Response
```

## New Directory Structure

```
ai-text-opt/
├── config.py                      # Unified configuration
├── gemini_rag_chatbot.py          # Updated main (refactored)
├── database/
│   ├── __init__.py
│   └── pgvector_store.py          # Async pgvector client
├── knowledge/
│   ├── __init__.py
│   ├── graph_store.py             # NetworkX graph + persistence
│   └── graph_builder.py           # Entity/relationship extraction
├── search/
│   ├── __init__.py
│   ├── federated_search.py        # Multi-store search + RRF
│   └── faiss_adapter.py           # FAISS wrapper
├── llm/
│   ├── __init__.py
│   ├── adapter.py                 # Gemini/Mistral abstraction
│   └── mistral_client.py          # Mistral API client
└── tests/
    └── test_revamp.py             # New component tests
```

## Implementation Steps

### Step 1: Configuration Layer
**File:** `config.py`

- Create `AppConfig` dataclass consolidating all settings
- Load from `.env` with validation
- Support feature flags: `USE_PGVECTOR`, `USE_FEDERATED_SEARCH`, `KG_ENABLED`
- Define `EmbeddingSourceConfig` for each embedding file

### Step 2: Mistral API Integration
**Files:** `llm/mistral_client.py`, `llm/adapter.py`

- `MistralClient`: Async httpx client for Mistral API
- `MistralConfig`: API key, model, temperature from env
- `LLMRouter`: Routes to Gemini or Mistral based on config/request
- `GeminiAdapter` + `MistralAdapter`: Unified `generate()` interface

### Step 3: Database Setup (Supabase)
**File:** `database/pgvector_store.py`

- Create Supabase project (free tier: 500MB, pgvector included)
- SQL schema for `embeddings_768d` table with vector column
- `PgVectorStore`: Async connection pool with asyncpg
- `search_similar()`: Cosine similarity search
- `bulk_insert()`: Migrate existing 768D embeddings

**Schema:**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE embeddings_768d (
    id SERIAL PRIMARY KEY,
    source_file VARCHAR(255),
    text TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    metadata JSONB DEFAULT '{}'
);
CREATE INDEX ON embeddings_768d USING ivfflat (embedding vector_cosine_ops);
```

### Step 4: Federated Search
**Files:** `search/federated_search.py`, `search/faiss_adapter.py`

- `VectorStore` protocol: `search()`, `dimension`, `name`
- `FAISSAdapter`: Wrap existing FAISS for 384D embeddings
- `FederatedSearchEngine`: Query all stores, fuse with RRF
- Reciprocal Rank Fusion: `score = Σ(weight / (k + rank))`

### Step 5: Knowledge Graph
**Files:** `knowledge/graph_store.py`, `knowledge/graph_builder.py`

- `KnowledgeGraph`: NetworkX DiGraph with JSON persistence
- `GraphBuilder`: Extract entities from documents (regex-based)
- `enrich_query_context()`: Add related concepts to RAG prompt
- Build initial graph from existing CSV documents

### Step 6: Refactor Main Chatbot
**File:** `gemini_rag_chatbot.py`

- Use `AppConfig` for all configuration
- Initialize `LLMRouter` with Gemini + Mistral (if keys present)
- Replace single FAISS with `FederatedSearchEngine`
- Add knowledge graph enrichment to context building
- Update prompt template to include graph context

### Step 7: Update Dependencies
**File:** `environment.yml`

Add:
```yaml
- networkx
- pip:
  - asyncpg
  - pgvector
  - httpx
```

### Step 8: Update .env.example
Add new variables:
- `MISTRAL_API_KEY`, `MISTRAL_MODEL`
- `SUPABASE_DB_URL` or `NEON_DB_URL`
- `USE_PGVECTOR`, `USE_FEDERATED_SEARCH`
- `KG_ENABLED`, `KG_PERSISTENCE_PATH`
- `DEFAULT_LLM_PROVIDER`

### Step 9: Migration Script
**File:** `scripts/migrate_embeddings.py`

- Load 768D embeddings from `ideas_with_embeddings.csv`
- Normalize vectors (current ones have norm ~11.4)
- Bulk insert to Supabase pgvector table

### Step 10: Update Tests
**File:** `test_gemini_rag.py` + `tests/test_revamp.py`

- Test LLM routing (Gemini/Mistral switching)
- Test federated search with mock stores
- Test knowledge graph persistence
- Test RRF fusion deduplication

## Critical Files to Modify

| File | Action | Purpose |
|------|--------|---------|
| `gemini_rag_chatbot.py` | Refactor | Use new components |
| `environment.yml` | Update | Add new dependencies |
| `.env.example` | Update | Document new variables |
| `test_gemini_rag.py` | Update | Add new test cases |

## Critical Files to Create

| File | Purpose |
|------|---------|
| `config.py` | Unified configuration |
| `llm/adapter.py` | LLM abstraction layer |
| `llm/mistral_client.py` | Mistral API client |
| `database/pgvector_store.py` | PostgreSQL vector store |
| `search/federated_search.py` | Multi-store search |
| `search/faiss_adapter.py` | FAISS wrapper |
| `knowledge/graph_store.py` | Knowledge graph |
| `knowledge/graph_builder.py` | Entity extraction |

## Environment Variables (New)

```bash
# Mistral
MISTRAL_API_KEY=your_key
MISTRAL_MODEL=mistral-small-latest
DEFAULT_LLM_PROVIDER=gemini  # or "mistral"

# Database
SUPABASE_DB_URL=postgresql://...
USE_PGVECTOR=true

# Knowledge Graph
KG_ENABLED=true
KG_PERSISTENCE_PATH=knowledge_graph.json

# Features
USE_FEDERATED_SEARCH=true
```

## Verification Plan

1. **Unit Tests**: Run `pytest tests/` for new components
2. **Integration Test**: Run `python test_gemini_rag.py` (updated)
3. **Manual Test**:
   - Start chatbot: `python gemini_rag_chatbot.py`
   - Test query with `--provider mistral` flag
   - Verify federated results show sources from both stores
   - Check `knowledge_graph.json` is created/updated
4. **Database Verify**: Query Supabase dashboard to confirm embeddings inserted

## Backward Compatibility

- Default config works without Mistral/Supabase keys
- `USE_PGVECTOR=false` (default) uses FAISS-only mode
- `KG_ENABLED=false` disables graph enrichment
- Existing `.env` files continue to work

---

## Detailed Component Specifications

### Mistral Client (`llm/mistral_client.py`)

```python
@dataclass
class MistralConfig:
    api_key: str
    model: str = "mistral-small-latest"
    base_url: str = "https://api.mistral.ai/v1"
    max_tokens: int = 512
    temperature: float = 0.7

class MistralClient:
    async def generate(self, prompt: str, system_prompt: str = None) -> str
    async def generate_stream(self, prompt: str) -> AsyncIterator[str]
```

### LLM Adapter (`llm/adapter.py`)

```python
class LLMProvider(Enum):
    GEMINI = "gemini"
    MISTRAL = "mistral"

class LLMRouter:
    def register(self, adapter: BaseLLMAdapter, is_default: bool = False)
    async def generate(self, prompt: str, provider: LLMProvider = None) -> LLMResponse
```

### pgvector Store (`database/pgvector_store.py`)

```python
class PgVectorStore:
    async def initialize(self) -> None
    async def search_similar(self, embedding: np.ndarray, limit: int = 5) -> List[SearchResult]
    async def bulk_insert(self, texts: List[str], embeddings: np.ndarray) -> int
    async def close(self) -> None
```

### Federated Search (`search/federated_search.py`)

```python
class FederatedSearchEngine:
    def register_store(self, store: VectorStore, weight: float = 1.0)
    def register_embedding_model(self, dimension: int, model: EmbeddingModel)
    async def search(self, query: str, limit: int = 5) -> List[SearchResult]
```

### Knowledge Graph (`knowledge/graph_store.py`)

```python
class KnowledgeGraph:
    def add_node(self, node: GraphNode) -> None
    def add_edge(self, edge: GraphEdge) -> None
    def get_related_concepts(self, concept_ids: List[str], max_hops: int = 2) -> List[Tuple]
    def enrich_query_context(self, query: str, retrieved_texts: List[str]) -> Dict
    def save_to_disk(self) -> None
```

### Unified Config (`config.py`)

```python
@dataclass
class AppConfig:
    # LLM
    google_api_key: str
    mistral_api_key: str
    default_llm_provider: str = "gemini"

    # Database
    supabase_db_url: str
    use_pgvector: bool = False

    # Knowledge Graph
    kg_enabled: bool = True
    kg_persistence_path: Path

    # RAG
    top_k: int = 5
    score_threshold: float = 0.3
    use_federated_search: bool = True

    @classmethod
    def from_env(cls) -> "AppConfig"
    def validate(self) -> List[str]
```
