# Local Ollama + Weaviate Pipeline

Two-phase workflow:
1. **Phase 1 (Now)**: Embed CSV data with local Ollama (no Docker required)
2. **Phase 2 (Later)**: Import embeddings to Weaviate (when Docker is available)

## Phase 1: Local Embedding (No Docker)

```bash
# Run with local Ollama instance
python embed_local.py
```

**Output:**
- `ideas_with_vectors.pkl` - Pickled embeddings for Python
- `ideas_with_vectors.json` - JSON format ready for Weaviate

**Requirements:**
- Ollama running: `ollama serve`
- dolphin-phi:2.7b model available

## Phase 2: Weaviate Import (Requires Docker)

When you have Docker installed and running:

```bash
# Start services
docker compose up -d

# Install client
pip install -U "weaviate-client"

# Import embeddings
python weaviate_import.py
```

**Output:**
- Embeddings stored in Weaviate vector database
- Accessible at `http://localhost:8080`

## Files

| File | Purpose |
|------|---------|
| `embed_local.py` | Create embeddings with local Ollama |
| `weaviate_import.py` | Import embeddings to Weaviate |
| `docker-compose.yml` | Docker services config |
| `weaviate_pipeline.py` | Full end-to-end pipeline (with Docker) |

## Workflow Example

```bash
# TODAY (no Docker needed)
$ ollama serve  # Terminal 1
$ python embed_local.py  # Terminal 2
# Output: ideas_with_vectors.json

# LATER (when Docker available)
$ docker compose up -d
$ python weaviate_import.py
# Embeddings now in Weaviate
```

## Configuration

Edit these variables in the scripts:

**embed_local.py:**
```python
CSV_FILE = "151_ideas_updated2.csv"  # Change CSV file
EMBEDDING_MODEL = "dolphin-phi:2.7b"  # Change model
```

**weaviate_import.py:**
```python
JSON_INPUT = "ideas_with_vectors.json"  # Input file
COLLECTION_NAME = "Ideas"  # Weaviate collection
```

## Status Check

```bash
# Is Ollama running?
curl http://localhost:11434/api/tags

# Is Weaviate running? (Docker required)
curl http://localhost:8080/v1/.well-known/ready
```

## Data Format

**ideas_with_vectors.json structure:**
```json
[
  {
    "data": { "col1": "value1", "col2": "value2" },
    "embedding": [0.123, 0.456, ...],
    "text_preview": "combined text..."
  },
  ...
]
```
