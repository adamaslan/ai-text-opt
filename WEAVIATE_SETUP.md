# Weaviate + Ollama Pipeline Setup

Minimal integration of Weaviate vector database with Ollama (dolphin-phi:2.7b) to process 151_ideas_updated2.csv

## Prerequisites

- Docker & Docker Compose installed
- Python 3.8+ with mamba fin-ai1 environment
- `weaviate-client` package

## Quick Start

### 1. Start Weaviate & Ollama

```bash
docker-compose up -d
```

Wait 30 seconds for services to initialize.

### 2. Pull the Ollama Model

```bash
docker compose exec ollama ollama pull dolphin-phi:2.7b
```

### 3. Install Python Dependencies

```bash
mamba activate fin-ai1
pip install -U "weaviate-client"
```

### 4. Run the Pipeline

```bash
python weaviate_pipeline.py
```

## What It Does

1. **Loads CSV** - Reads 151_ideas_updated2.csv  
2. **Creates Collection** - Sets up Weaviate collection with Ollama embeddings
3. **Vectorizes Data** - Uses dolphin-phi:2.7b to create embeddings
4. **Semantic Search** - Performs example searches on the data

## Configuration

Edit the constants in `weaviate_pipeline.py`:
- `CSV_FILE` - Input CSV file path
- `COLLECTION_NAME` - Weaviate collection name
- `EMBEDDING_MODEL` - Ollama model to use
- `OLLAMA_ENDPOINT` - Ollama API endpoint
- `WEAVIATE_URL` - Weaviate URL

## Verify Services

```bash
# Check Weaviate
curl http://localhost:8080/v1/.well-known/ready

# Check Ollama
curl http://localhost:11434/api/tags
```

## Stop Services

```bash
docker-compose down
```

## Troubleshooting

**Connection refused**: Wait 30s for services to start  
**Model not found**: Run the ollama pull command above  
**Out of memory**: Reduce batch_size in pipeline script
