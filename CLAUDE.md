# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **text optimization and embeddings project** built with Marimo notebooks and Python, leveraging local LLMs via Ollama and transformer-based models for text analysis, generation, and semantic embeddings.

The core workflow consists of:
1. **Multi-phase agent framework** - Sequential text processing with local LLM (Ollama)
2. **Embedding generation** - Combined embeddings from Ollama and SentenceTransformer models
3. **Batch processing** - Handling CSV data with progress tracking and checkpoints
4. **RAG (Retrieval-Augmented Generation)** - Multiple branches exploring different RAG approaches

## Technology Stack

- **Marimo**: Interactive notebooks for cell-based Python applications (`ma5.py`, `roberta-emb.py`)
- **Ollama**: Local LLM inference with model pulling/verification
- **SentenceTransformers**: GPU-accelerated embedding generation
- **Asyncio**: Parallel processing with retry logic and exponential backoff
- **Pickle**: Data persistence for intermediate results (`.pkl` files)
- **CSV**: Input data format for batch processing

## Project Structure

### Core Application Files

- **`ma5.py`** - Marimo app with multi-phase agent framework using Ollama for text generation
  - BaseAgent class for timeout-aware generation
  - TherapeuticResponse dataclass for response tracking
  - OllamaClient with model verification and pulling
  - Multiple specialized agents (IntimacyContextAnalyzer, IntimacyActionGenerator, etc.)

- **`roberta-emb.py`** - Marimo app for embedding generation
  - EmbeddingGenerator for combined embeddings (Ollama + SentenceTransformer)
  - EmbeddingCache for caching strategies
  - EmbeddingConfig for YAML/env-based configuration
  - Async batch processing with checkpointing

### Data Files

- **CSV files**: `151_ideas_updated2.csv`, `151qa2.csv`, `ideas_with_embeddings.csv`
- **Pickle files**: Intermediate results and checkpoints (`.pkl` files)
- **Jupyter notebooks** (`*.ipynb`): Development and experimentation notebooks

### Branching Strategy

The repository has many experimental branches exploring different approaches:
- `main` - Primary branch
- `synth1` - Current development branch (contains `ma5.py` and `roberta-emb.py`)
- RAG branches (`rag1-5`, `rob-rag2/3/5a`, `doperag1/2`) - Different RAG implementations
- Embedding branches (`emb3`, `emb4`) - Embedding optimization experiments
- Local model branches (`ollama-local`, `ollama-local2`) - Local inference variants

## Common Development Tasks

### Running Marimo Applications

Marimo applications are interactive notebooks that run as web apps:

```bash
# Run the main text optimization app
marimo run ma5.py

# Run the embedding generator
marimo run roberta-emb.py

# Edit in Marimo IDE
marimo edit ma5.py
```

### Working with Ollama

The project requires a running Ollama service:

```bash
# Start Ollama (typically on localhost:11434)
ollama serve

# Check available models
ollama list

# Run specific model
ollama run dolphin-phi:2.7b
```

The default Ollama model used is:
- **Dolphin-Phi 2.7B**: `dolphin-phi:2.7b` (for both generation and embeddings)

### Processing Data

When working with CSV data:

```bash
# Input files expected: `151_ideas_updated2.csv` or similar
# Output: embeddings stored in pickle format
# Configuration: Can use `embedding_config.yaml` or environment variables
```

Example configuration via env vars:
```bash
export EMBEDDING_INPUT_FILE="input.pkl"
export EMBEDDING_OUTPUT_FILE="embeddings.pkl"
export EMBEDDING_OLLAMA_MODEL="dolphin-phi:2.7b"
export EMBEDDING_BATCH_SIZE="32"
export EMBEDDING_USE_GPU="true"
```

## Architecture Patterns

### Agent Framework Pattern

The project follows a **multi-phase agent pattern**:

1. **BaseAgent** - Foundation class with `safe_generate()` method
   - Wraps Ollama client calls with timeout handling
   - Manages retry logic and error tracking
   - Returns `TherapeuticResponse` dataclass with metadata

2. **Specialized Agents** - Inherit from BaseAgent
   - `IntimacyContextAnalyzer` - Analyzes user input for desires/communication
   - `IntimacyActionGenerator` - Creates action plans from analysis
   - `IntimacyCustomizer` - Refines actions based on preferences
   - `IntensitySpecialist` - Creates ultra-intense variants
   - Each agent focuses on a single transformation step

3. **Pipeline Execution**
   - Agents process output from previous step
   - Results serialized to pickle files for checkpoint/recovery
   - Safe loading with mock data fallback for missing files

### Embedding Generation Pattern

- **EmbeddingConfig** - Centralized configuration (YAML or env-based)
- **EmbeddingCache** - Dual-cache system (in-memory + persistent disk)
- **EmbeddingGenerator** - Async batch processor with:
  - Parallel embedding requests to Ollama
  - GPU acceleration for SentenceTransformer
  - Progress checkpointing every N items
  - Exponential backoff retry strategy

### Error Handling

- **Timeouts** - ThreadPoolExecutor with max_wait constraints
- **Retries** - Configurable max_retries with exponential backoff
- **Graceful Degradation** - Returns zero vectors on embedding failure
- **JSON Parsing** - Robust fallback parsing (try full → try substring extraction)
- **Resource Cleanup** - Context managers for HTTP clients and file handles

## Key Configuration Values

### OllamaClient

```python
model_name = "dolphin-phi:2.7b"
base_url = "http://localhost:11434"
max_retries = 5
request_timeout = 300  # seconds
```

### EmbeddingGenerator

```python
batch_size = 32              # Items per batch
max_retries = 3              # API retry attempts
timeout = 60                 # HTTP request timeout
use_gpu = True               # GPU acceleration
num_workers = 4              # Parallel workers
save_interval = 100          # Checkpoint every N items
```

### BaseAgent

```python
retry_count = 3
max_wait = 300               # Seconds to wait for generation
```

## Important Notes

### Data Persistence

- All results are pickled to files for recovery/resumption
- Checkpoint files (`.checkpoint.pkl`) enable resuming interrupted processes
- Cache files separate Ollama and SentenceTransformer embeddings

### Async/Concurrency

- All Ollama calls are async via httpx.AsyncClient
- SentenceTransformer calls are synchronous (GPU-accelerated internally)
- Combined embeddings use `asyncio.gather()` for parallel processing
- ThreadPoolExecutor used for timeout management

### Model Management

- Ollama models are auto-pulled if missing (`_verify_model()` → `_pull_model()`)
- Model status checks include name matching with substring search
- Progress logged during model pulls

### Performance Considerations

- GPU detection automatic (CUDA if available, else CPU)
- Batch processing with configurable batch size
- Caching at multiple levels (in-memory + disk)
- Checkpoint saves prevent reprocessing on resume
