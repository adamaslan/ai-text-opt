# Gemini RAG Chatbot - Setup & Usage Guide

## Overview

The **Gemini RAG Chatbot** is a question-answering system that combines:
- **Google Gemini 2.0 Flash** for intelligent responses
- **High-quality MiniLM embeddings** (384D) for semantic search
- **FAISS vector store** for fast retrieval from 367 Q&A documents
- **LangChain** for RAG orchestration

This provides better response quality than local LLMs while requiring no model downloads.

## Quick Start

### 1. Get Your Gemini API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy your API key (looks like: `AIzaSy...`)

### 2. Configure the Chatbot

```bash
# Navigate to project directory
cd /Users/adamaslan/code/ai-text-opt

# Copy template to .env
cp .env.example .env

# Edit .env and add your API key
nano .env
# Or use your preferred editor: vim, code, etc.
```

Your `.env` should look like:
```bash
GOOGLE_API_KEY=AIzaSyXxXxXxXxXxXxXxXxXxXxXxXxXxXxXxXx
```

### 3. Install Dependencies

```bash
# Using mamba (recommended - already have ai-text-opt environment)
mamba activate ai-text-opt
pip install python-dotenv langchain-google-genai

# Or update the environment from environment.yml
mamba env update -f environment.yml
```

### 4. Run the Chatbot

```bash
python gemini_rag_chatbot.py
```

Expected output:
```
======================================================================
Initializing Gemini RAG Chatbot...
======================================================================

📚 Loading embeddings...
   ✓ Loaded 367 documents
   ✓ Embedding dimension: 384D

🔍 Creating vector store...
   ✓ FAISS index created with 367 entries

🧠 Connecting to Gemini API...
   ✓ Connected to gemini-2.0-flash-exp

⛓️  Building RAG pipeline...
   ✓ RAG chain ready

======================================================================
🤖 Gemini RAG Chatbot - Interactive Mode
======================================================================
Model: gemini-2.0-flash-exp
Knowledge Base: embeddings/151qa2_with_embeddings.csv
Retrieval: Top 5 results (score > 0.3)

Type your question or 'exit' to quit.
======================================================================

🤔 You:
```

### 5. Ask Questions

```
🤔 You: What is the meaning of life?

🤖 Gemini: Based on the knowledge base...
[Response with source citations]

📚 Sources:
  1. ...
  2. ...
  3. ...
```

Type `exit` to quit.

---

## Configuration

All settings are configured via `.env` file:

### Required
- `GOOGLE_API_KEY` - Your Gemini API key (required)

### Optional
- `GEMINI_MODEL` - Model choice
  - `gemini-2.0-flash-exp` (default, fastest, experimental)
  - `gemini-1.5-pro` (best quality, slower, higher API cost)
  - `gemini-1.5-flash` (production-grade, balanced)

- `RAG_TOP_K` - Number of context documents to retrieve (default: 5)

- `RAG_SCORE_THRESHOLD` - Minimum relevance score (default: 0.3)
  - Range: 0.0-1.0
  - Lower = more results (broader search)
  - Higher = fewer results (stricter matching)

- `MAX_TOKENS` - Maximum response length (default: 512)

- `TEMPERATURE` - Response creativity (default: 0.7)
  - 0.0 = deterministic (consistent)
  - 1.0 = balanced (default)
  - 2.0 = very creative

Example configuration for longer responses:
```bash
MAX_TOKENS=1024
TEMPERATURE=0.5
RAG_TOP_K=8
```

---

## Architecture

```
User Query
    ↓
[Gemini Embedding] ← all-MiniLM-L6-v2
    ↓
[FAISS Retriever] ← 367 Q&A documents (384D)
    ↓
[Top 5 Results] (score > 0.3)
    ↓
[Prompt Template] + [User Query] + [Context]
    ↓
[Gemini API] ← gemini-2.0-flash-exp
    ↓
[Generated Response]
    ↓
[Display with Sources]
```

### Components

**Embeddings**: 151qa2_with_embeddings.csv
- 367 Q&A entries
- 384-dimensional MiniLM embeddings
- Unit-normalized (perfect for cosine similarity)
- Generated with `setup_embeddings.sh`

**Vector Store**: FAISS
- Fast similarity search (O(log n) with hashing)
- CPU-based (no GPU required)
- Normalized L2 distance metric

**LLM**: Google Gemini API
- Cloud-based (no local model needed)
- Streaming API with 2M context window
- Free tier: 60 requests/minute

---

## Features

### ✓ Smart Context-Aware Responses
- Only answers from provided knowledge base
- Cites sources for transparency
- Admits when information is missing

### ✓ Fast Retrieval
- Semantic similarity search
- Configurable relevance thresholds
- Top-k document ranking

### ✓ Production-Ready
- Error handling for API failures
- Configuration validation
- Graceful degradation

### ✓ Developer-Friendly
- LangChain integration
- Source code readable and documented
- Easy to modify prompts or configuration

---

## Troubleshooting

### Error: "GOOGLE_API_KEY not found"
**Solution**: Ensure `.env` file exists and contains `GOOGLE_API_KEY=your_key`

```bash
# Check if .env exists
test -f .env && echo ".env exists" || echo ".env missing"

# Show current value (redacted)
grep GOOGLE_API_KEY .env | cut -d= -f1
```

### Error: "Embeddings file not found"
**Solution**: Generate embeddings using setup script

```bash
bash setup_embeddings.sh
```

### Error: "API quota exceeded"
**Solution**: Wait or upgrade API plan
- Free tier: 60 requests/minute
- Paid plans: Higher limits

### Error: "Connection timeout"
**Solution**: Check internet connection and Gemini API status
```bash
# Test internet
ping google.com

# Check API key validity (test with a simple query)
python gemini_rag_chatbot.py
```

### Slow Responses
**Solution**: Reduce `MAX_TOKENS` or change model

```bash
# Use faster model
GEMINI_MODEL=gemini-2.0-flash-exp MAX_TOKENS=256 python gemini_rag_chatbot.py
```

---

## Comparison: Gemini RAG vs Previous Implementations

| Feature | Gemini RAG | rob-rag1.ipynb |
|---------|-----------|----------------|
| **LLM** | Gemini API (cloud) | GPT-Neo 125M (local) |
| **Quality** | Excellent | Basic |
| **Speed** | 2-4s/query | 10-30s/query |
| **Embeddings** | MiniLM 384D (normalized) | RoBERTa 768D (not normalized) |
| **Retrieval** | Good discrimination | Poor (similarity 0.98+) |
| **No local model** | ✓ | ✗ (requires download) |
| **Configuration** | Via .env | Hardcoded |
| **Context window** | 2M tokens | 256 tokens |
| **Cost** | Free tier available | Free (local) |

---

## Advanced Usage

### Custom Prompt Template
Edit the `PROMPT_TEMPLATE` variable in `gemini_rag_chatbot.py`:

```python
PROMPT_TEMPLATE = """Your custom instructions here...

Context: {context}
Question: {question}

Answer:"""
```

### Using Different Embeddings
To use a different embedding dataset:

1. Ensure CSV has `text` and `Embeddings` columns
2. Update `EMBEDDINGS_CSV` in `ChatbotConfig`:
```python
EMBEDDINGS_CSV: str = "path/to/your_embeddings.csv"
```

### Programmatic Usage (not just CLI)
```python
from gemini_rag_chatbot import (
    load_embeddings_from_csv,
    create_vector_store,
    create_gemini_llm,
    create_rag_chain,
    ChatbotConfig
)

# Initialize components
texts, embeddings = load_embeddings_from_csv(ChatbotConfig.EMBEDDINGS_CSV)
vector_store = create_vector_store(texts, embeddings)
llm = create_gemini_llm()
qa_chain = create_rag_chain(vector_store, llm)

# Query programmatically
result = qa_chain({"query": "Your question here"})
print(result["result"])
for doc in result["source_documents"]:
    print(f"Source: {doc.page_content}")
```

---

## API Costs

**Google Gemini Free Tier**:
- 60 requests/month
- gemini-2.0-flash-exp
- No credit card required

**Paid Tiers**:
- $0.075 per 1M input tokens
- $0.30 per 1M output tokens
- Volume discounts available

Typical query: ~200 input + 200 output tokens = ~$0.00003

---

## Environment Setup Details

### System Requirements
- Python 3.11+
- ~500 MB disk space (embeddings + models)
- ~2 GB RAM during execution
- Internet connection (for Gemini API)

### Mamba Environment
```bash
# Create/update environment
mamba env create -f environment.yml

# Activate
mamba activate ai-text-opt

# List packages
mamba list
```

### Dependencies
```
python-dotenv        # Load .env configuration
langchain            # RAG orchestration (installed via pip)
langchain-google-genai # Gemini integration
sentence-transformers # Embeddings (for queries)
faiss-cpu            # Vector search
numpy, pandas        # Data processing
```

---

## Next Steps

1. ✓ Configure `.env` with API key
2. ✓ Run `python gemini_rag_chatbot.py`
3. ✓ Ask questions and verify responses
4. ✓ Check sources for context quality
5. Optional: Adjust `RAG_TOP_K` or `TEMPERATURE` for different behavior

---

## Documentation

- **Implementation**: `/docs/gemini-rag-setup.md` (this file)
- **RAG Architecture**: `/docs/roberta-rag1-notebook-summary.md`
- **Embeddings Analysis**: `/docs/csv-files-comparison.md`
- **Embeddings Generation**: `/docs/embeddings-generation-summary.md`
- **Source Code**: `/gemini_rag_chatbot.py` (well-documented)

---

## Support

For issues:
1. Check `.env` file for typos
2. Verify API key is valid
3. Ensure embeddings exist: `ls -lah embeddings/151qa2_with_embeddings.csv`
4. Check internet connection
5. Review error messages for specific issues

## License

This project follows the repository's standard license.
