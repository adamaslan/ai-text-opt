# roberta-rag1.ipynb - RAG Chatbot with RoBERTa & Local LLMs

## Overview

A Jupyter notebook implementing a **Retrieval-Augmented Generation (RAG) chatbot** that combines:
- **Vector embeddings** from CSV data (RoBERTa embeddings)
- **FAISS vector store** for semantic similarity search
- **Local LLM backends** for response generation
- **LangChain orchestration** for RAG pipeline

The notebook explores multiple LLM configurations (DeepSeek vs GPT-Neo) and documents iterative improvements ("works responses not great - better rag needed").

---

## Architecture Overview

```
CSV Data (ideas_with_embeddings.csv)
    ↓
[Load Embeddings] → RoBERTa 768D vectors
    ↓
[FAISS Vector Store] ← L2 normalization
    ↓
[RAG Retriever] ← k=5, score_threshold=0.4
    ↓
[LLM Pipeline] → GPT-Neo or DeepSeek
    ↓
[Custom Prompt Template] → Philosophical analysis
    ↓
[Chat Interface] → Interactive Q&A with sources
```

---

## Implementation Details

### Phase 1: Data Loading & Embedding Processing

**Function: `load_embeddings(csv_path)`**

Purpose: Load pre-computed embeddings from CSV and validate format

```python
# Input: ideas_with_embeddings.csv with columns:
# - Cleaned_Ideas: preprocessed text
# - Embeddings: stringified list format "[0.123, -0.456, ...]"

# Steps:
1. Read CSV into DataFrame
2. Parse Embeddings column:
   - Strip brackets "[]"
   - Remove newlines
   - Convert to numpy array (float32)
3. Validate all embeddings have 768 dimensions (RoBERTa-base standard)
4. Return: (texts_list, numpy_array_embeddings)
```

**Error Handling:**
- Column validation (must have 'Cleaned_Ideas' and 'Embeddings')
- Embedding dimension validation (768D expected)
- Detailed error messages for invalid entries

---

### Phase 2: FAISS Vector Store Creation

**Configuration:**

```python
embedding_model = HuggingFaceEmbeddings(
    model_name="roberta-base",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

vector_store = FAISS.from_embeddings(
    text_embeddings=list(zip(texts, embeddings)),
    embedding=embedding_model,
    normalize_L2=True  # CRITICAL: Enables cosine similarity
)
```

**Key Parameters:**
- **Model**: RoBERTa-base (for consistency with embeddings)
- **Device**: CPU (cross-platform compatibility)
- **Normalization**: FAISS applies L2 normalization for cosine similarity
- **Index type**: FAISS Flat index (exact nearest neighbor search)

**Performance:**
- Index size: 150 entries (all ideas from CSV)
- Search type: Cosine similarity
- Query configuration: `k=5, score_threshold=0.4`

---

### Phase 3: LLM Backend Configurations

The notebook implements **three versions** with different LLM choices:

#### Version 1: DeepSeek with 4-bit Quantization

```python
model_name = "deepseek-ai/deepseek-llm-1.5b-chat"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # nf4 quantization type
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True    # Double quantization
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)
```

**Characteristics:**
- Model size: 1.5B parameters (chat-optimized variant)
- Quantization: 4-bit NF4 (reduces memory ~75%)
- Compute type: FP16 (half-precision for speed)
- Memory footprint: ~600MB (with quantization)

#### Version 2: GPT-Neo 125M (Final Implementation)

```python
model_name = "EleutherAI/gpt-neo-125M"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)
```

**Characteristics:**
- Model size: 125M parameters (ultra-lightweight)
- Quantization: None (full precision for this small model)
- Compute type: FP32
- Memory footprint: ~250MB
- Trade-off: Smaller, faster, but lower quality responses

#### Version 3: Duplicate Configuration (Cell 2)

Cell 2 repeats Version 2 (GPT-Neo 125M) with notation "old v - pop up", suggesting earlier prototype.

---

### Phase 4: Pipeline Configuration

**Text Generation Pipeline:**

```python
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id
)
```

**Parameter Tuning:**
- **max_new_tokens**: 256 (limit response length)
- **temperature**: 0.7 (moderate creativity, not too random)
- **top_p**: 0.9 (nucleus sampling - keep top 90% probability mass)
- **repetition_penalty**: 1.1 (mild penalty to avoid repetition)
- **do_sample**: True (stochastic sampling, not greedy)
- **return_full_text**: False (return only new tokens, not prompt)

**LangChain Wrapper:**
```python
llm = HuggingFacePipeline(pipeline=pipe)
```

---

### Phase 5: Custom Prompt Template

```python
template = """### Instruction:
Analyze this philosophical concept using the provided context.
If unsure, state "I don't have sufficient information."

### Context:
{context}

### Question:
{question}

### Response:
"""
```

**Design:**
- **Structure-based**: Uses markdown-style markers for clarity
- **Domain-specific**: Targets philosophical analysis
- **Graceful uncertainty**: Instructs model to admit when unsure
- **Variables**: {context} injected from retriever, {question} from user

**Prompt Template Setup:**
```python
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
    template_format="f-string"
)
```

---

### Phase 6: RetrievalQA Chain

```python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "score_threshold": 0.4}
    ),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt,
        "document_prompt": PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}"
        )
    },
    verbose=True
)
```

**Configuration:**
- **chain_type**: "stuff" (concatenate all retrieved docs into single prompt)
- **retriever**: FAISS with cosine similarity
- **k=5**: Retrieve top 5 matching ideas
- **score_threshold=0.4**: Minimum relevance score
- **return_source_documents**: Enable source attribution
- **verbose=True**: Log intermediate steps

**Execution Flow:**
```
User Query
    ↓
[FAISS Retriever] → Top 5 relevant ideas (k=5)
    ↓
[Prompt Template] → Combine context + question
    ↓
[LLM Pipeline] → Generate response (max 256 tokens)
    ↓
[Response Processing] → Extract answer + source docs
    ↓
[Display] → Answer + top 3 sources with scores
```

---

### Phase 7: Interactive Chat Interface

```python
def run_chat():
    print("Chatbot initialized. Type 'exit' to quit.")
    while True:
        try:
            query = input("\nUser: ").strip()
            if query.lower() in ["exit", "quit"]:
                break

            if not query:
                print("Please enter a valid question")
                continue

            result = qa_chain({"query": query})

            # Extract response
            response = result['result'].split("### Assistant Response:")[-1].strip()
            print(f"\nAssistant: {response}")

            # Display top sources with metadata
            print("\nTop Sources:")
            for i, doc in enumerate(result['source_documents'][:3], 1):
                excerpt = doc.page_content[:150].replace("\n", " ") + "..."
                score = doc.metadata.get('score', 0)
                print(f"{i}. {excerpt} (Score: {score:.2f})")

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break
        except Exception as e:
            print(f"Error processing request: {str(e)}")
```

**Features:**
- **REPL loop**: Continuous Q&A interaction
- **Input validation**: Rejects empty queries
- **Graceful exit**: Handles "exit", "quit", Ctrl+C
- **Source attribution**: Shows top 3 matching ideas with relevance scores
- **Error resilience**: Catches exceptions and continues
- **Response parsing**: Extracts LLM output from structured prompt

---

## Iterative Development Notes

### Evolution Across Cells

**Cell 0 (Initial Attempt - DeepSeek):**
- Model: DeepSeek 1.5B with 4-bit quantization
- Status: Works but "responses not great"
- Note: High memory overhead, complex quantization setup

**Cell 1 (Simplified - GPT-Neo):**
- Model: GPT-Neo 125M (full precision)
- Status: "works responses not great - better rag needed"
- Improvement: Lightweight, no quantization complexity
- Insight: Quality issue traced to RAG retrieval, not just LLM

**Cell 2 (Archive - Duplicate GPT-Neo):**
- Annotation: "old v - pop up"
- Content: Identical to Cell 1
- Purpose: Version control/backup

### Development Insight

The progression indicates the developer recognized **retrieval quality** as the bottleneck, not just model size. The notebook comments suggest:
1. Initial DeepSeek approach was resource-heavy
2. Downsizing to GPT-Neo didn't help much
3. Root cause: FAISS retrieval returning low-relevance contexts
4. Future improvement: Better embedding model (higher discriminative power)

---

## Technical Stack

| Component | Implementation | Details |
|-----------|---|---|
| **Embeddings** | RoBERTa-base | Pre-computed, 768D, from CSV |
| **Vector DB** | FAISS | CPU-based, L2-normalized cosine similarity |
| **Retrieval** | LangChain Retriever | Top-k=5, score threshold=0.4 |
| **LLM (v1)** | DeepSeek 1.5B | 4-bit quantized, chat-optimized |
| **LLM (v2)** | GPT-Neo 125M | Full precision, ultra-lightweight |
| **Pipeline** | HuggingFace Transformers | Text-generation task with custom params |
| **Orchestration** | LangChain | RetrievalQA chain, custom prompts |
| **Interface** | CLI (REPL) | Interactive chat with source display |

---

## Known Limitations & Issues

### 1. Embedding Quality Problem
- FAISS searches often return poor matches (relates to CSV embedding analysis)
- High inter-vector similarity (0.98+) reduces discriminative power
- Score threshold=0.4 likely too lenient, allows weak matches

### 2. LLM Response Quality
- Both DeepSeek (1.5B) and GPT-Neo (125M) produce weak answers
- Small models struggle with philosophical reasoning
- Likely underfitting to domain-specific concepts

### 3. Prompt Parsing Fragility
```python
response = result['result'].split("### Assistant Response:")[-1].strip()
```
- Assumes marker appears in output (may not if LLM ignores template)
- Could fail if model generates different format
- No fallback logic

### 4. Missing Features
- No conversation memory (each query independent)
- No document filtering by quality/relevance
- No hybrid search (semantic + keyword)
- No caching of retriever results

---

## Recommended Improvements

### 1. Fix Embedding Quality
```python
# Normalize embeddings before FAISS
from sklearn.preprocessing import normalize
embeddings = normalize(embeddings, norm='l2', axis=1)
```

### 2. Upgrade Embedding Model
```python
# Use better discriminative model
HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # Better semantic understanding
    model_kwargs={'device': 'cuda'}  # GPU acceleration
)
```

### 3. Upgrade LLM
```python
# Better local model or API-backed
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# or use OpenAI/Anthropic API for production
```

### 4. Robustify Response Parsing
```python
response = result.get('result', '')
# More defensive parsing without string split
if isinstance(response, str):
    response = response.strip()
```

### 5. Add Retrieval Debugging
```python
# Log retrieval quality
for doc in result['source_documents']:
    print(f"Score: {doc.metadata.get('score', 0):.4f}")
    # Filter out low-quality matches
```

---

## Configuration Summary

### Deployment Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Embedding Model | RoBERTa-base | Standard transformer baseline |
| Embedding Dim | 768 | RoBERTa native output size |
| Vector Store | FAISS | Fast approximate/exact search |
| Normalization | L2 (FAISS) | Enables cosine similarity |
| Retrieval k | 5 | Balance precision/coverage |
| Score Threshold | 0.4 | Permissive (may need raising) |
| Max tokens | 256 | Prevent verbose responses |
| Temperature | 0.7 | Moderate creativity |
| Top-p | 0.9 | Nucleus sampling stability |

---

## Status & Verdict

**Development Status**: Early prototype with known quality issues

**Working**: ✓
- FAISS indexing and retrieval
- LangChain pipeline assembly
- Chat interface and source display
- Both LLM backends load successfully

**Not Working Well**: ✗
- Response quality (too generic/weak)
- Retrieval relevance (high similarity scores despite poor matches)
- Domain understanding (philosophical analysis weak)

**Next Steps**:
1. Debug FAISS retrieval quality (check score distribution)
2. Regenerate embeddings with better model
3. Upgrade LLM to 7B+ parameter model
4. Add retrieval evaluation metrics
5. Consider API-backed LLM (Claude, GPT-4) for better reasoning
