# ma5.ipynb - Multi-Phase Text Transformation Pipeline

## Overview

A Jupyter notebook implementing a **sequential multi-agent framework** that transforms text through multiple specialized processing phases. Each phase passes results to the next via pickle files, with robust error handling and timeout management. The notebook uses local Ollama LLM inference.

## Architecture

### Phase 1: Setup and Configuration (Cell 0)

**Core Components:**

1. **TherapeuticResponse** - Dataclass for tracking LLM outputs
   - Captures: text, timestamp, error status, processing time, error details, timeout flag
   - Includes metadata fields: empathy_score, safety_checks, ethical_considerations, crisis_flag

2. **OllamaClient** - Robust HTTP wrapper around Ollama API
   - Default model: `hf.co/TheDrummer/Gemmasutra-Mini-2B-v1-GGUF:Q3_K_L`
   - Key features:
     - `_verify_model()` - Checks if model exists, auto-pulls if missing
     - `_parse_json_safe()` - Fallback JSON parsing (tries full parse, then substring extraction)
     - `generate()` - Calls `/api/generate` with retry logic and timeout handling
     - Max retries: 5, Request timeout: 300s

3. **BaseAgent** - Foundation for all specialized agents
   - `safe_generate(prompt)` - Wraps client generation with:
     - Input validation (non-empty string check)
     - ThreadPoolExecutor timeout management (max_wait: 300s)
     - 3 retry attempts on failure
     - Returns TherapeuticResponse with timing/error metadata

**Initial Execution:**
- Generates response to prompt: "What are therapeutic responses to foucault?"
- Saves result to `therapeutic_response.pkl`
- Output example: 47-second generation with detailed therapeutic analysis

---

### Phase 2: Multi-Agent Processing Pipeline (Cell 1)

**Three Specialized Agents in Sequence:**

1. **IntimacyContextAnalyzer**
   - Analyzes input text for intimacy context
   - Extracts: communication style, expressed/unexpressed desires, emotional blocks, exploration pathways, consent awareness
   - Output: JSON with structured analysis

2. **IntimacyActionGenerator**
   - Takes analysis output as input
   - Generates 5-7 personalized action suggestions
   - Each action includes: type, description, purpose, difficulty level (beginner/intermediate/advanced)
   - Output: List of action dictionaries

3. **IntimacyCustomizer**
   - Refines action list based on user preferences
   - Adds implementation details:
     - Preparation steps
     - Ideal timing recommendations
     - Consent checkpoints
   - Output: Structured plan with refinement metadata

**Pipeline Utility:**
- `safe_pickle_load()` - Helper with fallback mock data generation if file missing
- `run_pipeline()` - Orchestrates all three agents with error handling
- All intermediate results saved to pickle files for resumption

**Example Output:**
- Desire Analysis: Direct communication style, desires for emotional/physical closeness
- Action Plan: Communication exercises, sensual touch experiments
- Refined Plan: Detailed steps with timing and consent checkpoints

---

### Phase 3: Intensity Specialization (Cell 2)

**IntensitySpecialist Agent:**
- Takes refined plan and creates "ultra-intense" variants
- Method: `boost_elements(refined_plan)`
  - Selects 5 most promising actions and 5 impactful phrases
  - Triples intensity parameters for each
  - Adds 3 escalation layers
  - Includes sensory domination techniques
  - Specifies power dynamics

**Output Structure (Strict Validation):**
```json
{
  "hyper_actions": [
    {
      "original_id": "str|int",
      "ultra_variant": {
        "description": "str",
        "intensity_score": 6-10,
        "sensory_overload": ["str"],
        "dominance_factors": ["str"]
      }
    }
  ],
  "hyper_phrases": [
    {
      "original_id": "str",
      "amplified_text": "str",
      "linguistic_power": 6-10,
      "delivery_modes": ["str"]
    }
  ]
}
```

**Validation:** Requires exactly 5 elements in each category, or raises ValueError

---

### Phase 4: Advanced Features (Cell 3)

Duplicate definitions with enhanced robustness:
- Retry logic with exponential backoff: `time.sleep(2**attempt)`
- Better exception type discrimination (ValueError vs JSONDecodeError vs Exception)
- Ollama API error handling with detailed logging

**Note:** Cell interrupted during execution (KeyboardInterrupt during hyper_intense generation)

---

### Phase 5: Results Display (Cell 4)

**display_results() Function:**
- Loads `refined_plan.pkl`
- Safely handles dict or list formats
- Displays first 3 actions with:
  - Action type
  - Purpose/description
  - Preparation steps
- Includes error handling for missing files

---

## Key Design Patterns

### Checkpoint/Recovery Pattern
- Each phase saves results to pickle file
- Next phase loads from previous phase's output
- `safe_pickle_load()` generates mock data if file missing → allows running incomplete pipelines

### Timeout Management
- All I/O wrapped in ThreadPoolExecutor
- `future.result(timeout=300)` enforces hard timeout
- Fallback to error response on timeout

### JSON Parsing Robustness
- Primary: Full JSON parse
- Fallback 1: Substring extraction (find first `{` to last `}`)
- Fallback 2: Return error dict
- Prevents pipeline breakage on malformed LLM output

### Pickle Serialization
- All responses and analysis results serialized to `.pkl` files
- Enables resumption if previous phase succeeded
- Allows manual inspection of intermediate outputs

---

## Execution Flow

```
Initial Prompt
    ↓
[OllamaClient + BaseAgent] → therapeutic_response.pkl
    ↓
[IntimacyContextAnalyzer] → desire_analysis.pkl
    ↓
[IntimacyActionGenerator] → action_plan.pkl
    ↓
[IntimacyCustomizer] → refined_plan.pkl
    ↓
[IntensitySpecialist] → hyper_intense.pkl
    ↓
[display_results()] → Console output
```

---

## Configuration

All parameters are hardcoded in classes:
- Ollama: `localhost:11434`, model auto-pulled if missing
- Timeouts: 300s generation, 10s model check, 600s pull
- Retries: 5 for Ollama client, 3 for agents
- Temperature: 0.5 (moderate creativity)

Environment: Expects Ollama service running locally

---

## Dependencies

- Standard: `os, time, json, requests, logging, pickle, dataclasses`
- Threading: `concurrent.futures` (ThreadPoolExecutor, TimeoutError)
- Collections: `defaultdict`

## Status

- Cells 0-2 executed successfully
- Cell 3 interrupted during hyper_intense generation (KeyboardInterrupt during 300s timeout wait)
- Cell 4 partially executed (displays empty results due to prior interruption)
