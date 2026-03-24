#!/bin/bash

# AI Text Optimization - Embeddings Setup & Generation
# This script follows mamba-rules.md guidelines for environment management
# Usage: ./setup_embeddings.sh or source setup_embeddings.sh

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="ai-text-opt"
EMBEDDINGS_DIR="$PROJECT_DIR/embeddings"

echo "========================================================================"
echo "AI Text Optimization - Environment Setup & Embedding Generation"
echo "========================================================================"

# Create embeddings directory
mkdir -p "$EMBEDDINGS_DIR"
echo "✓ Embeddings directory: $EMBEDDINGS_DIR"

# Initialize mamba/conda
echo ""
echo "Initializing conda/mamba environment..."
if command -v mamba &> /dev/null; then
    PKG_MANAGER="mamba"
    echo "✓ Using mamba (preferred)"
elif command -v conda &> /dev/null; then
    PKG_MANAGER="conda"
    echo "✓ Using conda (fallback)"
else
    echo "✗ Error: Neither mamba nor conda found"
    echo "  Please install micromamba or conda-forge"
    exit 1
fi

# Initialize conda shell integration if needed
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc" 2>/dev/null || true
fi

# Create environment from environment.yml if it doesn't exist
if ! $PKG_MANAGER info --envs | grep -q "^$ENV_NAME "; then
    echo ""
    echo "Creating environment from environment.yml..."
    echo "This installs all packages in a single command for optimal stability."
    $PKG_MANAGER env create -f "$PROJECT_DIR/environment.yml" -y
    echo "✓ Environment created successfully"
else
    echo "✓ Environment '$ENV_NAME' exists, using existing"
fi

# Activate environment and run embedding generation
echo ""
echo "Activating environment: $ENV_NAME"
echo ""
echo "========================================================================"
echo "Setup complete! Running embedding generation for 151qa2.csv..."
echo "========================================================================"
echo ""

# Use mamba run to execute Python in the environment
$PKG_MANAGER run -n "$ENV_NAME" python3 << 'PYTHON_EOF'
import csv
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import time
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(PROJECT_DIR, '151qa2.csv')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'embeddings')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, '151qa2_with_embeddings.csv')

print("="*70)
print("EMBEDDING GENERATION FOR 151qa2.csv")
print("="*70)

# Load CSV
print(f"\nLoading: {CSV_PATH}")
if not os.path.exists(CSV_PATH):
    print(f"✗ Error: File not found: {CSV_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH)
texts = df['text'].tolist()
print(f"✓ Loaded {len(texts)} entries")

# Initialize embedding model
print("\nInitializing embedding model...")
print("Model: all-MiniLM-L6-v2")
print("  - Better semantic understanding than RoBERTa")
print("  - Lower dimensionality (384D) vs RoBERTa (768D)")
print("  - Optimized for retrieval tasks")

model = SentenceTransformer('all-MiniLM-L6-v2')
model_dim = model.get_sentence_embedding_dimension()
print(f"✓ Model dimension: {model_dim}D")

# Generate embeddings
print("\nGenerating embeddings (this may take a minute)...")
start_time = time.time()
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
elapsed = time.time() - start_time

print(f"✓ Completed in {elapsed:.2f} seconds")
print(f"✓ Shape: {embeddings.shape}")

# Create output dataframe
output_df = df.copy()
output_df['Embeddings'] = [list(emb) for emb in embeddings]

# Save to CSV
print(f"\nSaving: {OUTPUT_FILE}")
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"✓ Saved successfully")

# Generate statistics
print("\n" + "="*70)
print("EMBEDDING STATISTICS")
print("="*70)

# Calculate norms
import math
norms = [math.sqrt(np.sum(emb**2)) for emb in embeddings]

print(f"\nVector Norms (Magnitude):")
print(f"  Min:  {min(norms):.4f}")
print(f"  Max:  {max(norms):.4f}")
print(f"  Mean: {np.mean(norms):.4f}")
print(f"  Std:  {np.std(norms):.4f}")

# Check unit normalization
unit_count = sum(1 for n in norms if 0.95 < n < 1.05)
print(f"  Unit-normalized vectors: {unit_count}/{len(norms)}")

# Value distribution
all_values = embeddings.flatten()
print(f"\nValue Distribution:")
print(f"  Min:    {np.min(all_values):.6f}")
print(f"  Max:    {np.max(all_values):.6f}")
print(f"  Mean:   {np.mean(all_values):.6f}")
print(f"  Median: {np.median(all_values):.6f}")

# Similarity analysis
print(f"\nCosine Similarity (first 5 items):")
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = []
for i in range(min(5, len(embeddings)-1)):
    for j in range(i+1, min(5, len(embeddings))):
        sim = cosine_sim(embeddings[i], embeddings[j])
        similarities.append(sim)
        print(f"  Pair {i},{j}: {sim:.4f}")

if similarities:
    print(f"  Average: {np.mean(similarities):.4f}")

print("\n" + "="*70)
print(f"✓ Embeddings generated successfully!")
print(f"  Output: {OUTPUT_FILE}")
print(f"  Entries: {len(embeddings)}")
print(f"  Dimension: {model_dim}D")
print("="*70)

PYTHON_EOF

echo ""
echo "✓ Embedding generation complete!"
echo ""
echo "Next steps:"
echo "  1. Use embeddings with RAG: roberta-rag1.ipynb"
echo "  2. Compare embedding quality: docs/csv-files-comparison.md"
echo "  3. Run inference: mamba run -n $ENV_NAME python your_script.py"
echo ""
echo "To reactivate this environment later:"
echo "  mamba activate $ENV_NAME"
echo ""
