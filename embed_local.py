#!/usr/bin/env python3
"""
Minimal Ollama Embedding Pipeline (No Docker Required)
Processes CSV, creates embeddings with local Ollama, stores results
Ready to integrate with Weaviate later
"""

import csv
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import requests
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CSV_FILE = "151_ideas_updated2.csv"
OUTPUT_PKL = "ideas_with_vectors.pkl"
OUTPUT_JSON = "ideas_with_vectors.json"
OLLAMA_ENDPOINT = "http://localhost:11434"
EMBEDDING_MODEL = "dolphin-phi:2.7b"


def verify_ollama() -> bool:
    """Check if Ollama is running"""
    try:
        resp = requests.get(f"{OLLAMA_ENDPOINT}/api/tags", timeout=5)
        if resp.status_code == 200:
            logger.info("✓ Ollama is running")
            return True
    except Exception as e:
        logger.error(f"✗ Ollama not responding: {e}")
    return False


def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from Ollama"""
    try:
        resp = requests.post(
            f"{OLLAMA_ENDPOINT}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text[:2000]},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json().get("embedding")
    except Exception as e:
        logger.warning(f"Embedding error for '{text[:50]}...': {e}")
    return None


def load_csv_data(file_path: str) -> List[Dict]:
    """Load CSV file and return list of dictionaries"""
    if not Path(file_path).exists():
        logger.error(f"CSV file not found: {file_path}")
        return []
    
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row:
                    data.append(row)
        logger.info(f"✓ Loaded {len(data)} rows from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return []


def process_and_embed(data: List[Dict]) -> List[Dict]:
    """Process CSV data and add embeddings"""
    processed = []
    total = len(data)
    
    for idx, item in enumerate(data, 1):
        # Create text to embed (combine key fields)
        text_parts = [str(v) for v in item.values() if v]
        text = " ".join(text_parts[:500])  # Limit length
        
        # Get embedding
        embedding = get_embedding(text)
        
        # Store result
        result = {
            "original_data": item,
            "embedding": embedding,
            "embedding_text": text[:200]
        }
        processed.append(result)
        
        # Progress
        if idx % 10 == 0:
            logger.info(f"Processed {idx}/{total} items")
            time.sleep(0.5)  # Avoid overwhelming Ollama
    
    logger.info(f"✓ Processed {len(processed)} items with embeddings")
    return processed


def save_results(data: List[Dict]) -> None:
    """Save results to pickle and JSON"""
    # Save as pickle
    try:
        with open(OUTPUT_PKL, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"✓ Saved to {OUTPUT_PKL}")
    except Exception as e:
        logger.error(f"Error saving pickle: {e}")
    
    # Save as JSON (for Weaviate)
    try:
        json_data = []
        for item in data:
            json_item = {
                "data": item["original_data"],
                "embedding": item["embedding"],
                "text_preview": item["embedding_text"]
            }
            json_data.append(json_item)
        
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved to {OUTPUT_JSON}")
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")


def main():
    """Main pipeline"""
    logger.info("=" * 50)
    logger.info("Ollama Embedding Pipeline (No Docker Required)")
    logger.info("=" * 50)
    
    # Check Ollama
    if not verify_ollama():
        logger.error("\n⚠️  Ollama is not running!")
        logger.error("Start with: ollama serve")
        return
    
    # Load CSV
    data = load_csv_data(CSV_FILE)
    if not data:
        logger.error("No data loaded. Exiting.")
        return
    
    # Process and embed
    logger.info(f"\nEmbedding {len(data)} items with {EMBEDDING_MODEL}...")
    processed = process_and_embed(data)
    
    # Save results
    save_results(processed)
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("✓ Pipeline Complete!")
    logger.info("=" * 50)
    logger.info(f"Items processed: {len(processed)}")
    logger.info(f"Pickle output: {OUTPUT_PKL}")
    logger.info(f"JSON output: {OUTPUT_JSON} (ready for Weaviate)")
    logger.info("\nWhen Docker is ready, import to Weaviate with:")
    logger.info("  docker compose up -d")
    logger.info("  python weaviate_import.py")


if __name__ == "__main__":
    main()
