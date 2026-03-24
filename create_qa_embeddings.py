#!/usr/bin/env python3
"""
Embedding Generator for 151qa2.csv
Uses nomic-embed-text:latest
"""

import csv
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import requests
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CSV_FILE = "151qa2.csv"
OUTPUT_PKL = "qa_embeddings.pkl"
OUTPUT_JSON = "qa_embeddings.json"
OLLAMA_ENDPOINT = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
BATCH_SIZE = 20
DELAY = 0.2


def verify_ollama() -> bool:
    """Check if Ollama is running"""
    try:
        resp = requests.get(f"{OLLAMA_ENDPOINT}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m['name'] for m in data.get('models', [])]
            logger.info(f"✓ Ollama is running with {len(models)} models")
            
            if any(EMBEDDING_MODEL in m for m in models):
                logger.info(f"✓ Model {EMBEDDING_MODEL} found")
                return True
            else:
                logger.error(f"✗ Model {EMBEDDING_MODEL} not found")
                return False
    except Exception as e:
        logger.error(f"✗ Ollama not responding: {e}")
    return False


def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding"""
    try:
        resp = requests.post(
            f"{OLLAMA_ENDPOINT}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json().get("embedding")
    except Exception as e:
        logger.warning(f"Embedding error: {e}")
    return None


def load_csv(file_path: str) -> List[Dict]:
    """Load CSV file"""
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        return []
    
    try:
        data = []
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


def process_embeddings(data: List[Dict]) -> List[Dict]:
    """Create embeddings"""
    results = []
    total = len(data)
    
    for idx, item in enumerate(data, 1):
        texts = [str(v) for v in item.values() if v]
        text = " ".join(texts)[:1000]
        
        embedding = get_embedding(text)
        
        result = {
            "id": idx,
            "data": item,
            "embedding": embedding,
            "text_sample": text[:100]
        }
        results.append(result)
        
        if idx % BATCH_SIZE == 0:
            logger.info(f"Processed {idx}/{total} items ({100*idx//total}%)")
        
        time.sleep(DELAY)
    
    logger.info(f"✓ Embedded {len(results)} items")
    return results


def save_outputs(data: List[Dict], pkl_file: str, json_file: str):
    """Save results"""
    # Pickle
    try:
        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)
        size = Path(pkl_file).stat().st_size / (1024*1024)
        logger.info(f"✓ Saved {pkl_file} ({size:.2f} MB)")
    except Exception as e:
        logger.error(f"Error saving pickle: {e}")
    
    # JSON
    try:
        json_data = [
            {
                "id": item['id'],
                "properties": item['data'],
                "vector": item['embedding'],
                "text_preview": item['text_sample']
            }
            for item in data
        ]
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        size = Path(json_file).stat().st_size / (1024*1024)
        logger.info(f"✓ Saved {json_file} ({size:.2f} MB)")
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")


def main():
    logger.info("=" * 60)
    logger.info("Embedding Generator (Q&A CSV)")
    logger.info("=" * 60)
    
    if not verify_ollama():
        logger.error("\n⚠️  Setup incomplete. Exiting.")
        return
    
    data = load_csv(CSV_FILE)
    if not data:
        logger.error("No data loaded.")
        return
    
    logger.info(f"\nCreating embeddings for {len(data)} Q&A items...")
    logger.info(f"Model: {EMBEDDING_MODEL}\n")
    
    results = process_embeddings(data)
    
    logger.info("\nSaving results...")
    save_outputs(results, OUTPUT_PKL, OUTPUT_JSON)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Complete!")
    logger.info("=" * 60)
    logger.info(f"Items processed: {len(results)}")
    logger.info(f"\nOutputs:")
    logger.info(f"  - {OUTPUT_PKL}")
    logger.info(f"  - {OUTPUT_JSON} (for Weaviate)")


if __name__ == "__main__":
    main()
