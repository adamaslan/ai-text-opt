#!/usr/bin/env python3
"""
Embed Grief & Reflection Text
Splits thoughts_on_grief.txt into semantic chunks and creates embeddings
"""

import re
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
TEXT_FILE = "thoughts_on_grief.txt"
OUTPUT_PKL = "grief_thoughts_embeddings.pkl"
OUTPUT_JSON = "grief_thoughts_embeddings.json"
OLLAMA_ENDPOINT = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
BATCH_SIZE = 10
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


def load_text(file_path: str) -> str:
    """Load text file"""
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        return ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"✓ Loaded {len(text)} characters from {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return ""


def chunk_by_ideas(text: str) -> List[Dict]:
    """Split text by numbered ideas (1 - idea text)"""
    chunks = []
    
    # Split by numbered items
    pattern = r'(\d+)\s*[-–]\s*(.+?)(?=\d+\s*[-–]|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for idea_num, idea_text in matches:
        chunk = {
            "idea_number": int(idea_num),
            "title": f"Idea {idea_num}: Grief & Reflection",
            "content": idea_text.strip()[:1000],  # Limit length
            "full_content": idea_text.strip()
        }
        chunks.append(chunk)
    
    logger.info(f"✓ Split into {len(chunks)} idea chunks")
    return chunks


def process_embeddings(chunks: List[Dict]) -> List[Dict]:
    """Create embeddings for chunks"""
    results = []
    total = len(chunks)
    
    for idx, chunk in enumerate(chunks, 1):
        # Use content for embedding
        text = chunk['content'] or chunk['full_content']
        
        embedding = get_embedding(text)
        
        result = {
            "id": chunk['idea_number'],
            "idea_number": chunk['idea_number'],
            "title": chunk['title'],
            "content_preview": chunk['content'],
            "full_content": chunk['full_content'],
            "embedding": embedding
        }
        results.append(result)
        
        if idx % BATCH_SIZE == 0:
            logger.info(f"Processed {idx}/{total} ideas ({100*idx//total}%)")
        
        time.sleep(DELAY)
    
    logger.info(f"✓ Embedded {len(results)} ideas")
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
                "idea_number": item['idea_number'],
                "title": item['title'],
                "content": item['content_preview'],
                "full_content": item['full_content'],
                "vector": item['embedding']
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
    logger.info("Grief & Reflection Text Embeddings")
    logger.info("=" * 60)
    
    if not verify_ollama():
        logger.error("\n⚠️  Setup incomplete. Exiting.")
        return
    
    # Load text
    text = load_text(TEXT_FILE)
    if not text:
        logger.error("No text loaded.")
        return
    
    # Chunk by ideas
    chunks = chunk_by_ideas(text)
    if not chunks:
        logger.error("No chunks created.")
        return
    
    logger.info(f"\nCreating embeddings for {len(chunks)} ideas...")
    logger.info(f"Model: {EMBEDDING_MODEL}\n")
    
    results = process_embeddings(chunks)
    
    logger.info("\nSaving results...")
    save_outputs(results, OUTPUT_PKL, OUTPUT_JSON)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Complete!")
    logger.info("=" * 60)
    logger.info(f"Ideas embedded: {len(results)}")
    logger.info(f"\nOutputs:")
    logger.info(f"  - {OUTPUT_PKL}")
    logger.info(f"  - {OUTPUT_JSON} (for Weaviate)")
    logger.info(f"\nYou now have embeddings for:")
    for r in results[:5]:
        logger.info(f"  Idea {r['idea_number']}: {r['content_preview'][:60]}...")


if __name__ == "__main__":
    main()
