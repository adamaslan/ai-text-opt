#!/usr/bin/env python3
"""
Embed Love & Connection Ideas
Splits love_and_connection_ideas.txt into semantic sections and creates embeddings
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
TEXT_FILE = "love_and_connection.txt"
OUTPUT_PKL = "love_connection_embeddings.pkl"
OUTPUT_JSON = "love_connection_embeddings.json"
OLLAMA_ENDPOINT = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
BATCH_SIZE = 10
DELAY = 0.2


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
        logger.info(f"✓ Loaded {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return ""


def chunk_by_themes(text: str) -> List[Dict]:
    """Split text into thematic chunks"""
    chunks = []
    
    # Split by major themes/sections marked by headers or double newlines
    sections = re.split(r'\n\n+', text)
    
    for idx, section in enumerate(sections, 1):
        section = section.strip()
        if len(section) < 50:  # Skip very short sections
            continue
        
        # Extract theme/title from first line or create one
        lines = section.split('\n')
        title = lines[0] if lines[0] else f"Section {idx}"
        
        chunk = {
            "section_number": idx,
            "title": title[:100],
            "content": section[:1000],
            "full_content": section
        }
        chunks.append(chunk)
    
    logger.info(f"✓ Split into {len(chunks)} thematic sections")
    return chunks


def process_embeddings(chunks: List[Dict]) -> List[Dict]:
    """Create embeddings"""
    results = []
    total = len(chunks)
    
    for idx, chunk in enumerate(chunks, 1):
        text = chunk['content'] or chunk['full_content']
        embedding = get_embedding(text)
        
        result = {
            "id": chunk['section_number'],
            "section_number": chunk['section_number'],
            "title": chunk['title'],
            "content_preview": chunk['content'],
            "full_content": chunk['full_content'],
            "embedding": embedding
        }
        results.append(result)
        
        if idx % BATCH_SIZE == 0:
            logger.info(f"Processed {idx}/{total} sections ({100*idx//total}%)")
        
        time.sleep(DELAY)
    
    logger.info(f"✓ Embedded {len(results)} sections")
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
                "section_number": item['section_number'],
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
    logger.info("Love & Connection Text Embeddings")
    logger.info("=" * 60)
    
    if not verify_ollama():
        logger.error("\n⚠️  Ollama not running")
        return
    
    text = load_text(TEXT_FILE)
    if not text:
        return
    
    chunks = chunk_by_themes(text)
    if not chunks:
        logger.error("No chunks created")
        return
    
    logger.info(f"\nCreating embeddings for {len(chunks)} sections...")
    logger.info(f"Model: {EMBEDDING_MODEL}\n")
    
    results = process_embeddings(chunks)
    
    logger.info("\nSaving results...")
    save_outputs(results, OUTPUT_PKL, OUTPUT_JSON)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Complete!")
    logger.info("=" * 60)
    logger.info(f"Sections embedded: {len(results)}")
    logger.info(f"\nOutputs:")
    logger.info(f"  - {OUTPUT_PKL}")
    logger.info(f"  - {OUTPUT_JSON} (for Weaviate/ChromaDB)")


if __name__ == "__main__":
    main()
