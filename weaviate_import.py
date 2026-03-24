#!/usr/bin/env python3
"""
Import Embeddings to Weaviate
Run this after Docker is available
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

try:
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
JSON_INPUT = "ideas_with_vectors.json"
COLLECTION_NAME = "Ideas"
WEAVIATE_URL = "http://localhost:8080"


def load_embeddings(file_path: str) -> List[Dict]:
    """Load embeddings from JSON"""
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✓ Loaded {len(data)} embeddings from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        return []


def connect_weaviate() -> Optional:
    """Connect to Weaviate"""
    if not WEAVIATE_AVAILABLE:
        logger.error("weaviate-client not installed. Run: pip install -U 'weaviate-client'")
        return None
    
    try:
        client = weaviate.connect_to_local()
        logger.info("✓ Connected to Weaviate")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        logger.error("Make sure Docker is running: docker compose up -d")
        return None


def create_collection(client) -> bool:
    """Create collection for embeddings"""
    try:
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)
            logger.info(f"Deleted existing {COLLECTION_NAME}")
        
        ideas = client.collections.create(
            name=COLLECTION_NAME,
            properties=[
                Property(name="title", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="text_preview", data_type=DataType.TEXT),
            ],
            vector_index_config=Configure.VectorIndex.hnsw(),
        )
        logger.info(f"✓ Created {COLLECTION_NAME} collection")
        return True
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return False


def import_vectors(client, data: List[Dict]) -> int:
    """Import vectors and data to Weaviate"""
    ideas = client.collections.get(COLLECTION_NAME)
    imported = 0
    
    try:
        with ideas.batch.fixed_size(batch_size=50) as batch:
            for item in data:
                properties = {
                    "title": str(list(item["data"].values())[0])[:200] if item["data"] else "Untitled",
                    "content": " ".join([str(v)[:100] for v in item["data"].values()])[:500],
                    "text_preview": item.get("text_preview", "")[:200],
                }
                
                vector = item.get("embedding")
                if vector:
                    batch.add_object(
                        properties=properties,
                        vector=vector
                    )
                    imported += 1
        
        logger.info(f"✓ Imported {imported} items to Weaviate")
        return imported
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return imported


def main():
    """Main import pipeline"""
    logger.info("=" * 50)
    logger.info("Import Embeddings to Weaviate")
    logger.info("=" * 50)
    
    # Load embeddings
    data = load_embeddings(JSON_INPUT)
    if not data:
        logger.error("No embeddings loaded. Exiting.")
        return
    
    # Connect to Weaviate
    client = connect_weaviate()
    if not client:
        logger.error("Cannot connect to Weaviate. Exiting.")
        logger.error("Start services with: docker compose up -d")
        return
    
    try:
        # Create collection
        if not create_collection(client):
            return
        
        # Import data
        imported = import_vectors(client, data)
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("✓ Import Complete!")
        logger.info("=" * 50)
        logger.info(f"Items imported: {imported}")
        logger.info(f"Query Weaviate at: {WEAVIATE_URL}")
        
    finally:
        client.close()
        logger.info("Closed Weaviate connection")


if __name__ == "__main__":
    main()
