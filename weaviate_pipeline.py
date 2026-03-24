#!/usr/bin/env python3
"""
Minimal Weaviate + Ollama Pipeline
Reads CSV data, creates Weaviate collection, vectorizes with Ollama
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional

import weaviate
from weaviate.classes.config import Configure

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CSV_FILE = "151_ideas_updated2.csv"
COLLECTION_NAME = "Ideas"
OLLAMA_ENDPOINT = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"  # Specialized embedding model
GENERATIVE_MODEL = "dolphin-phi:2.7b"  # For RAG (optional)
WEAVIATE_URL = "http://localhost:8080"


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
        logger.info(f"Loaded {len(data)} rows from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return []


def connect_weaviate() -> Optional[weaviate.Client]:
    """Connect to local Weaviate instance"""
    try:
        client = weaviate.connect_to_local()
        logger.info("Connected to Weaviate")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        logger.info("Make sure Weaviate is running: docker-compose up -d")
        return None


def create_collection(client: weaviate.Client) -> bool:
    """Create Ideas collection with Ollama vectorization"""
    try:
        # Delete if exists
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)
            logger.info(f"Deleted existing {COLLECTION_NAME} collection")
        
        # Create collection with Ollama embeddings
        ideas = client.collections.create(
            name=COLLECTION_NAME,
            vector_config=Configure.Vectors.text2vec_ollama(
                api_endpoint=OLLAMA_ENDPOINT,
                model=EMBEDDING_MODEL,
            ),
        )
        logger.info(f"Created {COLLECTION_NAME} collection with Ollama embeddings")
        return True
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return False


def import_data(client: weaviate.Client, data: List[Dict]) -> int:
    """Import CSV data into Weaviate collection"""
    ideas = client.collections.get(COLLECTION_NAME)
    imported_count = 0
    
    try:
        with ideas.batch.fixed_size(batch_size=50) as batch:
            for item in data:
                # Use first two columns as properties
                properties = {}
                for key, value in item.items():
                    if key and value:  # Skip empty fields
                        properties[key] = str(value)[:500]  # Limit field length
                
                if properties:
                    batch.add_object(properties=properties)
                    imported_count += 1
        
        logger.info(f"Imported {imported_count} objects into Weaviate")
        return imported_count
    except Exception as e:
        logger.error(f"Failed to import data: {e}")
        return imported_count


def semantic_search(client: weaviate.Client, query: str, limit: int = 5) -> List[Dict]:
    """Perform semantic search on imported data"""
    ideas = client.collections.get(COLLECTION_NAME)
    results = []
    
    try:
        response = ideas.query.near_text(
            query=query,
            limit=limit,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
        )
        
        for obj in response.objects:
            result = {
                "distance": obj.metadata.distance,
                "properties": obj.properties
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} results for query: {query}")
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def main():
    """Main pipeline"""
    logger.info("Starting Weaviate + Ollama Pipeline")
    
    # Load CSV
    data = load_csv_data(CSV_FILE)
    if not data:
        logger.error("No data loaded. Exiting.")
        return
    
    # Connect to Weaviate
    client = connect_weaviate()
    if not client:
        logger.error("Cannot connect to Weaviate. Exiting.")
        return
    
    try:
        # Create collection
        if not create_collection(client):
            logger.error("Failed to create collection. Exiting.")
            return
        
        # Import data
        imported = import_data(client, data)
        if imported == 0:
            logger.error("No data imported. Exiting.")
            return
        
        # Example searches
        queries = [
            "How to improve relationships?",
            "Communication tips",
            "Personal growth strategies",
        ]
        
        logger.info("\n=== Semantic Search Results ===")
        for query in queries:
            logger.info(f"\nQuery: '{query}'")
            results = semantic_search(client, query, limit=3)
            for i, result in enumerate(results, 1):
                logger.info(f"  Result {i} (distance: {result['distance']:.4f})")
                # Print first property
                for key, value in list(result['properties'].items())[:1]:
                    logger.info(f"    {key}: {str(value)[:100]}...")
        
        logger.info("\n✅ Pipeline completed successfully!")
        
    finally:
        client.close()
        logger.info("Closed Weaviate connection")


if __name__ == "__main__":
    main()
