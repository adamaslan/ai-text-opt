#!/usr/bin/env python3
"""
Weaviate Migration
Move ChromaDB collections to Weaviate
Run this when Docker is available
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
    print("⚠️  weaviate-client not installed. Run: pip install -U 'weaviate-client'")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
WEAVIATE_URL = "http://localhost:8080"
MIGRATION_FILES = {
    "Ideas": "weaviate_ideas.json",
    "QA": "weaviate_qa.json",
    "GriefThoughts": "weaviate_grief.json"
}


class WeaviateMigration:
    """Migrates data from ChromaDB export to Weaviate"""
    
    def __init__(self, url: str = WEAVIATE_URL):
        self.url = url
        self.client = None
    
    def connect(self) -> bool:
        """Connect to Weaviate"""
        if not WEAVIATE_AVAILABLE:
            logger.error("weaviate-client not installed")
            return False
        
        try:
            self.client = weaviate.connect_to_local()
            logger.info("✓ Connected to Weaviate")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            logger.error("Make sure Docker is running: docker compose up -d")
            return False
    
    def load_migration_file(self, file_path: str) -> List[Dict]:
        """Load prepared Weaviate format JSON"""
        if not Path(file_path).exists():
            logger.warning(f"File not found: {file_path}")
            return []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"✓ Loaded {len(data)} items from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            return []
    
    def create_collection(self, collection_name: str) -> bool:
        """Create Weaviate collection"""
        try:
            if self.client.collections.exists(collection_name):
                self.client.collections.delete(collection_name)
                logger.info(f"Deleted existing {collection_name}")
            
            self.client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="text_content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="text_preview", data_type=DataType.TEXT),
                ]
            )
            logger.info(f"✓ Created {collection_name} collection")
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def import_data(self, collection_name: str, data: List[Dict]) -> int:
        """Import data to Weaviate"""
        collection = self.client.collections.get(collection_name)
        imported = 0
        
        try:
            with collection.batch.fixed_size(batch_size=50) as batch:
                for item in data:
                    properties = {
                        "title": item.get("id", ""),
                        "text_content": item.get("text_preview", ""),
                        "source": "chromadb_migration",
                        "text_preview": item.get("text_preview", "")[:200],
                    }
                    
                    vector = item.get("vector")
                    if vector:
                        batch.add_object(
                            properties=properties,
                            vector=vector
                        )
                        imported += 1
            
            logger.info(f"✓ Imported {imported} items to {collection_name}")
            return imported
        except Exception as e:
            logger.error(f"Import error: {e}")
            return imported
    
    def migrate_collection(self, collection_name: str, file_path: str) -> bool:
        """Full migration: create + import"""
        data = self.load_migration_file(file_path)
        if not data:
            return False
        
        if not self.create_collection(collection_name):
            return False
        
        imported = self.import_data(collection_name, data)
        return imported > 0


def main():
    logger.info("=" * 60)
    logger.info("Weaviate Migration from ChromaDB")
    logger.info("=" * 60)
    
    if not WEAVIATE_AVAILABLE:
        logger.error("\n⚠️  Install weaviate-client:")
        logger.error("  pip install -U 'weaviate-client'")
        return
    
    # Initialize migration
    migration = WeaviateMigration()
    
    if not migration.connect():
        logger.error("\n⚠️  Cannot connect to Weaviate")
        logger.error("Start Docker: docker compose up -d")
        return
    
    # Migrate collections
    logger.info("\nMigrating collections...")
    migrated = 0
    
    for collection_name, file_path in MIGRATION_FILES.items():
        if Path(file_path).exists():
            logger.info(f"\nMigrating {collection_name}...")
            if migration.migrate_collection(collection_name, file_path):
                migrated += 1
        else:
            logger.warning(f"Skipping {collection_name} - no migration file")
    
    # Summary
    logger.info("\n" + "=" * 60)
    if migrated > 0:
        logger.info("✓ Migration Complete!")
        logger.info("=" * 60)
        logger.info(f"Collections migrated: {migrated}")
        logger.info(f"Weaviate available at: {WEAVIATE_URL}")
        logger.info(f"\nQuery your data with:")
        logger.info(f"  http://localhost:8080/v1/graphql")
    else:
        logger.error("✗ No collections migrated")
    
    migration.client.close()


if __name__ == "__main__":
    main()
