#!/usr/bin/env python3
"""
ChromaDB Setup - Local Semantic Search
Designed to be easily migrated to Weaviate later
"""

import json
import logging
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import chromadb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CHROMA_DB_PATH = "./chromadb_storage"
EMBEDDING_FILES = {
    "ideas": "ideas_embeddings.json",
    "qa_data": "qa_embeddings.json",
    "grief": "grief_thoughts_embeddings.json",
    "love_connection_ideas": "love_connection_ideas_embeddings.json",
    "love_connection": "love_connection_embeddings.json"
}


class ChromaDBSetup:
    """Manages ChromaDB with Weaviate migration in mind"""
    
    def __init__(self, db_path: str = CHROMA_DB_PATH):
        """Initialize ChromaDB client"""
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Persistent local storage
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
        )
        logger.info(f"✓ ChromaDB initialized at {self.db_path}")
    
    def load_json(self, file_path: str) -> List[Dict]:
        """Load embeddings from JSON"""
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"✓ Loaded {len(data)} items from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            return []
    
    def create_collection(self, collection_name: str, data: List[Dict]) -> bool:
        """Create collection with data"""
        try:
            # Delete if exists
            if collection_name in [c.name for c in self.client.list_collections()]:
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted existing {collection_name} collection")
            
            # Create collection
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Add data
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for item in data:
                item_id = str(item.get("id", item.get("idea_number", "")))
                
                # Get vector
                vector = item.get("vector") or item.get("embedding")
                if not vector:
                    logger.warning(f"Skipping {item_id} - no embedding")
                    continue
                
                embeddings.append(vector)
                ids.append(item_id)
                
                # Get text content
                text = item.get("content") or item.get("properties", {})
                if isinstance(text, dict):
                    text = " ".join([str(v) for v in text.values()])
                documents.append(str(text)[:2000])
                
                # Metadata (Weaviate-compatible schema)
                metadata = {
                    "id": item_id,
                    "title": item.get("title", ""),
                    "source": collection_name,
                    "text_preview": str(item.get("content_preview", ""))[:200]
                }
                metadatas.append(metadata)
            
            # Batch add
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"✓ Created {collection_name} collection with {len(ids)} items")
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def export_for_weaviate(self, collection_name: str, output_file: str) -> bool:
        """Export collection in Weaviate-compatible format"""
        try:
            collection = self.client.get_collection(collection_name)
            
            # Get all data
            all_data = collection.get(include=["metadatas", "documents", "embeddings"])
            
            # Convert to Weaviate format
            weaviate_data = []
            for idx, doc_id in enumerate(all_data['ids']):
                emb = all_data['embeddings'][idx]
                item = {
                    "id": doc_id,
                    "properties": all_data['metadatas'][idx],
                    "vector": emb.tolist() if hasattr(emb, 'tolist') else emb,
                    "text_preview": all_data['documents'][idx][:200]
                }
                weaviate_data.append(item)
            
            # Save
            with open(output_file, 'w') as f:
                json.dump(weaviate_data, f, indent=2)
            
            logger.info(f"✓ Exported {len(weaviate_data)} items to {output_file} (Weaviate format)")
            return True
        except Exception as e:
            logger.error(f"Error exporting: {e}")
            return False
    
    def quality_report(self, collection_name: str) -> Dict:
        """Assess embedding quality for a collection.

        Metrics:
        - zero_count: embeddings with near-zero norm (degenerate)
        - norm_mean / norm_std: distribution of embedding magnitudes
        - avg_pairwise_cosine: mean cosine similarity among a sample of pairs
          (higher = more clustered; ~0 = spread out; negative = spread/antipodal)
        - nn_distance_mean: mean distance to each item's nearest neighbour
          (lower = tighter clusters)
        """
        try:
            collection = self.client.get_collection(collection_name)
            all_data = collection.get(include=["embeddings"])
            raw = all_data["embeddings"]
            if raw is None or len(raw) == 0:
                logger.warning(f"No embeddings found in {collection_name}")
                return {}

            # Convert to plain Python lists
            vecs: List[List[float]] = [
                v.tolist() if hasattr(v, "tolist") else list(v) for v in raw
            ]
            n = len(vecs)
            dim = len(vecs[0])

            # --- norms ---
            def norm(v: List[float]) -> float:
                return math.sqrt(sum(x * x for x in v))

            norms = [norm(v) for v in vecs]
            zero_count = sum(1 for nrm in norms if nrm < 1e-6)
            norm_mean = sum(norms) / n
            norm_std = math.sqrt(sum((nrm - norm_mean) ** 2 for nrm in norms) / n)

            # --- cosine similarity helpers ---
            def cosine(a: List[float], b: List[float], na: float, nb: float) -> float:
                if na < 1e-9 or nb < 1e-9:
                    return 0.0
                dot = sum(x * y for x, y in zip(a, b))
                return dot / (na * nb)

            # Sample up to 200 items to keep runtime fast
            sample_size = min(n, 200)
            step = max(1, n // sample_size)
            sample_vecs = vecs[::step][:sample_size]
            sample_norms = norms[::step][:sample_size]
            m = len(sample_vecs)

            # Pairwise cosine over sample
            pair_sims: List[float] = []
            for i in range(m):
                for j in range(i + 1, m):
                    pair_sims.append(
                        cosine(sample_vecs[i], sample_vecs[j],
                               sample_norms[i], sample_norms[j])
                    )

            avg_pairwise_cosine = sum(pair_sims) / len(pair_sims) if pair_sims else 0.0

            # Nearest-neighbour distance (cosine distance = 1 - cosine_sim)
            nn_distances: List[float] = []
            for i in range(m):
                best = 2.0  # max cosine distance
                for j in range(m):
                    if i == j:
                        continue
                    sim = cosine(sample_vecs[i], sample_vecs[j],
                                 sample_norms[i], sample_norms[j])
                    dist = 1.0 - sim
                    if dist < best:
                        best = dist
                nn_distances.append(best)

            nn_distance_mean = sum(nn_distances) / len(nn_distances) if nn_distances else 0.0

            report = {
                "collection": collection_name,
                "total_embeddings": n,
                "dimension": dim,
                "zero_embeddings": zero_count,
                "norm_mean": round(norm_mean, 4),
                "norm_std": round(norm_std, 4),
                "avg_pairwise_cosine": round(avg_pairwise_cosine, 4),
                "nn_distance_mean": round(nn_distance_mean, 4),
            }

            logger.info(f"\n--- Quality Report: {collection_name} ---")
            logger.info(f"  Items         : {n}  |  Dim: {dim}")
            logger.info(f"  Zero embeds   : {zero_count}")
            logger.info(f"  Norm mean±std : {norm_mean:.4f} ± {norm_std:.4f}")
            logger.info(f"  Avg pairwise cosine sim : {avg_pairwise_cosine:.4f}  "
                        f"(0=spread, 1=collapsed)")
            logger.info(f"  Avg NN cosine distance  : {nn_distance_mean:.4f}  "
                        f"(lower=tighter clusters)")
            return report

        except Exception as e:
            logger.error(f"Quality report error for {collection_name}: {e}")
            return {}

    def search(self, collection_name: str, query: str, limit: int = 5,
               query_embedding: Optional[List[float]] = None) -> List[Dict]:
        """Semantic search. Provide query_embedding (768-dim) for vector search,
        or omit to do a text search using ChromaDB's default embedder."""
        try:
            collection = self.client.get_collection(collection_name)
            if query_embedding is not None:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                results = collection.query(
                    query_texts=[query],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"]
                )
            
            # Format results
            output = []
            for i in range(len(results['ids'][0])):
                result = {
                    "id": results['ids'][0][i],
                    "distance": results['distances'][0][i],
                    "text": results['documents'][0][i][:200],
                    "metadata": results['metadatas'][0][i]
                }
                output.append(result)
            
            return output
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []


def main():
    logger.info("=" * 60)
    logger.info("ChromaDB Setup with Weaviate Migration")
    logger.info("=" * 60)
    
    # Initialize
    chroma = ChromaDBSetup()
    
    # Import datasets
    logger.info("\nImporting datasets...")
    for collection_name, file_path in EMBEDDING_FILES.items():
        data = chroma.load_json(file_path)
        if data:
            chroma.create_collection(collection_name, data)
        else:
            logger.warning(f"Skipping {collection_name} - no data")
    
    # Quality reports
    logger.info("\n" + "=" * 60)
    logger.info("Embedding Quality Reports")
    logger.info("=" * 60)
    for collection_name in EMBEDDING_FILES.keys():
        chroma.quality_report(collection_name)

    # Export for Weaviate migration
    logger.info("\nPreparing migration files...")
    for collection_name in EMBEDDING_FILES.keys():
        output_file = f"weaviate_{collection_name}.json"
        chroma.export_for_weaviate(collection_name, output_file)
    
    # Example searches
    logger.info("\n" + "=" * 60)
    logger.info("Example Searches")
    logger.info("=" * 60)
    
    # Load sample embeddings to use as query vectors (same 768-dim space)
    sample_embeddings = {}
    for collection_name, file_path in EMBEDDING_FILES.items():
        data = chroma.load_json(file_path)
        if data:
            vec = data[0].get("vector") or data[0].get("embedding")
            if vec:
                sample_embeddings[collection_name] = vec

    example_queries = [
        ("ideas", "death and grief"),
        ("qa_data", "communication technology"),
        ("grief", "compassion kindness"),
    ]

    for collection, query in example_queries:
        logger.info(f"\nQuery: '{query}' in {collection}")
        query_emb = sample_embeddings.get(collection)
        results = chroma.search(collection, query, limit=2, query_embedding=query_emb)
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. (distance: {result['distance']:.4f})")
            logger.info(f"     {result['text']}...")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Setup Complete!")
    logger.info("=" * 60)
    logger.info(f"Database: {CHROMA_DB_PATH}/")
    logger.info(f"Collections: {list(EMBEDDING_FILES.keys())}")
    logger.info(f"\nMigration files ready:")
    for collection in EMBEDDING_FILES.keys():
        logger.info(f"  - weaviate_{collection}.json")
    logger.info(f"\nWhen ready to migrate to Weaviate:")
    logger.info(f"  1. docker compose up -d")
    logger.info(f"  2. python weaviate_migrate.py")


if __name__ == "__main__":
    main()
