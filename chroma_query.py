#!/usr/bin/env python3
"""
ChromaDB Query Interface
Search across all collections and assess embedding quality
"""

import logging
import math
from pathlib import Path
from typing import List, Dict, Optional
import chromadb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHROMA_DB_PATH = "./chromadb_storage"


class ChromaQuery:
    """Query and quality-check interface for ChromaDB"""

    def __init__(self, db_path: str = CHROMA_DB_PATH):
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collections = {c.name: c for c in self.client.list_collections()}
        logger.info("✓ Connected to ChromaDB")
        logger.info(f"✓ Collections: {list(self.collections.keys())}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_all(self, query: str, limit: int = 5) -> Dict[str, List]:
        """Search across all collections using the provided query text."""
        all_results = {}
        for collection_name, collection in self.collections.items():
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"],
                )
                formatted = [
                    {
                        "id": results["ids"][0][i],
                        "distance": results["distances"][0][i],
                        "text": results["documents"][0][i][:150],
                        "source": collection_name,
                    }
                    for i in range(len(results["ids"][0]))
                ]
                all_results[collection_name] = formatted
            except Exception as e:
                logger.error(f"Error searching {collection_name}: {e}")
        return all_results

    def search_collection(self, collection_name: str, query: str, limit: int = 5,
                          query_embedding: Optional[List[float]] = None) -> List:
        """Search a specific collection.

        Pass query_embedding (same dim as stored vectors) for proper semantic search.
        Falls back to ChromaDB's built-in text embedder otherwise (may dim-mismatch).
        """
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                logger.error(f"Collection not found: {collection_name}")
                return []

            if query_embedding is not None:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"],
                )
            else:
                results = collection.query(
                    query_texts=[query],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"],
                )

            return [
                {
                    "id": results["ids"][0][i],
                    "distance": round(results["distances"][0][i], 4),
                    "text": results["documents"][0][i][:200],
                    "metadata": results["metadatas"][0][i],
                }
                for i in range(len(results["ids"][0]))
            ]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    # ------------------------------------------------------------------
    # Quality assessment
    # ------------------------------------------------------------------

    def quality_report(self, collection_name: str) -> Dict:
        """Assess embedding quality for one collection.

        Metrics
        -------
        zero_embeddings      Embeddings with near-zero norm (degenerate / failed).
        norm_mean / norm_std Distribution of vector magnitudes — high std may
                             indicate inconsistent generation.
        avg_pairwise_cosine  Mean cosine similarity across a random sample of pairs.
                             Near 0  → well-spread embeddings (good).
                             Near 1  → all embeddings collapsed to same region (bad).
        nn_distance_mean     Mean cosine distance to each item's nearest neighbour.
                             Lower   → tighter semantic clusters (good for retrieval).
                             Higher  → embeddings are isolated / not grouping.
        """
        try:
            collection = self.collections.get(collection_name)
            if not collection:
                logger.error(f"Collection not found: {collection_name}")
                return {}

            all_data = collection.get(include=["embeddings"])
            raw = all_data["embeddings"]
            if raw is None or len(raw) == 0:
                logger.warning(f"No embeddings in {collection_name}")
                return {}

            vecs: List[List[float]] = [
                v.tolist() if hasattr(v, "tolist") else list(v) for v in raw
            ]
            n = len(vecs)
            dim = len(vecs[0])

            def norm(v: List[float]) -> float:
                return math.sqrt(sum(x * x for x in v))

            def cosine(a: List[float], b: List[float], na: float, nb: float) -> float:
                if na < 1e-9 or nb < 1e-9:
                    return 0.0
                return sum(x * y for x, y in zip(a, b)) / (na * nb)

            norms = [norm(v) for v in vecs]
            zero_count = sum(1 for nrm in norms if nrm < 1e-6)
            norm_mean = sum(norms) / n
            norm_std = math.sqrt(sum((nrm - norm_mean) ** 2 for nrm in norms) / n)

            # Sample up to 200 items for pairwise metrics
            sample_size = min(n, 200)
            step = max(1, n // sample_size)
            sv = vecs[::step][:sample_size]
            sn = norms[::step][:sample_size]
            m = len(sv)

            pair_sims: List[float] = []
            for i in range(m):
                for j in range(i + 1, m):
                    pair_sims.append(cosine(sv[i], sv[j], sn[i], sn[j]))
            avg_pairwise = sum(pair_sims) / len(pair_sims) if pair_sims else 0.0

            nn_dists: List[float] = []
            for i in range(m):
                best = 2.0
                for j in range(m):
                    if i == j:
                        continue
                    d = 1.0 - cosine(sv[i], sv[j], sn[i], sn[j])
                    if d < best:
                        best = d
                nn_dists.append(best)
            nn_mean = sum(nn_dists) / len(nn_dists) if nn_dists else 0.0

            report = {
                "collection": collection_name,
                "total_embeddings": n,
                "dimension": dim,
                "zero_embeddings": zero_count,
                "norm_mean": round(norm_mean, 4),
                "norm_std": round(norm_std, 4),
                "avg_pairwise_cosine": round(avg_pairwise, 4),
                "nn_distance_mean": round(nn_mean, 4),
            }

            print(f"\n{'=' * 50}")
            print(f"  Quality Report: {collection_name}")
            print(f"{'=' * 50}")
            print(f"  Items              : {n}")
            print(f"  Dimension          : {dim}")
            print(f"  Zero embeddings    : {zero_count}  {'⚠ bad' if zero_count else '✓'}")
            print(f"  Norm  mean ± std   : {norm_mean:.4f} ± {norm_std:.4f}")
            print(f"  Avg pairwise sim   : {avg_pairwise:.4f}  "
                  f"{'(spread well)' if avg_pairwise < 0.7 else '(may be collapsed)'}")
            print(f"  Avg NN distance    : {nn_mean:.4f}  "
                  f"{'(tight clusters)' if nn_mean < 0.3 else '(loosely clustered)'}")
            return report

        except Exception as e:
            logger.error(f"Quality report error for {collection_name}: {e}")
            return {}

    def quality_report_all(self) -> Dict[str, Dict]:
        """Run quality_report on every collection and return all results."""
        return {name: self.quality_report(name) for name in self.collections}


# ------------------------------------------------------------------
# Interactive CLI
# ------------------------------------------------------------------

HELP = """
Commands:
  all <query>              Search all collections by query text
  <collection> <query>     Search a specific collection
  quality                  Run quality report on all collections
  quality <collection>     Run quality report on one collection
  collections              List available collections
  quit                     Exit
"""


def main():
    q = ChromaQuery()

    print("\n" + "=" * 60)
    print("ChromaDB Query Interface")
    print("=" * 60)
    print(f"Collections: {list(q.collections.keys())}")
    print(HELP)

    while True:
        try:
            user_input = input("Search> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            if user_input.lower() == "collections":
                print(f"Available: {list(q.collections.keys())}")
                continue
            if user_input.lower() == "quality":
                q.quality_report_all()
                continue
            if user_input.lower().startswith("quality "):
                coll = user_input.split(" ", 1)[1].strip()
                q.quality_report(coll)
                continue

            parts = user_input.split(" ", 1)
            if len(parts) < 2:
                print("Usage: <collection> <query>  or  all <query>")
                continue

            collection, search_query = parts[0], parts[1]

            if collection == "all":
                results = q.search_all(search_query, limit=5)
                for coll_name, items in results.items():
                    print(f"\n{coll_name.upper()}:")
                    for i, item in enumerate(items, 1):
                        print(f"  {i}. dist={item['distance']:.4f}  {item['text'][:120]}...")
            else:
                results = q.search_collection(collection, search_query, limit=5)
                if results:
                    print(f"\n{collection.upper()} Results:")
                    for i, item in enumerate(results, 1):
                        print(f"  {i}. dist={item['distance']:.4f}  {item['text'][:120]}...")
                else:
                    print(f"No results in '{collection}'")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
