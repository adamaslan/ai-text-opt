#!/usr/bin/env python3
"""
Knowledge Graph – cross-collection semantic map.

Builds a graph where:
  - Nodes  = documents from all ChromaDB collections
  - Edges  = cosine similarity above a threshold (default 0.75)

Outputs:
  1. knowledge_graph.json  – full adjacency list + node metadata
  2. Console summary       – most connected nodes, cross-collection bridges

Usage:
    python knowledge_graph.py
    python knowledge_graph.py --threshold 0.70 --sample 50
    python knowledge_graph.py --output my_graph.json
    python knowledge_graph.py --visualize          # requires: pip install networkx matplotlib
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHROMA_DB_PATH = str(Path(__file__).parent / "chromadb_storage")
DEFAULT_THRESHOLD = 0.75   # cosine similarity: 1 = identical, 0 = orthogonal
DEFAULT_SAMPLE = 100       # max items per collection to keep graph manageable
DEFAULT_OUTPUT = "knowledge_graph.json"


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine(a: List[float], b: List[float], na: Optional[float] = None, nb: Optional[float] = None) -> float:
    na = na or _norm(a)
    nb = nb or _norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


# ---------------------------------------------------------------------------
# Graph node / edge types
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("id", "collection", "title", "tags", "text_preview", "vector", "norm")

    def __init__(
        self,
        node_id: str,
        collection: str,
        title: str,
        tags: str,
        text_preview: str,
        vector: List[float],
    ):
        self.id = node_id
        self.collection = collection
        self.title = title
        self.tags = tags
        self.text_preview = text_preview
        self.vector = vector
        self.norm = _norm(vector)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "collection": self.collection,
            "title": self.title,
            "tags": self.tags,
            "text_preview": self.text_preview,
        }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """
    Loads all ChromaDB collections, computes pairwise cosine similarity,
    and exposes the resulting graph as an adjacency list.
    """

    def __init__(
        self,
        db_path: str = CHROMA_DB_PATH,
        threshold: float = DEFAULT_THRESHOLD,
        sample: int = DEFAULT_SAMPLE,
    ):
        self.threshold = threshold
        self.sample = sample
        self._client = chromadb.PersistentClient(path=db_path)
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[Dict]] = defaultdict(list)  # node_id → [{target, similarity, cross_collection}]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Pull vectors from all collections into self.nodes."""
        collections = self._client.list_collections()
        if not collections:
            raise RuntimeError(f"No collections found at {CHROMA_DB_PATH}")

        logger.info("Collections: %s", [c.name for c in collections])

        for coll_obj in collections:
            name = coll_obj.name
            collection = self._client.get_collection(name)
            total = collection.count()
            logger.info("  Loading '%s' (%d items, sampling up to %d)…", name, total, self.sample)

            data = collection.get(
                include=["embeddings", "documents", "metadatas"],
                limit=self.sample,
            )

            ids = data.get("ids") or []
            embeddings = data.get("embeddings") or []
            metadatas = data.get("metadatas") or []

            if not embeddings:
                logger.warning("  No embeddings in '%s' — skipping", name)
                continue

            for doc_id, vec, meta in zip(ids, embeddings, metadatas):
                meta = meta or {}
                v = vec.tolist() if hasattr(vec, "tolist") else list(vec)
                if not v or _norm(v) < 1e-6:
                    continue
                node = Node(
                    node_id=f"{name}::{doc_id}",
                    collection=name,
                    title=meta.get("title", doc_id),
                    tags=meta.get("tags", ""),
                    text_preview=meta.get("text_preview", "")[:200],
                    vector=v,
                )
                self.nodes[node.id] = node

        logger.info("Loaded %d nodes across %d collections", len(self.nodes), len(collections))

    # ------------------------------------------------------------------
    # Edge computation
    # ------------------------------------------------------------------

    def build_edges(self) -> None:
        """Compute all pairwise cosine similarities above threshold."""
        node_list = list(self.nodes.values())
        n = len(node_list)
        edge_count = 0

        logger.info("Computing pairwise similarities for %d nodes…", n)

        for i in range(n):
            a = node_list[i]
            for j in range(i + 1, n):
                b = node_list[j]
                sim = _cosine(a.vector, b.vector, a.norm, b.norm)
                if sim >= self.threshold:
                    cross = a.collection != b.collection
                    self.edges[a.id].append({
                        "target": b.id,
                        "similarity": round(sim, 4),
                        "cross_collection": cross,
                    })
                    self.edges[b.id].append({
                        "target": a.id,
                        "similarity": round(sim, 4),
                        "cross_collection": cross,
                    })
                    edge_count += 1

        logger.info("Found %d edges at threshold %.2f", edge_count, self.threshold)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "metadata": {
                "threshold": self.threshold,
                "total_nodes": len(self.nodes),
                "total_edges": sum(len(v) for v in self.edges.values()) // 2,
                "collections": list({n.collection for n in self.nodes.values()}),
            },
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": dict(self.edges),
        }

    def save(self, output_path: str = DEFAULT_OUTPUT) -> None:
        graph_dict = self.to_dict()
        with open(output_path, "w") as f:
            json.dump(graph_dict, f, indent=2)
        logger.info("✓ Graph saved to %s", output_path)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def most_connected(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Return nodes sorted by degree (number of edges)."""
        degrees = [(nid, len(edges)) for nid, edges in self.edges.items()]
        degrees.sort(key=lambda x: x[1], reverse=True)
        return degrees[:top_n]

    def cross_collection_bridges(self, top_n: int = 10) -> List[Tuple[str, str, float]]:
        """Return the strongest cross-collection edges."""
        bridges: List[Tuple[str, str, float]] = []
        seen: set = set()
        for nid, edges in self.edges.items():
            for edge in edges:
                if not edge["cross_collection"]:
                    continue
                key = tuple(sorted([nid, edge["target"]]))
                if key in seen:
                    continue
                seen.add(key)
                bridges.append((nid, edge["target"], edge["similarity"]))
        bridges.sort(key=lambda x: x[2], reverse=True)
        return bridges[:top_n]

    def collection_cluster_report(self) -> Dict[str, Dict]:
        """Intra- vs inter-collection edge counts per collection."""
        report: Dict[str, Dict] = defaultdict(lambda: {"intra": 0, "inter": 0, "nodes": 0})
        for node in self.nodes.values():
            report[node.collection]["nodes"] += 1
        for nid, edges in self.edges.items():
            src_coll = self.nodes[nid].collection
            for edge in edges:
                tgt_coll = self.nodes[edge["target"]].collection
                if src_coll == tgt_coll:
                    report[src_coll]["intra"] += 1
                else:
                    report[src_coll]["inter"] += 1
        # Halve intra counts (each edge counted twice)
        for v in report.values():
            v["intra"] //= 2
        return dict(report)

    def print_summary(self) -> None:
        print("\n" + "=" * 65)
        print("  KNOWLEDGE GRAPH SUMMARY")
        print("=" * 65)

        meta = self.to_dict()["metadata"]
        print(f"  Nodes      : {meta['total_nodes']}")
        print(f"  Edges      : {meta['total_edges']}  (threshold ≥ {self.threshold})")
        print(f"  Collections: {', '.join(meta['collections'])}")

        # Per-collection cluster report
        cluster = self.collection_cluster_report()
        print("\n  Collection stats:")
        print(f"  {'Collection':<30}  {'Nodes':>5}  {'Intra':>5}  {'Inter':>5}")
        print("  " + "-" * 50)
        for coll, stats in sorted(cluster.items()):
            print(f"  {coll:<30}  {stats['nodes']:>5}  {stats['intra']:>5}  {stats['inter']:>5}")

        # Most connected nodes
        print("\n  Most connected nodes (by degree):")
        for nid, deg in self.most_connected(10):
            node = self.nodes[nid]
            print(f"    [{node.collection}] {node.title[:50]:<50}  deg={deg}")

        # Cross-collection bridges
        bridges = self.cross_collection_bridges(10)
        if bridges:
            print("\n  Strongest cross-collection bridges:")
            for src, tgt, sim in bridges:
                src_n = self.nodes[src]
                tgt_n = self.nodes[tgt]
                print(
                    f"    sim={sim:.4f}  "
                    f"[{src_n.collection}] {src_n.title[:35]:<35}  ↔  "
                    f"[{tgt_n.collection}] {tgt_n.title[:35]}"
                )
        else:
            print("\n  No cross-collection bridges found at this threshold.")
            print("  Try --threshold 0.60 to discover weaker links.")

        print("=" * 65)


# ---------------------------------------------------------------------------
# Optional visualisation (requires networkx + matplotlib)
# ---------------------------------------------------------------------------

def visualize(graph: KnowledgeGraph, output_path: str = "knowledge_graph.png") -> None:
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("\n[visualize] Install dependencies: pip install networkx matplotlib")
        return

    G = nx.Graph()

    # Assign a colour per collection
    collections = list({n.collection for n in graph.nodes.values()})
    colour_map = {c: cm.tab10(i / max(len(collections), 1)) for i, c in enumerate(collections)}

    for nid, node in graph.nodes.items():
        G.add_node(nid, label=node.title[:25], collection=node.collection)

    for nid, edges in graph.edges.items():
        for edge in edges:
            if nid < edge["target"]:  # avoid duplicates
                G.add_edge(nid, edge["target"], weight=edge["similarity"])

    node_colours = [colour_map[graph.nodes[n].collection] for n in G.nodes()]

    plt.figure(figsize=(18, 14))
    pos = nx.spring_layout(G, k=0.6, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=node_colours, node_size=120, alpha=0.85)
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.8)

    # Label only high-degree nodes
    degrees = dict(G.degree())
    labels = {n: G.nodes[n]["label"] for n in G.nodes() if degrees[n] >= 3}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=colour_map[c], markersize=10, label=c)
        for c in collections
    ]
    plt.legend(handles=legend_handles, loc="upper left", fontsize=8)
    plt.title("Knowledge Graph – Cross-Collection Semantic Map", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n✓ Graph visualisation saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic knowledge graph from ChromaDB")
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold for edges (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=DEFAULT_SAMPLE,
        help=f"Max items per collection (default: {DEFAULT_SAMPLE})",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Generate a PNG visualisation (requires networkx + matplotlib)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    graph = KnowledgeGraph(
        threshold=args.threshold,
        sample=args.sample,
    )
    graph.load()
    graph.build_edges()
    graph.save(args.output)
    graph.print_summary()

    if args.visualize:
        viz_path = args.output.replace(".json", ".png")
        visualize(graph, output_path=viz_path)


if __name__ == "__main__":
    main()
