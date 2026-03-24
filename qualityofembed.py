#!/usr/bin/env python3
"""
qualityofembed.py — Embedding Quality Assessment Tool

Usage:
    python qualityofembed.py <embeddings.json> [embeddings2.json ...]
    python qualityofembed.py --chroma <collection_name>
    python qualityofembed.py --chroma-all

Accepts JSON files where each item has a "vector" or "embedding" key.
Also works directly against a ChromaDB collection.

Metrics reported:
  zero_embeddings      Items with near-zero norm (failed / missing embeddings).
  norm_mean / std      Distribution of vector magnitudes.
  avg_pairwise_cosine  Mean cosine sim across sampled pairs.
                       ~0 = well spread (good).  ~1 = collapsed (bad).
  nn_distance_mean     Mean cosine distance to nearest neighbour.
                       Low = tight semantic clusters.  High = isolated items.
  coverage_score       Fraction of embedding space covered (estimated via
                       std of per-dimension variance). Higher = better.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Pure-math helpers (no numpy dependency)
# ---------------------------------------------------------------------------

def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine(a: List[float], b: List[float], na: float, nb: float) -> float:
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


# ---------------------------------------------------------------------------
# Core quality computation
# ---------------------------------------------------------------------------

def compute_quality(vecs: List[List[float]], label: str = "embeddings") -> Dict:
    """Compute quality metrics for a list of embedding vectors.

    Args:
        vecs:  List of float vectors (all must have the same dimension).
        label: Name shown in the printed report.

    Returns:
        Dict of metric name → value.
    """
    n = len(vecs)
    if n == 0:
        print(f"[{label}] No vectors to evaluate.")
        return {}

    dim = len(vecs[0])
    norms = [_norm(v) for v in vecs]
    zero_count = sum(1 for nrm in norms if nrm < 1e-6)

    norm_mean = _mean(norms)
    norm_std = _std(norms, norm_mean)

    # ---- Sample up to 300 items for O(n²) metrics ----
    sample_size = min(n, 300)
    step = max(1, n // sample_size)
    sv = vecs[::step][:sample_size]
    sn = norms[::step][:sample_size]
    m = len(sv)

    # Pairwise cosine similarity
    pair_sims: List[float] = []
    for i in range(m):
        for j in range(i + 1, m):
            pair_sims.append(_cosine(sv[i], sv[j], sn[i], sn[j]))
    avg_pairwise = _mean(pair_sims)
    pairwise_std = _std(pair_sims, avg_pairwise)

    # Nearest-neighbour cosine distance
    nn_dists: List[float] = []
    for i in range(m):
        best = 2.0
        for j in range(m):
            if i == j:
                continue
            d = 1.0 - _cosine(sv[i], sv[j], sn[i], sn[j])
            if d < best:
                best = d
        nn_dists.append(best)
    nn_mean = _mean(nn_dists)
    nn_std = _std(nn_dists, nn_mean)

    # Per-dimension variance (coverage estimate)
    dim_variances: List[float] = []
    for d in range(dim):
        col = [v[d] for v in sv]
        mu = _mean(col)
        dim_variances.append(_std(col, mu) ** 2)
    avg_dim_var = _mean(dim_variances)
    # Coverage score: std of per-dim variances (high = uneven use of space,
    # low = uniform coverage). We invert so higher score = better coverage.
    var_std = _std(dim_variances, avg_dim_var)
    coverage_score = round(1.0 - min(var_std / (avg_dim_var + 1e-9), 1.0), 4)

    # ---- Interpret metrics ----
    def grade(value: float, low: float, high: float,
              prefer: str = "low") -> str:
        """Return OK / WARN / BAD based on thresholds."""
        if prefer == "low":
            return "OK" if value <= low else ("WARN" if value <= high else "BAD")
        else:
            return "OK" if value >= high else ("WARN" if value >= low else "BAD")

    zero_grade = "OK" if zero_count == 0 else ("WARN" if zero_count / n < 0.05 else "BAD")
    pairwise_grade = grade(avg_pairwise, 0.5, 0.8, prefer="low")
    nn_grade = grade(nn_mean, 0.2, 0.5, prefer="low")
    coverage_grade = grade(coverage_score, 0.5, 0.7, prefer="high")

    report = {
        "label": label,
        "total": n,
        "dimension": dim,
        "zero_embeddings": zero_count,
        "zero_grade": zero_grade,
        "norm_mean": round(norm_mean, 4),
        "norm_std": round(norm_std, 4),
        "avg_pairwise_cosine": round(avg_pairwise, 4),
        "pairwise_std": round(pairwise_std, 4),
        "pairwise_grade": pairwise_grade,
        "nn_distance_mean": round(nn_mean, 4),
        "nn_distance_std": round(nn_std, 4),
        "nn_grade": nn_grade,
        "coverage_score": coverage_score,
        "coverage_grade": coverage_grade,
    }

    _print_report(report)
    return report


def _print_report(r: Dict) -> None:
    w = 54
    print("\n" + "=" * w)
    print(f"  Embedding Quality Report: {r['label']}")
    print("=" * w)
    print(f"  Items        : {r['total']}")
    print(f"  Dimension    : {r['dimension']}")
    print()

    z = r["zero_embeddings"]
    pct = 100 * z / r["total"] if r["total"] else 0
    print(f"  Zero embeds  : {z} ({pct:.1f}%)  [{r['zero_grade']}]")
    print(f"  Norm mean±std: {r['norm_mean']} ± {r['norm_std']}")
    print()

    print(f"  Avg pairwise cosine sim : {r['avg_pairwise_cosine']}  [{r['pairwise_grade']}]")
    print(f"    std                   : {r['pairwise_std']}")
    _bar("  spread", 1.0 - r["avg_pairwise_cosine"])
    print()

    print(f"  Avg NN cosine distance  : {r['nn_distance_mean']}  [{r['nn_grade']}]")
    print(f"    std                   : {r['nn_distance_std']}")
    _bar("  cluster tightness", 1.0 - r["nn_distance_mean"])
    print()

    print(f"  Coverage score          : {r['coverage_score']}  [{r['coverage_grade']}]")
    _bar("  coverage", r["coverage_score"])

    print()
    print("  Grades: OK = good  WARN = acceptable  BAD = investigate")
    print("=" * w)


def _bar(label: str, value: float, width: int = 20) -> None:
    """Print a simple ASCII progress bar."""
    filled = int(max(0.0, min(1.0, value)) * width)
    bar = "█" * filled + "░" * (width - filled)
    print(f"  {label:<22} [{bar}] {value:.2f}")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_json_file(path: str) -> Tuple[List[List[float]], str]:
    """Load vectors from a JSON embeddings file."""
    p = Path(path)
    if not p.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return [], p.stem

    with open(p) as f:
        data = json.load(f)

    vecs = []
    for item in data:
        v = item.get("vector") or item.get("embedding")
        if v:
            vecs.append(list(v))
    return vecs, p.stem


def load_chroma_collection(collection_name: str,
                            db_path: str = "./chromadb_storage") -> Tuple[List[List[float]], str]:
    """Load vectors from a ChromaDB collection."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=db_path)
        col = client.get_collection(collection_name)
        all_data = col.get(include=["embeddings"])
        raw = all_data["embeddings"]
        if raw is None:
            raw = []
        vecs = [v.tolist() if hasattr(v, "tolist") else list(v) for v in raw]
        return vecs, collection_name
    except Exception as e:
        print(f"ChromaDB error: {e}", file=sys.stderr)
        return [], collection_name


def load_all_chroma(db_path: str = "./chromadb_storage") -> List[Tuple[List[List[float]], str]]:
    """Load vectors from every collection in a ChromaDB database."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
        results = []
        for col in collections:
            vecs, label = load_chroma_collection(col.name, db_path)
            results.append((vecs, label))
        return results
    except Exception as e:
        print(f"ChromaDB error: {e}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess embedding quality from JSON files or ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="*", help="JSON embedding file(s) to evaluate")
    parser.add_argument("--chroma", metavar="COLLECTION",
                        help="Evaluate a ChromaDB collection by name")
    parser.add_argument("--chroma-all", action="store_true",
                        help="Evaluate all ChromaDB collections")
    parser.add_argument("--db-path", default="./chromadb_storage",
                        help="Path to ChromaDB storage (default: ./chromadb_storage)")
    args = parser.parse_args()

    if not args.files and not args.chroma and not args.chroma_all:
        parser.print_help()
        sys.exit(0)

    if args.files:
        for path in args.files:
            vecs, label = load_json_file(path)
            if vecs:
                compute_quality(vecs, label)

    if args.chroma:
        vecs, label = load_chroma_collection(args.chroma, args.db_path)
        if vecs:
            compute_quality(vecs, label)

    if args.chroma_all:
        for vecs, label in load_all_chroma(args.db_path):
            if vecs:
                compute_quality(vecs, label)


if __name__ == "__main__":
    main()
