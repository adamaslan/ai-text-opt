#!/usr/bin/env python3
"""
benchmark_embeddings.py — Compare embedding models on your actual data.

Tests models available via Ollama and (optionally) sentence-transformers,
then prints a ranked comparison using the same metrics as qualityofembed.py.

Usage:
    python benchmark_embeddings.py                     # uses built-in sample texts
    python benchmark_embeddings.py --file ideas_embeddings.json --n 30
    python benchmark_embeddings.py --models nomic-embed-text mxbai-embed-large
"""

import argparse
import json
import math
import time
import urllib.request
import urllib.error
from typing import List, Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Math helpers (no numpy)
# ---------------------------------------------------------------------------

def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))

def _cosine(a, b, na, nb) -> float:
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def _std(vals: List[float], mu: float) -> float:
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0

# ---------------------------------------------------------------------------
# Embedding via Ollama
# ---------------------------------------------------------------------------

OLLAMA_BASE = "http://localhost:11434"

def ollama_available() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return True
    except Exception:
        return False

def list_ollama_embed_models() -> List[str]:
    """Return Ollama models that look like embedding models."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5) as r:
            data = json.load(r)
        models = [m["name"] for m in data.get("models", [])]
        # Keep models that contain embed-related keywords
        embed_keywords = ["embed", "nomic", "mxbai", "minilm", "bge", "e5", "gte"]
        embed_models = [m for m in models if any(k in m.lower() for k in embed_keywords)]
        return embed_models if embed_models else models  # fall back to all if none match
    except Exception:
        return []

def embed_ollama(model: str, texts: List[str]) -> Tuple[List[List[float]], float]:
    """Embed texts with an Ollama model. Returns (vectors, seconds_elapsed)."""
    vectors = []
    start = time.time()
    for text in texts:
        payload = json.dumps({"model": model, "input": text}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            emb = data.get("embeddings", [[]])[0]
            vectors.append(list(emb))
        except Exception as e:
            print(f"    [warn] embed failed for model {model}: {e}")
            vectors.append([])
    elapsed = time.time() - start
    return vectors, elapsed

# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_metrics(vecs: List[List[float]]) -> Dict:
    """Same metrics as qualityofembed.py, returned as a dict."""
    vecs = [v for v in vecs if len(v) > 0]
    n = len(vecs)
    if n == 0:
        return {}

    norms = [_norm(v) for v in vecs]
    zero_count = sum(1 for nrm in norms if nrm < 1e-6)
    norm_mean = _mean(norms)
    norm_std = _std(norms, norm_mean)

    # Pairwise cosine
    pair_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_sims.append(_cosine(vecs[i], vecs[j], norms[i], norms[j]))
    avg_pairwise = _mean(pair_sims)

    # Nearest-neighbour distance
    nn_dists = []
    for i in range(n):
        best = 2.0
        for j in range(n):
            if i == j:
                continue
            d = 1.0 - _cosine(vecs[i], vecs[j], norms[i], norms[j])
            if d < best:
                best = d
        nn_dists.append(best)
    nn_mean = _mean(nn_dists)

    # Semantic discrimination: std of pairwise sims
    # High std → model can distinguish similar vs dissimilar text (good)
    pair_std = _std(pair_sims, avg_pairwise)

    # Coverage via per-dim variance uniformity
    dim = len(vecs[0])
    dim_vars = []
    for d in range(dim):
        col = [v[d] for v in vecs]
        mu = _mean(col)
        dim_vars.append(_std(col, mu) ** 2)
    avg_var = _mean(dim_vars)
    var_std = _std(dim_vars, avg_var)
    coverage = round(1.0 - min(var_std / (avg_var + 1e-9), 1.0), 4)

    return {
        "n": n,
        "dim": dim,
        "zero": zero_count,
        "norm_mean": round(norm_mean, 4),
        "norm_std": round(norm_std, 4),
        "avg_pairwise": round(avg_pairwise, 4),
        "pair_std": round(pair_std, 4),       # discrimination ability
        "nn_dist": round(nn_mean, 4),
        "coverage": coverage,
    }

def overall_score(m: Dict) -> float:
    """Single 0-1 score combining all metrics (higher = better)."""
    if not m:
        return 0.0
    # spread (lower pairwise sim = better spread, capped at 0.8)
    spread = 1.0 - min(m["avg_pairwise"], 0.8) / 0.8
    # discrimination (higher pair_std = model can tell texts apart)
    discrim = min(m["pair_std"] / 0.15, 1.0)
    # cluster tightness (lower nn_dist = tighter, capped)
    tightness = 1.0 - min(m["nn_dist"], 0.6) / 0.6
    coverage = m["coverage"]
    zero_penalty = 1.0 if m["zero"] == 0 else max(0.0, 1.0 - m["zero"] / m["n"])
    score = (spread * 0.25 + discrim * 0.30 + tightness * 0.20 + coverage * 0.15 + zero_penalty * 0.10)
    return round(score, 4)

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

SEMANTIC_PAIRS = [
    # (text_a, text_b, expected: "similar" | "different")
    ("grief and loss of a loved one", "mourning death and sadness", "similar"),
    ("meditation and mindfulness practice", "stock market investment strategy", "different"),
    ("love and romantic connection", "compassion and emotional intimacy", "similar"),
    ("philosophy of Nietzsche", "cooking recipes and food preparation", "different"),
    ("beauty in everyday moments", "aesthetic appreciation of nature", "similar"),
]

def semantic_discrimination_score(model: str) -> Optional[float]:
    """
    Score how well a model separates similar vs different pairs.
    Returns mean(similar_sims) - mean(different_sims). Higher = better.
    """
    sim_scores = []
    diff_scores = []
    for a, b, label in SEMANTIC_PAIRS:
        vecs, _ = embed_ollama(model, [a, b])
        if len(vecs) != 2 or not vecs[0] or not vecs[1]:
            continue
        na, nb = _norm(vecs[0]), _norm(vecs[1])
        sim = _cosine(vecs[0], vecs[1], na, nb)
        if label == "similar":
            sim_scores.append(sim)
        else:
            diff_scores.append(sim)
    if not sim_scores or not diff_scores:
        return None
    return round(_mean(sim_scores) - _mean(diff_scores), 4)

def run_benchmark(texts: List[str], models: List[str]) -> None:
    print(f"\n{'=' * 64}")
    print(f"  Embedding Model Benchmark")
    print(f"  Texts: {len(texts)}  |  Models: {len(models)}")
    print(f"{'=' * 64}\n")

    results = []

    for model in models:
        print(f"  Testing: {model} ...", flush=True)
        vecs, elapsed = embed_ollama(model, texts)
        good_vecs = [v for v in vecs if v]
        if not good_vecs:
            print(f"    [skip] no valid embeddings returned\n")
            continue

        metrics = compute_metrics(good_vecs)
        metrics["model"] = model
        metrics["elapsed_s"] = round(elapsed, 2)
        metrics["speed"] = round(len(good_vecs) / elapsed, 2)  # items/sec

        print(f"    discrim pairs ...", flush=True)
        sep = semantic_discrimination_score(model)
        metrics["separation"] = sep  # similar_sim - different_sim

        metrics["score"] = overall_score(metrics)
        results.append(metrics)
        print(f"    dim={metrics['dim']}  pairwise={metrics['avg_pairwise']}  "
              f"nn_dist={metrics['nn_dist']}  separation={sep}  score={metrics['score']}\n")

    if not results:
        print("No results collected.")
        return

    results.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n{'=' * 64}")
    print(f"  RANKED RESULTS")
    print(f"{'=' * 64}")
    header = f"  {'Model':<32} {'Dim':>4} {'Score':>6} {'Sep':>6} {'Spread':>7} {'Discrim':>8} {'NN dist':>8} {'Speed':>8}"
    print(header)
    print(f"  {'-'*62}")
    for i, r in enumerate(results):
        sep_str = f"{r['separation']:+.3f}" if r["separation"] is not None else "  N/A"
        spread = round(1.0 - r["avg_pairwise"], 4)
        print(f"  {'#'+str(i+1)+' '+r['model']:<32} {r['dim']:>4} {r['score']:>6.3f} "
              f"{sep_str:>6} {spread:>7.4f} {r['pair_std']:>8.4f} {r['nn_dist']:>8.4f} "
              f"{r['speed']:>6.2f}/s")

    print(f"\n  Legend:")
    print(f"    Score    = composite quality score (higher = better)")
    print(f"    Sep      = similar_pairs_sim - different_pairs_sim (higher = better)")
    print(f"    Spread   = 1 - avg_pairwise_cosine (higher = more spread)")
    print(f"    Discrim  = std of pairwise sims (higher = more discriminative)")
    print(f"    NN dist  = avg nearest-neighbour cosine distance (lower = tighter clusters)")
    print(f"    Speed    = items embedded per second")
    print(f"\n  Winner: {results[0]['model']}  (score={results[0]['score']})")

    # ChromaDB / Weaviate recommendation
    print(f"\n{'=' * 64}")
    print(f"  ChromaDB / Weaviate Recommendation")
    print(f"{'=' * 64}")
    best = results[0]
    print(f"  Best model : {best['model']}")
    print(f"  Dimension  : {best['dim']}")
    print(f"  Notes:")
    if best["dim"] == 768:
        print(f"    - 768-dim is the standard for nomic / roberta-base models")
        print(f"    - Good balance of quality and storage cost")
    elif best["dim"] >= 1024:
        print(f"    - High-dim ({best['dim']}) = more expressive but slower HNSW indexing")
        print(f"    - Consider mxbai-embed-large or similar for production")
    elif best["dim"] <= 384:
        print(f"    - Low-dim ({best['dim']}) = fast retrieval, lower quality")
        print(f"    - Fine for large corpora where speed matters")
    sep = best.get("separation")
    if sep is not None:
        if sep > 0.2:
            print(f"    - Separation score {sep:+.3f}: excellent semantic discrimination")
        elif sep > 0.1:
            print(f"    - Separation score {sep:+.3f}: good semantic discrimination")
        else:
            print(f"    - Separation score {sep:+.3f}: weak — model may not distinguish topics well")
    print(f"{'=' * 64}\n")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_TEXTS = [
    "grief and the loss of a loved one, mourning and sadness",
    "meditation techniques for mindfulness and present-moment awareness",
    "romantic love, connection, and emotional intimacy",
    "Nietzsche's philosophy on power, will, and morality",
    "beauty in everyday life and aesthetic appreciation",
    "pragmatism and practical philosophy in daily decisions",
    "Camus and the absurdist philosophy of meaning",
    "tolerance, exploitation, and the slave mentality",
    "full expression of one's true self and identity",
    "compassion, kindness, and human connection",
    "death, impermanence, and the acceptance of mortality",
    "technology, communication, and modern relationships",
    "intentions versus outcomes in ethical decision making",
    "focus, attention, and the practice of concentration",
    "pettiness, jealousy, and interpersonal conflict",
]

def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama embedding models on your data.")
    parser.add_argument("--models", nargs="+", help="Specific Ollama model names to test")
    parser.add_argument("--file", help="JSON embeddings file to pull sample texts from")
    parser.add_argument("--n", type=int, default=15, help="Number of texts to use (default: 15)")
    args = parser.parse_args()

    if not ollama_available():
        print("Ollama is not running at localhost:11434. Start it with: ollama serve")
        return

    # Determine models to test
    if args.models:
        models = args.models
    else:
        models = list_ollama_embed_models()
        if not models:
            print("No embedding models found in Ollama. Pull one first, e.g.:")
            print("  ollama pull nomic-embed-text")
            return
        print(f"  Auto-detected embedding models: {models}")

    # Determine texts
    if args.file:
        import pathlib
        p = pathlib.Path(args.file)
        if not p.exists():
            print(f"File not found: {args.file}")
            return
        with open(p) as f:
            data = json.load(f)
        texts = []
        for item in data[: args.n]:
            t = item.get("content") or item.get("properties", {})
            if isinstance(t, dict):
                t = " ".join(str(v) for v in t.values())
            texts.append(str(t)[:400])
    else:
        texts = DEFAULT_TEXTS[: args.n]

    run_benchmark(texts, models)


if __name__ == "__main__":
    main()
