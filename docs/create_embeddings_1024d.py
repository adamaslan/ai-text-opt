#!/usr/bin/env python3
"""
1024-Dimensional Embedding Generator

Uses one of several high-quality models that output 1024D vectors:
- intfloat/e5-large-v2 (State-of-the-art, recommended)
- BAAI/bge-large-en-v1.5 (Production-grade)
- jinaai/jina-embeddings-v2-large-en (8K context window)

Features:
- Asynchronous batch processing
- GPU acceleration
- Checkpoint recovery
- 1024D dense vectors
- Progress tracking
- Per-batch retry with exponential backoff
- Safe JSON-based embedding serialization
"""

import ast
import json
import os
import pickle
import time
import logging
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("embedding_1024d.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("embedding_1024d")

MAX_BATCH_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds; doubles each retry


@dataclass
class EmbeddingConfig:
    """Configuration for 1024D embedding generation."""
    input_file: Path
    output_file: Path
    model_name: str = "intfloat/e5-large-v2"  # 1024D, best quality
    batch_size: int = 16
    use_gpu: bool = True
    num_workers: int = 4
    cache_dir: Optional[Path] = None
    save_interval: int = 50  # Save checkpoint every N rows
    text_column: str = "Ideas"

    def __post_init__(self) -> None:
        self.input_file = Path(self.input_file)
        self.output_file = Path(self.output_file)
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.save_interval < 1:
            raise ValueError("save_interval must be >= 1")

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "EmbeddingConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        for path_field in ("input_file", "output_file", "cache_dir"):
            if path_field in config_data and config_data[path_field]:
                config_data[path_field] = Path(config_data[path_field])

        return cls(**config_data)

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Load configuration from environment variables."""
        return cls(
            input_file=Path(os.getenv("EMBEDDING_INPUT_FILE", "151_ideas_updated2.csv")),
            output_file=Path(os.getenv("EMBEDDING_OUTPUT_FILE", "ideas_with_embeddings_1024d.csv")),
            model_name=os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2"),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
            use_gpu=os.getenv("EMBEDDING_USE_GPU", "true").lower() in ("true", "1", "yes"),
            num_workers=int(os.getenv("EMBEDDING_NUM_WORKERS", "4")),
            cache_dir=Path(os.getenv("EMBEDDING_CACHE_DIR")) if os.getenv("EMBEDDING_CACHE_DIR") else None,
            save_interval=int(os.getenv("EMBEDDING_SAVE_INTERVAL", "50")),
            text_column=os.getenv("EMBEDDING_TEXT_COLUMN", "Ideas"),
        )


def _parse_embedding(value: object) -> Optional[np.ndarray]:
    """Safely parse a stored embedding string back to ndarray.

    Accepts JSON arrays or Python-repr lists — never calls eval() on raw strings.
    Returns None if parsing fails.
    """
    if isinstance(value, np.ndarray):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
        return np.array(parsed, dtype=np.float32)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        parsed = ast.literal_eval(value)
        return np.array(parsed, dtype=np.float32)
    except (ValueError, SyntaxError):
        return None


class EmbeddingGenerator1024D:
    """Generates 1024-dimensional embeddings using state-of-the-art models."""

    MODEL_INFO: Dict[str, Dict[str, str]] = {
        "intfloat/e5-large-v2": {
            "dimension": "1024",
            "quality": "State-of-the-art",
            "context": "512 tokens",
            "size": "1.47 GB",
            "note": "Requires 'Passage: ' prefix for documents",
        },
        "BAAI/bge-large-en-v1.5": {
            "dimension": "1024",
            "quality": "Excellent",
            "context": "512 tokens",
            "size": "1.34 GB",
            "note": "No prefix needed, strong baseline",
        },
        "jinaai/jina-embeddings-v2-large-en": {
            "dimension": "1024",
            "quality": "Excellent",
            "context": "8192 tokens",
            "size": "1.52 GB",
            "note": "Handles very long texts well",
        },
    }

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

        if config.model_name in self.MODEL_INFO:
            info = self.MODEL_INFO[config.model_name]
            logger.info(
                "Model: %s  dim=%s  quality=%s  context=%s  size=%s",
                config.model_name, info["dimension"], info["quality"],
                info["context"], info["size"],
            )
        else:
            logger.warning("Model %s not in known list, attempting to load anyway", config.model_name)

        device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)

        logger.info("Loading model: %s", config.model_name)
        self.model = SentenceTransformer(config.model_name, device=device)

        test_embedding: np.ndarray = self.model.encode("test", convert_to_numpy=True)
        actual_dim = len(test_embedding)
        logger.info("Model loaded. Verified dimension: %dD", actual_dim)

        if actual_dim != 1024:
            logger.warning("Expected 1024D but got %dD", actual_dim)

        self.actual_dim = actual_dim

    def preprocess_text(self, text: str) -> str:
        """Preprocess text based on model requirements."""
        if not isinstance(text, str) or not text.strip():
            return ""
        if "e5" in self.config.model_name.lower():
            text = f"Passage: {text}"
        return text.strip()

    def generate_batch(self, texts: List[str]) -> np.ndarray:
        """Generate 1024D embeddings for a batch of texts with retry logic."""
        processed = [self.preprocess_text(t) for t in texts]

        for attempt in range(1, MAX_BATCH_RETRIES + 1):
            try:
                embeddings: np.ndarray = self.model.encode(
                    processed,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                return embeddings
            except Exception as exc:
                if attempt == MAX_BATCH_RETRIES:
                    raise RuntimeError(
                        f"Batch encoding failed after {MAX_BATCH_RETRIES} attempts: {exc}"
                    ) from exc
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "Batch encoding attempt %d/%d failed (%s). Retrying in %.1fs…",
                    attempt, MAX_BATCH_RETRIES, exc, wait,
                )
                time.sleep(wait)

        # Unreachable, but satisfies type checker
        raise RuntimeError("generate_batch: unexpected exit from retry loop")


def _load_checkpoint(
    checkpoint_path: Path, df: pd.DataFrame
) -> Tuple[pd.DataFrame, int]:
    """Return (dataframe_with_checkpoint_data, start_row_index)."""
    if not checkpoint_path.exists():
        return df, 0
    try:
        checkpoint_df = pd.read_csv(checkpoint_path, index_col=0)
        start_idx = int(checkpoint_df["embedding_1024d"].notna().sum())
        logger.info("Resuming from checkpoint at row %d", start_idx)
        return checkpoint_df, start_idx
    except Exception as exc:
        logger.warning("Failed to load checkpoint (%s); starting from scratch", exc)
        return df, 0


# ---------------------------------------------------------------------------
# ChromaDB Integration
# ---------------------------------------------------------------------------
# Free-tier strategy
#   - PersistentClient  : local disk, zero cost, unlimited collections/records
#   - Pre-computed embeddings passed directly via `embeddings=` — Chroma never
#     calls an external embedding API, so there are no per-token charges
#   - cosine distance   : correct metric for normalised sentence-transformer
#     vectors; avoids magnitude bias that would skew L2 or ip results
#   - HNSW tuning       : ef_construction=200 / max_neighbors=32 gives a good
#     recall/speed trade-off for datasets up to ~200k vectors on a single node
#   - Batched upserts   : avoids single-shot memory spikes; idempotent so reruns
#     never create duplicate records
# ---------------------------------------------------------------------------

CHROMA_COLLECTION_NAME = "ideas_1024d"
CHROMA_PERSIST_DIR = "chroma_db"           # relative to cwd; committed as .gitignore
CHROMA_UPSERT_BATCH = 200                  # records per upsert call
CHROMA_HNSW_EF_CONSTRUCTION = 200
CHROMA_HNSW_MAX_NEIGHBORS = 32
CHROMA_HNSW_EF_SEARCH = 100               # tunable post-creation


def _build_chroma_client(persist_dir: str) -> "chromadb.PersistentClient":
    """Return a PersistentClient, creating the directory if needed."""
    import chromadb  # local import — optional dependency

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    logger.info("ChromaDB PersistentClient ready at '%s'", persist_dir)
    return client


def _get_or_create_collection(
    client: "chromadb.PersistentClient",
    name: str,
    model_name: str,
) -> "chromadb.Collection":
    """Return a collection configured for cosine similarity on 1024D vectors.

    Uses get_or_create_collection so reruns are idempotent.  The embedding
    function is explicitly set to None because we pass pre-computed vectors —
    this prevents Chroma from attempting any external API call.
    """
    import chromadb

    collection = client.get_or_create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": CHROMA_HNSW_EF_CONSTRUCTION,
            "hnsw:M": CHROMA_HNSW_MAX_NEIGHBORS,
            "hnsw:search_ef": CHROMA_HNSW_EF_SEARCH,
            "embedding_model": model_name,
            "embedding_dimension": "1024",
        },
        embedding_function=None,  # we supply embeddings ourselves
    )
    logger.info(
        "Collection '%s' ready  (count=%d)", name, collection.count()
    )
    return collection


def _upsert_to_chroma(
    collection: "chromadb.Collection",
    texts: List[str],
    embeddings_array: np.ndarray,
    df: pd.DataFrame,
) -> None:
    """Upsert all records in batches.

    IDs are row-index strings ("row_0", "row_1", …) so reruns overwrite
    existing records rather than duplicating them.

    Metadata stored per record:
    - text_preview : first 200 chars for quick human inspection
    - row_index    : original CSV row for join-back queries
    - char_len     : character count (useful for where-filter debugging)
    """
    total = len(texts)
    for batch_start in range(0, total, CHROMA_UPSERT_BATCH):
        batch_end = min(batch_start + CHROMA_UPSERT_BATCH, total)
        ids = [f"row_{i}" for i in range(batch_start, batch_end)]
        docs = texts[batch_start:batch_end]
        embs = embeddings_array[batch_start:batch_end].tolist()
        metas = [
            {
                "row_index": i,
                "text_preview": texts[i][:200],
                "char_len": len(texts[i]),
            }
            for i in range(batch_start, batch_end)
        ]

        collection.upsert(
            ids=ids,
            documents=docs,
            embeddings=embs,
            metadatas=metas,
        )
        logger.info(
            "Upserted rows %d–%d into Chroma (%d/%d)",
            batch_start, batch_end - 1, batch_end, total,
        )

    logger.info(
        "ChromaDB collection '%s' final count: %d",
        collection.name, collection.count(),
    )


def query_chroma(
    query_texts: List[str],
    n_results: int = 10,
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    generator: Optional[EmbeddingGenerator1024D] = None,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
) -> Dict:
    """Query the local ChromaDB collection.

    Accepts raw text queries; embeddings are computed locally so no external
    API is called.  Optionally filter by metadata (``where``) or document
    content (``where_document``).

    Args:
        query_texts: One or more natural-language queries.
        n_results: Top-K neighbours to return per query.
        persist_dir: Path to the Chroma persistence directory.
        collection_name: Target collection name.
        generator: Pre-loaded EmbeddingGenerator1024D.  If None, loads the
            default model inline (slower for one-off calls).
        where: Chroma metadata filter, e.g. ``{"char_len": {"$gt": 50}}``.
        where_document: Chroma document filter, e.g.
            ``{"$contains": "keyword"}``.

    Returns:
        Raw Chroma QueryResult dict with keys: ids, documents, distances,
        metadatas.

    Example::

        results = query_chroma(
            ["romantic dinner ideas"],
            n_results=5,
            where={"char_len": {"$gt": 30}},
        )
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            print(f"{dist:.4f}  {doc[:80]}")
    """
    client = _build_chroma_client(persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=None
    )

    if generator is None:
        config = EmbeddingConfig.from_env()
        generator = EmbeddingGenerator1024D(config)

    query_embeddings = generator.generate_batch(query_texts).tolist()

    kwargs: Dict = dict(
        query_embeddings=query_embeddings,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where
    if where_document:
        kwargs["where_document"] = where_document

    return collection.query(**kwargs)


async def main() -> int:
    """Main function to run the 1024D embedding generator."""
    config_path = Path("embedding_config_1024d.yaml")
    if config_path.exists():
        config = EmbeddingConfig.from_yaml(config_path)
        logger.info("Loaded configuration from %s", config_path)
    else:
        config = EmbeddingConfig.from_env()
        logger.info("Loaded configuration from environment variables")

    logger.info("Input file : %s", config.input_file)
    logger.info("Output file: %s", config.output_file)
    logger.info("Batch size : %d", config.batch_size)

    if not config.input_file.exists():
        logger.error("Input file not found: %s", config.input_file)
        return 1

    # Load CSV
    try:
        df = pd.read_csv(config.input_file)
        logger.info("Loaded %d rows from %s", len(df), config.input_file)
        if config.text_column not in df.columns:
            raise ValueError(f"CSV missing required column: '{config.text_column}'")
        texts: List[str] = df[config.text_column].fillna("").tolist()
        logger.info("Extracted %d texts from column '%s'", len(texts), config.text_column)
    except Exception as exc:
        logger.error("Failed to load input data: %s", exc)
        return 1

    # Ensure output column exists
    if "embedding_1024d" not in df.columns:
        df["embedding_1024d"] = None

    # Load model
    start_time = time.time()
    try:
        generator = EmbeddingGenerator1024D(config)
    except Exception as exc:
        logger.error("Failed to initialise embedding model: %s", exc)
        return 1

    checkpoint_path = config.output_file.with_suffix(".checkpoint.csv")
    df, start_idx = _load_checkpoint(checkpoint_path, df)

    # Process batches
    rows_to_process = len(texts) - start_idx
    logger.info("Starting 1024D embedding generation for %d texts", rows_to_process)

    try:
        with tqdm(total=len(texts), initial=start_idx, desc="Generating 1024D embeddings") as pbar:
            for batch_start in range(start_idx, len(texts), config.batch_size):
                batch_end = min(batch_start + config.batch_size, len(texts))
                batch_texts = texts[batch_start:batch_end]

                batch_embeddings = generator.generate_batch(batch_texts)

                for i, emb in enumerate(batch_embeddings):
                    # Store as JSON string — safe to parse later without eval()
                    df.loc[batch_start + i, "embedding_1024d"] = json.dumps(emb.tolist())

                pbar.update(len(batch_texts))

                rows_done = batch_end - start_idx
                if rows_done % config.save_interval == 0 or batch_end == len(texts):
                    df.to_csv(checkpoint_path)
                    logger.info("Checkpoint saved at row %d", batch_end)

    except Exception as exc:
        logger.error("Processing failed: %s", exc)
        df.to_csv(checkpoint_path)
        logger.info("Partial progress saved to %s", checkpoint_path)
        return 1

    # Reconstruct embedding array from stored strings
    parsed: List[Optional[np.ndarray]] = [_parse_embedding(v) for v in df["embedding_1024d"]]
    bad_rows = [i for i, v in enumerate(parsed) if v is None]
    if bad_rows:
        logger.error("Failed to parse embeddings for %d rows: %s", len(bad_rows), bad_rows[:10])
        return 1

    embeddings_array = np.stack(parsed, axis=0).astype(np.float32)  # type: ignore[arg-type]
    logger.info("Generated embeddings shape: %s", embeddings_array.shape)

    if embeddings_array.shape[1] != 1024:
        logger.error("Unexpected embedding dimension: %d (expected 1024)", embeddings_array.shape[1])
        return 1

    logger.info("Verified: All embeddings are 1024D")

    # Save final CSV
    df.to_csv(config.output_file, index=False)
    logger.info("Results saved to %s", config.output_file)

    # Remove checkpoint only after successful final save
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    elapsed = time.time() - start_time
    logger.info(
        "Processed %d rows in %.2fs  |  mean L2 norm: %.4f",
        len(df), elapsed, float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
    )

    # Save pickled embeddings
    pkl_path = config.output_file.with_suffix(".pkl")
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "embeddings": embeddings_array,
                    "texts": texts,
                    "model": config.model_name,
                    "dimension": 1024,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("Pickled embeddings saved to %s", pkl_path)
    except Exception as exc:
        logger.warning("Could not write pickle file (%s); CSV output is still valid", exc)

    # ------------------------------------------------------------------
    # ChromaDB: upsert all embeddings into a local persistent collection
    # ------------------------------------------------------------------
    # Uses PersistentClient (free, unlimited) with pre-computed vectors so
    # Chroma never calls any external embedding API.
    try:
        chroma_client = _build_chroma_client(CHROMA_PERSIST_DIR)
        collection = _get_or_create_collection(chroma_client, CHROMA_COLLECTION_NAME, config.model_name)
        _upsert_to_chroma(collection, texts, embeddings_array, df)
    except ImportError:
        logger.warning(
            "chromadb not installed — skipping vector store upsert. "
            "Run: pip install chromadb"
        )
    except Exception as exc:
        logger.warning(
            "ChromaDB upsert failed (%s); embeddings are still saved to CSV/pkl", exc
        )

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        raise SystemExit(exit_code)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        raise SystemExit(130)
    except Exception as exc:
        logger.critical("Unhandled exception: %s", exc, exc_info=True)
        raise SystemExit(1)
