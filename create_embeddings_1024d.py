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
"""

import os
import pickle
import time
import logging
import asyncio
from dataclasses import dataclass
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


@dataclass
class EmbeddingConfig:
    """Configuration for 1024D embedding generation."""
    input_file: Path
    output_file: Path
    model_name: str = "intfloat/e5-large-v2"  # 1024D, best quality
    batch_size: int = 16  # Smaller batch for large models
    use_gpu: bool = True
    num_workers: int = 4
    cache_dir: Optional[Path] = None
    save_interval: int = 50  # Save every 50 items
    text_column: str = "Ideas"  # or "articles", "text", etc.

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "EmbeddingConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Convert string paths to Path objects
        for path_field in ["input_file", "output_file", "cache_dir"]:
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


class EmbeddingGenerator1024D:
    """Generates 1024-dimensional embeddings using state-of-the-art models."""

    MODEL_INFO = {
        "intfloat/e5-large-v2": {
            "dimension": 1024,
            "quality": "State-of-the-art",
            "context": "512 tokens",
            "size": "1.47 GB",
            "note": "Requires 'Passage: ' prefix for documents"
        },
        "BAAI/bge-large-en-v1.5": {
            "dimension": 1024,
            "quality": "Excellent",
            "context": "512 tokens",
            "size": "1.34 GB",
            "note": "No prefix needed, strong baseline"
        },
        "jinaai/jina-embeddings-v2-large-en": {
            "dimension": 1024,
            "quality": "Excellent",
            "context": "8192 tokens",
            "size": "1.52 GB",
            "note": "Handles very long texts well"
        },
    }

    def __init__(self, config: EmbeddingConfig):
        self.config = config

        # Validate model
        if config.model_name not in self.MODEL_INFO:
            logger.warning(f"Model {config.model_name} not in known list, attempting to load anyway")
        else:
            info = self.MODEL_INFO[config.model_name]
            logger.info(f"Model: {config.model_name}")
            logger.info(f"  Dimension: {info['dimension']}D")
            logger.info(f"  Quality: {info['quality']}")
            logger.info(f"  Context: {info['context']}")
            logger.info(f"  Size: {info['size']}")

        # Initialize device
        device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Load model
        logger.info(f"Loading model: {config.model_name}")
        self.model = SentenceTransformer(config.model_name, device=device)

        # Verify dimension
        test_embedding = self.model.encode("test")
        actual_dim = len(test_embedding)
        logger.info(f"✓ Model loaded successfully. Verified dimension: {actual_dim}D")

        if actual_dim != 1024:
            logger.warning(f"Expected 1024D but got {actual_dim}D")

    def preprocess_text(self, text: str) -> str:
        """Preprocess text based on model."""
        if not isinstance(text, str):
            return ""

        # For E5 models, add "Passage: " prefix to document texts
        if "e5" in self.config.model_name.lower():
            text = f"Passage: {text}"

        return text.strip()

    def generate_batch(self, texts: List[str]) -> np.ndarray:
        """Generate 1024D embeddings for a batch of texts."""
        # Preprocess texts
        processed_texts = [self.preprocess_text(t) for t in texts]

        # Generate embeddings
        embeddings = self.model.encode(
            processed_texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embeddings


async def main():
    """Main function to run the 1024D embedding generator."""
    # Load configuration
    config_path = Path("embedding_config_1024d.yaml")
    if config_path.exists():
        config = EmbeddingConfig.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = EmbeddingConfig.from_env()
        logger.info("Loaded configuration from environment variables")

    logger.info(f"Input file: {config.input_file}")
    logger.info(f"Output file: {config.output_file}")
    logger.info(f"Batch size: {config.batch_size}")

    # Check if input file exists
    if not config.input_file.exists():
        logger.error(f"Input file not found: {config.input_file}")
        return 1

    # Load CSV data
    try:
        df = pd.read_csv(config.input_file)
        logger.info(f"✓ Loaded {len(df)} rows from {config.input_file}")

        if config.text_column not in df.columns:
            raise ValueError(f"CSV missing required column: {config.text_column}")

        texts = df[config.text_column].tolist()
        logger.info(f"✓ Extracted {len(texts)} texts from column '{config.text_column}'")

    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return 1

    # Initialize embedding generator
    start_time = time.time()
    generator = EmbeddingGenerator1024D(config)

    # Checkpoint setup
    checkpoint_path = config.output_file.with_suffix(".checkpoint.csv")
    start_idx = 0

    # Resume from checkpoint if available
    if checkpoint_path.exists():
        try:
            checkpoint_df = pd.read_csv(checkpoint_path, index_col=0)
            start_idx = len(checkpoint_df)
            df = checkpoint_df.copy()
            logger.info(f"✓ Resuming from checkpoint at row {start_idx}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    # Process batches
    try:
        logger.info(f"Starting 1024D embedding generation for {len(texts) - start_idx} texts")

        embeddings_list = []

        with tqdm(total=len(texts), initial=start_idx, desc="Generating 1024D embeddings") as pbar:
            for batch_start in range(start_idx, len(texts), config.batch_size):
                batch_end = min(batch_start + config.batch_size, len(texts))
                batch_texts = texts[batch_start:batch_end]

                # Generate embeddings
                batch_embeddings = generator.generate_batch(batch_texts)
                embeddings_list.extend(batch_embeddings)

                # Update dataframe
                for i, emb in enumerate(batch_embeddings):
                    df.loc[batch_start + i, "embedding_1024d"] = str(emb.tolist())

                # Update progress
                pbar.update(len(batch_texts))

                # Save checkpoint
                if (batch_end - start_idx) % (config.batch_size * config.save_interval // config.batch_size) == 0:
                    df.to_csv(checkpoint_path)
                    logger.info(f"✓ Checkpoint saved at row {batch_end}")

        # Convert embeddings to proper format
        embeddings_array = np.array([eval(e) if isinstance(e, str) else e for e in df["embedding_1024d"]])
        logger.info(f"✓ Generated embeddings shape: {embeddings_array.shape}")

        # Verify dimension
        if embeddings_array.shape[1] != 1024:
            logger.error(f"Unexpected embedding dimension: {embeddings_array.shape[1]} (expected 1024)")
            return 1

        logger.info(f"✓ Verified: All embeddings are 1024D")

        # Save final results
        df.to_csv(config.output_file)

        # Remove checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        elapsed_time = time.time() - start_time
        logger.info(f"✓ Successfully processed {len(df)} rows in {elapsed_time:.2f} seconds")
        logger.info(f"✓ Results saved to {config.output_file}")
        logger.info(f"✓ Embedding shape: {embeddings_array.shape}")
        logger.info(f"✓ Mean magnitude: {np.mean(np.linalg.norm(embeddings_array, axis=1)):.4f}")

        # Save embeddings as pickle for later use
        pkl_path = config.output_file.with_suffix(".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({
                "embeddings": embeddings_array,
                "texts": texts,
                "model": config.model_name,
                "dimension": 1024
            }, f)
        logger.info(f"✓ Pickled embeddings saved to {pkl_path}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        df.to_csv(checkpoint_path)
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        exit(130)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        exit(1)
