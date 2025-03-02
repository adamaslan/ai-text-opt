import marimo

__generated_with = "0.11.13"
app = marimo.App()


@app.cell
def _():
    #!/usr/bin/env python3
    """
    Text Embedding Generator

    This script generates and combines embeddings from multiple models (Ollama API and SentenceTransformer)
    for text data, optimized for modern Python environments in 2025.

    Features:
    - Asynchronous and parallel processing
    - GPU acceleration
    - Robust error handling and logging
    - Configurable parameters via environment variables or config files
    - Progress tracking
    """

    import os
    import pickle
    import time
    import logging
    import asyncio
    import json
    from dataclasses import dataclass, field
    from functools import partial
    from pathlib import Path
    from typing import Dict, List, Optional, Tuple, Union, Any

    import httpx
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from tqdm.asyncio import tqdm_asyncio
    import yaml
    from contextlib import asynccontextmanager

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("embedding_generator.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("embedding_generator")

    @dataclass
    class EmbeddingConfig:
        """Configuration for embedding generation process."""
        input_file: Path
        output_file: Path
        ollama_model: str = "distilroberta"
        ollama_url: str = "http://localhost:11434/api/embeddings"
        sentence_transformer_model: str = "all-MiniLM-L6-v2"
        batch_size: int = 32
        max_retries: int = 3
        retry_delay: float = 1.0
        use_gpu: bool = True
        num_workers: int = 4
        timeout: int = 60  # HTTP request timeout in seconds
        cache_dir: Optional[Path] = None
        save_interval: int = 100  # Save progress every N items
    
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
                input_file=Path(os.getenv("EMBEDDING_INPUT_FILE", "input.pkl")),
                output_file=Path(os.getenv("EMBEDDING_OUTPUT_FILE", "embeddings.pkl")),
                ollama_model=os.getenv("EMBEDDING_OLLAMA_MODEL", "distilroberta"),
                ollama_url=os.getenv("EMBEDDING_OLLAMA_URL", "http://localhost:11434/api/embeddings"),
                sentence_transformer_model=os.getenv("EMBEDDING_SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"),
                batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
                max_retries=int(os.getenv("EMBEDDING_MAX_RETRIES", "3")),
                retry_delay=float(os.getenv("EMBEDDING_RETRY_DELAY", "1.0")),
                use_gpu=os.getenv("EMBEDDING_USE_GPU", "true").lower() in ("true", "1", "yes"),
                num_workers=int(os.getenv("EMBEDDING_NUM_WORKERS", "4")),
                timeout=int(os.getenv("EMBEDDING_HTTP_TIMEOUT", "60")),
                cache_dir=Path(os.getenv("EMBEDDING_CACHE_DIR")) if os.getenv("EMBEDDING_CACHE_DIR") else None,
                save_interval=int(os.getenv("EMBEDDING_SAVE_INTERVAL", "100")),
            )


    class EmbeddingCache:
        """Cache for storing and retrieving embeddings."""
    
        def __init__(self, cache_dir: Optional[Path] = None):
            self.cache_dir = cache_dir
            self.in_memory_cache = {}
        
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                self.ollama_cache_path = cache_dir / "ollama_cache.pkl"
                self.st_cache_path = cache_dir / "sentence_transformer_cache.pkl"
            
                # Load existing caches if they exist
                self.ollama_cache = self._load_cache(self.ollama_cache_path)
                self.st_cache = self._load_cache(self.st_cache_path)
            else:
                self.ollama_cache = {}
                self.st_cache = {}
    
        def _load_cache(self, path: Path) -> Dict[str, np.ndarray]:
            """Load cache from disk if it exists."""
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache from {path}: {e}")
            return {}
    
        def save_cache(self) -> None:
            """Save caches to disk."""
            if not self.cache_dir:
                return
        
            try:
                with open(self.ollama_cache_path, "wb") as f:
                    pickle.dump(self.ollama_cache, f)
            
                with open(self.st_cache_path, "wb") as f:
                    pickle.dump(self.st_cache, f)
                
                logger.info(f"Caches saved to {self.cache_dir}")
            except Exception as e:
                logger.error(f"Failed to save caches: {e}")
    
        def get_ollama_embedding(self, text: str) -> Optional[np.ndarray]:
            """Get Ollama embedding from cache."""
            return self.ollama_cache.get(text)
    
        def set_ollama_embedding(self, text: str, embedding: np.ndarray) -> None:
            """Set Ollama embedding in cache."""
            self.ollama_cache[text] = embedding
    
        def get_st_embedding(self, text: str) -> Optional[np.ndarray]:
            """Get SentenceTransformer embedding from cache."""
            return self.st_cache.get(text)
    
        def set_st_embedding(self, text: str, embedding: np.ndarray) -> None:
            """Set SentenceTransformer embedding in cache."""
            self.st_cache[text] = embedding


    class EmbeddingGenerator:
        """Generates and combines embeddings from multiple models."""
    
        def __init__(self, config: EmbeddingConfig):
            self.config = config
            self.cache = EmbeddingCache(config.cache_dir)
        
            # Initialize SentenceTransformer
            device = "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device} for SentenceTransformer")
            self.st_model = SentenceTransformer(config.sentence_transformer_model, device=device)
        
            # For batch processing with sentence-transformers
            self.st_model.max_seq_length = 512  # Adjust as needed
    
        @asynccontextmanager
        async def get_client(self):
            """Context manager for httpx client."""
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                yield client
    
        async def generate_ollama_embedding(self, text: str, client: httpx.AsyncClient) -> np.ndarray:
            """Generate embedding using Ollama API."""
            cached = self.cache.get_ollama_embedding(text)
            if cached is not None:
                return cached
        
            for attempt in range(self.config.max_retries):
                try:
                    response = await client.post(
                        self.config.ollama_url,
                        json={"model": self.config.ollama_model, "prompt": text}
                    )
                    response.raise_for_status()
                    data = response.json()
                    embedding = np.array(data.get("embedding", []))
                
                    if embedding.size == 0:
                        raise ValueError("Empty embedding received from Ollama API")
                
                    self.cache.set_ollama_embedding(text, embedding)
                    return embedding
            
                except Exception as e:
                    if attempt < self.config.max_retries - 1:
                        delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Ollama API error (attempt {attempt+1}/{self.config.max_retries}): {e}. Retrying in {delay:.2f}s")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Failed to generate Ollama embedding after {self.config.max_retries} attempts: {e}")
                        # Return zero vector of appropriate size on failure
                        # Using a size of 768 as a default for many transformer-based models
                        return np.zeros(768)
    
        def generate_sentence_transformer_embedding(self, text: str) -> np.ndarray:
            """Generate embedding using SentenceTransformer."""
            cached = self.cache.get_st_embedding(text)
            if cached is not None:
                return cached
        
            try:
                # SentenceTransformer automatically handles GPU acceleration if available
                embedding = self.st_model.encode(text, convert_to_numpy=True)
                self.cache.set_st_embedding(text, embedding)
                return embedding
            except Exception as e:
                logger.error(f"Failed to generate SentenceTransformer embedding: {e}")
                # Return zero vector of appropriate size
                return np.zeros(self.st_model.get_sentence_embedding_dimension())
    
        async def generate_combined_embedding(self, text: str, client: httpx.AsyncClient) -> np.ndarray:
            """Generate combined embedding from both models."""
            ollama_embedding = await self.generate_ollama_embedding(text, client)
            st_embedding = self.generate_sentence_transformer_embedding(text)
        
            # Combine embeddings (simple concatenation, can be extended with other methods)
            return np.concatenate([ollama_embedding, st_embedding])
    
        async def process_batch(self, texts: List[str], client: httpx.AsyncClient) -> List[np.ndarray]:
            """Process a batch of texts in parallel."""
            tasks = []
            for text in texts:
                task = self.generate_combined_embedding(text, client)
                tasks.append(task)
        
            return await asyncio.gather(*tasks)
    
        async def process_data(self, data: Dict[str, str]) -> Dict[str, np.ndarray]:
            """Process all text data and generate embeddings."""
            result = {}
            keys = list(data.keys())
            total_batches = (len(keys) + self.config.batch_size - 1) // self.config.batch_size
        
            async with self.get_client() as client:
                for i in range(0, len(keys), self.config.batch_size):
                    batch_keys = keys[i:i + self.config.batch_size]
                    batch_texts = [data[k] for k in batch_keys]
                
                    batch_result = await self.process_batch(batch_texts, client)
                
                    for key, embedding in zip(batch_keys, batch_result):
                        result[key] = embedding
                
                    # Save progress at intervals
                    if i > 0 and i % self.config.save_interval == 0:
                        self._save_checkpoint(result)
                        self.cache.save_cache()
        
            return result
    
        def _save_checkpoint(self, current_results: Dict[str, np.ndarray]) -> None:
            """Save current progress to a checkpoint file."""
            checkpoint_path = self.config.output_file.with_suffix(".checkpoint.pkl")
            try:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(current_results, f)
                logger.info(f"Checkpoint saved: {len(current_results)} items processed")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")


    async def main():
        """Main function to run the embedding generator."""
        # Load configuration (prioritize config file, fall back to env vars)
        config_path = Path("embedding_config.yaml")
        if config_path.exists():
            config = EmbeddingConfig.from_yaml(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            config = EmbeddingConfig.from_env()
            logger.info("Loaded configuration from environment variables")
    
        logger.info(f"Input file: {config.input_file}")
        logger.info(f"Output file: {config.output_file}")
    
        # Check if input file exists
        if not config.input_file.exists():
            logger.error(f"Input file not found: {config.input_file}")
            return 1
    
        # Load input data
        try:
            with open(config.input_file, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Loaded {len(data)} items from {config.input_file}")
        except Exception as e:
            logger.error(f"Failed to load input file: {e}")
            return 1
    
        # Check for checkpoint
        checkpoint_path = config.output_file.with_suffix(".checkpoint.pkl")
        result = {}
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "rb") as f:
                    result = pickle.load(f)
                logger.info(f"Resuming from checkpoint with {len(result)} items already processed")
            
                # Filter data to process only remaining items
                data = {k: v for k, v in data.items() if k not in result}
                logger.info(f"{len(data)} items remaining to process")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint, starting from scratch: {e}")
    
        # Initialize and run the embedding generator
        start_time = time.time()
        generator = EmbeddingGenerator(config)
    
        if not data:
            logger.info("No new data to process")
        else:
            logger.info(f"Starting embedding generation for {len(data)} items")
        
            # Process data
            new_results = await generator.process_data(data)
            result.update(new_results)
        
            # Save cache for future runs
            generator.cache.save_cache()
    
        # Save final results
        try:
            with open(config.output_file, "wb") as f:
                pickle.dump(result, f)
        
            # Remove checkpoint file if exists
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        
            elapsed_time = time.time() - start_time
            logger.info(f"Completed successfully. Processed {len(result)} items in {elapsed_time:.2f} seconds")
            logger.info(f"Results saved to {config.output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
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
    return (
        Any,
        Dict,
        EmbeddingCache,
        EmbeddingConfig,
        EmbeddingGenerator,
        List,
        Optional,
        Path,
        SentenceTransformer,
        Tuple,
        Union,
        asynccontextmanager,
        asyncio,
        dataclass,
        exit_code,
        field,
        httpx,
        json,
        logger,
        logging,
        main,
        np,
        os,
        partial,
        pickle,
        time,
        torch,
        tqdm_asyncio,
        yaml,
    )


if __name__ == "__main__":
    app.run()
