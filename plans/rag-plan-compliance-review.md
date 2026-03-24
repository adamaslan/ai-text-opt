# RAG Revamp Plan - Python Guidelines Compliance Review

This document identifies areas where the RAG revamp plan does not comply with Python Development Guidelines 2.

---

## ✅ Compliant Areas

### 1. Dependency Injection
All classes properly accept dependencies rather than creating them:
```python
class PgVectorStore:
    def __init__(self, config: PgVectorConfig):
        self.config = config  # ✅ Injected, not created internally
```

### 2. Protocols and Abstract Base Classes
Proper use of Protocol for interface definitions:
```python
class VectorStore(Protocol):  # ✅ Correct Protocol usage
    @abstractmethod
    async def search(...) -> List[SearchResult]: ...
```

### 3. Resource Management
Correct use of async context managers:
```python
@asynccontextmanager
async def get_client(self):  # ✅ Proper context manager
    async with httpx.AsyncClient(timeout=self.config.timeout) as client:
        yield client
```

### 4. Type Hints
Most signatures have proper type hints:
```python
async def search_similar(
    self,
    embedding: np.ndarray,
    dimension: int,
    limit: int = 5,
    threshold: float = 0.3,
) -> List[VectorSearchResult]:  # ✅ Fully typed
```

### 5. Async/Await Patterns
Proper async implementation:
```python
async def generate(self, prompt: str) -> str:  # ✅ Async where needed
    client = await self._get_client()
    response = await client.post(...)
```

---

## ❌ Non-Compliant Areas

### 1. Immutability Violations

**Issue**: Dataclasses are not frozen, violating "Favor Immutability" principle.

#### MistralConfig
```python
# ❌ Current (mutable)
@dataclass
class MistralConfig:
    api_key: str
    model: str = "mistral-small-latest"
    base_url: str = "https://api.mistral.ai/v1"
    max_tokens: int = 512
    temperature: float = 0.7

# ✅ Should be (immutable)
@dataclass(frozen=True)
class MistralConfig:
    api_key: str
    model: str = "mistral-small-latest"
    base_url: str = "https://api.mistral.ai/v1"
    max_tokens: int = 512
    temperature: float = 0.7
```

#### PgVectorConfig
```python
# ❌ Current
@dataclass
class PgVectorConfig:
    connection_string: str
    pool_min_size: int = 2
    pool_max_size: int = 10

# ✅ Should be
@dataclass(frozen=True)
class PgVectorConfig:
    connection_string: str
    pool_min_size: int = 2
    pool_max_size: int = 10
```

#### VectorSearchResult
```python
# ❌ Current
@dataclass
class VectorSearchResult:
    id: int
    text: str
    score: float
    metadata: dict
    source_file: str

# ✅ Should be
@dataclass(frozen=True)
class VectorSearchResult:
    id: int
    text: str
    score: float
    metadata: dict  # Consider using tuple or frozendict
    source_file: str
```

#### SearchResult
```python
# ❌ Current
@dataclass
class SearchResult:
    text: str
    score: float
    source: str
    metadata: Dict
    rank: int = 0

# ✅ Should be
@dataclass(frozen=True)
class SearchResult:
    text: str
    score: float
    source: str
    metadata: Dict  # Should be immutable type
    rank: int = 0
```

#### LLMResponse
```python
# ❌ Current
@dataclass
class LLMResponse:
    content: str
    provider: LLMProvider
    model: str
    usage: Optional[dict] = None

# ✅ Should be
@dataclass(frozen=True)
class LLMResponse:
    content: str
    provider: LLMProvider
    model: str
    usage: Optional[dict] = None  # Consider frozen type
```

#### GraphNode and GraphEdge
```python
# ❌ Current
@dataclass
class GraphNode:
    id: str
    label: str
    node_type: str
    properties: Dict = field(default_factory=dict)

# ✅ Should be
@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    node_type: str
    properties: Dict = field(default_factory=dict)  # Consider tuple of tuples
```

#### AppConfig
```python
# ❌ Current
@dataclass
class AppConfig:
    project_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    google_api_key: str = ""
    # ... mutable

# ✅ Should be
@dataclass(frozen=True)
class AppConfig:
    project_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    google_api_key: str = ""
    # ... immutable
```

**Fix**: Add `frozen=True` to ALL dataclasses representing data/config.

---

### 2. Missing Custom Exception Hierarchy

**Issue**: No domain-specific exceptions defined. Plan uses generic exceptions.

```python
# ❌ Current (in plan)
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found")  # Generic exception

# ✅ Should have custom hierarchy
class RAGError(Exception):
    """Base exception for RAG system errors."""
    pass

class ConfigurationError(RAGError):
    """Raised when configuration is invalid."""
    pass

class APIKeyMissingError(ConfigurationError):
    """Raised when required API key is missing."""
    def __init__(self, key_name: str):
        self.key_name = key_name
        super().__init__(f"{key_name} not found in environment")

class VectorStoreError(RAGError):
    """Raised when vector store operations fail."""
    pass

class DatabaseConnectionError(VectorStoreError):
    """Raised when database connection fails."""
    pass

class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""
    pass

class LLMProviderError(RAGError):
    """Raised when LLM provider is unavailable or fails."""
    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason
        super().__init__(f"LLM provider '{provider}' failed: {reason}")

class KnowledgeGraphError(RAGError):
    """Raised when knowledge graph operations fail."""
    pass
```

**Fix**: Create `exceptions.py` module with full hierarchy.

---

### 3. Missing or Incomplete Docstrings

**Issue**: Many classes/methods lack Google-style docstrings.

#### MistralClient.generate
```python
# ❌ Current (no docstring)
async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
    client = await self._get_client()
    # ...

# ✅ Should be
async def generate(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Generate completion from Mistral API.

    Args:
        prompt: User prompt to complete
        system_prompt: Optional system instruction to set behavior

    Returns:
        Generated text completion

    Raises:
        LLMProviderError: If API request fails or returns error
        ConfigurationError: If API key is invalid

    Examples:
        >>> client = MistralClient(config)
        >>> response = await client.generate("What is Python?")
        >>> print(response)
        'Python is a high-level programming language...'
    """
```

#### PgVectorStore.search_similar
```python
# ❌ Current (incomplete docstring)
async def search_similar(
    self,
    embedding: np.ndarray,
    dimension: int,
    limit: int = 5,
    threshold: float = 0.3,
) -> List[VectorSearchResult]:
    """
    Search for similar vectors using cosine distance.

    Args:
        embedding: Query embedding vector
        dimension: 384 or 768 (determines table)
        limit: Maximum results
        threshold: Minimum similarity score

    Returns:
        List of VectorSearchResult ordered by similarity
    """

# ✅ Should be (complete with Raises and Examples)
async def search_similar(
    self,
    embedding: np.ndarray,
    dimension: int,
    limit: int = 5,
    threshold: float = 0.3,
) -> List[VectorSearchResult]:
    """Search for similar vectors using cosine distance.

    Queries the pgvector database for vectors similar to the query embedding.
    Uses cosine similarity (1 - cosine_distance) for ranking.

    Args:
        embedding: Query embedding vector (must match dimension)
        dimension: Embedding dimension (384 or 768). Determines which table to query.
        limit: Maximum number of results to return. Must be positive.
        threshold: Minimum similarity score (0.0-1.0). Results below this are filtered.

    Returns:
        List of VectorSearchResult ordered by similarity score (descending).
        Empty list if no results meet threshold.

    Raises:
        VectorStoreError: If database query fails
        ValueError: If dimension is not 384 or 768
        ValueError: If embedding shape doesn't match dimension
        DatabaseConnectionError: If connection pool is exhausted

    Examples:
        >>> store = PgVectorStore(config)
        >>> await store.initialize()
        >>> embedding = np.random.randn(384)
        >>> results = await store.search_similar(embedding, dimension=384, limit=5)
        >>> for result in results:
        ...     print(f"{result.text[:50]}: {result.score:.3f}")
    """
```

#### FederatedSearchEngine.search
```python
# ❌ Current (minimal docstring)
async def search(
    self,
    query: str,
    limit: Optional[int] = None,
) -> List[SearchResult]:
    """
    Execute federated search across all stores.

    Args:
        query: User query string
        limit: Maximum results (default: config.final_limit)

    Returns:
        Fused and ranked results
    """

# ✅ Should be (complete)
async def search(
    self,
    query: str,
    limit: Optional[int] = None,
) -> List[SearchResult]:
    """Execute federated search across all registered vector stores.

    Queries all registered stores in parallel, generates appropriate embeddings
    for each dimension, and fuses results using Reciprocal Rank Fusion (RRF).

    The RRF algorithm assigns scores based on rank position across stores,
    which handles different similarity score scales naturally.

    Args:
        query: User query string to search for
        limit: Maximum number of final results. If None, uses config.final_limit.

    Returns:
        List of SearchResult objects ranked by RRF score. Results are
        deduplicated (same text from multiple stores appears once with
        combined score). Empty list if no results found or no stores registered.

    Raises:
        EmbeddingError: If query embedding generation fails
        VectorStoreError: If all stores fail (partial failures are tolerated)

    Examples:
        >>> engine = FederatedSearchEngine(config)
        >>> engine.register_store(faiss_store)
        >>> engine.register_store(pgvector_store)
        >>> results = await engine.search("What is Python?", limit=5)
        >>> for result in results:
        ...     print(f"[{result.source}] {result.text[:50]}: {result.score:.3f}")
    """
```

**Fix**: Add complete Google-style docstrings to ALL public methods.

---

### 4. Missing None Safety Checks

**Issue**: Code doesn't guard against None values explicitly in some places.

#### AppConfig.from_env
```python
# ❌ Current (doesn't check for None explicitly)
@classmethod
def from_env(cls, env_path: Optional[Path] = None) -> "AppConfig":
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    # ... loads config

# ✅ Should check None and validate
@classmethod
def from_env(cls, env_path: Optional[Path] = None) -> "AppConfig":
    """Load configuration from environment variables.

    Args:
        env_path: Optional path to .env file. If None, searches standard locations.

    Returns:
        AppConfig instance with validated settings

    Raises:
        ConfigurationError: If required variables are missing or invalid
    """
    if env_path is not None:
        if not env_path.exists():
            raise ConfigurationError(f".env file not found: {env_path}")
        load_dotenv(env_path)
    else:
        load_dotenv()

    # Build config
    config = cls(
        project_dir=Path(os.getenv("PROJECT_DIR", Path(__file__).parent)),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        # ...
    )

    # Validate
    errors = config.validate()
    if errors:
        raise ConfigurationError(f"Invalid configuration: {'; '.join(errors)}")

    return config
```

#### LLMRouter.generate
```python
# ❌ Current
async def generate(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
) -> LLMResponse:
    provider = provider or self.default_provider

    if provider not in self.adapters:
        raise ValueError(f"Provider {provider} not registered")

# ✅ Should be
async def generate(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
) -> LLMResponse:
    """Generate response using specified or default LLM provider.

    Args:
        prompt: User prompt
        system_prompt: Optional system instruction
        provider: LLM provider to use. If None, uses default.

    Returns:
        LLMResponse with generated content

    Raises:
        LLMProviderError: If provider not registered or generation fails
        ValueError: If no default provider configured
    """
    # Guard against None
    if provider is None:
        if self.default_provider is None:
            raise LLMProviderError(
                "none",
                "No default provider configured and no provider specified"
            )
        provider = self.default_provider

    if provider not in self.adapters:
        raise LLMProviderError(
            str(provider),
            f"Provider not registered. Available: {list(self.adapters.keys())}"
        )

    return await self.adapters[provider].generate(prompt, system_prompt)
```

**Fix**: Add explicit None checks with early returns and clear error messages.

---

### 5. Specific Exception Handling Not Shown

**Issue**: Plan shows generic exception handling in some places.

#### PgVectorStore._setup_connection
```python
# ❌ Current (not shown in plan, but typical pattern)
async def _setup_connection(self, conn: asyncpg.Connection) -> None:
    try:
        await register_vector(conn)
    except Exception as e:  # Too broad
        logger.error(f"Failed to register vector: {e}")
        raise

# ✅ Should be specific
async def _setup_connection(self, conn: asyncpg.Connection) -> None:
    """Register vector type for connection.

    Args:
        conn: asyncpg connection to configure

    Raises:
        DatabaseConnectionError: If vector extension registration fails
    """
    try:
        await register_vector(conn)
    except asyncpg.PostgresError as e:
        raise DatabaseConnectionError(
            f"Failed to register pgvector extension: {e}"
        ) from e
    except ImportError as e:
        raise ConfigurationError(
            "pgvector package not installed. Run: pip install pgvector"
        ) from e
```

#### MistralClient.generate
```python
# ❌ Current (in plan)
async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
    client = await self._get_client()

    response = await client.post(
        "/chat/completions",
        json={...},
    )
    response.raise_for_status()  # Raises generic HTTPStatusError

# ✅ Should be
async def generate(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Generate completion from Mistral API."""
    client = await self._get_client()

    try:
        response = await client.post(
            "/chat/completions",
            json={...},
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            raise LLMProviderError(
                "mistral",
                "Authentication failed. Check MISTRAL_API_KEY"
            ) from e
        elif e.response.status_code == 429:
            raise LLMProviderError(
                "mistral",
                "Rate limit exceeded. Please retry later"
            ) from e
        elif e.response.status_code >= 500:
            raise LLMProviderError(
                "mistral",
                f"Mistral API server error: {e.response.status_code}"
            ) from e
        else:
            raise LLMProviderError(
                "mistral",
                f"HTTP error {e.response.status_code}: {e.response.text}"
            ) from e
    except httpx.TimeoutException as e:
        raise LLMProviderError(
            "mistral",
            f"Request timed out after {self.config.timeout}s"
        ) from e
    except httpx.NetworkError as e:
        raise LLMProviderError(
            "mistral",
            f"Network error: {e}"
        ) from e

    data = response.json()
    return data["choices"][0]["message"]["content"]
```

**Fix**: Replace generic exception handling with specific exception types.

---

### 6. Early Returns Not Consistently Applied

**Issue**: Some validation logic uses nested ifs instead of early returns.

#### AppConfig.validate
```python
# ❌ Current (nested structure)
def validate(self) -> List[str]:
    errors = []

    if not self.google_api_key and not self.mistral_api_key:
        errors.append("At least one LLM API key required")

    if self.default_llm_provider == "gemini" and not self.google_api_key:
        errors.append("GOOGLE_API_KEY required when default is 'gemini'")

    if self.use_pgvector:
        if not self.supabase_db_url and not self.neon_db_url:
            errors.append("Database URL required when USE_PGVECTOR is true")

    return errors

# ✅ Should use early returns for complex validation
def validate(self) -> None:
    """Validate configuration.

    Raises:
        ConfigurationError: If any validation check fails
    """
    # Check at least one LLM is configured
    if not self.google_api_key and not self.mistral_api_key:
        raise ConfigurationError(
            "At least one LLM API key required (GOOGLE_API_KEY or MISTRAL_API_KEY)"
        )

    # Check default provider has API key
    if self.default_llm_provider == "gemini":
        if not self.google_api_key:
            raise ConfigurationError(
                "GOOGLE_API_KEY required when DEFAULT_LLM_PROVIDER='gemini'"
            )
    elif self.default_llm_provider == "mistral":
        if not self.mistral_api_key:
            raise ConfigurationError(
                "MISTRAL_API_KEY required when DEFAULT_LLM_PROVIDER='mistral'"
            )
    else:
        raise ConfigurationError(
            f"Invalid DEFAULT_LLM_PROVIDER: {self.default_llm_provider}. "
            f"Must be 'gemini' or 'mistral'"
        )

    # Check database if pgvector enabled
    if self.use_pgvector:
        if not self.supabase_db_url and not self.neon_db_url:
            raise ConfigurationError(
                "Database URL required when USE_PGVECTOR=true. "
                "Set SUPABASE_DB_URL or NEON_DB_URL"
            )

    # Check embedding source files exist (for FAISS sources)
    for source in self.embedding_sources:
        if source.store_type == "faiss":
            if not source.file_path.exists():
                raise ConfigurationError(
                    f"Embedding file not found: {source.file_path}"
                )
```

**Fix**: Use early returns with explicit error messages.

---

### 7. Magic Numbers

**Issue**: Some configuration values are hardcoded.

#### PgVectorStore SQL query
```python
# ❌ Current (magic number in SQL)
query = f"""
SELECT
    id, text,
    1 - (embedding <=> $1::vector) as score,
    metadata, source_file
FROM {table}
WHERE 1 - (embedding <=> $1::vector) >= $2
ORDER BY embedding <=> $1::vector
LIMIT $3
"""

# ✅ Should use constants
# At module level
DEFAULT_SIMILARITY_OPERATOR = "<=>"  # Cosine distance
SIMILARITY_TO_SCORE_FORMULA = "1 - (embedding <=> $1::vector)"

# In method
query = f"""
SELECT
    id, text,
    {SIMILARITY_TO_SCORE_FORMULA} as score,
    metadata, source_file
FROM {table}
WHERE {SIMILARITY_TO_SCORE_FORMULA} >= $2
ORDER BY embedding {DEFAULT_SIMILARITY_OPERATOR} $1::vector
LIMIT $3
"""
```

#### FederatedSearchEngine RRF constant
```python
# ❌ Current (magic number in config)
@dataclass
class FederatedSearchConfig:
    rrf_k: int = 60  # What does 60 mean?

# ✅ Should document and use named constant
# At module level
RRF_DEFAULT_K = 60  # Standard RRF constant from literature
# Lower values give more weight to top results
# Higher values smooth out rank differences

@dataclass(frozen=True)
class FederatedSearchConfig:
    """Configuration for federated search with RRF fusion."""
    rrf_k: int = RRF_DEFAULT_K
    per_store_limit: int = 10
    final_limit: int = 5
    min_score: float = 0.3
```

**Fix**: Replace magic numbers with named constants with comments.

---

### 8. Missing Logging Best Practices

**Issue**: Inconsistent logging patterns.

```python
# ❌ Current (inconsistent)
logger.info("Loaded graph: {self.graph.number_of_nodes()} nodes")  # f-string in wrong place

# ✅ Should use % formatting for performance
logger.info(
    "Loaded graph: %d nodes, %d edges",
    self.graph.number_of_nodes(),
    self.graph.number_of_edges()
)

# ❌ Current (missing context)
logger.error(f"Failed to load graph: {e}")

# ✅ Should include context and use exception logging
logger.exception(
    "Failed to load knowledge graph from %s",
    self.config.persistence_path,
    exc_info=True
)
```

**Fix**: Use `logger.info("message %s", var)` format consistently.

---

## Summary of Required Changes

### High Priority (Breaks guidelines)
1. ✅ Add `frozen=True` to all dataclasses representing data/config
2. ✅ Create custom exception hierarchy in `exceptions.py`
3. ✅ Add complete Google-style docstrings to all public methods
4. ✅ Add explicit None safety checks with early returns
5. ✅ Replace generic `Exception` catching with specific types

### Medium Priority (Best practices)
6. ✅ Replace magic numbers with named constants
7. ✅ Improve validation to use early returns
8. ✅ Standardize logging format (use % formatting)

### Low Priority (Nice to have)
9. Consider using `tuple` instead of `list` where order matters but mutability isn't needed
10. Add more inline comments for complex logic (RRF algorithm, etc.)

---

## Implementation Checklist

When implementing the RAG revamp, ensure:

- [ ] All config dataclasses use `@dataclass(frozen=True)`
- [ ] All result dataclasses use `@dataclass(frozen=True)`
- [ ] Custom exception hierarchy defined in `exceptions.py`
- [ ] All exceptions raised are domain-specific
- [ ] All public methods have complete Google-style docstrings
- [ ] All Optional parameters have None checks
- [ ] Early returns used for validation and error cases
- [ ] Magic numbers replaced with named constants
- [ ] Logging uses % formatting, not f-strings
- [ ] Type hints on all public APIs
- [ ] Context managers used for all resources
- [ ] Specific exception handling (no bare `except Exception`)
