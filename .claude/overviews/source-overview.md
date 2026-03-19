```
INDEX
- CORE SCHEMA: src/emb/schema.py
- I/O: src/emb/io.py
- CACHING: src/emb/cache.py
- CHUNKING: src/emb/chunking.py
- EMBEDDING: src/emb/embed.py
- CONTEXTUALIZATION: src/emb/contextualize.py
- SEARCH: src/emb/search.py
- CLI: src/emb/cli.py
- PROJECT: pyproject.toml
```

### CORE SCHEMA

**File**: `src/emb/schema.py`
Defines the core data structures for entries and indices.
- **Key Classes**:
  - `Entry`: Dataclass for a single text unit. Requires `id` and `text`. Optional fields include `source`, `title`, `date`, `metadata`, `content_hash`, and `embedding`.
  - `Index`: A container for a list of `Entry` objects and index-level metadata.
- **Dependencies**: None.

### I/O

**File**: `src/emb/io.py`
Handles reading and writing entries to/from JSONL files.
- **Key Functions**:
  - `read_jsonl()`: Reads `Entry` objects from a JSONL file or stdin, with validation.
  - `write_jsonl()`: Writes a list of `Entry` objects to a JSONL file or stdout.
- **Dependencies**: `emb.schema`.

### CACHING

**File**: `src/emb/cache.py`
Implements a persistent, numpy-backed cache for text embeddings.
- **Key Classes**:
  - `EmbeddingCache`: Manages a mapping from content hashes to embedding vectors. Saves to `index.json` (hashes) and `vectors.npy` (embeddings). Handles dimension mismatches.
- **Dependencies**: `numpy`.

### CHUNKING

**File**: `src/emb/chunking.py`
Provides sentence-aware text chunking strategies.
- **Key Functions**:
  - `chunk_text()`: Splits long text into overlapping chunks, respecting sentence boundaries.
  - `chunk_entries()`: Applies `chunk_text` to `Entry` objects, creating child entries with `parent_id` metadata.
  - `multiscale_chunk_entries()`: Chunks entries at multiple token scales (e.g., 200, 500), tagging each chunk with its scale for advanced retrieval.
- **Dependencies**: `emb.schema`.

### EMBEDDING

**File**: `src/emb/embed.py`
Core engine for generating text embeddings with multiple backends.
- **Key Classes**:
  - `EmbeddingEngine`: Orchestrates embedding generation.
- **Key Features**:
  - **Backends**: Supports `sentence-transformers` and `ollama`.
  - **Caching**: Integrates with `EmbeddingCache` to avoid re-embedding.
  - **Checkpointing**: Periodically saves the cache during long embedding jobs.
  - **Hashing**: Uses `compute_content_hash` on entry text to create cache keys.
- **Dependencies**: `emb.schema`, `emb.cache`, `numpy`, `sentence-transformers`.

### CONTEXTUALIZATION

**File**: `src/emb/contextualize.py`
Enhances entries by prepending an LLM-generated context summary.
- **Key Functions**:
  - `contextualize_batch()`: Asynchronously calls an LLM (via `litellm`) for a batch of entries to generate and prepend context.
  - `contextualize_sync()`: Synchronous wrapper for the above.
- **Key Features**:
  - Skips already contextualized entries.
  - Uses bounded concurrency for API calls.
- **Dependencies**: `emb.schema`, `litellm`.

### SEARCH

**File**: `src/emb/search.py`
A multi-faceted search engine combining dense and sparse retrieval methods.
- **Key Classes**:
  - `SearchEngine`: Loads an index and performs searches.
- **Key Features**:
  - **Dense Search**: Cosine similarity search using `numpy`.
  - **BM25 Search**: Lazy-builds an in-memory `sqlite-fts5` index for keyword search.
  - **Hybrid Fusion**: Combines dense and BM25 ranks using Reciprocal Rank Fusion (RRF).
  - **Reranking**: Uses a `sentence-transformers` `CrossEncoder` for more accurate final ranking.
  - **Freshness Weighting**: Applies exponential decay based on entry date.
  - **Chunk Deduplication**: Groups results by `parent_id` to show only the best-matching chunk from a document.
- **Dependencies**: `emb.schema`, `numpy`, `sentence-transformers`, `python-dateutil`.

### CLI

**File**: `src/emb/cli.py`
Provides the command-line interface for all library functions.
- **Key Commands**:
  - `embed`: Creates an index from JSONL, with options for chunking and caching.
  - `search`: Queries an index, with flags for hybrid, reranking, freshness, and an interactive mode.
  - `contextualize`: Runs the contextualization process on a JSONL file.
  - `merge`: Combines multiple indices into one, deduplicating by entry ID.
  - `info`: Displays statistics about an index.
- **Dependencies**: All other `emb.*` modules, `typer`, `rich`.

### PROJECT

**File**: `pyproject.toml`
Defines project metadata, dependencies, and entry points.
- **Key Sections**:
  - `[project]`: Core dependencies like `numpy`, `sentence-transformers`, `typer`.
  - `[project.optional-dependencies]`: Separates dependencies for features like reranking (`rerank`) and contextualization (`contextualize`).
  - `[project.scripts]`: Defines the `emb` command-line tool.

### TESTING

**Files**: `tests/*.py`
Provides unit and integration tests for each module in `src/emb/`.
- **Key Files**:
  - `test_embed.py`: Tests embedding engine, caching, and hashing.
  - `test_chunking.py`: Tests sentence-aware and multi-scale chunking logic.
  - `test_search.py`: Tests dense, hybrid, filtering, and ranking features.
  - `test_cli.py`: Tests CLI commands like `merge` and `info`.
- **Dependencies**: `pytest`.
