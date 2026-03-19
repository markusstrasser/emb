# emb

Embed, index, and search text corpora. Dense + BM25 hybrid search, RRF fusion, cross-encoder reranking, freshness weighting, spreading activation.

## Install

```bash
uv tool install git+https://github.com/markusstrasser/emb
```

## CLI

```bash
# Embed JSONL into a searchable index (split format: JSONL + mmap numpy)
emb embed input.jsonl -o my_index/

# Search
emb search my_index/ "your query"
emb search my_index/ "your query" --hybrid --rerank -k 20

# Interactive search REPL
emb search my_index/ --interactive

# Index info
emb info my_index/

# Convert between formats
emb convert legacy.json split_dir/    # JSON → split
emb convert split_dir/ output.json    # split → JSON

# Legacy JSON format (not recommended for large indices)
emb embed input.jsonl -o index.json --format json
```

### Input format

One JSON object per line (`input.jsonl`):

```json
{"id": "doc1", "text": "Your document text", "source": "blog", "title": "My Post", "date": "2025-01-15"}
{"id": "doc2", "text": "Another document", "source": "notes", "title": "Meeting Notes"}
```

Required: `id`, `text`. Optional: `source`, `title`, `date`, `metadata` (dict).

## Python API

```python
from emb.search import SearchEngine

# Load from split index directory
engine = SearchEngine("my_index/")

# Basic search
results = engine.search("machine learning", top_k=10)

# Hybrid search (dense + BM25) with provenance tracking
results = engine.search(
    "machine learning",
    hybrid=True,
    rerank=True,
    freshness_weight=0.3,
    provenance=True,
)
for r in results:
    print(f"[{r['source']}] {r['title']} sim={r['similarity']:.3f}")
    print(f"  provenance: {r['provenance']}")
```

### Build from pre-loaded data

```python
from emb.search import SearchEngine
from emb.schema import Entry
import numpy as np

entries = [Entry(id="1", text="hello", source="test")]
embeddings = np.random.randn(1, 768).astype(np.float32)

engine = SearchEngine.from_data(entries, embeddings)
```

### Extension points

```python
# Custom entry filter
results = engine.search("query", entry_filter=lambda e: e.metadata.get("channel") == "authored")

# Custom dedup key (e.g. group podcast chunks by episode)
results = engine.search("query", dedup_key=lambda r: r['entry'].metadata.get("video_id"))

# Post-processors (e.g. spreading activation)
from emb.search import NeighborIndex

neighbors = NeighborIndex(engine.entries, key_extractor=lambda e: [f"repo:{e.metadata.get('repo')}"])
results = engine.search("query", post_processors=[neighbors.as_post_processor(engine.entries)])

# Source group expansion
from emb.search import expand_sources
groups = {"health": {"research", "docs", "healthkit"}}
sources = expand_sources({"health"}, groups)  # → {"research", "docs", "healthkit"}
```

### Split index format

```python
from emb.index import write_index, read_index, check_staleness

# Write
write_index(entries, embeddings, "my_index/", {"embedding_model": "gte-modernbert-base"})

# Read (with mmap for zero-copy embeddings)
entries, embeddings, metadata = read_index("my_index/")

# Filtered read (only loads matching entries)
entries, embeddings, metadata = read_index("my_index/", sources={"git", "docs"})

# Check if source data is newer than index
stale = check_staleness("my_index/", "data/", pattern="*_parsed.json")
```

### Embedding engine

```python
from emb.embed import EmbeddingEngine
from emb.cache import EmbeddingCache

engine = EmbeddingEngine(model="Alibaba-NLP/gte-modernbert-base")
cache = EmbeddingCache.load("cache_dir/", dim=768)

entries = engine.embed_entries(entries, cache=cache, checkpoint_dir="cache_dir/")
```

## Index formats

**Split (default, recommended):**
```
my_index/
  metadata.json      # model, dim, sources, generated_at
  entries.jsonl       # one JSON per line (no embeddings)
  embeddings.npy      # float32 (N × dim), mmap-able
```

**JSON (legacy):** Single file with entries + embeddings inline. Works but 4x larger and can't be mmap'd.

## Models

Default: `Alibaba-NLP/gte-modernbert-base` (768d, 8K context). Downloaded on first use with confirmation prompt.

Reranker (optional, with `--rerank`): `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`.
