# emb

Embed, index, and search text (and multimodal) corpora. Dense + BM25 hybrid search, RRF/convex fusion, cross-encoder reranking, freshness weighting, spreading activation, all-pairs similarity, and a long-context read stage.

Backends: `sentence-transformers` (local, default), `ollama`, and `gemini` (multimodal — text + image/audio via Gemini Embedding 2). Every vector is L2-normalized regardless of backend; caches are namespaced by model slug so same-dimension models in different spaces never collide.

## Install

```bash
uv tool install git+https://github.com/markusstrasser/emb
```

## CLI

```bash
# Embed JSONL into an immutable generation and atomically activate it
emb embed input.jsonl -o my_index/

# Search
emb search my_index/ "your query"
emb search my_index/ "your query" --hybrid --rerank -k 20

# Interactive search REPL
emb search my_index/ --interactive

# Index info
emb info my_index/

# Find all high-similarity PAIRS within a corpus (duplicates, interference)
emb pairs my_index/ --threshold 0.85 -o pairs.jsonl
emb pairs my_index/ --threshold 0.50 --max-threshold 0.80   # interference band

# Locate-then-read: hybrid search, then span a long-context model over only the hits
emb read my_index/ "what did I conclude about X?" --top-k 200   # cost-preflighted

# Convert a legacy monolithic-JSON index to split format (read path)
emb convert legacy.json split_dir/    # JSON → split
emb convert split_dir/ output.json    # split → JSON
```

`emb embed` writes an immutable generation under `generations/` and atomically
activates it through `current.json`. Multimodal indexes (gemini backend) are built
via the Python API (`EmbeddingEngine.embed_media`); see below.

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

# Load the active, validated generation
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

### Immutable generation format

```python
from emb import read_generation, write_generation
from emb.generation import default_text_recipe

generation = write_generation(
    "my_index/", entries, embeddings,
    generation_id="20260715T120000Z",
    lane="text",
    embedding_model="Alibaba-NLP/gte-modernbert-base",
    recipes=[default_text_recipe("Alibaba-NLP/gte-modernbert-base", 768)],
    source_manifest={"fingerprints": {"docs": "sha256:..."}},
    generator={
        "app_git_commit": "...",
        "emb_git_commit": "...",
        "generator_version": "personal-text-v1",
    },
)

# Resolves current.json once, validates every digest/shape/recipe invariant,
# and pins one immutable generation.
entries, embeddings, manifest = read_generation("my_index/")
```

`generation.json` is the sole metadata authority. Writers stage
`<generation-id>.building` beside the final directory, fsync and validate it,
then atomically publish and activate it. `current.json` contains only the safe
generation ID and expected manifest digest. Activation of a prior generation is
an instant rollback; failed builds never alter the active pointer.

### Legacy split index conversion API

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

**Generation (default):**
```
my_index/
  generations/<generation-id>/
    generation.json   # sole identity, recipe, provenance, and digest authority
    entries.jsonl     # one JSON per line (no embeddings)
    embeddings.npy    # float32 (N × dim), mmap-able
  current.json        # atomically replaced activation pointer
```

**Split/JSON (legacy):** Offline conversion inputs retained by `emb convert` and
legacy readers. `emb embed` no longer writes either mutable format.

## Multimodal (Gemini Embedding 2)

```python
from emb.embed import EmbeddingEngine

engine = EmbeddingEngine(model="gemini-embedding-2-preview")  # needs emb[gemini] + GEMINI_API_KEY

# Text (each item is its own Content — required; a bare list[str] would fuse to one vector)
vecs = engine.embed_texts(["front/back card text", "another"])

# Multimodal: each item is (description, [(data_or_path, mime), ...]) fused into one embedding
M = engine.embed_media([
    ("stop sign card front/back", [("card.png", "image/png")]),  # text + image
    ("", [("diagram.png", "image/png")]),                         # image-only (e.g. occlusion)
])
```

## Models

Default: `Alibaba-NLP/gte-modernbert-base` (768d, 8K context, local). Downloaded on first use with confirmation prompt.

Multimodal: `gemini-embedding-2-preview` (768d, text + image/audio) via `emb[gemini]`.

Reranker (optional, with `--rerank`): `Alibaba-NLP/gte-reranker-modernbert-base` (149M,
passage-windowed MaxP). **Opt-in after eval, not free insurance** — on the intel corpus no
rerank config beat hybrid-alone (it demoted more than it rescued); validate on your corpus
before enabling. Evidence: `docs/HANDOFF.md` §3, `evals/retrieval_backend_bakeoff/EXPERIMENT.md` §4c.
