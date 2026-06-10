# txtai Deep Dive — Adopt, Extend, or Rip?

**Question:** Retire emb and adopt txtai (actively-developed, harness their ongoing work)? Or extend it, or rip out its best parts?
**Date:** 2026-06-10 | **Repo:** cloned to `~/Projects/best/txtai` (v9.10.0, 265 files / 31K LOC Python)
**Method:** Explore-agent architecture map (`.model-review/txtai-deepdive-map.md`) + first-hand verification of load-bearing claims + live smoke test (1K docs, hybrid + SQL + external backend) in `~/Projects/best/txtai/.scratch/`

## Verdict: DON'T ADOPT. Keep emb. Cherry-pick 2-3 parts. Watch as break-glass fallback.

The reframe that settles it: **"harness their ongoing work" is already how emb works.** emb is 2,200 LOC of glue over sentence-transformers (HuggingFace), SQLite FTS5, numpy, and google-genai — all actively developed by large teams. txtai is 31K LOC of one solo maintainer's glue over the *same engines* (1,872 of ~1,900 commits are davidmezzetti [DATA: git shortlog]). Adopting it swaps four big-team dependencies for one bus-factor-1 dependency that wraps them.

## Maintenance profile

| | emb | txtai |
|---|---|---|
| LOC owned | 2,200 | 31K (theirs) + a policy wrapper (ours) |
| Activity | as-needed | 80 commits / last 90 days — genuinely active [DATA] |
| Bus factor | 1 (Markus + agents) | **1 (davidmezzetti, 98.6% of commits)** [DATA] |
| Underlying engines | sentence-transformers, FTS5, numpy, google-genai | same + FAISS, NLTK, torch (required in base) |
| Base install | ~similar (torch via ST) | 614MB venv [DATA: probe] |
| License | MIT | Apache-2.0 |

## Smoke test — the decisive finding

1K-doc index, MiniLM, `content=True, hybrid=True` [DATA: `.scratch/smoke.py`, `smoke2.py`]:

```
hybrid search:                                 works, sane results
plain SQL filter:                              works
similar('optimization') AND source='papers':   0 results        ← !!
similar('optimization', 500) AND source=...:   3 results (correct)
```

**txtai's SQL+vector filtering is post-filter over the ANN candidate set.** With default candidate depth, metadata filters can silently wipe out *all* results; the documented user remedy is manual overfetch (`similar('query', 500)`). This is **exactly phenome's historical `fetch_k = top_k*5` bug shipped as designed behavior** — the thing emb's Phase-3 plan eliminates via true pre-filtering (emb filters the full candidate set *before* dense ranking; exact at 72K scale). On the single feature that matters most to phenome's agent queries (filtered semantic search), emb is *correct* and txtai is *truncation-prone*.

## Feature map (verified, with mapper corrections)

| emb feature | txtai equivalent | Notes |
|---|---|---|
| Hybrid BM25 + RRF | **NATIVE, richer** | 3 fusion methods: RRF, convex, log-odds (`scoring/`) |
| Cross-encoder rerank | **NATIVE** (mapper said missing — FALSE; verified `pipeline/text/reranker.py`, `crossencoder.py`) | same overfetch×10-then-rescore design as emb |
| ST backend | NATIVE | |
| Ollama backend | PARTIAL | via External transform or litellm |
| Gemini backend (google-genai direct) | **MISSING** | API route is **litellm** (`vectors/dense/litellm.py`) — the dependency evicted from emb in a9a8b7b (supply-chain incident). Clean alternative: External backend + our own google-genai fn |
| Content-hash embedding cache | **MISSING** | upsert re-embeds unchanged docs; "user must track content hash manually" [SOURCE: map §8] |
| Filters BEFORE ranking | **MISSING** | post-filter, see smoke test |
| Per-source freshness half-life | **MISSING** | no time-decay hook; workaround = post-process [SOURCE: map §6] |
| Spreading-activation boost | PARTIAL | semantic graph (NetworkX/openCypher) is a *separate query surface*, not integrated into ranking [SOURCE: map §7] |
| All-pairs band similarity (anki) | MISSING | |
| CAG / read-stage | NATIVE-ish | RAG pipeline, 5 LLM backends |
| Split index (jsonl + mmap npy) | DIFFERENT | FAISS + SQLite + config dir; fine but not mmap-simple |
| Multimodal | PARTIAL | CLIP images; no Gemini-2; External backend accepts precomputed vectors |
| MCP/API server | **NATIVE — emb lacks this** | FastAPI + fastapi-mcp |

Net: txtai's NATIVE wins are things emb already has working (hybrid, rerank, RAG). emb's differentiators (pre-filter exactness, content-hash cache, freshness, spreading activation in ranking, google-genai-direct) are MISSING or PARTIAL in txtai. An "extend" architecture (emb-as-policy-layer-on-txtai) would have to fork the search internals to fix post-filtering — a wrapper fighting the engine.

## Rip-out candidates (the actual value)

1. **Fusion math** (`scoring/` convex combination + log-odds alongside RRF) — small, drop-in upgrade to emb's RRF-only fusion. Worth taking in Phase 3.
2. **SPLADE learned-sparse** — txtai's implementation is a reference if/when emb's deferred SPLADE trigger fires.
3. **fastapi-mcp server pattern** — emb's plan already wants an agent-facing surface; txtai's API module is the design to copy (pattern, not code).
4. FAISS/quantization config patterns — reference for the deferred-ANN trigger (>500K entries).

## Break-glass condition (recorded for future sessions)

Adopt txtai only if: emb's maintenance burden materially grows (e.g., multimodal backend churn) AND txtai gains pre-filtering or phenome stops needing filtered queries AND the litellm route is replaced/acceptable. Until then: cherry-pick, don't migrate.

<!-- Rejected: full adoption (post-filter truncation = phenome's worst historical bug as design; bus factor 1; litellm dependency); emb-on-txtai extension (requires forking search internals to fix filtering). -->
